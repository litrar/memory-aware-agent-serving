"""Linux memory sampler that runs next to the agent workload.

Every `interval` seconds it writes one line to memory.jsonl, on the same clock as events.jsonl:
the llama-server process's memory and CPU use, system memory and CPU use, page faults and swap I/O
(cumulative counters from /proc), the server's slot state from /slots (on by default in llama-server),
NVIDIA GPU memory and utilization if nvidia-smi is present, and (if the server was started with
--metrics) its Prometheus metrics.
"""
import asyncio
import json
import os
import platform
import shutil
import subprocess
from urllib.parse import urlparse

import psutil

MiB = 2 ** 20
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}
# What the srv_*_mib process memory fields measure (Linux), saved in config.json with the host info
MEMORY_SOURCES = dict(
    rss="VmRSS: resident memory, including model weights mmap'ed from the GGUF file",
    anon="RssAnon: resident anonymous memory (heap, KV cache, compute buffers); excludes mmap'ed weights "
         "and anonymous pages swapped out",
    uss="Private_Clean + Private_Dirty (smaps_rollup): USS, pages mapped by this process only; "
        "includes mmap'ed weights when no other process maps the GGUF file",
    private_dirty="Private_Dirty (smaps_rollup): private pages the process wrote, which cannot be "
                  "dropped from RAM, only swapped",
    swap="Swap (smaps_rollup): the process's pages swapped out",
    peak_rss="VmHWM: the OS-tracked peak of VmRSS")


def _listens_on(proc, port):
    try:
        conns = proc.net_connections(kind="tcp") if hasattr(proc, "net_connections") else proc.connections(kind="tcp")
    except psutil.Error:
        return False
    return any(c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == port for c in conns)


def find_server_process(url, pid=None):
    """The llama-server process behind `url`, or None if it isn't running on this machine."""
    if pid:
        return psutil.Process(pid)
    u = urlparse(url)
    if u.hostname not in LOCAL_HOSTS:
        return None
    try:
        for c in psutil.net_connections(kind="tcp"):        # whoever is listening on the server port
            if c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == u.port and c.pid:
                return psutil.Process(c.pid)
    except (psutil.AccessDenied, psutil.NoSuchProcess):     # macOS needs root for the system-wide list
        pass
    candidates = [p for p in psutil.process_iter(["name"])
                  if any(s in (p.info["name"] or "").lower() for s in ("llama", "python")) and p.pid != os.getpid()]
    for p in candidates:                                    # ... but can list our own processes' sockets
        if _listens_on(p, u.port):
            return p
    return None


def describe_process(proc):
    """pid, name and command line (so the server's -c / -np flags end up in config.json)."""
    if proc is None:
        return None
    try:
        return dict(pid=proc.pid, name=proc.name(), cmdline=proc.cmdline())
    except psutil.Error:
        return dict(pid=proc.pid)


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout.strip()


def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def gpu_info():
    """GPUs on this machine (name, memory, driver), best effort; [] if none are found."""
    try:
        if shutil.which("nvidia-smi"):
            out = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
            return [dict(name=name.strip(), vram_mib=int(float(mem)), driver=drv.strip())
                    for name, mem, drv in (line.split(",") for line in out.splitlines())]
        if shutil.which("lspci"):
            return [dict(name=line.split(": ", 1)[1]) for line in _run(["lspci"]).splitlines()
                    if "VGA compatible controller" in line or "3D controller" in line]
    except (OSError, subprocess.SubprocessError, ValueError, KeyError):
        pass
    return []


def host_info():
    return dict(platform=platform.platform(), cpu=cpu_name(), logical_cpus=psutil.cpu_count(),
                ram_total_mib=round(psutil.virtual_memory().total / MiB), gpus=gpu_info(),
                process_memory_sources=MEMORY_SOURCES)


def _proc_status_kib(pid, keys):
    """Linux: fields of /proc/<pid>/status, in KiB."""
    out = {}
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            k, _, v = line.partition(":")
            if k in keys:
                out[k] = int(v.split()[0])
    return out


def _smaps_rollup_kib(pid):
    """Linux >= 4.14: the process's memory summed over all its mappings, in KiB. Unlike /proc/<pid>/status,
    reading it needs ptrace access, so it raises PermissionError for another user's process (sudo, Docker)."""
    out = {}
    with open(f"/proc/{pid}/smaps_rollup") as f:
        for line in f:
            k, _, v = line.partition(":")
            if v.strip().endswith("kB"):
                out[k] = int(v.split()[0])
    return out


def process_memory(proc):
    """The server process's memory in MiB, one field per entry of MEMORY_SOURCES, and why smaps_rollup
    (uss, private_dirty, swap) could not be read, or None."""
    out = {"rss": round(proc.memory_info().rss / MiB, 1)}
    if platform.system() != "Linux":
        return out, None                # /proc fields have no equivalent on macOS; keep RSS
    st = _proc_status_kib(proc.pid, ("RssAnon", "VmHWM"))
    try:
        sm, error = _smaps_rollup_kib(proc.pid), None
    except OSError as e:                     # no ptrace access, or a kernel older than 4.14
        sm, error = {}, repr(e)[:200]
    kib = dict(anon=st.get("RssAnon"), peak_rss=st.get("VmHWM"), private_dirty=sm.get("Private_Dirty"),
               uss=sm["Private_Clean"] + sm["Private_Dirty"] if "Private_Clean" in sm and "Private_Dirty" in sm else None,
               swap=sm.get("Swap"))
    out.update({k: round(v / 1024, 1) for k, v in kib.items() if v is not None})
    return out, error


def vmstat(keys=("pgmajfault", "pswpin", "pswpout")):
    """Linux: system-wide cumulative counters from /proc/vmstat (major page faults, pages swapped in / out)."""
    out = {}
    with open("/proc/vmstat") as f:
        for line in f:
            k, _, v = line.partition(" ")
            if k in keys:
                out[k] = int(v)
    return out


def major_faults(pid):
    """Linux: cumulative major page faults of one process (field 12 of /proc/<pid>/stat)."""
    with open(f"/proc/{pid}/stat") as f:
        return int(f.read().rsplit(")", 1)[1].split()[9])


async def nvidia_query():
    """Memory in use on all NVIDIA GPUs (where llama-server keeps weights and KV cache with -ngl) and their
    mean utilization, or None."""
    p = await asyncio.create_subprocess_exec("nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                                             "--format=csv,noheader,nounits",
                                             stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    out, _ = await p.communicate()
    rows = [[float(x) for x in line.split(",")] for line in out.decode().splitlines() if line.strip()]
    if not rows:
        return None
    return dict(gpu_used_mib=round(sum(m for m, _ in rows), 1), gpu_util_pct=round(sum(u for _, u in rows) / len(rows), 1))


def parse_prometheus(text):
    out = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            name, _, value = line.rpartition(" ")
            try:
                out[name] = float(value)
            except ValueError:
                pass
    return out


def slot_state(slots):
    """/slots -> how many slots are decoding, and the length of the token sequence each slot keeps.
    n_prompt_tokens is prompt.tokens.size(): prompt + output so far (generated tokens are appended when they
    are batched, before they are decoded), and an idle slot keeps its last conversation until it is reused or
    purged. It is a token count, not the KV cache's allocation or memory use."""
    per = [dict(id=s['id'], busy=bool(s.get('is_processing')),
                tokens=s.get('n_prompt_tokens', 0)) for s in slots]
    return dict(busy_slots=sum(s['busy'] for s in per),
                slot_context_tokens_busy=sum(s['tokens'] for s in per if s['busy']),
                slot_context_tokens_total=sum(s['tokens'] for s in per), slots=per)



class MemoryMonitor:
    def __init__(self, path, clock, interval=0.25, proc=None, client=None, url=None, scrape_every_s=1.0):
        self.path, self.clock, self.interval = path, clock, interval
        self.proc, self.client, self.url = proc, client, url
        self.scrape_every_s = scrape_every_s
        self.metrics_ok = False             # set in start() if the server exposes /metrics
        self.slots_ok = False               # ... and /slots
        self.nvidia = shutil.which("nvidia-smi") is not None
        self._warned_smaps = False
        self._next_scrape = 0.0
        self._stop = asyncio.Event()
        self._task = self._file = None
        self.endpoint_status = {}

    def sample(self):
        rec = {"t": round(self.clock(), 3)}
        try:                               # CPU use since the previous sample, 100% = all cores
            rec["sys_cpu_pct"] = psutil.cpu_percent(None)
        except (OSError, psutil.Error):
            pass
        try:
            rec.update({f"vm_{k}": v for k, v in vmstat().items()})
        except OSError:
            pass
        try:
            vm = psutil.virtual_memory()
            rec.update(sys_used_mib=round((vm.total - vm.available) / MiB, 1),
                       sys_avail_mib=round(vm.available / MiB, 1))
        except (OSError, psutil.Error) as e:
            rec['system_memory_error'] = repr(e)[:200]
        try:
            rec['swap_used_mib'] = round(psutil.swap_memory().used / MiB, 1)
        except (OSError, psutil.Error) as e:
            rec['swap_memory_error'] = repr(e)[:200]
        if self.proc:
            try:
                mem, smaps_error = process_memory(self.proc)
                rec.update({f"srv_{k}_mib": v for k, v in mem.items()})
                if smaps_error:
                    rec["srv_smaps_error"] = smaps_error
                    if not self._warned_smaps:
                        self._warned_smaps = True
                        print(f"warning: cannot read /proc/{self.proc.pid}/smaps_rollup ({smaps_error}); "
                              "srv_uss_mib, srv_private_dirty_mib and srv_swap_mib will be missing. "
                              "Run the agent as the user that runs llama-server.")
                rec["srv_cpu_pct"] = round(self.proc.cpu_percent(None) / (psutil.cpu_count() or 1), 1)
                if platform.system() == "Linux":
                    rec["srv_majflt"] = major_faults(self.proc.pid)
            except psutil.NoSuchProcess:
                rec["srv_exited"] = True
                self.proc = None
            except (psutil.AccessDenied, OSError) as e:
                rec["srv_memory_error"] = repr(e)[:200]
        rec["rss_mb"] = rec.pop("srv_rss_mib", None)
        return rec

    async def scrape(self, rec):
        """Adds the server's /metrics values, slot state and GPU memory to `rec`, at most once every scrape_every_s."""
        if rec["t"] < self._next_scrape:
            return
        self._next_scrape = rec["t"] + self.scrape_every_s
        if self.metrics_ok:
            try:
                r = await self.client.get(f"{self.url}/metrics", timeout=2)
                r.raise_for_status()
                rec["srv_metrics"] = parse_prometheus(r.text)
                # llama-server's own KV cache gauges; removed in b5441 (#13660), so null on newer builds
                rec["kv_cache_usage_ratio"] = rec["srv_metrics"].get("llamacpp:kv_cache_usage_ratio")
                rec["kv_cache_tokens"] = rec["srv_metrics"].get("llamacpp:kv_cache_tokens")
            except Exception as e:
                rec["metrics_error"] = repr(e)[:200]
        if self.slots_ok:
            try:
                r = await self.client.get(f"{self.url}/slots", timeout=2)
                r.raise_for_status()
                rec.update(slot_state(r.json()))
            except Exception as e:
                rec["error"] = repr(e)[:200]
        if self.nvidia:
            try:
                rec.update(await nvidia_query() or {})
            except (OSError, ValueError):
                self.nvidia = False

    def write(self, rec):
        self._file.write(json.dumps(rec) + "\n")
        self._file.flush()

    async def start(self):
        if self.client:
            for name in ("metrics", "slots"):
                try:
                    ok = (await self.client.get(f"{self.url}/{name}", timeout=5)).status_code == 200
                except Exception:
                    ok = False
                self.endpoint_status[name] = ok
                setattr(self, f"{name}_ok", ok)
        self._file = open(self.path, "w", encoding="utf-8")
        rec = self.sample()                # baseline, taken before any task starts
        rec["endpoint_status"] = self.endpoint_status
        await self.scrape(rec)
        self.write(rec)
        self._task = asyncio.create_task(self._run())

    async def _run(self):
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), self.interval)
            except asyncio.TimeoutError:
                pass
            rec = self.sample()
            await self.scrape(rec)
            self.write(rec)

    async def stop(self):
        self._stop.set()
        await self._task                   # _run takes one final sample on its way out
        self._file.close()
