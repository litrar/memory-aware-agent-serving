# Short/long context 校準報告

> 第二輪更新：針對接近 KV 壓力的任務，採約 2,000 document tokens、C=2 作測壓起點；第一輪約 1,100-token 的建議保留為較保守對照。完整 completion／error 分類與限制見第二輪。

日期：2026-10-23。這是兩個合成情境的開發校準，不是正式 scheduler 優化實驗或統計顯著性結論。

本文件彙整本輪可供團隊參考的設定、結果與限制；所有 `results/...`、`task-generator/calibration/...` 和程式路徑均以 repo 根目錄為起點。逐筆原始 logs 保存在本機 `results/`，該目錄被 Git 忽略，不會隨本報告一同分享；重現步驟見 [README 的 task generator 區塊](../README.md#task-generator-and-workload-calibration)。

## 第一輪：較保守的文件長度起點

在本次 **8,192-token 共享 KV、三輪 LLM、C=2、自然結束、每輪長回答上限 768 tokens** 的配置下，四類任務共用的起始建議為：

| 輸入文件類型 | 目標 document tokens | 英文草稿字數參考 | 本次實測文件 |
|---|---:|---:|---|
| Short | 約 200，先以 180–230 為製作範圍 | 約 130–180 words | 191/205 tokens；142/151 words |
| Long | 約 1,100，先以 1,000–1,200 為製作範圍 | 約 800–1,000 words | 1,075/1,125 tokens；853/889 words |

範圍是以本次實測點為中心的製作建議，並非整個範圍都已逐點驗證。**token 數是驗收條件；字數只協助起草。** 本報告中的固定文章已用相同模型 tokenizer 計數，執行固定 workload 時不必重算；若新增或修改文章，應對變更後的文字重新計數，並試跑完整 task。Short/long 是相對這個部署的工作量分類，不是模型能力上限的分類。

約 2,200-token 文件可作為額外的 LS 壓力層；約 3,200–3,400-token 文件可作為 LS 接近邊界的壓力層。約 4,300–4,500-token 文件可保留作超量案例。不要把這些較大數值直接套用到 LL：2,200-token 文件搭配長回答時，本次兩筆 task 都在最後一輪 context 超限。

選 1,100 而不是 2,200 作為共用 long-context 起點，是為了讓兩筆 LL 在 C=2 下仍有完成的空間；不能只用 LS 的成功範圍決定 LL 長度。這不是保證 C=4/8 或 EC2 上也可完成；正式六至八筆 workload 還需固定 task 集合後掃描 concurrency。

## 固定條件

- Apple M1，16 GiB RAM；本機 CPU 推論，四個 inference/batch threads，`-ngl 0`。
- Qwen3-4B Q8_0；llama.cpp commit `797de63b2fa7396dcc8caca76834c699ac84e1c3`，build `b1-797de63`。
- `--ctx-size 8192 -np 8 --kv-unified -ctk f16 -ctv f16 --fit off`；停用 context shifting 與 thinking。
- 自己控制的 task-level FIFO；同批兩筆 task 一起到達，固定 Cooling → Repair 順序；三輪 LLM、兩次一秒 tool delay。
- 固定 temperature=0、LLM seed=0、ignore_eos=false、prompt reuse=true。短回答每輪上限 192，長回答每輪上限 768 tokens。
- 每次獨立 run 重啟 server，preflight 確認沒有 retained slot tokens。server 使用 8081 隔離實驗端口；手動重現可使用 8080。
- 每個 HTTP request timeout=600 秒。server 載入／ready 等待不計入 runner 時間；runner 時間仍包含其設定讀取等少量開銷。
- 本次案例使用固定完整文章，未重複段落、未在測試途中生成文件、未針對 concurrency 修改原文或生成預算。

## 實測結果

每列是一次 run、兩筆 task。完成率只指 execution status=ok；不代表內容品質完全合格。slot tokens 是取樣到的邏輯 busy 序列加總；RSS 是程序 RSS。兩者皆非精確 KV-byte 使用率。

| Case | 文件 tokens（兩篇） | 第一輪 prompt tokens | 完成 | 時間 s | Peak logical busy tokens | Peak RSS MiB | 實際每輪輸出 tokens |
|---|---|---|---|---:|---:|---:|---|
| SS_1x_c2 | 205/191 | 362/348 | 2/2 | 58.0 | 1843 | 6471.0 | 91–113 |
| LS_1x_c2 | 1075/1125 | 1232/1282 | 2/2 | 98.6 | 3682 | 6784.3 | 98–122 |
| LS_2x_c2 | 2170/2234 | 2327/2391 | 2/2 | 166.3 | 5904 | 7116.2 | 106–131 |
| LS_3x_c2 | 3188/3360 | 3345/3517 | 2/2 | 262.5 | 7975 | 6253.5 | 83–144 |
| LS_4x_c1 | 4306/4504 | 4463/4661 | 2/2 | 339.1 | 5218 | 6855.6 | 92–126 |
| LS_4x_c2 | 4306/4504 | 4463/4661 | 0/2 | 135.8 | 8192 | 7508.0 | — |
| LL_1x_c2 | 1075/1125 | 1292/1342 | 2/2 | 370.5 | 5982 | 6881.5 | 427–474 |
| LL_2x_c2 | 2170/2234 | 2387/2451 | 0/2 | 447.8 | 8192 | 6457.6 | 418–474 |
| SS_1x_c1 | 205/191 | 362/348 | 2/2 | 75.1 | 949 | 6352.1 | 91–113 |
| SL_1x_c2 | 205/191 | 422/408 | 2/2 | 204.5 | 3630 | 6628.8 | 319–376 |
| LS_2x_c1 | 2170/2234 | 2327/2391 | 2/2 | 190.7 | 2983 | 6526.5 | 106–133 |

`SS/SL --scale 1` 是短文章；`LS/LL --scale 1` 是原始長文章，兩者的 scale=1 並非同一個長度。

約 3,300-token LS 在本次 C=2 run 仍完成，busy 邏輯峰值接近 8,192；這不是安全容量保證。4x LS 在 C=1 完成而 C=2 兩筆均失敗，支持並行序列競爭共享 context 容量的解讀。LL 2x 的前兩輪共四次 request 有正常回答，但兩筆 task 都未完成最後一輪，因此成功 task throughput 是零，不能用成功 request 數代替完整 task 完成量。

LL 1x 的實際例子：Cooling 第一輪 prompt=1,292；第二輪為 1,938（前輪加入 445-token 回答與約 201-token 新內容）；第三輪為 2,595，再生成 474 tokens，單筆末輪序列約 3,069 tokens。Repair 末輪約為 2,584+433=3,017。兩者各自末輪的需求合計約 6,086，是規劃上界式的加總，不是同時實測的 unique cells；本次取樣 peak busy 為 5,982。這說明只有約 1,100-token 的文件，經多輪長回答累積後仍能形成明顯負載。

本系列補測的 LS 2x C=1 用時約 190.7 秒，C=2 約 166.3 秒；SS C=1 約 75.1 秒，C=2 約 58.0 秒。這是單次觀察，兩組實際輸出也略有不同，不宣稱統計顯著加速。

不同長度的 RSS 沒有單調增加：OS resident pages、運算暫存、KV 預配置和其他記憶體管理都會影響觀察值。不能由這張表擬合「每多一個 token，RSS 增加多少」，也不能把 context 超限稱作物理 RAM OOM。本次固定版本的 `src/llama-kv-cache.cpp` 在初始化時依配置建立 KV tensors、配置 buffer 並清零（約 233–295 行）；較長 prompt 會使用既有池中的更多位置，不要求 RSS 等比例增加。因此本實驗中的 RSS 是 server 整體 RAM 足跡的背景指標，不是 prompt 導致 KV 增量占用的量測值；固定 server 設定下，應用每輪 prompt/生成長度、slot 狀態和 context 超限結果判斷工作負載壓力。busy-only slot 加總會略過等 tool 時留存的 idle cache，且可能重複計算共享前綴。

## 品質與自然生成的差異

人工檢視了完成 task 的最終摘要；主要活動數字及核心證據限制大致保留，但仍有錯誤，不把 HTTP 成功視為品質通過：

- LS 2x 的 Cooling 摘要把原文明確的 1–4 p.m.、4–7 p.m. 寫成上午／下午。這是未受來源支持的時段描述，應標為 factual error。
- SS 的 Cooling 摘要將「68 responses」稱作「68 visitors」；原文沒有提供獨立受訪者識別，應保留 responses 單位。
- SL 的 Cooling 中間回答把 54/9/5 個 survey responses 誤寫成 54%/9%/5%。這是明確的數量／比例錯誤；短文件＋長回答即使 execution=ok，也還沒有通過品質驗收。
- Repair SL 的部分回答把「沒有 final test 紀錄」寫成「沒有做 final test」。缺少紀錄不能證明測試從未發生，這個區分也應列入品質檢查。
- 部分 LS 回答超出 60–90-word 指令。300–400-word 的長回答也可能略短，均需保留實際長度。下表以 `str.split()` 計算 whitespace words，是一致的近似字數規則。

| Case | 已完成的 request 數 | 實際回答 words 範圍 | 超出要求區間的 request 數 | finish=length |
|---|---:|---|---:|---:|
| SS_1x_c2 | 6 | 62–81 | 0 | 0 |
| LS_1x_c2 | 6 | 71–89 | 0 | 0 |
| LS_2x_c2 | 6 | 80–98 | 2 | 0 |
| LS_3x_c2 | 6 | 62–109 | 3 | 0 |
| LS_4x_c1 | 6 | 68–99 | 2 | 0 |
| LS_4x_c2 | 0 | — | 0 | 0 |
| LL_1x_c2 | 6 | 317–362 | 0 | 0 |
| LL_2x_c2 | 4 | 291–341 | 1 | 0 |
| SS_1x_c1 | 6 | 62–81 | 0 | 0 |
| SL_1x_c2 | 6 | 236–280 | 6 | 0 |
| LS_2x_c1 | 6 | 80–98 | 3 | 0 |

失敗 task 的中間輸出仍保留；沒有用「回答太短」或「不符合預期」作事後排除。每筆 task 的 required_facts 仍是基本審查清單，另需標出捏造、錯誤單位、遺漏限制及重複句。此報告沒有宣稱完成完整的雙人品質評分或給出品質通過率。

## 如何製作下一批文件

上述 token 範圍是資源校準起點，並非四類 tasks 已全面品質合格。尤其 SL 仍要修正計數／比例錯誤及回答長度不合規；可先加強輸出指令與品質檢查，若需補充來源資料或改變篇幅，就要重新記錄 tokens 並做小型 pilot，不可直接宣稱原設定已驗證。

以 200/1,100 document tokens 作 short/long 起點，先產生六至八份內容不同的英文文件；保留可驗證的核心事實及一致回答要求。同一篇文件可用來配對 short/long response，但應另外標示同文重複 task，不能把 cache-friendly workload 當成獨立文章集合。

文件 tokens 不等於完整 prompt：例如 short response 的固定包裝，本次比文件多約 157 tokens；long response 包裝多約 217 tokens。後續輪次還會加上回答與 tool/指令內容。新文件的字數/token 比例可能不同，所以所有新文件都要 tokenize，不必把通用 preflight 驗證搬進 agent.py。

保守規劃可用 `單筆峰值 ≈ 第一輪完整 prompt + 各輪生成預算總和 + 後續指令/tool/格式增量`，再考慮同時保留的序列。但這是預算上界估計，不是 measured unique KV usage；正式判斷仍看每輪實際 context、完成／超限及記憶體紀錄。

## 重現與原始資料

README 的 reference server 指令及 calibration commands 是手動重現入口。每次 run 重啟相同 server，再跑一條命令；不要在同一次 run 的 LLM 步驟間重啟。重現可使用 `--url http://127.0.0.1:8081`，或改成預設 8080。兩次系列的實際 server 指令與每個案例指令均保存在 manifest：

- `results/context-calibration-20261023-145259/manifest.json`
- `results/context-calibration-20261023-175127/manifest.json`

對應目錄保存每個 case 的 server/client log；以下 run 另有 config、精確 workload、events 和 memory samples：

- `SS_1x_c2`：`results/pressure-demo/SS_1x_c2_1792792399513331000`
- `LS_1x_c2`：`results/pressure-demo/LS_1x_c2_1792792471143231000`
- `LS_2x_c2`：`results/pressure-demo/LS_2x_c2_1792792584395251000`
- `LS_3x_c2`：`results/pressure-demo/LS_3x_c2_1792792765440832000`
- `LS_4x_c1`：`results/pressure-demo/LS_4x_c1_1792793941803577000`
- `LS_4x_c2`：`results/pressure-demo/LS_4x_c2_1792796988345978000`
- `LL_1x_c2`：`results/pressure-demo/LL_1x_c2_1792798033900369000`
- `LL_2x_c2`：`results/pressure-demo/LL_2x_c2_1792801030656786000`
- `SS_1x_c1`：`results/pressure-demo/SS_1x_c1_1792803108137001000`
- `SL_1x_c2`：`results/pressure-demo/SL_1x_c2_1792803196827037000`
- `LS_2x_c1`：`results/pressure-demo/LS_2x_c1_1792803416061055000`

`results/` 不在 Git 內；共編者拿到 README 與本報告可重跑。若要審核這次既有原始結果，需要另外分享上述 results 目錄。

## 推論界線

這輪是單次、兩情境的開發校準，各條件沒有統計重複，也未平衡案例順序，背景負載與溫度未完全控制。輸出長度有變，latency 並非純輸入長度的函數。先前使用重複段落或 cache_prompt=false 的測試不併入本表；使用者先前的 LS 2x C=1（341.6 秒）也不當成本系列的直接因果對照。

新資料應在凍結長度與回答設定後獨立生成，避免把調整過的兩情境當作未見評估資料。C=4/8、混合六至八筆 tasks、不同工具延遲和 EC2 均需再確認；這份建議沒有保證其他部署的可行性。沒有刻意修改 FIFO 以使其失敗，也沒有刪除任何失敗結果。核心是描述基準行為與容量邊界，而不是宣稱 scheduler 已改善效能。

## 第二輪：接近 2,000 document tokens 的邊界與固定四筆 concurrency sweep

### 第二輪校準結論（固定四筆 LL）

文件實際為 1,953–2,020 tokens（1,639–1,731 英文 words），第一輪完整 prompt 為 2,170–2,237 tokens。每筆有三輪自然生成、每輪上限 768 tokens。下列完成數指 execution 成功，未代表品質全面通過。

| Task concurrency | 每 request timeout s | 完成／送入 | Context 超限 | Timeout | Peak logical busy tokens |
|---:|---:|---|---:|---:|---:|
| 1 | 600 | 4/4 | 0 | 0 | 4018 |
| 2 | 600 | 4/4 | 0 | 0 | 7877 |
| 3 | 600 | 0/4 | 2 | 2 | 10063 |
| 3 | 1200 | 1/4 | 3 | 0 | 8244 |
| 4 | 600 | 0/4 | 4 | 0 | 8192 |

- 高壓文件：以 **約 2000 document tokens** 為製作目標，約 1,600–1,750 英文 words；新文章需再 tokenize。C=2 在本批四筆皆完成且 peak logical busy=7,877，適合用作接近 8,192-token KV 的測壓起點，不是最大生成預算下的可靠性保證。
- 較保守的長文對照：1,000–1,200 document tokens（約 800–1,000 words）；較低壓短文：180–230 tokens（約 130–180 words）。
- C=1 作可靠完成／低並行對照；C=2 作本批 LL 的近界情境；C=3/4 保留為超量情境。本輪不將 C=2 宣稱為全域最快，耗時有明顯執行環境波動。
- 保留所有 C 的結果，包括完成全部 task 的 FIFO 設定；不要只挑失敗的 FIFO 作 Phase 2 比較基準。混合六至八筆或新文章需重新確認。

長期限 C=3 確認組為 1/4，context 超限 3 筆，timeout 0 筆，server peak busy slots=3。原始 600 秒那組與這組分列保存，避免把 timeout 或 backend request 未釋放的影響全部歸因 KV。

控制核對：所有固定四筆 run 的輸入 snapshot、順序、指令、seed、工具與輸出預算逐值一致；每次開始的 slot 均為空，n_ctx 均為 8,192。


這一輪將 document-token 目標限制在 1,600/1,800/2,000，而不是把前一輪約 2,200-token 的失敗點當成精確上限。server、長回答指令（每輪上限 768）、三輪步驟、tool delay、自然 EOS、cache 政策均維持不變。

先以相同兩情境在 C=2 加密長度，再選這三個目標中觀察到兩筆皆完成的最高目標，用固定四筆 task 測 C=1/2/3/4。不同 N 的列不當作單一因素比較。這是逐步選點的開發校準，並非事先固定的獨立評估或完整閾值搜尋。

表內 output tokens 是已成功回傳 usage 的 request 加總，包含其後 task 失敗前的成功 request；發生錯誤的 request 可能做過未回傳的部分計算，不能由此推論完整浪費工作量。不同 C 的實際輸出長度仍有變動，不把相同生成上限當成相同實際工作量。

四篇文件是兩個既有情境的受控變體：各有不同標題／地名與支持段落組合，保留核心數字；每篇完整、自洽且內部無重複段落。因為它們仍源自兩組事實，不能稱為四個獨立內容領域。保存於 `task-generator/calibration/boundary-v1.toml`，正式評估仍需新文章。

| 目標 document tokens | Task 數 N | Concurrency C | Request timeout s | 實際 document tokens | 完成／送入 | 時間 s | 成功 tasks/min | Peak logical busy tokens | Peak RSS MiB | 已回傳 output tokens |
|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| 1600 | 2 | 2 | 600 | 1610/1581 | 2/2 | 396.9 | 0.302 | 6855 | 6837.9 | 2609 |
| 1800 | 2 | 2 | 600 | 1792/1766 | 2/2 | 473.4 | 0.253 | 7336 | 7153.0 | 2675 |
| 2000 | 2 | 2 | 600 | 1985/1953 | 2/2 | 473.6 | 0.253 | 7849 | 6798.9 | 2724 |
| 2000 | 4 | 1 | 600 | 1985/1953/1973/2020 | 4/4 | 1129.4 | 0.213 | 4018 | 5094.3 | 5394 |
| 2000 | 4 | 2 | 600 | 1985/1953/1973/2020 | 4/4 | 1860.3 | 0.129 | 7877 | 6361.3 | 5324 |
| 2000 | 4 | 3 | 600 | 1985/1953/1973/2020 | 0/4 | 609.1 | 0.000 | 10063 | 5510.5 | 415 |
| 2000 | 4 | 4 | 600 | 1985/1953/1973/2020 | 0/4 | 269.4 | 0.000 | 8192 | 5599.9 | 0 |
| 2000 | 4 | 3 | 1200 | 1985/1953/1973/2020 | 1/4 | 937.2 | 0.064 | 8244 | 5371.3 | 2186 |

兩筆長度掃描選出的 nominal target 為 **2000 document tokens**。固定四筆測試中，完成全部 task 且觀察到最高完成 throughput 的 concurrency 為 **C=1**；這是單次觀察，不是跨負載的全域最優解。

### 壓力情境與後續文件設定

- 低負載對照：short 文件沿用約 180–230 document tokens（約 130–180 英文 words），搭配短回答；不要把這個文件長度誤當成完整多輪 context。
- 一般 long：前一輪約 1,000–1,200 document tokens（約 800–1,000 words），保留較多長回答累積餘裕。
- 接近邊界的 long：以 nominal 2000 tokens 為新文章起稿目標；本輪實際 1953–2020 document tokens，1639–1731 英文 words。此字數只反映本批素材，必須再 tokenize 驗收。
- 過載情境：固定同批四篇 long/long tasks，增加 C，依完成率、成功 throughput 與超限錯誤辨識；不是讓每筆文件單獨就不可能執行。
- 短文件＋長回答、長文件＋短回答是中間負載；生成長度可使它們交換壓力排序，不能只靠 doc 長度預測。

### 四類 workload 預期形成的場景

以下是依本次實際生成長度做的序列規劃估計，並非四組新資料的精確 unique-KV 量測。P0 會隨指令格式改變，最後峰值也取決於時間重疊：

| 類型 | Document tokens | 每輪實際輸出參考 | 單筆完整序列規劃值 | C=2 的預期用途 |
|---|---:|---:|---:|---|
| SS | 約 200 | 約 100 tokens | 約 900 tokens | 低負載對照 |
| SL | 約 200 | 約 350 tokens | 約 1,850 tokens | 提高 decode 工作及多輪歷史累積 |
| LS | 約 2,000 | 約 120 tokens | 約 2,800 tokens | 較大的 prefill 與起始 KV，後續增長較小 |
| LL | 約 2,000 | 約 450 tokens | 約 3,950 tokens | 兩筆約 7,900 tokens，逼近本次共享 KV 容量 |

這些是受控實驗的長短分類，不是模型能力的 32K/更長 context 定義。四類主測試可先用 short≈200、long≈2,000 的文件目標；需要較保守的 LL 控制組時，另保留 long≈1,100。新文件的實際輸出可能不同，因此在最終六至八筆混合 workload 中仍需確認，不能把這個估計表當成容量保證。

### C=3 的 timeout 與 backend occupancy

原始 C=3、600 秒期限那組含兩筆 request timeout 和兩筆 context 超限；當 client timeout 釋出 worker 並接下一筆 task 時，backend 舊 request 尚未完全結束，曾取樣到四個 server busy slots，即使 client task-level C=3。該時點 logical token 加總 10,063 不代表實體 KV 使用超過配置，而是進一步說明這個計數不是 unique physical cells。這組不能單獨當成乾淨的 C=3 記憶體上限證明，也不能把 timeout 都歸因 KV。

因此額外以相同四筆輸入、C=3、request timeout=1,200 秒做確認，和原始 600 秒結果分列保留；不是把不符合預期的結果刪除或替換。Backend timeout 取消／占用釋放的同步問題列為後續 agent runner 工作，本輪不在測到一半修改 runner 行為。

**為什麼 peak logical busy tokens 會是 10,063，大於 8,192？** 10,063 是 `/slots` 各 busy slot 回報的 `n_prompt_tokens` 相加，代表邏輯序列長度總和，不是已分配 KV cells 的精確數量。共享 prompt prefix 可能在多個 slot 的 token 計數中重複出現；因此邏輯總和可以超過共享實體池容量，不能解讀為 KV buffer 實際存了 10,063 個互異 token。原始 C=3 run 還有 timeout 後 client worker 先釋出、舊 backend request 尚未完全停止的競態，曾短暫觀察到 4 個 server busy slots。故這個 10,063 既不是「buffer 超額配置」，也不能用來計算占用百分比；它是一次不穩定失敗過程中的邏輯總和。實際容量壓力的證據是 server 回報 context 超限，而長 timeout 的 C=3 確認組也有 3 筆 context 超限；後者 peak logical sum 8,244 仍不是實體 KV 用量。現有量測沒有追蹤唯一 KV cells，也沒有逐一定位 10,063 中哪些 token 被重複或暫態計入；共享 prefix 是可能的解釋，尚不能斷言它就是這次超額計數的完整原因。不能從這些數字反推出精確的物理占用。

### 預算上界與自然生成壓力的區分

LL 這批固定格式的初始包裝約 217 tokens，後續兩輪新增指令/tool/格式約 201+183 tokens。若每輪都生成到 768 上限，單筆需求約為 `D + 217 + 201 + 183 + 3*768 = D + 2,905`。忽略少量實作差異和共享前綴，C=2 的保守條件是 `2*(D+2,905) <= 8,192`，得到 D 約不得高於 1,191。這解釋第一輪為何把 1,100 當成保守 long 起點。

本輪 1,600–2,000-token 的目標，是讓自然生成的典型回答逼近容量，**不是保證即使每輪都用滿生成上限仍可完成**。如果需要更可靠完成率，可用約 1,100-token 文件／C=2，或對較大的 LL 文件採 C=1。若需要展示 scheduler 的 admission 壓力，保留本輪較長文件和較高 C；不要以壓力案例的單次成功當成 SLA 保證。

### 解讀限制

本次 `--ctx-size 8192 -np 8 --kv-unified` 中，8 是 server 的請求 slot 數，8,192 是共享 KV 池的 token 容量；單個 slot 的 context 上限也回報為 8,192，但不是每個 slot 都獨享 8,192 個 KV 位置。client 的 C 則限制同時執行的完整 agent tasks，與 server slot 數分開設定。slot 的 token 計數是邏輯序列長度，KV cache 是保存其 K/V 狀態的記憶體結構，兩者的加總不可直接視為唯一實體 cells。

此處的「邊界」是指定 server 的共享 KV/context 容量邊界。不是模型宣稱的 context window，也不是整台機器 RAM 的精確使用率。slot-token 加總仍是邏輯估計，不能以它單獨宣稱精確的記憶體飽和百分比。

另觀察到執行時間明顯波動：相同兩筆 t2000 task 的第一輪，兩筆測試約 153/168 秒，四筆測試的同樣初始 request 約 333/374 秒，而輸出 tokens 相同。當時唯讀檢查顯示程序約 360% CPU、system-wide memory free percentage 55%，pmset 沒有記錄 performance/thermal 狀態；這些不足以確定慢速原因，診斷快照已保存。故這輪不以單次 timing 排名宣稱 concurrency 的因果最優值，建議區分容量壓力與可靠性用途。

文件長度之外，實際輸出會改變後續 prompt；同樣 target、不同文件與 concurrency，可能有不同完成數。若邊界下任何一筆失敗，不能把其餘完成的樣本挑出來報告。正式 baseline 應保留一般負載與過載負載，並另在獨立文章、不同到達順序與六至八筆混合 task 上確認。容量結論也不代表回答品質已通過；第一輪列出的數字／比例、缺失紀錄與篇幅問題仍待處理。

### 本輪原始資料

- Manifest、每列 server/client logs：`results/boundary-20261023-183621`
- target=1600, N=2, C=2：`results/pressure-demo/LL_t1600_n2_c2_1792805817105429000`
- target=1800, N=2, C=2：`results/pressure-demo/LL_t1800_n2_c2_1792806299133454000`
- target=2000, N=2, C=2：`results/pressure-demo/LL_t2000_n2_c2_1792807966733837000`
- target=2000, N=4, C=1：`results/pressure-demo/LL_t2000_n4_c1_1792808458474621000`
- target=2000, N=4, C=2：`results/pressure-demo/LL_t2000_n4_c2_1792810237080496000`
- target=2000, N=4, C=3：`results/pressure-demo/LL_t2000_n4_c3_1792812117785509000`
- target=2000, N=4, C=4：`results/pressure-demo/LL_t2000_n4_c4_1792812878197497000`
- target=2000, N=4, C=3：`results/pressure-demo/LL_t2000_n4_c3_1792813223475563000`

README 的所有 task-generator 說明集中在新增的單一區塊；先前 scheduler 文件原文逐字保留。本輪例子使用 `--workload task-generator/calibration/boundary-v1.toml --target-tokens TARGET --profile LL`，預設選四筆；`--limit 2` 只用於前面的長度加密。每個獨立 run 前重啟相同 server。

## Phase 1 候選文章與品質驗收規則

### 資料與狀態

`task-generator/source/phase1-articles.toml` 保存 30 篇不同主題的合成文章，每組 10 篇。輸入層級目標為 200／2,000／2,600 document tokens；每篇有唯一 ID、四項 `required_facts`、逐字 `evidence_quote` 及 `forbidden_claims`。量測使用同一個 Qwen3-4B Q8_0 tokenizer，`add_special=false`。保存的文件字串也經重新計數，並檢查全部文章沒有相同完整段落。必要事實分布於長文前、中、後段。

near 與 overload 都是部署相關的候選壓力標籤。2,000 是前輪測試起點；2,600 是為增加容量競爭新增的長度，尚未以本批文章證實其失敗率。三組均不保證特定 concurrency 的成敗。這批僅通過輸入 token／格式檢查，尚未完成 30 筆 C=1 全步驟推論及內容品質驗收，不應把它標成已驗收正式 benchmark。

同學可由 `agent.py` 直接讀取固定文章及其共用 task settings，並在每次 run 中保存展開後的 tasks snapshot。重現指令集中在 [README](../README.md#phase-1-candidate-articles-and-quality-rules)。本次 tokenizer 與完整初始 prompt 的驗收紀錄在本機 `results/phase1-input-validation/`；此目錄同樣不隨 Git 分享。

實際保存字串的輸入驗收結果如下（不是推論成敗結果）：

| Input tier | 篇數 | 實際 document tokens | 實際英文 words |
|---|---:|---:|---:|
| short | 10 | 199–204 | 159–168 |
| near | 10 | 1,966–2,034 | 1,738–1,836 |
| overload | 10 | 2,508–2,625 | 2,265–2,322 |

同時檢查了 30 篇 × 兩種回答指令的 60 種完整初始 prompt；它們包含文章以外的 system、指令及 chat template。這不涵蓋後續自然回答造成的歷史增長。

### 任務組合與回答控制

正式 source 檔保存共同回答指令、三輪 LLM、兩次一秒模擬 tool、seed=0 與 short/long response profiles；`agent.py` 直接讀取並展開固定 tasks。自然 EOS、temperature=0 與停用 thinking 沿用 runner 設定。主程式不新增自動品質評分器。

文章庫共有 30 篇；目前每輪預設從中抽 6 篇、不重複，依 SS/LL 交替順序進入 C=2 workers。組合定義與設計理由見 source 的 `task_settings.combinations` 註解；`--combination` 選組合，`--long-tier` 選 near 或 overload。相同 seed／篇數／long tier 的 LL-first、SS-first、SS-LL 使用相同 tasks，只改順序；SS 對 SL、LS 對 LL 可固定文章比較回答設定。比較不同 scheduler／C 時應保持選樣參數相同，或直接重播已保存的 workload snapshot。短回答要求 60–90 words、上限 192 tokens；長回答要求 300–400 words、上限 768 tokens。這是可見回答的篇幅／結構要求，不是要求模型做簡單或複雜的隱藏推理。

`required_facts`、證據引文、禁止主張和品質規則只放在評分資料與結果設定，不放進模型 messages。工具提供的是檢查流程，不提供正確答案。每篇文章的四項事實採二元語意評分；必須保留其關鍵數字、單位、範圍和證據限制，允許同義改寫。遺漏、矛盾和支持應分開標註，不用字面關鍵字出現就判通過。`forbidden_claims` 是錯誤範例而非完整錯誤清單；其他捏造也要記錄。

### 每輪回答的品質 metrics

| 指標 | 定義與規則 |
|---|---|
| Required-fact recall | 完整、正確表達的必要事實數／4；部分或錯誤表達不算完整支持。 |
| Contradiction count | 與原文明確矛盾的事實主張數，包含數字、單位、否定或範圍錯誤。 |
| Unsupported-claim count | 無原文支持的事實主張數；清楚標為建議且沒有假稱已實施的建議不算捏造。 |
| Numeric/unit accuracy | 正確且適用範圍一致的數字主張數／受檢查的數字主張數；分母為零記 N/A，必要數字遺漏仍扣 fact recall。 |
| Length compliance | `len(text.split())` 是否落在該 response profile 的 words 範圍；另保留 tokenizer 的實際 completion tokens。 |
| Factual pass | Recall=1，且 contradiction=0、unsupported=0。 |
| Strict pass | Factual pass、length compliance，且 finish_reason=stop；不要把 finish=length 的截斷輸出算通過。 |

每個成功返回的 LLM step 都可獨立填寫上述欄位；最終摘要的 pass rate 與「三輪全過」的 task pass rate 分列，不混用。若需要成本較低的首輪驗收，可先雙人審查跨三組的固定樣本以對齊判準，再檢查其餘回答；不可據此宣稱未審查的回答已通過。表中的語意評分尚未自動實作，不能由 HTTP 成功或 regex 命中代替。

Timeout、context 超限、其他錯誤以及缺少最終回答的 task 均保留，task 通過率分母使用全部送入 tasks。失敗 task 的未返回回答，其 request-level 品質欄位記為無資料，不能當成完整文字評零或假設沒有做計算；但 task-level strict pass 必定為 false。另報 `strictly passed tasks / experiment wall time` 作為品質與篇幅都合格的完成 throughput。所有已返回但過短、錯誤或被截斷的回答仍要納入審查。

### 凍結與比較限制

這些新主題與舊校準案例分開，但文章共用報告寫作框架，語言／文體多樣性有限。不同長度組使用不同主題，組間差異同時含題材差異，不能將結果全歸因文件長度；若要研究純長度的因果影響，需另設配對主題版本。本 corpus 的主要用途是固定混合負載，比較相同 tasks 在不同 scheduler／concurrency 下的表現。

先用既有 pilot 對齊評分準則與程式，再按預定流程檢查新資料。若因新資料的輸出修改文章或提示，應更新版本並重新記錄 input counts，不能仍稱其為未用於調整的評估資料。正式收集前凍結文件、指令、回答預算、到達順序、cache 政策和 server 設定；保留全部失敗，不為了得到預期壓力而事後挑選成敗樣本。真實完成率與輸出長度仍需實際推論才能確定。
