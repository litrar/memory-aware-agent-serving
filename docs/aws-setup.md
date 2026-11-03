# AWS setup

## Optional: AWS GPU host (`infra/`)

`infra/main.tf` creates one EC2 instance (default `g4dn.xlarge`, one NVIDIA T4) from the AWS Deep Learning Base GPU AMI (Ubuntu 22.04, NVIDIA driver and CUDA preinstalled) in the default VPC. The security group allows SSH only, from `ssh_cidr` and from the region's EC2 Instance Connect range; llama-server stays on localhost and is reached through an SSH tunnel.

Prerequisites: Terraform >= 1.5, AWS credentials (`aws configure`), an EC2 key pair in the target region, and a vCPU quota of at least 4 for "Running On-Demand G and VT instances" (spot runs need the separate spot quota). AWS Free-plan accounts can only launch Free Tier instance types; upgrade the account plan before launching a GPU instance.

```bash
cd infra
terraform init
terraform apply -var="key_name=<your-key-pair>" -var="ssh_cidr=$(curl -s https://checkip.amazonaws.com)/32"
terraform output ssh          # ssh command with -L 8080:localhost:8080
```

| Variable | Default | Notes |
|---|---|---|
| `key_name` | required | Existing EC2 key pair name |
| `region` | `us-east-2` | Must match where the key pair exists and where your account may launch EC2 |
| `instance_type` | `g4dn.xlarge` | |
| `ssh_cidr` | `0.0.0.0/0` | Narrow to your own IP (`x.x.x.x/32`) |
| `use_spot` | `false` | Cheaper but interruptible; needs spot quota |

On the instance, clone the repository and follow [Setup](../README.md#setup) (`scripts/build-llama-cpp.sh --install-deps` builds with CUDA). Copy results back before tearing down; `terraform destroy` deletes the root volume:

```bash
scp -r -i <path-to-key.pem> ubuntu@<instance-ip>:memory-aware-agent-serving/results ./results-ec2
terraform destroy -var="key_name=<your-key-pair>"   # pass the same -var values used for apply
```

GPU instances bill while running. `terraform.tfstate*`, `.terraform/` and `*.tfvars` hold account-specific data and are git-ignored; do not commit them, key files or AWS credentials.
