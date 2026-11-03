# Minimal single-GPU EC2 host for running llama-server.
#
#   terraform init
#   terraform apply -var="key_name=my-key"
#   terraform destroy        # GPU instances bill by the second, tear down when idle
#
# Defaults to g4dn.xlarge (1x NVIDIA T4 16 GB, 4 vCPU, 16 GiB RAM), the
# smallest/cheapest CUDA-capable instance type on AWS.

terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# ---------------------------------------------------------------- variables

variable "region" {
  description = "The org SCP only allows EC2 in us-east-2."
  type        = string
  default     = "us-east-2"
}

variable "instance_type" {
  type    = string
  default = "g4dn.xlarge"
}

variable "key_name" {
  description = "Name of an existing EC2 key pair for SSH access."
  type        = string
}

variable "ssh_cidr" {
  description = "CIDR allowed to SSH in. Narrow this to your own IP (x.x.x.x/32)."
  type        = string
  default     = "0.0.0.0/0"
}

variable "use_spot" {
  description = "Run as a spot instance (~60-70% cheaper, may be interrupted)."
  type        = bool
  default     = false
}

# ---------------------------------------------------------------- provider

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project = "cse291-agent-edge-serving"
    }
  }
}

# ---------------------------------------------------------------- lookups

# Default VPC/subnets avoid creating any networking resources.
data "aws_vpc" "default" {
  default = true
}

# Ubuntu 22.04 with NVIDIA driver + CUDA preinstalled, so no driver setup is needed.
data "aws_ami" "gpu" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) *"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

# ---------------------------------------------------------------- resources

resource "aws_security_group" "gpu" {
  name_prefix = "cse291-gpu-"
  description = "SSH only; reach llama-server via an SSH tunnel"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "gpu" {
  ami                    = data.aws_ami.gpu.id
  instance_type          = var.instance_type
  key_name               = var.key_name
  vpc_security_group_ids = [aws_security_group.gpu.id]

  dynamic "instance_market_options" {
    for_each = var.use_spot ? [1] : []
    content {
      market_type = "spot"
      spot_options {
        spot_instance_type             = "one-time"
        instance_interruption_behavior = "terminate"
      }
    }
  }

  # Size is left unset so the volume matches the AMI snapshot (the minimum allowed).
  root_block_device {
    volume_type           = "gp3"
    delete_on_termination = true
  }

  metadata_options {
    http_tokens = "required"
  }

  tags = {
    Name = "cse291-gpu"
  }
}

# ---------------------------------------------------------------- outputs

output "public_ip" {
  value = aws_instance.gpu.public_ip
}

output "ssh" {
  value = "ssh -i <path-to-${var.key_name}.pem> -L 8080:localhost:8080 ubuntu@${aws_instance.gpu.public_dns}"
}

output "ami" {
  value = data.aws_ami.gpu.name
}
