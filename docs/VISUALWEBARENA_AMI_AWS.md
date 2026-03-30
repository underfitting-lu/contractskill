# VisualWebArena AMI On AWS

This is the fastest path when you do not want to download the full VisualWebArena / WebArena website assets onto your local machine.

## What The AMI Does

An AMI is an AWS machine image. The official VisualWebArena AMI already contains the large website assets and Docker containers, so you launch a prebuilt EC2 instance instead of rebuilding the benchmark websites from scratch.

This repo assumes the default cloud layout:

- the agent also runs on the EC2 instance
- the benchmark websites also run on that same EC2 instance

## Recommended AWS Settings

As of March 11, 2026, prefer the official VisualWebArena AMI:

- Region: `us-east-2`
- AMI: `ami-080f6d73cfce497a1`
- Instance type: `t3a.xlarge`
- Root disk: `1000 GB gp3`

Security group inbound rules:

- `22`
- `80`
- `4399`
- `7770`
- `7780`
- `8023`
- `8888`
- `9980`
- `9999`

Allocate an Elastic IP if you want a stable public address.

Official references:

- VisualWebArena Docker README: https://github.com/web-arena-x/visualwebarena/blob/main/environment_docker/README.md
- WebArena Docker README: https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md

## EC2 Bring-Up

From the AWS console:

1. Open `EC2`.
2. Switch region to `us-east-2`.
3. Click `Launch instance`.
4. Select `ami-080f6d73cfce497a1`.
5. Choose `t3a.xlarge`.
6. Set the root volume to `1000 GB gp3`.
7. Choose or create an SSH key pair.
8. Configure the inbound ports listed above.
9. Launch the instance.
10. Associate an Elastic IP if needed.

SSH in:

```bash
ssh -i /path/to/your-key.pem ubuntu@<EC2_PUBLIC_IP>
```

If `ubuntu` does not work, check the AMI's default SSH username in EC2.

## Bring Up The Sites

On the EC2 instance:

```bash
docker ps -a
```

Then start the official site containers:

```bash
bash scripts/start_vwa_ami_sites.sh
```

If you also want the extra WebArena containers:

```bash
bash scripts/start_vwa_ami_sites.sh --with-webarena-extra
```

This script expects these AMI containers to exist:

- `postgis`
- `shopping_admin`
- `shopping`
- `classifieds`
- `reddit`
- `wikipedia`
- `homepage`

Default public ports:

- `classifieds -> 9980`
- `shopping -> 7770`
- `shopping_admin -> 7780`
- `reddit -> 9999`
- `wikipedia -> 8888`
- `homepage -> 4399`

## Prepare The Repo On EC2

Clone or copy this repo onto the EC2 instance:

```bash
git clone <your-repo-url> contractskill
cd contractskill
```

If you do not have a remote repo, copy the project archive or use `scp`.

Then create the Python environment:

```bash
bash scripts/setup_vwa_env.sh
```

If you want one command that does environment setup, site startup, `.env.vwa` generation, and readiness checks in sequence, use:

```bash
bash scripts/bootstrap_vwa_ami_agent.sh
```

If public IP auto-detection fails:

```bash
bash scripts/bootstrap_vwa_ami_agent.sh --host <EC2_PUBLIC_IP>
```

Set your GLM credentials:

```bash
export ZAI_API_KEY='<your-key>'
export ZHIPU_BASE_URL='https://open.bigmodel.cn/api/paas/v4/'
```

## Generate `.env.vwa`

Recommended:

```bash
bash scripts/write_vwa_ami_env.sh
```

This will:

- auto-detect the EC2 public IPv4 through AWS metadata when possible
- read `CLASSIFIEDS_RESET_TOKEN` from the running `classifieds` container
- write `.env.vwa`

If auto-detection fails, pass the host explicitly:

```bash
bash scripts/write_vwa_ami_env.sh --host <EC2_PUBLIC_IP>
```

Then load it:

```bash
source scripts/load_vwa_env.sh
```

You can also start from the template file [`.env.vwa.ami.example`](/mnt/c/home/lzj/contractskill/.env.vwa.ami.example).

The resulting `.env.vwa` will look like:

```bash
VWA_CLASSIFIEDS=http://<EC2_PUBLIC_IP>:9980
VWA_CLASSIFIEDS_RESET_TOKEN=<token>
VWA_SHOPPING=http://<EC2_PUBLIC_IP>:7770
VWA_REDDIT=http://<EC2_PUBLIC_IP>:9999
VWA_WIKIPEDIA=http://<EC2_PUBLIC_IP>:8888
VWA_HOMEPAGE=http://<EC2_PUBLIC_IP>:4399
```

## Validation

Run:

```bash
.venv_vwa/bin/python scripts/check_vwa_sites.py
.venv_vwa/bin/python scripts/check_vwa_env.py
```

Ready means:

- `scripts/check_vwa_sites.py` is all green
- `scripts/check_vwa_env.py` reports:
  - `playwright_browser_ready = yes`
  - `docker_ready = yes`
  - `required_env_vars_ready = yes`

## Run The First Benchmark

Smoke run:

```bash
.venv_vwa/bin/python run_vwa_experiment.py --baseline no_skill --split-path tasks/vwa_smoke_2.json
```

Then:

```bash
.venv_vwa/bin/python run_vwa_experiment.py --baseline no_skill --split-path tasks/vwa_dev_20.json
.venv_vwa/bin/python run_vwa_experiment.py --baseline no_skill --split-path tasks/vwa_main_100.json
```

After that, move to:

- `skill_no_repair`
- `text_only_rewrite`
- `contractskill`

## Troubleshooting

- If `docker` works but `check_vwa_sites.py` fails:
  - confirm the site containers are `Up`
  - confirm the EC2 security group allows the public ports
  - confirm `.env.vwa` uses the current Elastic IP or public IP
- If `write_vwa_ami_env.sh` fails to detect the host:
  - pass `--host <EC2_PUBLIC_IP>` manually
- If `CLASSIFIEDS_RESET_TOKEN` extraction fails:
  - confirm `classifieds` is running
  - run:
    ```bash
    docker exec classifieds sh -c 'cat /usr/src/app/.env | grep CLASSIFIEDS_RESET_TOKEN'
    ```
