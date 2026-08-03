# TAO Execution SDK

Optional Python layer on top of the [TAO skill bank](https://github.com/NVIDIA-TAO/tao-skill-bank). The skill bank works with just docker; install this SDK **only** if you need one of:

- Job handles with persistent state (SQLite)
- Background polling + status normalization
- S3 I/O auto-wrapping (`script_runner` inputs/outputs)
- Multi-node distributed training
- Direct Python-script jobs in an existing virtual environment
- DGX Cloud Lepton access (Lepton is API-first — SDK is the only path)
- Failure analysis (OOM classification, node events) on Lepton

If you're running locally on your own GPU, or on Brev via `brev exec docker run`, you don't need this SDK. The skill bank plugin is sufficient.

## What's in this repo

| Module | Role |
|---|---|
| `tao_sdk.brev_sdk.BrevSDK` | Wraps `brev exec docker run` with Job handles + state + S3 I/O |
| `tao_sdk.lepton_sdk.LeptonSDK` | Lepton job submission via the official `leptonai` SDK |
| `tao_sdk.job_store.JobStore` | SQLite persistent state for jobs across process restarts |
| `tao_sdk.monitor.JobMonitor` | Background daemon thread for status polling |
| `tao_sdk.io_wrapper`, `tao_sdk.script_runner` | Build shell harness for S3 input download + output upload |
| `tao_sdk.models` | Data classes: `Job`, `JobStatus`, `Checkpoint`, `CredentialError` |
| `tao_sdk.platforms.virtualenv.VirtualEnvSDK` | Runs isolated local Python jobs without a container or shell |
| `tao_sdk.handlers/` | Platform handler interface + Lepton implementation |

No skill-bank content lives in this repo. The skill bank is a separate artifact — this SDK only consumes its metadata at runtime.

## Install

```bash
pip install tao-sdk

# Platform extras (pick what you need):
pip install 'tao-sdk[lepton]'   # Lepton job submission
pip install 'tao-sdk[brev]'     # Brev handler (wraps brev CLI)
```

Development install:

```bash
git clone git@github.com:NVIDIA-TAO/tao-sdks.git
cd tao-sdk
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Credentials

All via environment variables. Load from `.env` at session start:

| Platform | Required env vars |
|---|---|
| Lepton | `LEPTON_WORKSPACE_ID`, `LEPTON_AUTH_TOKEN` |
| Brev | `BREV_API_TOKEN` (optional — manual `brev login` also works) |
| S3 I/O | `S3_BUCKET_NAME`, `ACCESS_KEY`, `SECRET_KEY`, `S3_ENDPOINT_URL` (optional for non-AWS) |
| Containers | `NGC_KEY`, `HF_TOKEN` (passed through into each `docker run`) |

Missing required vars raise `CredentialError` at SDK instantiation. The SDK **never** prompts for secrets — the caller is responsible for populating the environment.

## Usage

```python
from tao_sdk.lepton_sdk import LeptonSDK    # or BrevSDK
import os

sdk = LeptonSDK()

job = sdk.create_job(
    image='nvcr.io/nvidia/tao/tao-toolkit:6.26.3-pyt',
    command='visual_changenet train -e /tmp/spec.yaml',
    gpu_count=8,
    env_vars={'HF_TOKEN': os.environ['HF_TOKEN']},
    inputs={'/tmp/spec.yaml': 's3://bucket/specs/train.yaml'},
    outputs=['/results/'],
    resource_shape='gpu.h100.8',
    num_nodes=1,
)

print(job.id, job.results_dir)

# Poll
import time
while True:
    s = sdk.get_job_status(job.id)
    if s.status in ('Complete', 'Error', 'Canceled'):
        break
    time.sleep(30)

if s.status == 'Error':
    print(sdk.get_job_logs(job.id, tail=200))
    analysis = sdk.get_failure_analysis(job.id)   # Lepton only
    print(analysis)
```

For full API surface and patterns, see the skill bank's `platform/tao-sdk/SKILL.md`.

### Local virtual-environment jobs

Use the separate Python-script capability when an action is installed in a
local virtual environment instead of packaged as a container CLI:

```python
from tao_sdk.platforms.virtualenv import VirtualEnvSDK

sdk = VirtualEnvSDK(
    venv_path=".venv",
    work_dir=".tao/virtualenv",
)
job = sdk.create_python_job(
    script="train.py",
    specs={"train": {"epochs": 2}, "results_dir": "output"},
    config_format="yaml",
    script_args=["--config", "{config_path}"],
    outputs={"results_dir": {"type": "folder"}},
    gpu_count=1,
)
```

The SDK uses a job-local, standard-library Python supervisor so completion can
be recovered after an orchestrator restart. That supervisor invokes the
training argv as ``[.venv/bin/python, train.py, ...]`` without activating the
environment or using a shell. Each job receives its own spec, logs, durable
exit record, and results directory. Only local declared inputs are supported;
stage remote data before submission. A positive `gpu_count` without `gpu_ids`
is metadata only and preserves existing CUDA visibility.

## Typical workflow (agent-driven)

The SDK is designed to be used through an agent that reads the skill bank. The user doesn't write the Python above by hand — the agent constructs it from the skill + dataset intent:

```
User: "Fine-tune cosmos-rl on s3://bucket/datasets/my-data, 8 GPUs on Lepton"
  → Agent reads models/cosmos-rl/SKILL.md + references/model_info.yaml
  → Constructs spec TOML, assembles docker command
  → Calls sdk.create_job(...) with inputs/outputs mapped
  → Polls status, reports progress, surfaces failures
```

For the skill bank (the thing the agent reads), install the plugin:

```shell
/plugin marketplace add git@github.com:NVIDIA-TAO/tao-skill-bank.git
/plugin install tao-skill-bank@tao-skills
```

See the [skill bank README](https://github.com/NVIDIA-TAO/tao-skill-bank/blob/main/README.md) for the full install flow.

## Project structure

```
tao-sdk/
├── pyproject.toml
├── CLAUDE.md                      # agent instructions (setup, never-do rules)
├── .env.example                   # credential template
├── tao_sdk/
│   ├── __init__.py                # public API: Job, JobStatus, Checkpoint, CredentialError, JobStore
│   ├── brev_sdk.py                # BrevSDK — wraps brev CLI
│   ├── lepton_sdk.py              # LeptonSDK — uses official leptonai SDK
│   ├── job_store.py               # SQLite persistence
│   ├── monitor.py                 # background polling daemon
│   ├── io_wrapper.py              # shell harness for S3 I/O
│   ├── script_runner.py           # runs inside container, handles downloads/uploads
│   ├── platform_sdk.py            # PlatformSDK Protocol
│   ├── models.py                  # data classes
│   └── handlers/                  # platform handler implementations
│       ├── base.py                # ExecutionHandler + BaseHandler
│       ├── lepton.py              # LeptonHandler
│       └── registry.py            # handler discovery via entry points
├── platforms/
│   └── brev/                      # tao-sdk-brev companion package (separate wheel)
└── launch_*.py                    # example launch scripts
```

## Related repos

- [**tao-skills-external**](https://github.com/NVIDIA-TAO/tao-skill-bank) — the skill bank (primary artifact; plugin marketplace)

<!-- blossom CI smoke test: verify 7.1.0 OSS migration tests pass — DO NOT MERGE -->
