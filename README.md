# SDFLoRA

SDFLoRA is a research codebase that implements a dual-adapter LoRA design for federated foundation models. The core idea is to split adaptation into a global adapter (federated, shared across clients) and a local adapter (personalized, kept on each client), then fuse their outputs with lightweight mechanisms such as weighted sum, gating, or attention. The project also includes heterogeneous-rank aggregation and optional DP-SGD for privacy.


## Requirements
The scripts expect a Python environment with:
- `torch`
- `transformers`
- `peft`
- `numpy`
- `PyYAML`

Some scripts also assume the FedSA-LoRA / FederatedScope ecosystem is available and importable.

## Setup Notes
Several scripts and configs hardcode paths (for example `/home/...`). Update these paths for your environment before running:
- `code/run_dual_lora.py`
- `code/mmlu_evaluator.py`
- `code/example_mmlu_evaluation.py`
- `setting/*.yaml`

Search for `/home/` and replace with your local paths.

## Quick Start (Federated Training)
```bash
python code/run_dual_lora.py --cfg setting/dual_lora_config.yaml
```

### Heterogeneous Clients
```bash
python code/run_dual_lora.py --cfg setting/dual_lora_hetero_config.yaml
```

### DP-SGD Options
```bash
python code/run_dual_lora.py --cfg setting/dual_lora_config.yaml --enable-dp-sgd
python code/run_dual_lora.py --cfg setting/dual_lora_config.yaml --enable-dp-sgd --dp-epsilon 1.0 --dp-delta 1e-5
```

## MMLU Evaluation
Run a detailed evaluation with the provided config:
```bash
python code/mmlu_evaluator.py --config setting/mmlu_evaluation_config.yaml
```

Run multi-variant comparisons:
```bash
python code/run_mmlu_experiments.py --config setting/mmlu_evaluation_config.yaml --variants dual_fusion,global_only
```

There is also an interactive example script:
```bash
python code/example_mmlu_evaluation.py
```

## Configuration Highlights
- `setting/dual_lora_config.yaml`: baseline dual LoRA training with DP-SGD.
- `setting/dual_lora_hetero_config.yaml`: heterogeneous client ranks.
- `setting/mmlu_evaluation_config.yaml`: MMLU evaluation and variant definitions.

## Results and Artifacts
- Training outputs are typically written into `exp/`.
- Evaluation outputs are placed under `mmlu_evaluation_results/`.

## Notes
This is research-oriented code. Paths and dependencies may require manual adjustment for your environment. If you plan to reuse this in a new setup, start by updating all hardcoded paths in configs and scripts.
