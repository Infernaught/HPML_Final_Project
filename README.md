# HPML Project: GRPO Fine-Tuning under Hardware Constraints

## Team Information
- **Team Name**: T4 Titans
- **Members**:
  - Manush (UNI1)
  - [Add other team members if any]

---

## 1. Problem Statement

We fine-tune and run inference on two instruction-tuned LLMs—**DeepSeek-Coder-7B-Instruct-v1.5** (6.91B parameters) and **Phi-3.5-Mini-Instruct** (3.82B parameters)—for math reasoning under hardware constraints using a single NVIDIA T4 GPU. The task involves solving two math benchmarks: 
- **Countdown**: easier, requires generating equations under constraints.
- **AIME 2024**: harder, advanced competition-style math problems.

We apply **Group Relative Policy Optimization (GRPO)**, **Supervised Fine-Tuning (SFT)**, and **quantization** to ensure feasibility and performance.

---

## 2. Model Description

- **Models Used**: DeepSeek-Coder-7B-Instruct-v1.5, Phi-3.5-Mini-Instruct
- **Framework**: PyTorch
- **Custom Methods**:
  - Applied GRPO for fine-tuning on task-specific JSONL datasets.
  - Used SFT prior to GRPO for improved reward shaping.
  - Employed 4-bit and 8-bit quantization via bitsandbytes.
  - Integrated LoRA adapters uploaded to HuggingFace for modular fine-tuning.

---

## 3. Final Results Summary

| Metric               | Value (DeepSeek-AIME, Phi-Countdown) |
|----------------------|--------------------------------------|
| Final Accuracy (Task Score) | [Insert if available]              |
| Inference Latency    | [Insert if measured] ms              |
| Model Size           | 6.91B / 3.82B (quantized)            |
| Peak Memory Use      | Fits within 16 GB T4 VRAM            |
| Training Time/Epoch  | [Insert if measured]                 |
| Device               | NVIDIA T4 GPU                        |

(Insert above values once available from logs or wandb)

---

## 4. Reproducibility Instructions

### A. Requirements

```bash
pip install -r requirements.txt
```

B. Wandb Dashboard
View training and evaluation metrics here:
```bash
https://wandb.ai/mtt-hpml/projects
```
C. Specify for Training or For Inference or if Both
GRPO Training:
```bash
python3 grpo_training.py --model phi --quantize --task aime \
--train_dataset_path ../tasks/aime/aime_train_dataset.jsonl \
--eval_dataset_path ../tasks/aime/aime_eval_dataset.jsonl
```

SFT Training:
```bash
python3 sft_training.py --model deepseek --task countdown
```
GRPO after SFT:
```bash
python3 upload_adapter.py --adapter_path ./outputs/... \
--repo_id YOUR_HF_REPO_ID
python3 grpo_training.py --model deepseek --adapter YOUR_HF_REPO_ID ...
```
D. Evaluation
```bash
python eval.py --weights checkpoints/best_model.pth
```
for inference:
```bash
python3 query_model.py \
--model deepseek-ai/deepseek-coder-7b-instruct-v1.5 \
--adapter_repo TDani/deepseek_aime_n100_mcl_256_quantized \
--prompt_file ../tasks/aime/aime_eval_dataset.jsonl \
--output_file deepseek_outputs/output.jsonl
```

E. Quickstart: Minimum Reproducible Result

# Step 1: Set up environment
```bash
pip install -r requirements.txt
```
# Step 2: Download dataset
```bash
python3 ./tasks/scripts/get_countdown_dataset.py --base_model deepseek
```
# Step 3: Train
```bash
python3 grpo_training.py --model deepseek --quantize --task countdown \
--train_dataset_path ../tasks/countdown/countdown_train_dataset.jsonl
```
# Step 4: Evaluate
```bash
python eval.py --weights checkpoints/best_model.pth
```
5. Notes
All GRPO datasets are in tasks/, preprocessed via get_aime_dataset.py or get_countdown_dataset.py.
Fine-tuned adapters are uploaded via upload_adapter.py.
All logs and TensorBoard profiles (until May 16th) can be accessed at:
>> T4 vs. L4 Comparison
>> DeepSeek vs. Phi Comparison
Final scores are located in inference/scores/.
Profiling logs are saved in ./logs/profiler.

