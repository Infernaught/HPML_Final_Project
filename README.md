# HPML_Final_Project

To run the general flow of experiments:

For GRPO training only:
1. Run grpo_training.py using the desired base model

For SFT training only:
1. Run sft_training.py using the desired base model

For GRPO training after SFT pretraining:
1. Run sft_training.py using the desired base model
2. Create an empty HF model repo
3. (Be logged in with a token with HF write permissions) Run merge_adapter.py with your local saved adapter weights and the correct base model
4. Run grpo_training.py using your newly uploaded HF model as the base model.

For inference:
1. If you are testing a base model, you can simply pass in the HF path to query_model.py
2. If you are testing a trained adapter, you need to:
   1. Create an empty HF model repo
   2. (Be logged in with a token with HF write permissions) Run merge_adapter.py with your local saved adapter weights and the correct base model
   3. Pass in the new HF model path to query_model.py