import os
import torch
import torch_pruning as tp
from transformers import AutoModelForCausalLM

CHECKPOINT_PATH = "training/outputs/microsoft/Phi-3.5-mini-instruct/checkpoint-66"
SAVE_PATH = "training/outputs/microsoft/Phi-3.5-mini-instruct/pruned-checkpoint-66"
PRUNING_RATIO = 0.3  # 30% channels removed from linear layers

def load_model(checkpoint_path):
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    return model

def prune_model(model, example_input):
    ignored_layers = (torch.nn.Embedding, torch.nn.LayerNorm)

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=(example_input,),
        importance=tp.importance.MagnitudeImportance(p=2),  # L2 norm
        iterative_steps=1,
        ch_sparsity=PRUNING_RATIO,
        ignored_layers=ignored_layers,
        output_transform=lambda out: out.logits,
    )

    print("Pruning...")
    pruner.step()
    return model

def save_model(model, tokenizer, save_path):
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)

    model = load_model(CHECKPOINT_PATH)

    # Dummy input for pruning trace
    input_ids = tokenizer("The area of a circle is", return_tensors="pt").input_ids.to(model.device)

    # Prune
    pruned_model = prune_model(model, input_ids)

    # Save
    save_model(pruned_model, tokenizer, SAVE_PATH)
    print(f"Pruned model saved to {SAVE_PATH}")

if __name__ == "__main__":
    main()
