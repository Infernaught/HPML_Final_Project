import json
import os

def format_chat(prompt, answer):
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{prompt.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n{answer.strip()}"
    )

def convert_file(input_path, output_path):
    kept, skipped = 0, 0
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            try:
                data = json.loads(line)
                prompt = data.get("prompt", "").strip()
                answer = data.get("answer", "").strip()

                if not prompt or not answer:
                    skipped += 1
                    continue

                formatted = format_chat(prompt, answer)
                json.dump({"prompt": formatted}, outfile)
                outfile.write("\n")
                kept += 1
            except Exception:
                skipped += 1

    print(f"✅ Done: {kept} examples written, {skipped} skipped.\n→ Output: {output_path}")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../aime"))

    input_train = os.path.join(base_dir, "aime_train_dataset.jsonl")
    input_eval = os.path.join(base_dir, "aime_eval_dataset.jsonl")

    output_train = os.path.join(base_dir, "aime_train_dataset_deepseek.jsonl")
    output_eval = os.path.join(base_dir, "aime_eval_dataset_deepseek.jsonl")

    print("Converting training set...")
    convert_file(input_train, output_train)

    print("Converting eval set...")
    convert_file(input_eval, output_eval)
