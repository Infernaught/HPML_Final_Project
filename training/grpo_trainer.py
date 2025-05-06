from transformers import Trainer
import torch

# we are using this custom traine, since "trl import GRPOTrainer has issues with .generate for deepseek
# this one has crash safety as it skips bad data

class GRPOTrainer(Trainer):
    def __init__(self, reward_funcs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_funcs = reward_funcs
        self.tokenizer = kwargs.get("tokenizer")



    def compute_loss(self, model, inputs, return_outputs=False):
        # inputs["scores"] is a Python list of floats.
        scores = torch.tensor(inputs["scores"], device=model.device)
        # maximize mean reward == minimize -mean
        loss   = - scores.mean()
        if return_outputs:
            return loss, None
        return loss



    def training_step(self, model, inputs, optimizer):
        # 1) generate & score
        batch = self._generate_and_score_completions(inputs)

        # 2) debug check for bad token‐ids
        ids   = batch["input_ids"]
        vmax  = ids.max().item()
        vmin  = ids.min().item()
        V     = self.tokenizer.vocab_size
        assert 0 <= vmin <= vmax < V, (
            f"Invalid token id in batch: found [{vmin}, {vmax}], "
            f"but vocab_size={V}"
        )

        # 3) PG loss & backward
        loss = - torch.tensor(batch["scores"], device=model.device).mean()
        loss.backward()
        return loss



    def _generate_and_score_completions(self, batch):
        unmod = self.accelerator.unwrap_model(self.model)
        try:
            gen_ids = unmod.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=self.args.max_completion_length,
                do_sample=True, top_p=0.95, temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        except RuntimeError as e:
            print(f"⚠️ Skipping batch: {e}")
            batch["completions"] = ["<GEN_FAIL>"] * len(batch["input_ids"])
            batch["scores"]      = [0.0]            * len(batch["input_ids"])
            return batch

        batch["completions"] = self.tokenizer.batch_decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        batch["scores"]      = [
            self.reward_funcs(p, c)
            for p, c in zip(batch["prompt"], batch["completions"])
        ]
        return batch