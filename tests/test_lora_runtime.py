"""Offline miniature PEFT round-trip; requires the project's ML environment."""
import importlib.util
import unittest

AVAILABLE = all(importlib.util.find_spec(name) is not None
                for name in ("torch", "transformers", "peft"))


@unittest.skipUnless(AVAILABLE, "requires project torch, transformers and peft")
class LoraRuntimeTests(unittest.TestCase):
    def test_frozen_inputs_gradients_save_load_and_merge(self):
        import copy
        import torch
        from transformers import GPT2Config, GPT2Model
        from peft import (LoraConfig, TaskType, get_peft_model,
                          get_peft_model_state_dict, set_peft_model_state_dict)

        torch.manual_seed(7)
        base = GPT2Model(GPT2Config(n_embd=16, n_layer=1, n_head=2,
                                   n_positions=32, vocab_size=20, use_cache=False))
        del base.wte
        base.requires_grad_(False)
        base.gradient_checkpointing_disable()
        fresh = copy.deepcopy(base)
        config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=2,
                            lora_alpha=4, lora_dropout=0,
                            target_modules=["c_attn", "c_proj", "c_fc"], bias="none")
        model = get_peft_model(base, config)
        inputs = torch.randn(1, 5, 16)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.01)
        target = torch.randn_like(inputs)
        loss = (model(inputs_embeds=inputs).last_hidden_state - target).square().mean()
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum().item() > 0
                            for n, p in model.named_parameters() if "lora_" in n))
        self.assertTrue(all(p.grad is None for n, p in model.named_parameters() if "lora_" not in n))
        optimizer.step()
        model.eval()
        restored = get_peft_model(fresh, copy.deepcopy(config))
        set_peft_model_state_dict(restored, get_peft_model_state_dict(model))
        restored.eval()
        with torch.no_grad():
            expected = model(inputs_embeds=inputs).last_hidden_state
            torch.testing.assert_close(restored(inputs_embeds=inputs).last_hidden_state, expected)
            merged = restored.merge_and_unload().eval()
            torch.testing.assert_close(merged(inputs_embeds=inputs).last_hidden_state,
                                       expected, atol=1e-5, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
