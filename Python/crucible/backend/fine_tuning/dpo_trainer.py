"""
DPO Trainer — Direct Preference Optimisation.

DPO (Rafailov et al. 2023, "Direct Preference Optimization: Your Language Model
is Secretly a Reward Model") trains a language model to prefer one response over
another using a closed-form optimisation objective derived from RLHF's reward model.

WHY DPO INSTEAD OF RLHF
------------------------
RLHF requires three separate training phases:
  1. SFT: supervised fine-tune the base model
  2. RM:  train a reward model on (prompt, chosen, rejected) pairs
  3. PPO: reinforce the policy using the reward model as a signal

DPO eliminates steps 2 and 3 by showing that the optimal RLHF policy
has a closed form in terms of the reference model and the preferences:

  π*(y|x) ∝ π_ref(y|x) · exp(r*(y,x) / β)

...and re-parameterising the reward in terms of the log-ratio of the policy
to the reference. This means the preference data directly optimises the policy
without needing a reward model.

PRACTICAL ADVANTAGES
--------------------
- Much simpler to implement (one training phase)
- More stable training (no RL reward hacking / instability)
- Less compute (no reward model forward pass during training)
- Works well with small datasets (100–10,000 preference pairs)

DATASET FORMAT
--------------
Each sample must have three fields:
  prompt:   The input context / question
  chosen:   The better (preferred) response
  rejected: The worse (dispreferred) response

Example:
  {
    "prompt": "Explain gradient descent simply.",
    "chosen": "Gradient descent is like rolling a ball down a hill...",
    "rejected": "Gradient descent is ∇L(θ) repeated."
  }

MOCK MODE
---------
When model_id starts with "mock-", training is simulated without loading a model.
Useful for testing the pipeline without GPU or HuggingFace model downloads.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fine_tuning.config import DPOConfig
from fine_tuning.trainer import StepEvent, CompleteEvent, ErrorEvent, TrainingResult


def validate_dpo_samples(samples: list[dict]) -> list[str]:
    """
    Validates that all samples have the required DPO fields.
    Returns a list of error messages (empty = all valid).
    """
    errors = []
    for i, s in enumerate(samples[:10]):   # check first 10
        for field in ("prompt", "chosen", "rejected"):
            if field not in s or not str(s[field]).strip():
                errors.append(f"Sample {i}: missing or empty '{field}' field")
    return errors


class DPOTrainer:
    """
    Runs DPO fine-tuning on (prompt, chosen, rejected) preference pairs.

    Same interface as SFTTrainer: inject a progress queue, call run().
    Supports mock mode for testing.
    """

    def __init__(
        self,
        config: DPOConfig,
        job_id: str,
        progress_queue: Optional[asyncio.Queue] = None,
    ):
        self.config = config
        self.job_id = job_id
        self.queue  = progress_queue or asyncio.Queue()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _push(self, event) -> None:
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self.queue.put_nowait, event)

    async def run(self, samples: list[dict], output_dir: str) -> TrainingResult:
        self._loop = asyncio.get_event_loop()
        result = await self._loop.run_in_executor(
            None, self._run_sync, samples, output_dir
        )
        await self.queue.put(None)
        return result

    def _run_sync(self, samples: list[dict], output_dir: str) -> TrainingResult:
        start = time.monotonic()

        # Validate samples
        errors = validate_dpo_samples(samples)
        if errors:
            err = f"DPO dataset validation failed: {'; '.join(errors[:3])}"
            self._push(ErrorEvent(message=err))
            return TrainingResult(
                job_id=self.job_id, adapter_path="",
                final_loss=0.0, total_steps=0,
                elapsed_secs=time.monotonic() - start, error=err,
            )

        if self.config.model_id.startswith("mock-"):
            return self._mock_training(samples, output_dir, start)
        return self._real_training(samples, output_dir, start)

    # ── Real DPO training ─────────────────────────────────────────────────────

    def _real_training(self, samples: list[dict], output_dir: str, start: float) -> TrainingResult:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import LoraConfig, get_peft_model, TaskType
            from trl import DPOConfig as TRLDPOConfig, DPOTrainer as TRLDPOTrainer
            from datasets import Dataset as HFDataset

            cfg = self.config
            job_out = Path(output_dir) / self.job_id
            job_out.mkdir(parents=True, exist_ok=True)

            # Load tokeniser
            tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_id,
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )
            model.enable_input_require_grads()

            # Reference model (copy of model before LoRA — DPO needs the original)
            ref_model = AutoModelForCausalLM.from_pretrained(
                cfg.sft_model_path or cfg.model_id,
                trust_remote_code=True,
                torch_dtype=torch.float32,
            ) if cfg.sft_model_path else None

            # LoRA adapter
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=cfg.lora.rank,
                lora_alpha=cfg.lora.alpha,
                lora_dropout=cfg.lora.dropout,
                target_modules=cfg.lora.target_modules,
            )
            model = get_peft_model(model, lora_cfg)

            # Build HF dataset with the DPO fields
            hf_dataset = HFDataset.from_list([
                {"prompt": s["prompt"], "chosen": s["chosen"], "rejected": s["rejected"]}
                for s in samples
            ])

            trainer_ref = self

            from transformers import TrainerCallback, TrainerState, TrainerControl

            class ProgressCallback(TrainerCallback):
                def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kw):
                    if not logs:
                        return
                    loss = logs.get("loss") or logs.get("train_loss", 0.0)
                    trainer_ref._push(StepEvent(
                        step=state.global_step,
                        total_steps=max(1, state.max_steps),
                        epoch=state.epoch or 0.0,
                        loss=float(loss),
                        learning_rate=logs.get("learning_rate", cfg.learning_rate),
                        elapsed_secs=time.monotonic() - start,
                    ))

            dpo_cfg = TRLDPOConfig(
                output_dir=str(job_out / "checkpoints"),
                num_train_epochs=cfg.epochs,
                per_device_train_batch_size=cfg.batch_size,
                gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                learning_rate=cfg.learning_rate,
                beta=cfg.beta,
                max_prompt_length=cfg.max_prompt_length,
                max_length=cfg.max_length,
                report_to=[],
                logging_steps=10,
            )

            trainer = TRLDPOTrainer(
                model=model,
                ref_model=ref_model,
                args=dpo_cfg,
                train_dataset=hf_dataset,
                processing_class=tokenizer,
                callbacks=[ProgressCallback()],
            )
            trainer.train()

            adapter_path = str(job_out / "adapter")
            model.save_pretrained(adapter_path)
            tokenizer.save_pretrained(adapter_path)

            final_loss = float(trainer.state.log_history[-1].get("train_loss", 0.0)) \
                if trainer.state.log_history else 0.0
            elapsed = time.monotonic() - start

            self._push(CompleteEvent(
                adapter_path=adapter_path,
                total_steps=trainer.state.global_step,
                final_loss=final_loss,
                elapsed_secs=elapsed,
            ))
            return TrainingResult(
                job_id=self.job_id, adapter_path=adapter_path,
                final_loss=final_loss, total_steps=trainer.state.global_step,
                elapsed_secs=elapsed,
            )

        except Exception as exc:
            msg = f"DPO training failed: {exc}"
            self._push(ErrorEvent(message=msg))
            return TrainingResult(
                job_id=self.job_id, adapter_path="",
                final_loss=0.0, total_steps=0,
                elapsed_secs=time.monotonic() - start, error=msg,
            )

    # ── Mock training ─────────────────────────────────────────────────────────

    def _mock_training(self, samples: list[dict], output_dir: str, start: float) -> TrainingResult:
        """Simulates DPO training with a reward-margin decay curve."""
        job_out = Path(output_dir) / self.job_id
        adapter_path = str(job_out / "adapter")
        Path(adapter_path).mkdir(parents=True, exist_ok=True)

        (Path(adapter_path) / "adapter_config.json").write_text(
            f'{{"peft_type": "LORA", "r": {self.config.lora.rank}, '
            f'"base_model_name_or_path": "{self.config.model_id}", '
            f'"dpo_beta": {self.config.beta}}}',
            encoding="utf-8",
        )

        steps = max(1, len(samples) // max(1, self.config.batch_size))
        total = steps * self.config.epochs
        loss  = 0.693  # DPO starts near log(2) ≈ 0.693 (random preference)

        for step in range(1, total + 1):
            # DPO loss decays toward 0 as the model learns preferences
            loss = 0.693 * math.exp(-2.5 * step / total) + 0.05
            self._push(StepEvent(
                step=step, total_steps=total,
                epoch=step / steps,
                loss=round(loss, 4),
                learning_rate=self.config.learning_rate * (1 - step / total),
                elapsed_secs=time.monotonic() - start,
            ))
            time.sleep(0.01)

        elapsed = time.monotonic() - start
        self._push(CompleteEvent(
            adapter_path=adapter_path,
            total_steps=total,
            final_loss=round(loss, 4),
            elapsed_secs=elapsed,
        ))
        return TrainingResult(
            job_id=self.job_id, adapter_path=adapter_path,
            final_loss=round(loss, 4), total_steps=total, elapsed_secs=elapsed,
        )
