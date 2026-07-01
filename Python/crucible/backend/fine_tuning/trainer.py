"""
SFT Trainer — Supervised Fine-Tuning with LoRA / QLoRA.

ARCHITECTURE
------------
The trainer wraps TRL's SFTTrainer with PEFT's LoraConfig to produce
a parameter-efficient adapter that can be merged back into the base model
or deployed standalone.

TRAINING PIPELINE
-----------------
1. Load base model (fp32 / fp16 / 4-bit with QLoRA)
2. Wrap with LoRA adapter (PEFT)
3. Format training samples with the dataset formatter
4. Train with TRL SFTTrainer
5. Save adapter weights to output_dir/{job_id}/
6. Optionally push adapter to HuggingFace Hub

WHAT IS SAVED
-------------
Only the adapter weights (q_proj, v_proj, etc. delta matrices) are saved.
These are small: rank=16 on Phi-2 (~10MB). The base model weights are NOT
copied — they stay on HuggingFace Hub and are loaded at inference time.

This means the adapter artifact is portable and cheap to store. To deploy:
  model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
  model = PeftModel.from_pretrained(model, "./adapters/job_42/")

PROGRESS STREAMING
------------------
The trainer uses the same asyncio.Queue + WebSocket sentinel pattern as the
AutoML training runner. A custom TrainerCallback pushes step-level metrics
(loss, learning_rate, epoch progress) to the queue.

MOCK MODE
---------
When model_id starts with "mock-" (e.g. "mock-phi"), the trainer skips
model loading and runs a fake training loop. Used in tests and CI where
downloading a 2-3GB model is not feasible.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

from fine_tuning.config import FineTuningConfig
from fine_tuning.formatter import format_sample, validate_samples


# ── Progress event types ──────────────────────────────────────────────────────

@dataclass
class StepEvent:
    step: int
    total_steps: int
    epoch: float
    loss: float
    learning_rate: float
    elapsed_secs: float


@dataclass
class EpochEvent:
    epoch: int
    total_epochs: int
    train_loss: float
    eval_loss: Optional[float]


@dataclass
class CompleteEvent:
    adapter_path: str
    total_steps: int
    final_loss: float
    elapsed_secs: float
    hub_url: Optional[str] = None


@dataclass
class ErrorEvent:
    message: str


# ── Training result ───────────────────────────────────────────────────────────

@dataclass
class TrainingResult:
    job_id: str
    adapter_path: str
    final_loss: float
    total_steps: int
    elapsed_secs: float
    hub_url: Optional[str] = None
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


# ── SFT Trainer ───────────────────────────────────────────────────────────────

class SFTTrainer:
    """
    Runs LoRA / QLoRA supervised fine-tuning.

    Progress events are pushed to an asyncio.Queue so the caller can
    stream them to a WebSocket without blocking the event loop.
    """

    def __init__(
        self,
        config: FineTuningConfig,
        job_id: str,
        progress_queue: Optional[asyncio.Queue] = None,
    ):
        self.config = config
        self.job_id = job_id
        self.queue = progress_queue or asyncio.Queue()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _push(self, event) -> None:
        """Thread-safe push to asyncio queue from background thread."""
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self.queue.put_nowait, event)

    async def run(
        self,
        samples: list[dict],
        output_dir: str,
    ) -> TrainingResult:
        """
        Runs fine-tuning asynchronously.

        The heavy work (model loading, forward/backward passes) runs in a
        thread pool executor to avoid blocking the FastAPI event loop.
        """
        self._loop = asyncio.get_event_loop()
        start = time.monotonic()

        result = await self._loop.run_in_executor(
            None,
            self._run_sync,
            samples,
            output_dir,
            start,
        )
        await self.queue.put(None)  # sentinel — stream is finished
        return result

    def _run_sync(
        self,
        samples: list[dict],
        output_dir: str,
        start: float,
    ) -> TrainingResult:
        """Blocking training loop — runs in a thread."""

        # ── Validate samples first ────────────────────────────────────────
        errors = validate_samples(samples, self.config.dataset_format)
        if errors:
            err = f"Dataset validation failed: {'; '.join(errors[:3])}"
            self._push(ErrorEvent(message=err))
            return TrainingResult(
                job_id=self.job_id, adapter_path="",
                final_loss=0.0, total_steps=0,
                elapsed_secs=time.monotonic() - start, error=err,
            )

        # ── Mock mode for testing ─────────────────────────────────────────
        if self.config.model_id.startswith("mock-"):
            return self._mock_training(samples, output_dir, start)

        # ── Real training path ────────────────────────────────────────────
        try:
            return self._real_training(samples, output_dir, start)
        except Exception as exc:
            msg = f"Training failed: {exc}"
            self._push(ErrorEvent(message=msg))
            return TrainingResult(
                job_id=self.job_id, adapter_path="",
                final_loss=0.0, total_steps=0,
                elapsed_secs=time.monotonic() - start, error=msg,
            )

    # ── Real training ─────────────────────────────────────────────────────────

    def _real_training(
        self, samples: list[dict], output_dir: str, start: float
    ) -> TrainingResult:
        """Full LoRA/QLoRA training with TRL + PEFT."""

        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            TrainerCallback,
            TrainerState,
            TrainerControl,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTConfig, SFTTrainer as TRLSFTTrainer
        from datasets import Dataset as HFDataset

        cfg = self.config
        job_out = Path(output_dir) / self.job_id
        job_out.mkdir(parents=True, exist_ok=True)

        # ── Load tokeniser ────────────────────────────────────────────────
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # ── Load model (with optional 4-bit quantisation) ─────────────────
        model_kwargs: dict = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if cfg.fp16 else torch.float32,
        }

        use_qlora = cfg.use_qlora
        if use_qlora:
            try:
                from transformers import BitsAndBytesConfig
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=cfg.qlora.bnb_4bit_quant_type,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=cfg.qlora.bnb_4bit_use_double_quant,
                )
                model_kwargs["quantization_config"] = bnb_cfg
            except ImportError:
                use_qlora = False   # bitsandbytes not available — fall back silently

        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **model_kwargs)
        model.enable_input_require_grads()

        # ── LoRA adapter ──────────────────────────────────────────────────
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora.rank,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=cfg.lora.target_modules,
            bias=cfg.lora.bias,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

        # ── Format dataset ────────────────────────────────────────────────
        texts = [format_sample(s, cfg.dataset_format, tokenizer) for s in samples]
        hf_dataset = HFDataset.from_dict({"text": texts})

        # ── Training arguments ────────────────────────────────────────────
        total_steps = max(
            1,
            (len(samples) // (cfg.batch_size * cfg.gradient_accumulation_steps)) * cfg.epochs
        )

        sft_cfg = SFTConfig(
            output_dir=str(job_out / "checkpoints"),
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            warmup_ratio=cfg.warmup_ratio,
            lr_scheduler_type=cfg.lr_scheduler,
            optim=cfg.optim,
            fp16=cfg.fp16,
            bf16=cfg.bf16,
            logging_steps=cfg.logging_steps,
            save_steps=cfg.save_steps,
            max_seq_length=cfg.max_seq_len,
            report_to=[],           # disable wandb / tensorboard
            dataset_text_field="text",
        )

        # ── Progress callback ─────────────────────────────────────────────
        trainer_ref = self

        class ProgressCallback(TrainerCallback):
            def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kw):
                if logs is None:
                    return
                loss = logs.get("loss") or logs.get("train_loss", 0.0)
                lr   = logs.get("learning_rate", 0.0)
                trainer_ref._push(StepEvent(
                    step=state.global_step,
                    total_steps=total_steps,
                    epoch=state.epoch or 0.0,
                    loss=float(loss),
                    learning_rate=float(lr),
                    elapsed_secs=time.monotonic() - start,
                ))

        # ── Run training ──────────────────────────────────────────────────
        trl_trainer = TRLSFTTrainer(
            model=model,
            args=sft_cfg,
            train_dataset=hf_dataset,
            processing_class=tokenizer,
            callbacks=[ProgressCallback()],
        )
        trl_trainer.train()

        # ── Save adapter weights ──────────────────────────────────────────
        adapter_path = str(job_out / "adapter")
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)

        # ── Optional Hub push ─────────────────────────────────────────────
        hub_url = None
        if cfg.push_to_hub and cfg.hub_model_id:
            from config import settings
            token = getattr(settings, "huggingface_token", None)
            if token:
                model.push_to_hub(cfg.hub_model_id, token=token)
                hub_url = f"https://huggingface.co/{cfg.hub_model_id}"

        final_loss = float(trl_trainer.state.log_history[-1].get("train_loss", 0.0)) \
            if trl_trainer.state.log_history else 0.0

        elapsed = time.monotonic() - start
        self._push(CompleteEvent(
            adapter_path=adapter_path,
            total_steps=trl_trainer.state.global_step,
            final_loss=final_loss,
            elapsed_secs=elapsed,
            hub_url=hub_url,
        ))

        return TrainingResult(
            job_id=self.job_id,
            adapter_path=adapter_path,
            final_loss=final_loss,
            total_steps=trl_trainer.state.global_step,
            elapsed_secs=elapsed,
            hub_url=hub_url,
        )

    # ── Mock training (for tests and CI) ──────────────────────────────────────

    def _mock_training(
        self, samples: list[dict], output_dir: str, start: float
    ) -> TrainingResult:
        """
        Simulates training without loading any model.
        Emits realistic progress events for testing WebSocket streams.
        """
        import math

        job_out = Path(output_dir) / self.job_id
        job_out.mkdir(parents=True, exist_ok=True)
        adapter_path = str(job_out / "adapter")
        Path(adapter_path).mkdir(parents=True, exist_ok=True)

        # Write a minimal adapter_config.json so tests can verify the artifact exists
        (Path(adapter_path) / "adapter_config.json").write_text(
            '{"base_model_name_or_path": "mock-model", "peft_type": "LORA", '
            f'"r": {self.config.lora.rank}, "lora_alpha": {self.config.lora.alpha}}}',
            encoding="utf-8",
        )

        steps_per_epoch = max(1, len(samples) // max(1, self.config.batch_size))
        total_steps = steps_per_epoch * self.config.epochs
        loss = 2.5   # simulate realistic starting loss

        for step in range(1, total_steps + 1):
            # Exponential decay loss curve
            loss = 2.5 * math.exp(-3.0 * step / total_steps) + 0.1
            epoch = step / steps_per_epoch

            self._push(StepEvent(
                step=step,
                total_steps=total_steps,
                epoch=epoch,
                loss=round(loss, 4),
                learning_rate=self.config.learning_rate * (1 - step / total_steps),
                elapsed_secs=time.monotonic() - start,
            ))
            time.sleep(0.01)   # small delay to test streaming

        elapsed = time.monotonic() - start
        self._push(CompleteEvent(
            adapter_path=adapter_path,
            total_steps=total_steps,
            final_loss=round(loss, 4),
            elapsed_secs=elapsed,
        ))

        return TrainingResult(
            job_id=self.job_id,
            adapter_path=adapter_path,
            final_loss=round(loss, 4),
            total_steps=total_steps,
            elapsed_secs=elapsed,
        )
