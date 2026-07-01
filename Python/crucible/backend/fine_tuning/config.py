"""
Fine-tuning configuration for LoRA / QLoRA / SFT.

WHAT EACH PARAMETER CONTROLS
-----------------------------

BASE MODEL
  model_id: any HuggingFace model hub path, e.g. "microsoft/phi-3-mini-4k-instruct"
  or a local path. The base model is NOT saved locally — only the adapter weights
  (~10-100MB) are saved after training.

LORA HYPERPARAMETERS
  rank (r): controls the size of the trainable low-rank matrices.
    rank=4  → very small update, minimal GPU memory, least expressive
    rank=16 → moderate, good default for instruction tuning
    rank=64 → more expressive, closer to full fine-tune quality
    Most practitioners use 8 or 16.

  alpha: scaling factor. The effective LoRA contribution = (alpha/rank) × weight_update.
    Setting alpha=2×rank is a common heuristic that keeps the effective LR stable
    as you change rank.

  dropout: applied to the LoRA matrices during training. 0.05-0.1 is typical.
    More than 0.1 usually hurts.

  target_modules: which layers to insert LoRA into.
    Attention-only: ["q_proj", "v_proj"] — conservative, fewer params
    Attention + MLP: adds ["gate_proj", "up_proj", "down_proj"] — more expressive
    "all-linear" is a PEFT shortcut to target all Linear layers.

QKORA (4-bit quantisation)
  use_qlora: requires bitsandbytes + CUDA GPU. Loads the base model in NF4
  (NormalFloat4) precision before applying LoRA. Reduces GPU memory by ~4×,
  making 7B/13B fine-tuning feasible on 16GB/24GB consumer GPUs.
  Falls back to standard LoRA if bitsandbytes is not available.

TRAINING
  epochs:        full passes through the training data. 1-3 is usually enough
                 for instruction tuning; more causes overfitting.
  learning_rate: 2e-4 is a well-validated default for LoRA SFT.
  batch_size:    effective batch = batch_size × gradient_accumulation_steps.
                 Keep batch_size small (1-4) for memory, increase accumulation.
  max_seq_len:   truncates samples longer than this. Match the base model's
                 context window or set lower for speed.

DATASET FORMAT
  alpaca: {"instruction": "...", "input": "...", "output": "..."}
  sharegpt: {"conversations": [{"from": "human"/"gpt", "value": "..."}]}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""
    rank: int               = 16     # r in the paper
    alpha: int              = 32     # scaling = alpha / rank
    dropout: float          = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    bias: str               = "none"  # none | all | lora_only


@dataclass
class QLoRAConfig:
    """
    Quantisation settings for QLoRA.
    Ignored when bitsandbytes is not installed (CPU / no-GPU environments).
    """
    load_in_4bit: bool      = True
    bnb_4bit_quant_type: str = "nf4"          # NormalFloat4 — best for LLM weights
    bnb_4bit_compute_dtype: str = "float16"   # computation dtype during forward pass
    bnb_4bit_use_double_quant: bool = True    # double-quantise the constants (~0.4 bits extra savings)


@dataclass
class FineTuningConfig:
    """
    Full fine-tuning job configuration.
    All fields have sensible defaults that work on CPU for testing.
    """
    # ── Model ─────────────────────────────────────────────────────────────
    model_id: str           = "microsoft/phi-2"   # ~2.7B, permissive licence, runs on CPU
    dataset_format: str     = "alpaca"             # alpaca | sharegpt

    # ── Method ────────────────────────────────────────────────────────────
    method: str             = "sft"                # sft | dpo (DPO in Phase 6.4)
    use_qlora: bool         = False                # True requires bitsandbytes + GPU

    # ── LoRA ──────────────────────────────────────────────────────────────
    lora: LoRAConfig        = field(default_factory=LoRAConfig)
    qlora: QLoRAConfig      = field(default_factory=QLoRAConfig)

    # ── Training ──────────────────────────────────────────────────────────
    epochs: int             = 1
    learning_rate: float    = 2e-4
    batch_size: int         = 4
    gradient_accumulation_steps: int = 4     # effective batch = batch_size × this
    max_seq_len: int        = 512
    warmup_ratio: float     = 0.03
    lr_scheduler: str       = "cosine"       # cosine | linear | constant
    weight_decay: float     = 0.001
    optim: str              = "adamw_torch"  # adamw_torch | paged_adamw_8bit (GPU)
    fp16: bool              = False          # mixed precision — only useful on GPU
    bf16: bool              = False          # bf16 > fp16 on Ampere+ GPUs

    # ── Output ────────────────────────────────────────────────────────────
    output_dir: str         = "./data/fine_tuning"
    save_steps: int         = 50
    logging_steps: int      = 10
    eval_steps: int         = 50
    max_steps: int          = -1             # -1 = run all epochs

    # ── HuggingFace Hub ───────────────────────────────────────────────────
    push_to_hub: bool       = False
    hub_model_id: Optional[str] = None      # e.g. "yourname/my-fine-tuned-model"

    def validate(self) -> list[str]:
        """Returns a list of validation error messages, empty if valid."""
        errors = []
        if self.lora.rank < 1 or self.lora.rank > 256:
            errors.append(f"LoRA rank must be 1-256, got {self.lora.rank}")
        if self.lora.alpha < 1:
            errors.append(f"LoRA alpha must be >= 1, got {self.lora.alpha}")
        if not 0 <= self.lora.dropout < 1:
            errors.append(f"LoRA dropout must be 0-1, got {self.lora.dropout}")
        if self.epochs < 1:
            errors.append(f"Epochs must be >= 1, got {self.epochs}")
        if self.learning_rate <= 0:
            errors.append(f"Learning rate must be > 0, got {self.learning_rate}")
        if self.batch_size < 1:
            errors.append(f"Batch size must be >= 1, got {self.batch_size}")
        if self.dataset_format not in ("alpaca", "sharegpt"):
            errors.append(f"dataset_format must be 'alpaca' or 'sharegpt'")
        if self.method not in ("sft", "dpo"):
            errors.append(f"method must be 'sft' or 'dpo'")
        return errors


@dataclass
class DPOConfig:
    """
    Configuration for Direct Preference Optimisation (DPO).

    DPO trains the model to prefer "chosen" responses over "rejected" ones
    without a separate reward model. It directly optimises a policy from
    (prompt, chosen, rejected) triplets using an implicit reward signal.

    BETA PARAMETER
    --------------
    β controls how strongly the policy can deviate from the reference model.
      β = 0.1  → stays close to reference (good for mild alignment nudges)
      β = 0.5  → moderate divergence (default)
      β = 1.0  → large divergence (risks reward hacking on small datasets)

    MEMORY REQUIREMENT
    ------------------
    DPO requires 2× batch memory vs SFT because both chosen and rejected
    sequences are processed in each forward pass. Reduce batch_size to 1
    and increase gradient_accumulation_steps to compensate.
    """
    model_id: str       = "microsoft/phi-2"
    sft_model_path: Optional[str] = None   # SFT adapter to use as reference model

    # DPO-specific hyperparameters
    beta: float         = 0.1
    max_prompt_length: int = 256
    max_length: int     = 512

    # LoRA adapter
    lora: LoRAConfig    = field(default_factory=LoRAConfig)
    use_qlora: bool     = False

    # Training
    epochs: int         = 1
    learning_rate: float = 5e-5
    batch_size: int     = 2
    gradient_accumulation_steps: int = 8
    output_dir: str     = "./data/fine_tuning"
    push_to_hub: bool   = False
    hub_model_id: Optional[str] = None

    def validate(self) -> list[str]:
        errors = []
        if not 0.0 < self.beta <= 2.0:
            errors.append(f"DPO beta must be in (0, 2], got {self.beta}")
        if self.epochs < 1:
            errors.append(f"Epochs must be >= 1, got {self.epochs}")
        if self.learning_rate <= 0:
            errors.append(f"Learning rate must be > 0, got {self.learning_rate}")
        if self.batch_size < 1:
            errors.append(f"Batch size must be >= 1, got {self.batch_size}")
        return errors
