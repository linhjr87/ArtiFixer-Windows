# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LoRA fine-tuning options shared by model_training.train and model_eval.export_checkpoint_pt.

Kept free of torch/peft imports at module level so the export tool can build its
argument parser without importing the heavy training stack.
"""

import argparse

# Attention projections in every Wan self- and cross-attention block. peft matches
# these as module-name suffixes, so both attn1 and attn2 are covered in all blocks.
DEFAULT_LORA_TARGET_MODULES = ("to_q", "to_k", "to_v", "to_out.0")


def add_lora_opts(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--lora_rank",
        default=0,
        type=int,
        help="LoRA rank. 0 (default) disables LoRA and trains all parameters. With LoRA the "
        "base weights are frozen and only low-rank adapters train — this is what makes "
        "fine-tuning the 14B checkpoint fit on a single 32 GiB GPU (e.g. RTX 5090).",
    )
    parser.add_argument(
        "--lora_alpha",
        default=0,
        type=int,
        help="LoRA alpha scaling. 0 (default) means 2 * lora_rank.",
    )
    parser.add_argument("--lora_dropout", default=0.0, type=float)
    parser.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=list(DEFAULT_LORA_TARGET_MODULES),
        help="Module-name suffixes to adapt with LoRA.",
    )


def build_lora_config(args: argparse.Namespace):
    """Build a peft LoraConfig from parsed args. Requires args.lora_rank > 0."""
    from peft import LoraConfig

    if args.lora_rank <= 0:
        raise ValueError(f"build_lora_config requires lora_rank > 0, got {args.lora_rank}")
    return LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha if args.lora_alpha > 0 else 2 * args.lora_rank,
        lora_dropout=args.lora_dropout,
        target_modules=list(args.lora_target_modules),
        bias="none",
    )
