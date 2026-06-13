"""Lightweight shared CLI helpers for experiment stage scripts."""

import argparse


def positive_int(value):
    """Parse a positive integer for argparse CLI options."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc

    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")

    return parsed


def add_generation_args(parser, config_module):
    """Add shared sampling/generation CLI arguments to a parser."""

    parser.add_argument("--temperature", type=float, default=config_module.TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=config_module.TOP_P)
    parser.add_argument("--n", type=int, default=config_module.N)

    return parser


def add_baseline_output_args(parser, config_module):
    """Add shared Llama/Qwen baseline output path arguments to a parser."""

    parser.add_argument("--output_path_llama", type=str, default=config_module.OUTPUT_PATH_LLAMA)
    parser.add_argument("--output_path_qwen", type=str, default=config_module.OUTPUT_PATH_QWEN)

    return parser
