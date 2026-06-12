"""Lightweight shared CLI helpers for experiment stage scripts."""


def add_generation_args(parser, config_module):
    """Add shared sampling/generation CLI arguments to a parser."""

    parser.add_argument("--temperature", type=float, default=config_module.TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=config_module.TOP_P)
    parser.add_argument("--n", type=int, default=config_module.N)

    return parser
