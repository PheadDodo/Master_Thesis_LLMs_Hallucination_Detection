"""Lightweight shared model-loading helpers for Llama/Qwen stages."""

from dataclasses import dataclass
from typing import Any
from typing import Iterator
from typing import TypeAlias


LoadedModelTuple: TypeAlias = tuple[Any, Any, Any, Any]

__all__ = [
    "add_model_loading_args",
    "LoadedModelPair",
    "LoadedModelTuple",
    "load_llama_qwen_model_pair",
    "load_llama_qwen_models",
    "resolve_qwen_load_in_8bit",
]


def add_model_loading_args(parser, config_module):
    """Add shared Llama/Qwen model-loading CLI arguments to a parser."""

    parser.add_argument("--llm_llama_path", type=str, default=config_module.LLM_LLAMA_PATH)
    parser.add_argument("--llm_qwen_path", type=str, default=config_module.LLM_QWEN_PATH)
    parser.add_argument("--device_map", type=str, default=config_module.DEVICE_MAP)
    parser.add_argument("--qwen_device_map", type=str, default=config_module.QWEN_DEVICE_MAP)
    parser.add_argument("--load_in_8bit", dest="load_in_8bit", action="store_true")
    parser.add_argument("--no_load_in_8bit", dest="load_in_8bit", action="store_false")
    parser.set_defaults(load_in_8bit=config_module.LOAD_IN_8BIT)
    parser.add_argument("--qwen_load_in_8bit", dest="qwen_load_in_8bit", action="store_true")
    parser.add_argument("--no_qwen_load_in_8bit", dest="qwen_load_in_8bit", action="store_false")
    parser.set_defaults(qwen_load_in_8bit=config_module.QWEN_LOAD_IN_8BIT)

    return parser


@dataclass(frozen=True)
class LoadedModelPair:
    """Loaded Llama/Qwen models and tokenizers in the pipeline's tuple order."""

    llm_llama: Any
    tokenizer_llama: Any
    llm_qwen: Any
    tokenizer_qwen: Any

    def as_tuple(self) -> LoadedModelTuple:
        return self.llm_llama, self.tokenizer_llama, self.llm_qwen, self.tokenizer_qwen

    def __iter__(self) -> Iterator[Any]:
        return iter(self.as_tuple())


def _unpack_model_and_tokenizer(load_result, model_name, model_path):
    try:
        model, tokenizer, _ = load_result
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{model_name} loader for {model_path!r} must return "
            "(model, tokenizer, metadata)."
        ) from exc

    return model, tokenizer


def _load_model_and_tokenizer(
    load_llm_fn,
    model_name,
    model_path,
    load_in_8bit,
    dtype,
    device_map,
):
    return _unpack_model_and_tokenizer(
        load_llm_fn(
            model_path,
            load_in_8bit=load_in_8bit,
            dtype=dtype,
            device_map=device_map,
        ),
        model_name=model_name,
        model_path=model_path,
    )


def resolve_qwen_load_in_8bit(args, qwen_load_in_8bit=None):
    """Resolve Qwen 8-bit policy from an explicit override or parsed args."""

    if qwen_load_in_8bit is not None:
        return qwen_load_in_8bit

    return getattr(args, "qwen_load_in_8bit", False)


def load_llama_qwen_model_pair(args, load_llm_fn, dtype, qwen_load_in_8bit=None):
    """Load Llama and Qwen models into a named result object."""

    qwen_load_in_8bit = resolve_qwen_load_in_8bit(args, qwen_load_in_8bit)

    llm_llama, tokenizer_llama = _load_model_and_tokenizer(
        load_llm_fn=load_llm_fn,
        model_name="Llama",
        model_path=args.llm_llama_path,
        load_in_8bit=args.load_in_8bit,
        dtype=dtype,
        device_map=args.device_map,
    )

    llm_qwen, tokenizer_qwen = _load_model_and_tokenizer(
        load_llm_fn=load_llm_fn,
        model_name="Qwen",
        model_path=args.llm_qwen_path,
        load_in_8bit=qwen_load_in_8bit,
        dtype=dtype,
        device_map=args.qwen_device_map,
    )

    return LoadedModelPair(
        llm_llama=llm_llama,
        tokenizer_llama=tokenizer_llama,
        llm_qwen=llm_qwen,
        tokenizer_qwen=tokenizer_qwen,
    )


def load_llama_qwen_models(args, load_llm_fn, dtype, qwen_load_in_8bit=None):
    """Load Llama and Qwen models using the legacy tuple return shape."""

    return load_llama_qwen_model_pair(
        args,
        load_llm_fn=load_llm_fn,
        dtype=dtype,
        qwen_load_in_8bit=qwen_load_in_8bit,
    ).as_tuple()
