from dataclasses import dataclass


@dataclass(frozen=True)
class LoadedModelPair:
    llm_llama: object
    tokenizer_llama: object
    llm_qwen: object
    tokenizer_qwen: object

    def as_tuple(self):
        return self.llm_llama, self.tokenizer_llama, self.llm_qwen, self.tokenizer_qwen

    def __iter__(self):
        return iter(self.as_tuple())


def _model_and_tokenizer(load_result, model_name, model_path):
    try:
        model, tokenizer, _ = load_result
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{model_name} loader for {model_path!r} must return "
            "(model, tokenizer, metadata)."
        ) from exc

    return model, tokenizer


def load_llama_qwen_model_pair(args, load_llm_fn, dtype, qwen_load_in_8bit=False):
    llm_llama, tokenizer_llama = _model_and_tokenizer(
        load_llm_fn(
            args.llm_llama_path,
            load_in_8bit=args.load_in_8bit,
            dtype=dtype,
            device_map=args.device_map,
        ),
        model_name="Llama",
        model_path=args.llm_llama_path,
    )

    llm_qwen, tokenizer_qwen = _model_and_tokenizer(
        load_llm_fn(
            args.llm_qwen_path,
            load_in_8bit=qwen_load_in_8bit,
            dtype=dtype,
            device_map=args.qwen_device_map,
        ),
        model_name="Qwen",
        model_path=args.llm_qwen_path,
    )

    return LoadedModelPair(
        llm_llama=llm_llama,
        tokenizer_llama=tokenizer_llama,
        llm_qwen=llm_qwen,
        tokenizer_qwen=tokenizer_qwen,
    )


def load_llama_qwen_models(args, load_llm_fn, dtype, qwen_load_in_8bit=False):
    return load_llama_qwen_model_pair(
        args,
        load_llm_fn=load_llm_fn,
        dtype=dtype,
        qwen_load_in_8bit=qwen_load_in_8bit,
    ).as_tuple()
