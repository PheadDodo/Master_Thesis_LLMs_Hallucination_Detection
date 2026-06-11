def load_llama_qwen_models(args, load_llm_fn, dtype):
    llm_llama, tokenizer_llama, _ = load_llm_fn(
        args.llm_llama_path,
        load_in_8bit=args.load_in_8bit,
        dtype=dtype,
        device_map=args.device_map,
    )

    llm_qwen, tokenizer_qwen, _ = load_llm_fn(
        args.llm_qwen_path,
        load_in_8bit=False,
        dtype=dtype,
        device_map=args.qwen_device_map,
    )

    return llm_llama, tokenizer_llama, llm_qwen, tokenizer_qwen
