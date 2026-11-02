"""Cover long inputs with bounded token windows before making one embedding.

Short inputs use the encoder's original encode() path. Long inputs use the same
Transformer/pooling/Normalize modules through forward(), without decoding token
IDs back to text. Non-overlapping windows are averaged by their content-token
counts, then optionally L2-normalized. A model's own Normalize module still runs
for every window. This preserves tail coverage, not full-document attention.

The long-input path supports standard Hugging Face Transformer sentence models,
float embeddings, and sentence_embedding output. Prompted long inputs fail
explicitly: repeating or splitting a pooling-excluded prompt would otherwise
silently change the encoder's semantics. Short prompted inputs remain supported.
"""

from __future__ import annotations

from itertools import chain, islice


class WindowedEncoder:
    MAX_WINDOW_BATCH_SIZE = 32

    def __init__(self, encoder):
        self._encoder = encoder

    def __getattr__(self, name):
        return getattr(self._encoder, name)

    def _window_settings(self):
        tokenizer = getattr(self._encoder, "tokenizer", None)
        sequence_limit = getattr(self._encoder, "max_seq_length", None)
        if tokenizer is None or sequence_limit is None:
            # Deterministic test encoders and non-tokenizing encoders keep their
            # existing API, including return types and keyword arguments.
            return None
        required = ("num_special_tokens_to_add", "prepare_for_model", "pad")
        if not callable(tokenizer) or not all(callable(getattr(tokenizer, name, None)) for name in required):
            raise ValueError("长文本编码需要支持 token IDs 和 padding 的 Hugging Face tokenizer。")
        if not callable(getattr(self._encoder, "forward", None)):
            raise ValueError("长文本编码需要可调用的 sentence encoder forward。")
        sequence_limit = int(sequence_limit)
        special_count = int(tokenizer.num_special_tokens_to_add(pair=False))
        if special_count < 0 or sequence_limit <= special_count:
            raise ValueError("向量模型 max_seq_length 必须大于特殊 token 所需的长度。")
        first_module = getattr(self._encoder, "_first_module", lambda: None)()
        lower_case = bool(getattr(first_module, "do_lower_case", False))
        return tokenizer, sequence_limit, sequence_limit - special_count, lower_case

    def _prompt(self, kwargs):
        prompt = kwargs.get("prompt")
        if prompt is not None:
            return prompt
        name = kwargs.get("prompt_name")
        prompts = getattr(self._encoder, "prompts", {})
        if name is not None:
            if name not in prompts:
                raise ValueError(f"向量模型未配置 prompt_name: {name}")
            return prompts[name]
        default = getattr(self._encoder, "default_prompt_name", None)
        return prompts.get(default, "") if default is not None else ""

    @staticmethod
    def _tokenize(tokenizer, text, lower_case):
        # Match sentence_transformers.models.Transformer.tokenize preprocessing.
        text = text.strip()
        if lower_case:
            text = text.lower()
        return tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            verbose=False,
        )

    def encode(self, sentences, **kwargs):
        settings = self._window_settings()
        if settings is None:
            return self._encoder.encode(sentences, **kwargs)
        texts = [sentences] if isinstance(sentences, str) else list(sentences)
        if not texts:
            return self._encoder.encode(sentences, **kwargs)
        if not all(isinstance(text, str) for text in texts):
            raise TypeError("WindowedEncoder.encode 仅接受字符串或字符串列表。")
        tokenizer, sequence_limit, window_size, lower_case = settings
        prompt = self._prompt(kwargs) or ""
        # Tokenize one text at a time, retaining only lengths in this scan. Do
        # not retain a full corpus/batch of unbounded token-ID buffers.
        long_inputs = [
            len(self._tokenize(tokenizer, prompt + text, lower_case)["input_ids"]) > window_size
            for text in texts
        ]
        if not any(long_inputs):
            return self._encoder.encode(sentences, **kwargs)
        if prompt:
            raise ValueError("长文本窗口编码暂不支持 prompt/default_prompt_name；请使用无 prompt 的编码配置。")
        if kwargs.get("output_value", "sentence_embedding") != "sentence_embedding":
            raise ValueError("长文本窗口编码仅支持 output_value='sentence_embedding'。")
        if kwargs.get("precision", "float32") not in {"float32", None}:
            raise ValueError("长文本窗口编码仅支持浮点向量，不支持量化 precision。")
        batch_size = int(kwargs.get("batch_size", 32))
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0。")
        batch_size = min(batch_size, self.MAX_WINDOW_BATCH_SIZE)
        import torch

        vectors = [None] * len(texts)
        # Keep short-text encoding on the original encode path even when a
        # neighboring input requires windows; preserve original input order.
        short_indices = [i for i, is_long in enumerate(long_inputs) if not is_long]
        short_kwargs = {**kwargs, "convert_to_tensor": True, "convert_to_numpy": False}
        short_kwargs["batch_size"] = batch_size
        for offset in range(0, len(short_indices), batch_size):
            positions = short_indices[offset : offset + batch_size]
            encoded = self._encoder.encode([texts[i] for i in positions], **short_kwargs)
            self._validate_vectors(encoded, len(positions))
            for index, vector in zip(positions, encoded):
                vectors[index] = vector.detach().cpu()

        device = kwargs.get("device") or self._encoder.device
        self._encoder.to(device)
        self._encoder.eval()
        for index, is_long in enumerate(long_inputs):
            if is_long:
                tokenized = self._tokenize(tokenizer, texts[index], lower_case)
                vectors[index] = self._encode_windows(
                    tokenized, tokenizer, sequence_limit, window_size, batch_size, device, kwargs
                )

        embeddings = torch.stack(vectors)
        if kwargs.get("convert_to_tensor", False):
            embeddings = embeddings.to(device)
        elif kwargs.get("convert_to_numpy", True):
            if embeddings.dtype == torch.bfloat16:
                embeddings = embeddings.float()
            embeddings = embeddings.numpy()
        else:
            embeddings = [vector.to(device) for vector in embeddings]
        return embeddings[0] if isinstance(sentences, str) else embeddings

    @staticmethod
    def _validate_vectors(vectors, count):
        import torch

        if not isinstance(vectors, torch.Tensor) or vectors.ndim != 2 or vectors.shape[0] != count:
            raise ValueError("向量模型必须返回与输入窗口数量一致的二维张量。")
        if vectors.shape[1] == 0 or not bool(torch.isfinite(vectors).all()):
            raise ValueError("向量模型返回了空向量或非有限数值。")
        norms = torch.linalg.vector_norm(vectors.float(), dim=1)
        if not bool(torch.isfinite(norms).all()):
            raise ValueError("向量模型返回的向量范数溢出为非有限数值。")
        if bool((norms == 0).any()):
            raise ValueError("向量模型返回了零范数向量，无法进行可靠的相似度计算。")

    @staticmethod
    def _prepared_windows(tokenized, tokenizer, window_size):
        encodings = getattr(tokenized, "encodings", None)
        if encodings:
            # Generic PreTrainedTokenizerFast has its special-token template in
            # tokenizer.json. Its Python prepare_for_model() can ignore that
            # template and produce mismatched input_ids/token_type_ids. Process
            # the original Rust Encodings instead, retaining exact token IDs,
            # offsets and the tokenizer's real CLS/SEP/type-ID behavior.
            encoding = encodings[0]
            encoding.truncate(window_size, stride=0, direction="right")
            for window in chain((encoding,), encoding.overflowing):
                prepared = tokenizer.backend_tokenizer.post_process(window, add_special_tokens=True)
                features = {"input_ids": prepared.ids, "attention_mask": prepared.attention_mask}
                if "token_type_ids" in tokenizer.model_input_names:
                    features["token_type_ids"] = prepared.type_ids
                yield features, len(window.ids)
        else:
            # Slow tokenizers implement their special-token construction in
            # Python and have no tokenizers.Encoding/postprocessor backend.
            ids = tokenized["input_ids"]
            for start in range(0, len(ids), window_size):
                window = ids[start : start + window_size]
                yield tokenizer.prepare_for_model(
                    window, add_special_tokens=True, truncation=False, return_attention_mask=True
                ), len(window)

    def _encode_windows(self, tokenized, tokenizer, sequence_limit, window_size, batch_size, device, kwargs):
        import torch

        encode_options = {
            "prompt_name", "prompt", "batch_size", "show_progress_bar", "output_value", "precision",
            "convert_to_numpy", "convert_to_tensor", "device", "normalize_embeddings",
        }
        forward_kwargs = {key: value for key, value in kwargs.items() if key not in encode_options}
        total = None
        truncate_dim = getattr(self._encoder, "truncate_dim", None)
        prepared_windows = self._prepared_windows(tokenized, tokenizer, window_size)
        while windows := list(islice(prepared_windows, batch_size)):
            features = [item for item, _ in windows]
            if any(len(item["input_ids"]) > sequence_limit for item in features):
                raise ValueError("tokenizer 添加特殊 token 后超过模型窗口，已停止以避免截断源码。")
            if any(
                len(item[key]) != len(item["input_ids"])
                for item in features for key in ("attention_mask", "token_type_ids") if key in item
            ):
                raise ValueError("tokenizer 生成的 input_ids、attention_mask、token_type_ids 长度不一致。")
            batch = tokenizer.pad(features, padding=True, return_tensors="pt")
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.inference_mode():
                vectors = self._encoder.forward(batch, **forward_kwargs)["sentence_embedding"]
                if truncate_dim is not None:
                    vectors = vectors[:, :truncate_dim]
                self._validate_vectors(vectors, len(windows))
                vectors = vectors.detach().to(device="cpu", dtype=torch.float32)
                weights = torch.tensor([count for _, count in windows], dtype=torch.float32)
                weighted = (vectors * weights[:, None]).sum(dim=0)
                total = weighted if total is None else total + weighted
        # Count only content IDs. Added CLS/SEP/padding do not overweight the
        # final short window. Normalize modules inside forward still apply.
        average = total / len(tokenized["input_ids"])
        self._validate_vectors(average[None, :], 1)
        if kwargs.get("normalize_embeddings", False):
            average = torch.nn.functional.normalize(average, p=2, dim=0)
        return average
