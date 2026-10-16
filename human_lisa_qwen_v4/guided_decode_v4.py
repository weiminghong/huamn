from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import torch
import transformers
import transformers.tokenization_utils as tokenization_utils

from .schema_v4 import GUIDED_JSON_SCHEMA_V4, SPECIAL_TOKENS_V4


def _load_enforcer():
    if not hasattr(tokenization_utils, "PreTrainedTokenizerBase"):
        tokenization_utils.PreTrainedTokenizerBase = (
            transformers.PreTrainedTokenizerBase
        )
    try:
        from lmformatenforcer import JsonSchemaParser, TokenEnforcerTokenizerData
        from lmformatenforcer.integrations.transformers import (
            build_token_enforcer_tokenizer_data,
            build_transformers_prefix_allowed_tokens_fn,
        )
    except ImportError as error:
        raise RuntimeError(
            "v4 evaluation requires lm-format-enforcer>=0.11.3"
        ) from error
    return (
        JsonSchemaParser,
        TokenEnforcerTokenizerData,
        build_token_enforcer_tokenizer_data,
        build_transformers_prefix_allowed_tokens_fn,
    )


def build_guided_prefix_allowed_tokens_fn(
    tokenizer, prompt_width: int = 0
) -> Callable:
    (
        JsonSchemaParser,
        TokenEnforcerTokenizerData,
        build_tokenizer_data,
        build_prefix_fn,
    ) = _load_enforcer()
    tokenizer_data = build_tokenizer_data(tokenizer)
    regular_tokens = list(tokenizer_data.regular_tokens)
    regular_ids = {int(item[0]) for item in regular_tokens}
    for token in SPECIAL_TOKENS_V4:
        token_id = int(tokenizer.convert_tokens_to_ids(token))
        if token_id < 0:
            raise RuntimeError(f"guided token is missing: {token}")
        if token_id not in regular_ids:
            regular_tokens.append((token_id, token, False))
    tokenizer_data = TokenEnforcerTokenizerData(
        regular_tokens,
        tokenizer_data.decoder,
        tokenizer_data.eos_token_id,
        False,
        len(tokenizer),
    )
    base = build_prefix_fn(
        tokenizer_data, JsonSchemaParser(GUIDED_JSON_SCHEMA_V4)
    )
    prompt_width = int(prompt_width)
    if prompt_width < 0:
        raise ValueError("prompt width must be non-negative")
    if prompt_width == 0:
        return base

    def allowed(batch_id: int, sequence: torch.Tensor):
        return base(batch_id, sequence[prompt_width:])

    return allowed


def assert_target_allowed(tokenizer, target_text: str) -> None:
    prefix_fn = build_guided_prefix_allowed_tokens_fn(tokenizer)
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    prefix = [int(tokenizer.bos_token_id or tokenizer.eos_token_id)]
    allowed = prefix_fn(0, torch.tensor(prefix, dtype=torch.long))
    for position, token_id in enumerate(target_ids):
        token_id = int(token_id)
        if token_id not in allowed:
            token = tokenizer.decode([token_id])
            raise RuntimeError(
                f"schema blocks target token {position}: {token_id} {token!r}"
            )
        prefix.append(token_id)
        allowed = prefix_fn(0, torch.tensor(prefix, dtype=torch.long))
    if int(tokenizer.eos_token_id) not in allowed:
        raise RuntimeError("schema does not allow EOS after a complete target")


def person_object_spans(text: str) -> List[Tuple[int, int]]:
    key_position = text.find('"persons"')
    if key_position < 0:
        return []
    array_start = text.find("[", key_position)
    if array_start < 0:
        return []
    spans = []
    depth = 0
    object_start = -1
    in_string = False
    escaped = False
    for position in range(array_start + 1, len(text)):
        character = text[position]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            if depth == 0:
                object_start = position
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0 and object_start >= 0:
                spans.append((object_start, position + 1))
                object_start = -1
            elif depth < 0:
                return []
        elif character == "]" and depth == 0:
            break
    return spans


def mean_logprob_by_spans(
    tokenizer,
    generated_ids: Sequence[int],
    token_logprobs: Sequence[float],
    spans: Sequence[Tuple[int, int]],
) -> List[float]:
    encoded = tokenizer(
        tokenizer.decode(generated_ids, skip_special_tokens=False),
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    if [int(value) for value in encoded["input_ids"]] != [
        int(value) for value in generated_ids
    ]:
        raise RuntimeError("generated text does not round-trip through tokenizer")
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
    scores = []
    for span_start, span_end in spans:
        values = [
            float(logprob)
            for (start, end), logprob in zip(offsets, token_logprobs)
            if end > start and start < span_end and end > span_start
        ]
        scores.append(sum(values) / len(values) if values else float("-inf"))
    return scores
