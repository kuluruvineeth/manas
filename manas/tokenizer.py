from pathlib import Path

from jinja2 import Environment

VOCAB_SIZE = 6400

SPECIAL_TOKENS = [
    "<|endoftext|>", "<|im_start|>", "<|im_end|>",
    "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>",
    "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
    "<|audio_start|>", "<|audio_end|>", "<|audio_pad|>",
    "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>", "<tts_text_bos_single>",
]
CONTROL_TOKENS = ["<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>", "<think>", "</think>"]
BUFFER_TOKENS = [f"<|buffer{i}|>" for i in range(1, 10)]
ADDED_TOKENS = SPECIAL_TOKENS + CONTROL_TOKENS + BUFFER_TOKENS

PAD_TOKEN, BOS_TOKEN, EOS_TOKEN = SPECIAL_TOKENS[:3]
THINK_START, THINK_END = "<think>", "</think>"
TOOL_CALL_START, TOOL_CALL_END = "<tool_call>", "</tool_call>"

CHAT_TEMPLATE = Path(__file__).with_name("chat_template.jinja").read_text(encoding="utf-8")

_env = Environment(keep_trailing_newline=True)
_template = _env.from_string(CHAT_TEMPLATE)


def token_id(token):
    return ADDED_TOKENS.index(token)


def render_chat(messages, tools=None, add_generation_prompt=False, open_thinking=False):
    return _template.render(
        messages=messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        open_thinking=open_thinking,
    )


def added_tokens_decoder():
    return {
        str(index): {
            "content": token,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": token in SPECIAL_TOKENS,
        }
        for index, token in enumerate(ADDED_TOKENS)
    }


def tokenizer_config():
    return {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": added_tokens_decoder(),
        "additional_special_tokens": [t for t in SPECIAL_TOKENS if t != PAD_TOKEN],
        "bos_token": BOS_TOKEN,
        "clean_up_tokenization_spaces": False,
        "eos_token": EOS_TOKEN,
        "legacy": True,
        "model_max_length": 131072,
        "pad_token": PAD_TOKEN,
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "unk_token": PAD_TOKEN,
        "image_token": "<|image_pad|>",
        "audio_token": "<|audio_pad|>",
        "video_token": "<|video_pad|>",
        "vision_bos_token": "<|vision_start|>",
        "vision_eos_token": "<|vision_end|>",
        "audio_bos_token": "<|audio_start|>",
        "audio_eos_token": "<|audio_end|>",
        "chat_template": CHAT_TEMPLATE,
        "tokenizer_class": "PreTrainedTokenizerFast",
    }
