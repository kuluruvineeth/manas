from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    repo: str
    kind: str
    transform: str
    license: str
    why: str
    config: str | None = None
    split: str = "train"
    weight: float | None = None
    streaming: bool = True


SOURCES = [
    Source(
        repo="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        kind="pretrain",
        transform="text",
        license="odc-by",
        why="quality-filtered educational web text; the strongest general English base for a small model",
        weight=0.65,
    ),
    Source(
        repo="HuggingFaceTB/smollm-corpus",
        config="cosmopedia-v2",
        kind="pretrain",
        transform="text",
        license="odc-by",
        why="synthetic textbook-style articles; dense knowledge per token, ideal at sub-100M scale",
        weight=0.25,
    ),
    Source(
        repo="roneneldan/TinyStories",
        kind="pretrain",
        transform="text",
        license="cdla-sharing-1.0",
        why="simple narratives that give tiny models early grammatical coherence",
        weight=0.10,
    ),
    Source(
        repo="HuggingFaceTB/smoltalk",
        config="all",
        kind="sft",
        transform="messages",
        license="not stated on card (HuggingFaceTB release)",
        why="1M multi-turn conversations curated specifically for sub-1B models",
    ),
    Source(
        repo="teknium/OpenHermes-2.5",
        kind="sft",
        transform="from_value",
        license="not stated on card (compilation of variously-licensed sets)",
        why="1M diverse instruction conversations; breadth of tasks and styles",
    ),
    Source(
        repo="HuggingFaceH4/ultrachat_200k",
        split="train_sft",
        kind="sft",
        transform="messages",
        license="mit",
        why="200k long multi-turn dialogues; strengthens conversational depth",
    ),
    Source(
        repo="NousResearch/hermes-function-calling-v1",
        config="func_calling",
        kind="sft",
        transform="hermes_tools",
        license="apache-2.0",
        why="tool-calling conversations already using <tool_call>/<tool_response> tags",
        streaming=False,
    ),
    Source(
        repo="NousResearch/hermes-function-calling-v1",
        config="func_calling_singleturn",
        kind="sft",
        transform="hermes_tools",
        license="apache-2.0",
        why="single-turn tool-calling; clean minimal examples of the call format",
        streaming=False,
    ),
    Source(
        repo="NousResearch/hermes-function-calling-v1",
        config="glaive_func_calling",
        kind="sft",
        transform="hermes_tools",
        license="apache-2.0",
        why="curated glaive subset in hermes tagging; adds tool diversity",
        streaming=False,
    ),
    Source(
        repo="glaiveai/glaive-function-calling-v2",
        kind="sft",
        transform="glaive",
        license="apache-2.0",
        why="113k function-calling dialogues; the volume backbone of tool-use training",
    ),
    Source(
        repo="open-thoughts/OpenThoughts-114k",
        kind="sft",
        transform="thought_split",
        license="apache-2.0",
        why="reasoning traces split into reasoning_content + answer; only short traces fit a small context",
    ),
    Source(
        repo="HuggingFaceH4/ultrafeedback_binarized",
        split="train_prefs",
        kind="dpo",
        transform="chosen_rejected",
        license="mit",
        why="canonical chosen/rejected preference pairs; schema-identical to our DPO contract",
        streaming=False,
    ),
    Source(
        repo="lavita/ChatDoctor-HealthCareMagic-100k",
        kind="medical",
        transform="input_output",
        license="not stated on card (mirror of ChatDoctor data)",
        why="real patient-question medical QA for the domain-adapter LoRA demo",
        streaming=False,
    ),
    Source(
        repo="openai/gsm8k",
        config="main",
        kind="math",
        transform="gsm8k",
        license="mit",
        why="grade-school word problems with verifiable numeric answers; ground truth for agentic RL",
        streaming=False,
    ),
    Source(
        repo="cais/mmlu",
        config="all",
        split="validation",
        kind="exam",
        transform="multiple_choice",
        license="mit",
        why="exam-format alignment; validation split only, never test",
        streaming=False,
    ),
    Source(
        repo="cais/mmlu",
        config="all",
        split="dev",
        kind="exam",
        transform="multiple_choice",
        license="mit",
        why="exam-format alignment; dev split only, never test",
        streaming=False,
    ),
    Source(
        repo="allenai/ai2_arc",
        config="ARC-Easy",
        kind="exam",
        transform="arc",
        license="cc-by-sa-4.0",
        why="science multiple choice; train split only",
        streaming=False,
    ),
    Source(
        repo="allenai/ai2_arc",
        config="ARC-Challenge",
        kind="exam",
        transform="arc",
        license="cc-by-sa-4.0",
        why="harder science multiple choice; train split only",
        streaming=False,
    ),
    Source(
        repo="allenai/openbookqa",
        config="main",
        kind="exam",
        transform="openbookqa",
        license="unknown (per dataset card)",
        why="open-book science questions; train split only",
        streaming=False,
    ),
]


def by_kind(kind):
    return [s for s in SOURCES if s.kind == kind]


def kinds():
    return sorted({s.kind for s in SOURCES})
