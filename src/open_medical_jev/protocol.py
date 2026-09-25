"""Protocol — exact prompt construction for the two readout structures.

Part of Open Medical Jev (Apache-2.0).

The prompt strings below were verified token-by-token against the reference
implementation (HuggingFace ``apply_chat_template(..., enable_thinking=False)``
of the Qwen3.5 family; see ``docs/protocol.md`` for the rendered golden
example). Keep this file in sync with ``docs/protocol.md`` if you change it —
prompt drift changes results.

Structures
----------
* ``pair``   — one candidate option per forward pass:
  "Question: ...\\n\\nCandidate answer: ...\\n\\nIs the candidate answer
  correct? Reply yes or no."  → read yes/no probabilities of the first
  output token.  (This is the primary "yes/no probability" contract.)
* ``choice`` — all options listed in one forward pass:
  "A. ... B. ..." → read the probability mass on each option letter.
"""

from __future__ import annotations

__all__ = [
    "PAIR_SYS", "PAIR_TAIL", "LEAD", "CHOICE_SYS", "CTX_CAP",
    "head_text", "pair_user_text", "pair_prompt_string", "choice_prompt_string",
]

CTX_CAP = 6000

# ---------------------------------------------------------------------------
# pair structure (primary contract)
# ---------------------------------------------------------------------------

PAIR_SYS = ("You are a careful medical expert. You will be given a question and one candidate "
            "answer. Judge whether that candidate answer correctly answers the question.")

PAIR_TAIL = "\n\nIs the candidate answer correct? Reply yes or no."

#: Prefix appended after the assistant header; the model's next token is read.
LEAD = "Answer: "

#: Empty thinking block produced by ``enable_thinking=False`` in the Qwen3.5
#: chat template (verified 2026-09-25). The ``pair`` string must include it.
THINK_EMPTY = "<think>\n\n</think>\n\n"


def head_text(item: dict) -> str:
    """Question header, with an optional context block (capped at CTX_CAP)."""
    ctx = (item.get("context") or "").strip()
    if ctx:
        return f"Question: {item['input']}\n\nContext:\n{ctx[:CTX_CAP]}\n"
    return f"Question: {item['input']}\n"


def pair_user_text(item: dict, option_text: str) -> str:
    return f"{head_text(item)}\nCandidate answer: {option_text}" + PAIR_TAIL


def pair_prompt_string(item: dict, option_text: str) -> str:
    """Full ChatML string for one candidate-option judgment (pair structure)."""
    return (
        f"<|im_start|>system\n{PAIR_SYS}<|im_end|>\n"
        f"<|im_start|>user\n{pair_user_text(item, option_text)}<|im_end|>\n"
        f"<|im_start|>assistant\n{THINK_EMPTY}" + LEAD
    )


# ---------------------------------------------------------------------------
# choice structure (single-pass, option letters)
# ---------------------------------------------------------------------------

CHOICE_SYS = ("You are a licensed medical professional taking a national licensing exam. "
              "Answer each question with the single best option.")


def choice_prompt_string(item: dict) -> str:
    """Full ChatML string for a single-pass choice readout (choice structure)."""
    opts = "\n".join(f"{o['key']}. {o['text']}" for o in item["options"])
    labs = [o["key"] for o in item["options"]]
    head = f"<|im_start|>system\n{CHOICE_SYS}\n<|im_end|>\n<|im_start|>user\n"
    tail = ("\n\nAnswer with exactly one option letter: " + ", ".join(labs)
            + ". Output only the letter.<|im_end|>\n<|im_start|>assistant\n")
    return head + item["input"] + "\n\n" + opts + tail
