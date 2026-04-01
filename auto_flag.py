"""
auto_flag.py - Automatic question flagging
"""
from __future__ import annotations
import re
from typing import List


RE_ANS = re.compile(
    r"(?:ans(?:wer)?|correct|key)\s*[:\-]?\s*\(?([A-Da-d])\)?",
    re.IGNORECASE,
)

RE_NOISE = re.compile(
    r"©\s*[Mm]arrow|marrow\.in|\bSOLVE\b|\bNEXT\b|"
    r"\bBOOKMARK\b|\bCOMPLETE\b|\b[a-f0-9]{20,}\b|"
    r"\[\s*\d{1,3}%\s*\]",
    re.IGNORECASE,
)


def auto_flag_question(q: dict) -> dict:
    flags = list(q.get("flags", []))

    stem = q.get("question", "")
    opts = q.get("options", {})
    ans  = q.get("answer",  "")
    exp  = q.get("explanation", "")

    # Missing answer
    if not ans:
        _add(flags, "missing_answer")

    # Missing options
    if not opts.get("A") and not opts.get("B"):
        _add(flags, "missing_options")

    # Short stem
    if len(stem.strip()) < 15:
        _add(flags, "short_stem")

    # Noise in stem
    if RE_NOISE.search(stem):
        _add(flags, "noise_in_stem")

    # Noise in options
    for letter, text in opts.items():
        if text and RE_NOISE.search(text):
            _add(flags, f"noise_in_opt_{letter}")

    # Noise in explanation
    if exp and RE_NOISE.search(exp):
        _add(flags, "noise_in_explanation")

    # Answer not in options
    if ans and opts and ans not in opts:
        _add(flags, "answer_not_in_options")

    # Empty correct option
    if ans and opts.get(ans, "").strip() == "":
        _add(flags, "correct_option_empty")

    # Bleeding: answer line in stem
    if RE_ANS.search(stem):
        _add(flags, "bleed_answer_in_stem")

    # Bleeding: answer line in option text
    for letter, text in opts.items():
        if text and RE_ANS.search(text):
            _add(flags, f"bleed_answer_in_opt_{letter}")

    q["flags"] = flags
    return q


def auto_flag_all(questions: List[dict]) -> List[dict]:
    return [auto_flag_question(q) for q in questions]


def flag_summary(questions: List[dict]) -> dict:
    summary: dict = {}
    for q in questions:
        for flag in q.get("flags", []):
            summary[flag] = summary.get(flag, 0) + 1
    return summary


def _add(flags: list, flag: str) -> None:
    if flag not in flags:
        flags.append(flag)
        