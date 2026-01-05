# -*- coding: utf-8 -*-
"""
prompt_builder.py

Builds the "instructions" string for the OpenAI Responses API.
대부분의 텍스트는 /prompts/*.txt에서 관리해서 톤 수정이 쉽도록 함.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

PROMPTS_DIR = Path(__file__).parent / "prompts"

def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def build_instructions() -> str:
    """Build the "instructions" string for the Responses API.

    Profiles
      - PROMPT_PROFILE=noble (default): original NOBLE counseling-style stack
      - PROMPT_PROFILE=cs: professional B2B customer support style (no therapy letters)

    Order matters: core -> optional letters -> output contract.
    """
    parts: List[str] = []

    profile = (os.getenv("PROMPT_PROFILE", "noble") or "noble").strip().lower()

    if profile == "cs":
        core = PROMPTS_DIR / "system_core_cs.txt"
    else:
        core = PROMPTS_DIR / "system_core.txt"

    if core.exists():
        parts.append(_read(core).strip())

    # Letters are counseling-flavored; skip them in CS profile.
    if profile != "cs":
        letters_dir = PROMPTS_DIR / "letters"
        if letters_dir.exists():
            for name in (
                "noble_letter.txt",
                "healing_letter.txt",
                "shadow_letter.txt",
                "taylor_letter.txt",
            ):
                p = letters_dir / name
                if p.exists():
                    parts.append(_read(p).strip())

    contract = PROMPTS_DIR / "output_contract.txt"
    if contract.exists():
        parts.append(_read(contract).strip())

    return "\n\n".join([p for p in parts if p]).strip()
