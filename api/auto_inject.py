# -*- coding: utf-8 -*-
"""
auto_inject.py

- scenarios.txt 라인에서 "GARDEN|..." 같은 도메인 프리픽스 파싱
- 시나리오 텍스트 기반 언어 감지(휴리스틱 + 선택적 langdetect)
- qa.language / qa.metaphor_pack 자동 주입
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

ALLOWED_DOMAINS = ["NAV", "SEA", "ARCH", "MUSIC", "GARDEN", "MOUNTAIN", "THEATER"]

# --- Language heuristics (fast, deterministic) ---
_RE_AR = re.compile(r"[\u0600-\u06FF]")
_RE_KO = re.compile(r"[가-힣]")
_RE_JA = re.compile(r"[ぁ-んァ-ン]")
_RE_ZH = re.compile(r"[\u4e00-\u9fff]")
_RE_LATIN = re.compile(r"[A-Za-z]")

def parse_scenario_prefix(line: str) -> Tuple[Optional[str], str]:
    """
    Allows scenarios like:
      ARCH|I feel ...
      GARDEN|"quoted..."
    Returns (domain_or_none, scenario_text_without_prefix).
    """
    if not isinstance(line, str):
        return None, str(line)

    for d in ALLOWED_DOMAINS:
        prefix = d + "|"
        if line.startswith(prefix):
            return d, line[len(prefix):].lstrip()
    return None, line

def detect_language(text: str) -> str:
    """
    Returns one of: ko, ja, zh-Hans, ar, en, es
    """
    if not isinstance(text, str):
        return "en"

    t = text.strip()
    if not t:
        return "en"

    if _RE_AR.search(t):
        return "ar"
    if _RE_KO.search(t):
        return "ko"
    if _RE_JA.search(t):
        return "ja"
    if _RE_ZH.search(t):
        # Hans/Hant 구분은 여기서 안 함 (기본 Hans)
        return "zh-Hans"

    # Latin이면 langdetect가 있으면 en/es 분리 시도
    if _RE_LATIN.search(t):
        try:
            from langdetect import detect  # type: ignore
            lang = detect(t)
            if lang in ("en", "es"):
                return lang
        except Exception:
            pass
        return "en"

    return "en"

def inject_qa_defaults(scenario_obj: Dict[str, Any], language: str, kw_pack_version: Optional[str]) -> None:
    """
    Mutates scenario_obj in-place.
      - qa.language: language
      - qa.metaphor_pack: "metaphor_keywords@<version>" (if version known)
    """
    qa = scenario_obj.setdefault("qa", {})
    if not isinstance(qa, dict):
        scenario_obj["qa"] = {}
        qa = scenario_obj["qa"]

    if isinstance(language, str) and language:
        qa["language"] = language

    if kw_pack_version:
        qa["metaphor_pack"] = f"metaphor_keywords@{kw_pack_version}"
