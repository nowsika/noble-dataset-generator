# domain_router.py
import re
from typing import Dict, List, Tuple

def _normalize(text: str) -> str:
    return (text or "").strip()

def _tokenize_for_match(text: str) -> str:
    # Keep it simple: lowercase, collapse whitespace.
    t = _normalize(text).lower()
    t = re.sub(r"\s+", " ", t)
    return t

def score_domains(
    scenario_text: str,
    domain_keywords: Dict[str, List[str]],
    case_insensitive: bool = True,
) -> Dict[str, int]:
    text = _normalize(scenario_text)
    if case_insensitive:
        text = _tokenize_for_match(text)

    scores: Dict[str, int] = {}
    for domain, kws in domain_keywords.items():
        s = 0
        for kw in kws:
            if not kw:
                continue
            key = kw.lower() if case_insensitive else kw
            if key in text:
                s += 1
        scores[domain] = s
    return scores

def pick_domain(
    scenario_text: str,
    domains: List[str],
    domain_keywords: Dict[str, List[str]],
    min_score_to_lock: int = 2,
    fallback_domain: str = "NAV",
) -> Tuple[str, Dict[str, int]]:
    scores = score_domains(scenario_text, domain_keywords, case_insensitive=True)

    # Choose max score
    best_domain = fallback_domain
    best_score = -1
    for d in domains:
        sc = scores.get(d, 0)
        if sc > best_score:
            best_score = sc
            best_domain = d

    # If not enough signal, return empty marker (caller decides fallback policy)
    if best_score < min_score_to_lock:
        return "", scores

    return best_domain, scores
