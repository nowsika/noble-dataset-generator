# -*- coding: utf-8 -*-
"""cs_presets.py

B2B SaaS Customer Support (CS) dataset presets.

Why this exists
  - The default template in this repo is therapy-flavored (tags like therapy/anxiety).
  - For a commercial CS dataset, we want business-relevant tags/signals and a
    professional support tone.

What it does
  - Classify a scenario into one of 6 CS categories using deterministic keyword rules.
  - Apply category presets for:
      * tags
      * signals (shadow/fuel/gravity)
      * context_state.tone_hint

Notes
  - "signals" uses the existing 7-archetype taxonomy (Wrath/Greed/...)
    even though it's not CS-native. We keep it consistent and moderately varied.
  - This module does NOT attempt ML classification.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


CS_CATEGORIES = (
    "billing",
    "account_security",
    "api_integrations",
    "incident_performance",
    "compliance_procurement",
    "deescalation_retention",
)


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def classify_cs_category(scenario: str) -> str:
    """Best-effort deterministic classifier for English CS scenarios."""
    t = _norm(scenario)

    # Compliance / legal / procurement
    if any(k in t for k in (
        "soc 2", "iso", "gdpr", "dpa", "subprocessor", "vendor", "questionnaire",
        "pen test", "rto", "rpo", "sla", "service credits", "baa", "hipaa",
        "privacy impact", "data residency", "cmk", "customer-managed keys", "coi",
        "contract clause", "nda", "audit evidence", "wcag",
    )):
        return "compliance_procurement"

    # Incidents / outages / performance
    if any(k in t for k in (
        "down", "outage", "incident", "rca", "postmortem", "latency", "timeout",
        "degraded", "blank page", "crash", "performance", "status page", "hotfix",
        "maintenance", "bug", "data integrity", "data loss", "security concern",
    )):
        return "incident_performance"

    # API / integrations / webhooks
    if any(k in t for k in (
        "api", "webhook", "oauth", "token", "rate limit", "429", "500",
        "zapier", "slack integration", "integration", "payload", "signature",
        "pagination", "bulk import", "export", "sftp", "cors", "idempotency",
        "streaming", "endpoint", "status endpoint",
    )):
        return "api_integrations"

    # Account / access / security (SSO/MFA/SCIM)
    if any(k in t for k in (
        "login", "log in", "mfa", "2fa", "sso", "saml", "scim", "idp",
        "permissions", "role", "audit logs", "ip allow", "allowlist", "token",
        "compromised", "password", "session", "domain", "workspace ownership",
    )):
        return "account_security"

    # De-escalation / churn / angry customer
    if any(k in t for k in (
        "unacceptable", "cancel", "churn", "competitor", "manager", "escalation",
        "refund immediately", "publicly", "post about", "key account", "we lost revenue",
        "admit fault", "stop closing", "conflicting answers",
    )):
        return "deescalation_retention"

    # Billing / invoices / payment terms
    if any(k in t for k in (
        "invoice", "charged", "charge", "billing", "refund", "credit memo", "po number",
        "net-", "net 30", "net-30", "ach", "wire", "payment failed", "tax-exempt",
        "vat", "receipt", "quote", "renewal", "seat", "add-on",
    )):
        return "billing"

    # Default
    return "billing"


def cs_presets(category: str) -> Tuple[Dict, List[str]]:
    """Return (patch_dict, tags) for scenario_obj."""
    cat = category if category in CS_CATEGORIES else "billing"

    # Base defaults
    base = {
        "signals": {
            "shadow": {"archetype": "Pride", "subtype": "pride_status_domination", "severity": "S2_harmful"},
            "fuel": {"primary": "fuel_security", "secondary": "fuel_respect"},
            "gravity_vectors": ["protect_future_self", "preserve_dignity", "seek_relief_now"],
        },
        "context_state": {
            "risk": {"self_harm": "LOW", "violence": "LOW", "child_safety": "LOW"},
            "tone_hint": {"softness": "MED", "clarity": "HIGH", "compression": "MED"},
        },
    }

    tags: List[str] = ["b2b", "customer_support", "saas", cat]

    if cat == "billing":
        base["signals"]["shadow"] = {"archetype": "Greed", "subtype": "greed_financial_scam", "severity": "S2_harmful"}
        base["signals"]["fuel"] = {"primary": "fuel_security", "secondary": "fuel_justice"}
        tags += ["billing", "invoice", "refund", "payments"]

    elif cat == "account_security":
        base["signals"]["shadow"] = {"archetype": "Pride", "subtype": "pride_gaslighting_control", "severity": "S2_harmful"}
        base["signals"]["fuel"] = {"primary": "fuel_security", "secondary": "fuel_respect"}
        base["signals"]["gravity_vectors"] = ["assert_control", "protect_future_self", "preserve_dignity"]
        tags += ["access", "security", "sso", "mfa", "roles"]
        base["context_state"]["tone_hint"] = {"softness": "MED", "clarity": "HIGH", "compression": "MED"}

    elif cat == "api_integrations":
        base["signals"]["shadow"] = {"archetype": "Sloth", "subtype": "sloth_avoidance", "severity": "S2_harmful"}
        base["signals"]["fuel"] = {"primary": "fuel_growth", "secondary": "fuel_security"}
        base["signals"]["gravity_vectors"] = ["protect_future_self", "assert_control", "seek_relief_now"]
        tags += ["api", "integrations", "webhook", "oauth", "debugging"]
        base["context_state"]["tone_hint"] = {"softness": "LOW", "clarity": "HIGH", "compression": "HIGH"}

    elif cat == "incident_performance":
        base["signals"]["shadow"] = {"archetype": "Wrath", "subtype": "wrath_verbal_abuse", "severity": "S2_harmful"}
        base["signals"]["fuel"] = {"primary": "fuel_security", "secondary": "fuel_relief"}
        base["signals"]["gravity_vectors"] = ["seek_relief_now", "assert_control", "protect_future_self"]
        tags += ["incident", "outage", "performance", "escalation", "sla"]
        base["context_state"]["tone_hint"] = {"softness": "LOW", "clarity": "HIGH", "compression": "HIGH"}

    elif cat == "compliance_procurement":
        base["signals"]["shadow"] = {"archetype": "Greed", "subtype": "greed_data_extraction", "severity": "S2_harmful"}
        base["signals"]["fuel"] = {"primary": "fuel_security", "secondary": "fuel_justice"}
        base["signals"]["gravity_vectors"] = ["protect_future_self", "assert_control", "preserve_dignity"]
        tags += ["compliance", "procurement", "security_review", "legal"]
        base["context_state"]["tone_hint"] = {"softness": "LOW", "clarity": "HIGH", "compression": "HIGH"}

    elif cat == "deescalation_retention":
        base["signals"]["shadow"] = {"archetype": "Wrath", "subtype": "wrath_verbal_abuse", "severity": "S2_harmful"}
        base["signals"]["fuel"] = {"primary": "fuel_respect", "secondary": "fuel_security"}
        base["signals"]["gravity_vectors"] = ["preserve_dignity", "seek_relief_now", "assert_control"]
        tags += ["deescalation", "retention", "churn_risk", "complaint"]
        base["context_state"]["tone_hint"] = {"softness": "MED", "clarity": "HIGH", "compression": "HIGH"}

    # de-dup tags preserving order
    seen = set()
    tags = [t for t in tags if not (t in seen or seen.add(t))]
    return base, tags


def apply_cs_presets(scenario_obj: Dict, scenario_text: str) -> None:
    """Mutate scenario_obj in-place with CS presets."""
    cat = classify_cs_category(scenario_text)
    patch, tags = cs_presets(cat)

    # Overwrite (these are intended to replace the therapy defaults in template)
    scenario_obj["signals"] = patch["signals"]
    scenario_obj["tags"] = tags
    scenario_obj.setdefault("context_state", {})
    scenario_obj["context_state"]["risk"] = patch["context_state"]["risk"]
    scenario_obj["context_state"]["tone_hint"] = patch["context_state"]["tone_hint"]

    qa = scenario_obj.setdefault("qa", {})
    if isinstance(qa, dict):
        qa["dataset_profile"] = "cs_b2b"
        qa["scenario_category"] = cat
