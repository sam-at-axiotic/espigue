#!/usr/bin/env python3
"""A/B metrics for two espigue synthesis.yaml files (baseline vs candidate)."""
import sys, re, collections

try:
    import yaml
except ImportError:
    sys.exit("pyyaml missing: pip install pyyaml")


def load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def metrics(doc):
    claims = doc.get("claims") or []
    quotes = [q for c in claims for q in (c.get("quotes") or [])]
    verified = [q for q in quotes if q.get("status") == "verified"]
    sources = {s for c in claims for s in (c.get("sources") or [])}
    support = collections.Counter((c.get("support_level") or "none") for c in claims)
    grades = collections.Counter((c.get("evidence_grade") or "none") for c in claims)
    narrative = doc.get("narrative") or ""
    words = len(narrative.split())
    sections = re.findall(r"^## +(.+)$", narrative, re.M)
    cx = len(re.findall(r"\[C\d+", narrative))
    gaps = doc.get("gaps") or []
    gap_types = collections.Counter((g.get("gap_type") or "untyped") for g in gaps)
    return {
        "claims": len(claims),
        "claims_with_quote": sum(1 for c in claims if c.get("quotes")),
        "quotes": len(quotes),
        "quotes_verified": len(verified),
        "verified_rate": (len(verified) / len(quotes)) if quotes else 0.0,
        "distinct_sources_cited": len(sources),
        "claims_multi_source": sum(1 for c in claims if len(c.get("sources") or []) >= 2),
        "counterarguments": sum(len(c.get("counterarguments") or []) for c in claims),
        "support_levels": dict(support),
        "evidence_grades": dict(grades),
        "narrative_words": words,
        "narrative_sections": len(sections),
        "section_headings": sections,
        "inline_cx_citations": cx,
        "areas_agreement": len(doc.get("areas_of_agreement") or []),
        "areas_disagreement": len(doc.get("areas_of_disagreement") or []),
        "uncertainties": len(doc.get("uncertainties") or []),
        "gaps": len(gaps),
        "gap_types": dict(gap_types),
        "model": doc.get("model"),
        "prompt_version": doc.get("prompt_version"),
        "generated_at": doc.get("generated_at"),
    }


def show(name, m):
    print(f"\n=== {name} ===")
    for k, v in m.items():
        if k == "section_headings":
            print(f"{k}:")
            for h in v:
                print(f"  - {h}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    a, b = sys.argv[1], sys.argv[2]
    ma, mb = metrics(load(a)), metrics(load(b))
    show("A (baseline)", ma)
    show("B (candidate)", mb)
