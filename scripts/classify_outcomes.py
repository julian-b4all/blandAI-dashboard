"""
BlandAI Call Outcome Classifier
================================
Classifies each call into an honest outcome category based on multiple signals:
- Transfer status
- Call duration
- Who ended the call
- Transcript word count
- Summary content analysis

Categories:
  TRANSFERRED     — Call was transferred to human agent
  RESOLVED        — High confidence the issue was resolved by AI
  LIKELY_RESOLVED — Moderate confidence of resolution
  ABANDONED       — Caller hung up early, no resolution evidence
  NO_INTERACTION  — No meaningful conversation occurred
  UNRESOLVED      — Evidence the issue was NOT resolved
  SYSTEM_ERROR    — System/billing error, not a real support call
  AMBIGUOUS       — Can't determine outcome

Outputs:
  data/calls_classified.csv — Full dataset with outcome classification
"""

import json
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DETAIL_DIR = DATA_DIR / "calls_detail"

# Inbound numbers
INBOUND_NUMBERS = {
    "+14153246039": "Applicant Pathway (Original)",
    "+17144521864": "Full Support Pathway",
    "+17146134173": "Spanish Pathway",
    "+16504139279": "Julian Applicant Pathway",
}

# Summary signals for classification
RESOLVED_SIGNALS = [
    "resolved", "guided", "provided instructions", "directed them",
    "helped", "explained", "informed", "walked through", "instructed",
    "successfully", "confirmed", "answered", "clarified",
    "advised", "recommended", "showed how",
]
UNRESOLVED_SIGNALS = [
    "unable to", "could not", "cut short", "abruptly ended",
    "hung up", "disconnected", "frustrated", "declined assistance",
    "refused", "not enough information to generate",
    "insufficient balance", "poor connection", "static",
    "unresponsive", "failed to", "struggled",
]


def classify_call(d):
    """Classify a single call into an outcome category with confidence."""
    length = d.get("call_length", 0) or 0
    ended_by = d.get("call_ended_by", "") or "unknown"
    transferred = bool(d.get("transferred_to"))
    concat = d.get("concatenated_transcript", "") or ""
    words = len(concat.split()) if concat else 0
    summary = (d.get("summary", "") or "").lower()

    # --- TRANSFERRED ---
    if transferred:
        return "TRANSFERRED", "high"

    # --- SYSTEM ERROR (insufficient balance, etc.) ---
    if "insufficient balance" in summary or "auto-recharge" in summary:
        return "SYSTEM_ERROR", "high"

    # --- NO INTERACTION ---
    if words == 0 or length < 0.15:
        return "NO_INTERACTION", "high"
    if words < 30 and length < 0.4:
        return "NO_INTERACTION", "medium"

    # Check summary signals
    has_resolved = any(s in summary for s in RESOLVED_SIGNALS)
    has_unresolved = any(s in summary for s in UNRESOLVED_SIGNALS)

    # --- ASSISTANT ENDED (AI closed the call) ---
    if ended_by == "ASSISTANT":
        if length < 0.3:
            return "NO_INTERACTION", "medium"
        if has_unresolved and not has_resolved:
            return "UNRESOLVED", "medium"
        if has_resolved and not has_unresolved:
            return "RESOLVED", "high"
        if has_resolved and has_unresolved:
            return "AMBIGUOUS", "low"
        # Assistant ended with decent length, no strong signals
        if length >= 1.0:
            return "LIKELY_RESOLVED", "medium"
        return "AMBIGUOUS", "low"

    # --- USER ENDED ---
    if ended_by == "USER":
        # Very short = abandoned
        if length < 0.5:
            return "ABANDONED", "high"
        if length < 1.0:
            if has_resolved:
                return "LIKELY_RESOLVED", "medium"
            return "ABANDONED", "medium"

        # 1-2 min range
        if length < 2.0:
            if has_resolved and not has_unresolved:
                return "LIKELY_RESOLVED", "medium"
            if has_unresolved:
                return "UNRESOLVED", "medium"
            return "ABANDONED", "low"

        # > 2 min — longer engagement
        if has_resolved and not has_unresolved:
            return "LIKELY_RESOLVED", "medium"
        if has_unresolved and not has_resolved:
            return "UNRESOLVED", "medium"
        if has_resolved and has_unresolved:
            return "AMBIGUOUS", "low"

        # Long call, user ended, no clear signals
        if length >= 5.0:
            return "AMBIGUOUS", "low"
        return "LIKELY_RESOLVED", "low"

    # --- UNKNOWN ended_by ---
    return "AMBIGUOUS", "low"


def main():
    print("Classifying call outcomes...")

    if not DETAIL_DIR.exists():
        print("ERROR: No detail files found.")
        return

    detail_files = sorted(DETAIL_DIR.glob("*.json"))
    rows = []

    for path in detail_files:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)

        if "call_id" not in d and len(d) == 1:
            d = list(d.values())[0]

        outcome, confidence = classify_call(d)

        # Determine pathway
        to_number = d.get("to", "")
        from_number = d.get("from", "")
        inbound = d.get("inbound", None)
        pathway_number = to_number if inbound else from_number
        pathway_name = INBOUND_NUMBERS.get(pathway_number, "Unknown")

        concat = d.get("concatenated_transcript", "") or ""
        transcripts = d.get("transcripts", []) or []

        rows.append({
            "call_id": d.get("call_id", path.stem),
            "created_at": d.get("created_at", ""),
            "call_length_min": d.get("call_length", 0) or 0,
            "ended_by": d.get("call_ended_by", ""),
            "inbound": d.get("inbound", ""),
            "from": from_number,
            "to": to_number,
            "pathway_name": pathway_name,
            "pathway_id": d.get("pathway_id", ""),
            "pathway_version": d.get("pathway_version", ""),
            "transferred": bool(d.get("transferred_to")),
            "transferred_to": d.get("transferred_to", ""),
            "outcome": outcome,
            "confidence": confidence,
            "transcript_words": len(concat.split()) if concat else 0,
            "user_turns": sum(1 for t in transcripts if t.get("user") == "user"),
            "agent_turns": sum(1 for t in transcripts if t.get("user") == "assistant"),
            "summary": (d.get("summary", "") or "")[:500],
            "price": d.get("price", 0) or 0,
            "error_message": (d.get("error_message", "") or "")[:200],
        })

    rows.sort(key=lambda r: r.get("created_at", ""))

    # Write CSV
    csv_path = DATA_DIR / "calls_classified.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {csv_path}")

    # ── Stats ───────────────────────────────────────────────────────────────

    print(f"\n=== OUTCOME CLASSIFICATION ({len(rows)} calls) ===\n")

    # Overall
    from collections import Counter
    outcomes = Counter(r["outcome"] for r in rows)
    for outcome in ["RESOLVED", "LIKELY_RESOLVED", "TRANSFERRED", "ABANDONED",
                     "NO_INTERACTION", "UNRESOLVED", "SYSTEM_ERROR", "AMBIGUOUS"]:
        count = outcomes.get(outcome, 0)
        pct = count / len(rows) * 100
        print(f"  {outcome:20s}: {count:4d} ({pct:5.1f}%)")

    # Confidence breakdown
    print(f"\n=== CONFIDENCE LEVELS ===\n")
    conf = Counter(r["confidence"] for r in rows)
    for c in ["high", "medium", "low"]:
        print(f"  {c}: {conf.get(c, 0)}")

    # By pathway
    print(f"\n=== BY PATHWAY ===\n")
    pathways = {}
    for r in rows:
        pathways.setdefault(r["pathway_name"], []).append(r)

    for name, calls in sorted(pathways.items(), key=lambda x: -len(x[1])):
        total = len(calls)
        resolved = sum(1 for c in calls if c["outcome"] == "RESOLVED")
        likely = sum(1 for c in calls if c["outcome"] == "LIKELY_RESOLVED")
        transferred = sum(1 for c in calls if c["outcome"] == "TRANSFERRED")
        abandoned = sum(1 for c in calls if c["outcome"] == "ABANDONED")
        no_int = sum(1 for c in calls if c["outcome"] == "NO_INTERACTION")
        unresolved = sum(1 for c in calls if c["outcome"] == "UNRESOLVED")
        sys_err = sum(1 for c in calls if c["outcome"] == "SYSTEM_ERROR")
        ambig = sum(1 for c in calls if c["outcome"] == "AMBIGUOUS")

        # Resolution rates
        meaningful = total - no_int - sys_err  # calls with real interaction
        ai_resolved = resolved + likely
        resolution_rate = ai_resolved / meaningful * 100 if meaningful > 0 else 0
        strict_rate = resolved / meaningful * 100 if meaningful > 0 else 0

        avg_len = sum(c["call_length_min"] for c in calls) / max(total, 1)

        print(f"  {name}")
        print(f"    Total: {total} | Meaningful: {meaningful} | Avg: {avg_len:.1f} min")
        print(f"    Resolved: {resolved} | Likely Resolved: {likely} | Transferred: {transferred}")
        print(f"    Abandoned: {abandoned} | No Interaction: {no_int} | Unresolved: {unresolved}")
        print(f"    System Error: {sys_err} | Ambiguous: {ambig}")
        print(f"    Resolution rate (strict):  {strict_rate:.1f}%  (RESOLVED / meaningful)")
        print(f"    Resolution rate (liberal): {resolution_rate:.1f}%  (RESOLVED + LIKELY / meaningful)")
        print()


if __name__ == "__main__":
    main()
