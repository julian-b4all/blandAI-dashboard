"""Classify BlandAI calls into the CallDriverAnalysis taxonomy.

Uses Claude Haiku to classify each call's TRANSCRIPT into the 24-component taxonomy
(component + symptom_category) defined in ../CallDriverAnalysis/taxonomy/components.json.

Reads full transcripts from data/calls_detail/*.json for accurate classification,
falls back to summary from calls_classified.csv for calls without detail files.

Usage:
    python scripts/classify_drivers.py              # Classify all unclassified calls
    python scripts/classify_drivers.py --force      # Re-classify all calls
    python scripts/classify_drivers.py --sample 50  # Classify a random sample of 50

Output: data/calls_drivers.csv
"""

import json
import csv
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv
import os

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT.parent / ".env"
TAXONOMY_PATH = PROJECT_ROOT.parent / "CallDriverAnalysis" / "taxonomy" / "components.json"
CLASSIFIED_PATH = PROJECT_ROOT / "data" / "calls_classified.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "calls_drivers.csv"
DETAIL_DIR = PROJECT_ROOT / "data" / "calls_detail"

load_dotenv(ENV_PATH, override=True)

# ── Load taxonomy ──────────────────────────────────────────────────────────

def load_taxonomy():
    """Load and format taxonomy for the classification prompt."""
    with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
        tax = json.load(f)

    components = []
    component_symptoms = {}  # name -> list of symptom IDs

    for section in ["shared_components", "operator_components", "applicant_components"]:
        for comp in tax.get(section, []):
            name = comp["name"]
            desc = comp["description"]
            applies = ", ".join(comp["applies_to"])
            symptoms = [s["id"] for s in comp.get("symptoms", [])]
            symptom_desc = "; ".join(
                f'{s["id"]} ({s["description"]})'
                for s in comp.get("symptoms", [])
            )
            components.append(
                f"- **{name}** (applies to: {applies}): {desc}\n"
                f"  Symptom categories: {symptom_desc}"
            )
            component_symptoms[name] = symptoms

    # Add escape valve
    components.append(
        "- **other**: Use ONLY when the issue genuinely does not fit any component above."
    )
    component_symptoms["other"] = ["other"]
    component_symptoms["call routing"] = [
        "no_live_interaction", "audio_or_language_barrier",
        "misrouted", "disconnected", "other",
    ]

    return "\n".join(components), component_symptoms


def build_classification_prompt(taxonomy_text):
    """Build the system prompt for classification."""
    return f"""You are a call analyst for Biometrics4ALL, a fingerprinting services company.
Given a transcript of a call handled by the BlandAI voice AI agent "Alex", you will:
1. Classify the call into the taxonomy below.
2. Assess whether the caller's issue was actually resolved.

## Taxonomy (24 components)

{taxonomy_text}

## Classification Rules
1. Read the transcript carefully. Identify what the caller's PRIMARY issue is.
2. Choose the BEST-FIT component for that primary issue. Most calls fit an existing component.
3. Choose one symptom_category from that component's symptom list.
4. Determine caller_type: "operator" (service center running LiveScan equipment),
   "applicant" (person getting fingerprinted), or "unknown" (can't tell).
5. If the transcript is empty, very short with no real interaction, or unintelligible,
   use component="call routing" with symptom_category="no_live_interaction".
6. If the call covers multiple topics, classify by the PRIMARY issue
   (what the caller originally called about or spent the most time on).

## Resolution Assessment Rules
Assess the ACTUAL outcome based on what happened in the conversation:
- "resolved" — The AI agent provided a clear, correct answer or walked the caller through
  a solution. The caller's issue appears addressed. The caller acknowledged understanding
  or expressed satisfaction.
- "partially_resolved" — The AI gave some useful guidance but the caller's issue wasn't
  fully addressed, OR the caller seemed uncertain, OR the AI gave correct info but the
  caller needs to take further action that may or may not work.
- "transferred" — The call was escalated/transferred to a human agent.
- "abandoned" — The caller hung up or stopped responding before the issue was addressed.
  Includes very short calls where the caller left quickly.
- "unresolved" — The AI failed to help. Wrong information, couldn't understand the issue,
  went in circles, or the caller expressed frustration and left without resolution.
- "no_interaction" — No meaningful conversation occurred (silence, test call, <5 words
  from caller).

Be SKEPTICAL of the AI agent's politeness. "Alex" always says helpful-sounding things.
Focus on whether the caller's ACTUAL PROBLEM was addressed, not whether the AI sounded nice.
If the AI gave generic advice ("visit our website", "call back later") without solving the
specific issue, that is NOT resolved.

## Output Format
Respond with ONLY a JSON object (no markdown, no explanation):
{{"component": "...", "symptom_category": "...", "caller_type": "operator|applicant|unknown", "resolution": "resolved|partially_resolved|transferred|abandoned|unresolved|no_interaction", "resolution_reason": "one sentence explaining why you chose this resolution status"}}"""


# ── Transcript loading ────────────────────────────────────────────────────

def load_transcript(call_id):
    """Load full transcript from calls_detail JSON file."""
    detail_path = DETAIL_DIR / f"{call_id}.json"
    if not detail_path.exists():
        return None
    with open(detail_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d.get("concatenated_transcript", "") or ""


# ── Classification ─────────────────────────────────────────────────────────

VALID_RESOLUTIONS = {
    "resolved", "partially_resolved", "transferred",
    "abandoned", "unresolved", "no_interaction",
}


def classify_call(client, system_prompt, text, model="claude-haiku-4-5-20251001"):
    """Classify a single call from its transcript or summary."""
    if not text or text.strip() == "" or len(text.split()) < 5:
        return {
            "component": "call routing",
            "symptom_category": "no_live_interaction",
            "caller_type": "unknown",
            "resolution": "no_interaction",
            "resolution_reason": "no meaningful transcript",
        }

    response = client.messages.create(
        model=model,
        max_tokens=300,
        system=system_prompt,
        messages=[{"role": "user", "content": f"Classify this call:\n\n{text[:4000]}"}],
    )

    raw = response.content[0].text.strip()
    # Parse JSON from response
    try:
        # Handle potential markdown wrapping
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)

        # Validate resolution value
        resolution = result.get("resolution", "unresolved")
        if resolution not in VALID_RESOLUTIONS:
            resolution = "unresolved"

        return {
            "component": result.get("component", "other").replace("_", " "),
            "symptom_category": result.get("symptom_category", "other"),
            "caller_type": result.get("caller_type", "unknown"),
            "resolution": resolution,
            "resolution_reason": result.get("resolution_reason", ""),
        }
    except (json.JSONDecodeError, IndexError):
        print(f"  [WARN] Failed to parse: {raw[:100]}")
        return {
            "component": "other",
            "symptom_category": "other",
            "caller_type": "unknown",
            "resolution": "unresolved",
            "resolution_reason": "parse error",
        }


def backfill_and_write(results):
    """Backfill missing metadata from calls_classified.csv and write output."""
    # Backfill metadata for any rows missing key fields.
    meta_fields = ["created_at", "call_length_min", "ended_by",
                   "pathway_name", "pathway_version", "outcome"]
    classified_lookup = {}
    with open(CLASSIFIED_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            classified_lookup[row["call_id"]] = row

    backfilled = 0
    for r in results:
        source = classified_lookup.get(r["call_id"])
        if not source:
            continue
        for field in meta_fields:
            if not r.get(field, "").strip() and source.get(field, "").strip():
                r[field] = source[field]
                backfilled += 1
    if backfilled:
        print(f"  Backfilled {backfilled} missing metadata fields from calls_classified.csv")

    # Write output — collect all field names from results to preserve extra columns
    base_fieldnames = [
        "call_id", "created_at", "call_length_min", "ended_by",
        "pathway_name", "pathway_version", "outcome",
        "component", "symptom_category", "caller_type",
        "resolution", "resolution_reason", "summary",
    ]
    extra = []
    for r in results:
        for k in r:
            if k not in base_fieldnames and k not in extra:
                extra.append(k)
    fieldnames = base_fieldnames + extra

    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        results.sort(key=lambda r: r.get("created_at", ""))
        writer.writerows(results)

    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Total rows: {len(results)}")


def main():
    parser = argparse.ArgumentParser(description="Classify BlandAI calls into taxonomy")
    parser.add_argument("--force", action="store_true", help="Re-classify all calls")
    parser.add_argument("--sample", type=int, help="Classify a random sample of N calls")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001", help="Model to use")
    args = parser.parse_args()

    # Load existing classifications
    existing = {}
    if OUTPUT_PATH.exists() and not args.force:
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing[row["call_id"]] = row
        print(f"Loaded {len(existing)} existing classifications")

    # Load call metadata from calls_classified.csv
    with open(CLASSIFIED_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        calls = list(reader)
    print(f"Loaded {len(calls)} calls from calls_classified.csv")

    # Filter to unclassified calls
    if not args.force:
        calls = [c for c in calls if c["call_id"] not in existing]
        print(f"{len(calls)} calls need classification")

    if args.sample and len(calls) > args.sample:
        import random
        random.seed(42)
        calls = random.sample(calls, args.sample)
        print(f"Sampled {len(calls)} calls")

    if not calls:
        print("Nothing to classify — checking for metadata gaps...")
        results = list(existing.values())
        backfill_and_write(results)
        return

    # Build prompt
    taxonomy_text, component_symptoms = load_taxonomy()
    system_prompt = build_classification_prompt(taxonomy_text)

    # Initialize client
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Classify calls
    results = list(existing.values())  # Start with existing
    errors = 0
    transcript_used = 0
    summary_used = 0
    start_time = time.time()

    for i, call in enumerate(calls):
        call_id = call["call_id"]
        summary = call.get("summary", "")
        outcome = call.get("outcome", "")

        # Prefer full transcript over BlandAI summary
        transcript = load_transcript(call_id)
        if transcript and len(transcript.split()) > 10:
            classify_text = transcript
            transcript_used += 1
        else:
            classify_text = summary
            summary_used += 1

        try:
            classification = classify_call(
                client, system_prompt, classify_text, model=args.model,
            )
        except Exception as e:
            print(f"  [{i+1}] ERROR classifying {call_id[:8]}: {e}")
            classification = {
                "component": "other",
                "symptom_category": "other",
                "caller_type": "unknown",
                "resolution": "unresolved",
                "resolution_reason": f"classification error: {str(e)[:80]}",
            }
            errors += 1
            # If credit balance error, stop early — no point continuing
            if "credit balance" in str(e).lower():
                print("\n  STOPPING: API credit balance too low. Top up and re-run.")
                break
            time.sleep(1)

        # Normalize component name (underscores to spaces)
        classification["component"] = classification["component"].replace("_", " ")

        # Validate symptom_category against component
        comp = classification["component"]
        sym = classification["symptom_category"]
        valid_syms = component_symptoms.get(comp, [])
        if valid_syms and sym not in valid_syms:
            classification["symptom_category"] = "other"

        results.append({
            "call_id": call_id,
            "created_at": call.get("created_at", ""),
            "call_length_min": call.get("call_length_min", ""),
            "ended_by": call.get("ended_by", ""),
            "pathway_name": call.get("pathway_name", ""),
            "pathway_version": call.get("pathway_version", ""),
            "outcome": outcome,
            "component": comp,
            "symptom_category": classification["symptom_category"],
            "caller_type": classification["caller_type"],
            "resolution": classification["resolution"],
            "resolution_reason": classification["resolution_reason"],
            "summary": summary[:500],
        })

        if (i + 1) % 25 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(calls) - i - 1) / rate
            print(f"  [{i+1}/{len(calls)}] {rate:.1f} calls/sec, ~{remaining:.0f}s remaining")

    backfill_and_write(results)

    elapsed = time.time() - start_time
    print(f"\nDone! Classified {len(calls)} calls in {elapsed:.1f}s")
    print(f"  Transcripts used: {transcript_used}")
    print(f"  Summary fallback: {summary_used}")
    print(f"  Errors: {errors}")


if __name__ == "__main__":
    main()
