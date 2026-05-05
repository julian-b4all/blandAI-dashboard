"""
BlandAI Data Pipeline — Pull All Calls
=======================================
Stage 1: Pull call list (metadata only, no transcripts)
Stage 2: Pull full details for each call (transcripts, pathway_logs, variables, etc.)

Outputs:
  data/calls_list.json       — raw API response from list endpoint
  data/calls_detail/         — one JSON per call (call_id.json)
  data/calls_summary.csv     — flat CSV with key fields for analysis
"""

import json
import csv
import time
import sys
from pathlib import Path
from datetime import datetime

import requests
from dotenv import load_dotenv
import os

# ── Config ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT.parent / ".env"
DATA_DIR = PROJECT_ROOT / "data"
DETAIL_DIR = DATA_DIR / "calls_detail"

API_BASE = "https://api.bland.ai/v1"
RATE_LIMIT_DELAY = 0.25  # seconds between detail requests (be polite)

# Our inbound numbers for reference
INBOUND_NUMBERS = {
    "+14153246039": "Applicant Pathway (Original)",
    "+17144521864": "Full Support Pathway",
    "+17146134173": "Spanish Pathway",
    "+16504139279": "Julian Applicant Pathway",
}


def load_api_key():
    """Load BLAND_API_KEY from shared .env"""
    load_dotenv(ENV_PATH, override=True)
    key = os.getenv("BLAND_API_KEY")
    if not key:
        print(f"ERROR: BLAND_API_KEY not found in {ENV_PATH}")
        sys.exit(1)
    return key


def get_headers(api_key):
    return {"authorization": api_key}


# ── Stage 1: Pull Call List ─────────────────────────────────────────────────

def pull_call_list(api_key):
    """Pull all calls from the API with pagination."""
    headers = get_headers(api_key)
    all_calls = []
    batch_size = 1000
    offset = 0

    print("Stage 1: Pulling call list...")

    while True:
        params = {
            "limit": batch_size,
            "from": offset,
            "to": offset + batch_size,
            "ascending": True,
            "sort_by": "created_at",
        }

        resp = requests.get(f"{API_BASE}/calls", headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()

        calls = data.get("calls", [])
        total = data.get("total_count", 0)

        all_calls.extend(calls)
        print(f"  Fetched {len(all_calls)}/{total} calls")

        # Stop only when this batch returned nothing
        if not calls:
            break

        offset += batch_size
        time.sleep(0.5)

    # Deduplicate by call_id (in case pagination overlapped)
    seen = set()
    unique = []
    for c in all_calls:
        cid = c.get("call_id") or c.get("c_id")
        if cid and cid not in seen:
            seen.add(cid)
            unique.append(c)
    if len(unique) < len(all_calls):
        print(f"  Deduplicated: {len(all_calls)} → {len(unique)}")
    all_calls = unique

    # Save raw list
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    list_path = DATA_DIR / "calls_list.json"
    with open(list_path, "w", encoding="utf-8") as f:
        json.dump({"total_count": len(all_calls), "calls": all_calls}, f, indent=2, ensure_ascii=False)

    print(f"  Saved {len(all_calls)} calls to {list_path}")
    return all_calls


# ── Stage 2: Pull Call Details ──────────────────────────────────────────────

def pull_call_details(api_key, calls):
    """Pull full details for each call. Skips already-downloaded calls."""
    headers = get_headers(api_key)
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)

    total = len(calls)
    skipped = 0
    fetched = 0
    errors = 0

    print(f"\nStage 2: Pulling details for {total} calls...")

    for i, call in enumerate(calls):
        call_id = call.get("call_id") or call.get("c_id")
        if not call_id:
            print(f"  WARNING: No call_id in record {i}")
            errors += 1
            continue

        detail_path = DETAIL_DIR / f"{call_id}.json"

        # Skip if already downloaded
        if detail_path.exists():
            skipped += 1
            continue

        try:
            resp = requests.get(f"{API_BASE}/calls/{call_id}", headers=headers)
            resp.raise_for_status()
            detail = resp.json()

            with open(detail_path, "w", encoding="utf-8") as f:
                json.dump(detail, f, indent=2, ensure_ascii=False)

            fetched += 1

            if (fetched + skipped) % 50 == 0 or fetched + skipped == total:
                print(f"  Progress: {fetched + skipped}/{total} ({fetched} new, {skipped} cached)")

        except requests.exceptions.HTTPError as e:
            print(f"  ERROR on {call_id}: {e}")
            errors += 1
        except Exception as e:
            print(f"  ERROR on {call_id}: {e}")
            errors += 1

        time.sleep(RATE_LIMIT_DELAY)

    print(f"  Done: {fetched} fetched, {skipped} cached, {errors} errors")


# ── Stage 3: Build Summary CSV ─────────────────────────────────────────────

def build_summary_csv():
    """Read all detail JSONs and produce a flat CSV for analysis."""
    print("\nStage 3: Building summary CSV...")

    if not DETAIL_DIR.exists():
        print("  ERROR: No detail files found. Run stages 1-2 first.")
        return

    detail_files = sorted(DETAIL_DIR.glob("*.json"))
    if not detail_files:
        print("  ERROR: No detail files found.")
        return

    rows = []
    for path in detail_files:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)

        # Handle nested response — some APIs wrap in a top-level key
        if "call_id" not in d and len(d) == 1:
            d = list(d.values())[0]

        # Extract transcript stats
        transcripts = d.get("transcripts", []) or []
        user_turns = sum(1 for t in transcripts if t.get("user") == "user")
        agent_turns = sum(1 for t in transcripts if t.get("user") == "assistant")
        total_turns = user_turns + agent_turns

        # Determine which inbound number was called
        to_number = d.get("to", "")
        from_number = d.get("from", "")
        inbound = d.get("inbound", None)

        # For inbound calls, "to" is our number; for outbound, "from" is our number
        pathway_number = to_number if inbound else from_number
        pathway_name = INBOUND_NUMBERS.get(pathway_number, "Unknown")

        # Concatenated transcript length (word count)
        concat = d.get("concatenated_transcript", "") or ""
        transcript_words = len(concat.split()) if concat else 0

        # Check for transfer
        transferred = bool(d.get("transferred_to"))
        transferred_to = d.get("transferred_to", "")

        # Call ended by
        ended_by = d.get("call_ended_by", "")

        # Summary
        summary = d.get("summary", "") or ""

        # Pathway info
        pathway_id = d.get("pathway_id", "")
        pathway_version = d.get("pathway_version", "")
        pathway_logs = d.get("pathway_logs", "") or ""
        if isinstance(pathway_logs, list):
            pathway_log_lines = len(pathway_logs)
        elif isinstance(pathway_logs, str):
            pathway_log_lines = len(pathway_logs.split("\n")) if pathway_logs else 0
        else:
            pathway_log_lines = 0

        # Variables captured during call
        variables = d.get("variables", {}) or {}
        variable_keys = ", ".join(variables.keys()) if variables else ""

        # Status fields
        status = d.get("status", "")
        queue_status = d.get("queue_status", "")
        answered_by = d.get("answered_by", "")
        error_message = d.get("error_message", "") or ""

        # Cost
        price = d.get("price", 0) or 0

        rows.append({
            "call_id": d.get("call_id", path.stem),
            "created_at": d.get("created_at", ""),
            "started_at": d.get("started_at", ""),
            "call_length_min": d.get("call_length", 0),
            "corrected_duration_sec": d.get("corrected_duration", ""),
            "status": status,
            "queue_status": queue_status,
            "answered_by": answered_by,
            "completed": d.get("completed", ""),
            "inbound": inbound,
            "from": from_number,
            "to": to_number,
            "pathway_number": pathway_number,
            "pathway_name": pathway_name,
            "pathway_id": pathway_id,
            "pathway_version": pathway_version,
            "ended_by": ended_by,
            "transferred": transferred,
            "transferred_to": transferred_to,
            "user_turns": user_turns,
            "agent_turns": agent_turns,
            "total_turns": total_turns,
            "transcript_words": transcript_words,
            "summary": summary[:500],  # truncate for CSV readability
            "variable_keys": variable_keys,
            "pathway_log_lines": pathway_log_lines,
            "price": price,
            "error_message": error_message[:200],
            "record": d.get("record", False),
            "voice_id": d.get("voice_id", ""),
        })

    # Sort by created_at
    rows.sort(key=lambda r: r.get("created_at", ""))

    # Write CSV
    csv_path = DATA_DIR / "calls_summary.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"  Saved {len(rows)} rows to {csv_path}")

    # Print quick stats
    print("\n-- Quick Stats --")
    print(f"  Total calls: {len(rows)}")

    # By pathway
    by_pathway = {}
    for r in rows:
        name = r["pathway_name"]
        by_pathway.setdefault(name, []).append(r)

    for name, calls in sorted(by_pathway.items(), key=lambda x: -len(x[1])):
        completed = sum(1 for c in calls if c["status"] == "completed")
        transferred = sum(1 for c in calls if c["transferred"])
        avg_len = sum(c["call_length_min"] or 0 for c in calls) / max(len(calls), 1)
        print(f"  {name}: {len(calls)} calls, {completed} completed, {transferred} transferred, avg {avg_len:.1f} min")

    # By status
    print("\n  By status:")
    by_status = {}
    for r in rows:
        by_status.setdefault(r["status"], 0)
        by_status[r["status"]] += 1
    for status, count in sorted(by_status.items(), key=lambda x: -x[1]):
        print(f"    {status or '(empty)'}: {count}")

    # By answered_by
    print("\n  By answered_by:")
    by_answered = {}
    for r in rows:
        by_answered.setdefault(r["answered_by"] or "(empty)", 0)
        by_answered[r["answered_by"] or "(empty)"] += 1
    for ans, count in sorted(by_answered.items(), key=lambda x: -x[1]):
        print(f"    {ans}: {count}")

    # Transfer stats
    total_completed = sum(1 for r in rows if r["status"] == "completed")
    total_transferred = sum(1 for r in rows if r["transferred"])
    total_inbound = sum(1 for r in rows if r["inbound"])
    print(f"\n  Inbound: {total_inbound} | Completed: {total_completed} | Transferred: {total_transferred}")
    if total_completed > 0:
        resolution_rate = (total_completed - total_transferred) / total_completed * 100
        print(f"  Estimated resolution rate (completed - transferred): {resolution_rate:.1f}%")


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    api_key = load_api_key()

    # Stage 1: Pull call list
    calls = pull_call_list(api_key)

    # Stage 2: Pull call details (with caching)
    pull_call_details(api_key, calls)

    # Stage 3: Build summary CSV
    build_summary_csv()

    print("\nDone! Files saved to:", DATA_DIR)
