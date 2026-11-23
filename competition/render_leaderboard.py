"""Render the public leaderboard without interpreting participant text as HTML."""
from __future__ import annotations

import csv
import html
import io
import json
from pathlib import Path

from encryption.crypto import atomic_write

COLUMNS = ["rank", "team", "type", "model", "f1_score"]


def csv_safe(value: object) -> str:
    value = str(value)
    # Spreadsheet import must not turn participant text into a formula.
    if value.lstrip().startswith(("=", "+", "-", "@")) or value.startswith(("\t", "\r", "\n")):
        return "'" + value
    return value


def markdown_safe(value: object) -> str:
    return html.escape(str(value), quote=True).replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def ranked_records(records: list[dict]) -> list[dict]:
    ordered = sorted(records, key=lambda entry: (-entry["score"], entry["timestamp_utc"], entry["team"].casefold()))
    previous_score = None
    rank = 0
    result = []
    for index, entry in enumerate(ordered, 1):
        if previous_score is None or entry["score"] != previous_score:
            rank = index
        previous_score = entry["score"]
        result.append({"rank": rank, **entry})
    return result


PAGE = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Competition leaderboard</title>
<style>body{font:16px system-ui,sans-serif;max-width:1200px;margin:3rem auto;padding:0 1rem;background:#f8fafc;color:#172033}h1{font-size:2rem}table{border-collapse:collapse;width:100%;background:white}th,td{text-align:left;padding:.8rem;border-bottom:1px solid #dbe3ed}th{background:#e8eef7}.scroll{overflow-x:auto}.muted{color:#526077}a{color:#1859aa}button,input{font:inherit}</style>
</head><body><h1>Competition leaderboard</h1><p class="muted" id="status">Loading verified submissions…</p>
<p><a href="leaderboard.csv">Download CSV</a> · Scores are ranked using full precision; equal scores share a rank.</p>
<div class="scroll"><table><thead><tr><th>Rank</th><th>Team</th><th>Type</th><th>Model</th><th>F1 Score</th></tr></thead><tbody id="entries"></tbody></table></div>
<script>
fetch('leaderboard.json',{cache:'no-store'}).then(response=>{if(!response.ok)throw Error('Unable to load leaderboard');return response.json()}).then(data=>{
 const body=document.getElementById('entries');
 for(const entry of data){const row=document.createElement('tr');
   for(const value of [entry.rank,entry.team,entry.type,entry.model,Number(entry.score).toFixed(6)]){
     const cell=document.createElement('td');cell.textContent=String(value);row.appendChild(cell);
   }body.appendChild(row);
 }
 document.getElementById('status').textContent=data.length ? data[0].benchmark_id+' · '+data.length+' valid submissions' : 'No valid submissions yet.';
}).catch(()=>{document.getElementById('status').textContent='The leaderboard could not be loaded. Please use the CSV link.'});
</script></body></html>
'''


def render(root: Path, records: list[dict] | None = None) -> None:
    root = Path(root)
    if records is None:
        ledger = root / "leaderboard/records.json"
        records = json.loads(ledger.read_text(encoding="utf-8")) if ledger.exists() else []
    ranked = ranked_records(records)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    for entry in ranked:
        row = {name: entry.get(name, "") for name in COLUMNS}
        row["f1_score"] = repr(entry["score"])
        writer.writerow({name: csv_safe(value) for name, value in row.items()})
    csv_contents = buffer.getvalue().encode("utf-8")
    lines = ["# Leaderboard", "", "Higher scores are better. Exact full-precision ties share a rank.", "",
             "| Rank | Team | Type | Model | F1 Score |", "| --- | --- | --- | --- | --- |"]
    for entry in ranked:
        cells = [entry["rank"], entry["team"], entry["type"], entry["model"], format(entry["score"], ".6f")]
        lines.append("| " + " | ".join(markdown_safe(cell) for cell in cells) + " |")
    atomic_write(root / "leaderboard/leaderboard.csv", csv_contents, overwrite=True)
    atomic_write(root / "leaderboard/leaderboard.md", ("\n".join(lines) + "\n").encode(), overwrite=True)
    atomic_write(root / "docs/leaderboard.csv", csv_contents, overwrite=True)
    atomic_write(root / "docs/leaderboard.json", (json.dumps(ranked, indent=2, allow_nan=False) + "\n").encode(), overwrite=True)
    atomic_write(root / "docs/index.html", PAGE.encode(), overwrite=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    render(parser.parse_args().root)
