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
<title>__TITLE__</title>
<style>
:root{--bg:#f8fafc;--panel:#fff;--text:#172033;--muted:#526077;--border:#dbe3ed;--accent:#1859aa}
*{box-sizing:border-box}body{margin:0;font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial;background:var(--bg);color:var(--text)}
.wrap{max-width:1100px;margin:0 auto;padding:18px}.title{text-align:center;padding:10px 0}h1{margin:0 0 6px;font-size:28px}.sub{margin:0;color:var(--muted)}
.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:14px;box-shadow:0 4px 14px rgba(23,32,51,.06);margin:8px 0}
.controls{display:grid;grid-template-columns:1.5fr .8fr .8fr 1.1fr;gap:12px;align-items:end}.control label{display:block;font-size:12px;color:var(--muted);margin:0 0 6px}
input,select{width:100%;padding:10px;border-radius:8px;border:1px solid var(--border);background:#fff;color:var(--text);font:inherit}input:focus,select:focus{outline:none;border-color:var(--accent)}
.checks{display:flex;flex-wrap:wrap;gap:8px}.checks label{display:flex;gap:6px;align-items:center;border:1px solid var(--border);padding:6px 8px;border-radius:999px;background:#fff;user-select:none}.checks input{width:auto}
.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;background:#fff}th,td{padding:.8rem;text-align:left;border-bottom:1px solid var(--border);white-space:nowrap}th{position:sticky;top:0;background:#e8eef7;font-size:12px;color:var(--muted);cursor:pointer}tbody tr:hover{background:#f1f5f9}.rank{color:var(--accent);font-weight:600}.score{font-variant-numeric:tabular-nums}.foot{margin-top:10px;color:var(--muted);font-size:12px}@media(max-width:800px){.controls{grid-template-columns:1fr 1fr}}
</style></head><body><header class="wrap"><div class="title"><h1>__TITLE__</h1><p class="sub">__SUBTITLE__</p></div></header>
<main class="wrap"><section class="controls card"><div class="control"><label for="search">Search</label><input id="search" type="search" placeholder="Search team or model…"></div><div class="control"><label for="modelFilter">Model</label><select id="modelFilter"><option value="all">All</option></select></div><div class="control"><label for="typeFilter">Type</label><select id="typeFilter"><option value="all">All</option></select></div><div class="control"><label>Columns</label><div class="checks" id="columnToggles"></div></div></section>
<section class="card"><div class="table-wrap"><table id="board"><thead><tr><th data-key="rank">Rank</th><th data-key="team">Team</th><th data-key="type">Type</th><th data-key="model">Model</th><th data-key="score">F1 Score</th></tr></thead><tbody></tbody></table></div><div class="foot" id="status"></div></section></main>
<script>
const state={rows:[],filtered:[],sortKey:'score',sortDir:'desc',hidden:new Set()};
const columns=[['team','Team'],['type','Type'],['model','Model'],['score','F1 Score']];
function render(){const body=document.querySelector('#board tbody');body.replaceChildren();state.filtered.forEach((row,index)=>{const tr=document.createElement('tr');const values=[['rank',index?row.rank:row.rank],['team',row.team],['type',row.type],['model',row.model],['score',Number(row.score).toFixed(6)]];values.forEach(([key,value])=>{const cell=document.createElement('td');cell.dataset.key=key;cell.textContent=String(value);if(key==='rank')cell.className='rank';if(key==='score')cell.classList.add('score');if(state.hidden.has(key))cell.style.display='none';tr.appendChild(cell)});body.appendChild(tr)});document.querySelectorAll('#board th').forEach(th=>th.style.display=state.hidden.has(th.dataset.key)?'none':'');document.getElementById('status').textContent=state.filtered.length?`${state.filtered.length} result(s)`:'No results'}
function filter(){const query=document.getElementById('search').value.toLowerCase().trim(),model=document.getElementById('modelFilter').value,type=document.getElementById('typeFilter').value;let rows=state.rows.filter(row=>(model==='all'||row.model===model)&&(type==='all'||row.type===type)&&(!query||`${row.team} ${row.model} ${row.type}`.toLowerCase().includes(query)));const dir=state.sortDir==='asc'?1:-1;rows.sort((a,b)=>{let x=a[state.sortKey],y=b[state.sortKey];if(state.sortKey==='score')return(Number(x)-Number(y))*dir;return String(x).localeCompare(String(y))*dir});state.filtered=rows;render()}
function options(id,key){const select=document.getElementById(id);[...new Set(state.rows.map(row=>row[key]).filter(Boolean))].sort().forEach(value=>{const option=document.createElement('option');option.value=option.textContent=value;select.appendChild(option)})}
function toggles(){const box=document.getElementById('columnToggles');columns.forEach(([key,label])=>{const labelNode=document.createElement('label'),check=document.createElement('input');check.type='checkbox';check.checked=true;check.addEventListener('change',()=>{check.checked?state.hidden.delete(key):state.hidden.add(key);render()});labelNode.append(check,document.createTextNode(label));box.appendChild(labelNode)})}
fetch('leaderboard.json',{cache:'no-store'}).then(response=>response.ok?response.json():Promise.reject()).then(rows=>{state.rows=rows;options('modelFilter','model');options('typeFilter','type');toggles();document.getElementById('search').addEventListener('input',filter);document.getElementById('modelFilter').addEventListener('change',filter);document.getElementById('typeFilter').addEventListener('change',filter);document.querySelectorAll('#board th').forEach(th=>th.addEventListener('click',()=>{const key=th.dataset.key;if(state.sortKey===key)state.sortDir=state.sortDir==='asc'?'desc':'asc';else{state.sortKey=key;state.sortDir=key==='score'?'desc':'asc'}filter()}));filter()}).catch(()=>{document.getElementById('status').textContent='Unable to load leaderboard.'});
</script></body></html>'''


def render(root: Path, records: list[dict] | None = None) -> None:
    root = Path(root).resolve()
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
    labels = {
        "DERMA": ("DERMA Challenge Leaderboard", "Dermoscopic Evaluation of Region Masks and Algorithms"),
        "TRACE": ("TRACE Challenge Leaderboard", "Tomato Recognition And Coding Evaluation"),
    }
    title, subtitle = labels.get(root.name, ("Leaderboard", ""))
    page = PAGE.replace("__TITLE__", html.escape(title)).replace("__SUBTITLE__", html.escape(subtitle))
    atomic_write(root / "docs/index.html", page.encode(), overwrite=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    render(parser.parse_args().root)
