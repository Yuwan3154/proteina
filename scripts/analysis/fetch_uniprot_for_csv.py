#!/usr/bin/env python3
"""Fetch UniProtKB annotations for protein IDs listed in a CSV and build an HTML dashboard.

Reads protein identifiers from a column (default protein_id), strips trailing -F<number>
suffix (AlphaFold-style fragment IDs), queries `GET https://rest.uniprot.org/uniprotkb/accessions`,
writes `uniprot_summary.tsv` + optional Plotly HTML per plan.

Example::

    conda activate proteina
    python scripts/analysis/fetch_uniprot_for_csv.py \\
        prediction/replica_compare_4-seq_vs_21-seq/compare_replicas_converged_passing_intersect.csv \\
        --plot-html ../prediction_out/uniprot_intersect.html
"""

from __future__ import annotations

import argparse
import csv
import gzip
import http.client
import json
import re
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

REST_HOST = "rest.uniprot.org"
USER_AGENT = "proteina-fetch-uniprot/1.0 (research; institutional)"

DEFAULT_FIELDS = (
    "accession,id,annotation_score,protein_existence,lit_pubmed_id,lit_doi_id"
)


def normalize_accession(raw_id: str) -> str:
    raw_id = raw_id.strip()
    raw_id = re.sub(r"-F\d+$", "", raw_id)
    return raw_id


def collect_ordered_pairs(csv_path: Path, column: str) -> Tuple[List[str], Dict[str, str]]:
    """Return ordered unique accessions and mapping accession -> first-seen source protein_id."""
    order: List[str] = []
    acc_first_pid: Dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if column not in (reader.fieldnames or []):
            raise ValueError(f"Column '{column}' not in CSV columns {reader.fieldnames}")
        for row in reader:
            pid = row.get(column)
            if not pid:
                continue
            pid = pid.strip()
            acc = normalize_accession(pid)
            if acc not in acc_first_pid:
                acc_first_pid[acc] = pid
                order.append(acc)
    return order, acc_first_pid


def load_configure_field_names() -> List[str]:
    url = "https://rest.uniprot.org/configure/uniprotkb/result-fields"
    groups = https_get_json(url)
    names: List[str] = []
    if not isinstance(groups, list):
        raise RuntimeError("configure/uniprotkb/result-fields: expected top-level JSON array")
    for grp in groups:
        if not isinstance(grp, dict):
            continue
        for f in grp.get("fields") or []:
            if isinstance(f, dict) and "name" in f:
                names.append(f["name"])
    deduped = sorted(set(names))
    if not deduped:
        raise RuntimeError("configure/uniprotkb/result-fields returned no field names")
    return deduped


def https_get_json(url: str) -> Any:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or REST_HOST
    path_q = urllib.parse.urlunparse(("", "", parsed.path or "/", "", parsed.query, ""))

    conn = http.client.HTTPSConnection(host, timeout=180)
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "User-Agent": USER_AGENT,
    }
    conn.request("GET", path_q, headers=headers)
    res = conn.getresponse()
    raw = res.read()
    conn.close()

    if res.status == 429:
        wait_s = res.getheader("Retry-After")
        sleep_for = int(wait_s) if wait_s and wait_s.isdigit() else 30
        time.sleep(sleep_for)
        return https_get_json(url)

    if res.status != 200:
        snippet = raw.decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"HTTP {res.status} for {url}: {snippet}")

    payload = raw
    while len(payload) >= 2 and payload[0] == 0x1F and payload[1] == 0x8B:
        payload = gzip.decompress(payload)
    return json.loads(payload.decode("utf-8"))


def chunk(seq: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def fetch_accessions_batch(
    accessions: Sequence[str],
    fields_csv: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Fetch UniProt entries via GET accessions endpoint."""
    q_fields = urllib.parse.quote(fields_csv, safe=",")
    acc_join = urllib.parse.quote(",".join(accessions), safe=",")
    url = (
        f"https://rest.uniprot.org/uniprotkb/accessions"
        f"?accessions={acc_join}&fields={q_fields}"
    )
    data = https_get_json(url)
    if "messages" in data and "results" not in data:
        raise RuntimeError(f"UniProt error response: {data}")

    results = data.get("results") or []
    missing: List[str] = []

    got = {r.get("primaryAccession"): r for r in results}
    for a in accessions:
        if a not in got:
            missing.append(a)
    return list(results), missing


def merge_extra_fields_rows(
    base_rows_by_acc: Dict[str, Dict[str, Any]],
    accessions: Sequence[str],
    field_batches: Iterable[List[str]],
) -> None:
    for fb in field_batches:
        fields_csv = "accession," + ",".join(fb)
        batch_results, _missing = fetch_accessions_batch(accessions, fields_csv)
        for r in batch_results:
            acc = r.get("primaryAccession")
            if not acc or acc not in base_rows_by_acc:
                continue
            for k, v in r.items():
                if k in {"extraAttributes"}:
                    continue
                if k == "primaryAccession":
                    continue
                if k == "references":
                    continue
                base_rows_by_acc[acc][k] = v


def citation_crossrefs(citation: Mapping[str, Any]) -> Tuple[str | None, str | None]:
    pmid = None
    doi = None
    for xr in citation.get("citationCrossReferences") or []:
        if not isinstance(xr, dict):
            continue
        db = xr.get("database")
        if db == "PubMed" and pmid is None:
            pmid = xr.get("id")
        if db == "DOI" and doi is None:
            doi = xr.get("id")
    return pmid, doi


def publication_derived(references: Any) -> Dict[str, Any]:
    pubmed_ids: List[str] = []
    doi_ids: List[str] = []
    titled_rows: List[Tuple[str | None, str]] = []

    seen_pubmed_title = set()

    for ref in references or []:
        if not isinstance(ref, dict):
            continue
        cit = ref.get("citation") or {}
        if not isinstance(cit, dict):
            continue
        pmid, doi = citation_crossrefs(cit)
        if pmid:
            pubmed_ids.append(pmid)
        if doi:
            doi_ids.append(doi)

        title = cit.get("title")
        if not title or not str(title).strip():
            continue
        dedupe_key = pmid if pmid else str(title).strip().lower()
        if dedupe_key in seen_pubmed_title:
            continue
        seen_pubmed_title.add(dedupe_key)
        titled_rows.append((pmid, str(title).strip()))

    sorted_pubmed = sorted(set(pubmed_ids))
    sorted_dois = sorted(set(doi_ids))
    titles_join = ", ".join(t for _, t in titled_rows)

    return {
        "pubmed_count": len(sorted_pubmed),
        "doi_count": len(sorted_dois),
        "publication_count": len(titled_rows),
        "lit_pubmed_id_flat": ";".join(sorted_pubmed),
        "lit_doi_id_flat": ";".join(sorted_dois),
        "publication_titles": titles_join,
    }


def parse_reviewed_label(entry_type: str | None) -> Tuple[bool | None, str]:
    if not entry_type:
        return None, "unknown"
    low = entry_type.lower()
    if "reviewed" in low and "unreviewed" not in low:
        return True, "Reviewed (Swiss-Prot)"
    if "unreviewed" in low:
        return False, "Unreviewed (TrEMBL)"
    return None, entry_type


def flatten_summary_rows(
    results_map: Mapping[str, Dict[str, Any]],
    ordered_accessions: Sequence[str],
    source_pid_for_acc: Mapping[str, str],
    extras_merge: Mapping[str, Mapping[str, str]] | None,
) -> List[Dict[str, Any]]:
    rows_out: List[Dict[str, Any]] = []
    for acc in ordered_accessions:
        row_base: Dict[str, Any] = {
            "query_accession": acc,
            "source_protein_id": source_pid_for_acc.get(acc, ""),
        }
        if extras_merge and row_base["source_protein_id"] in extras_merge:
            for ck, cv in extras_merge[row_base["source_protein_id"]].items():
                row_base[f"input_{ck}"] = cv

        res = results_map.get(acc)
        if res is None:
            row_base["fetch_status"] = "missing"
            rows_out.append(row_base)
            continue

        entry_type = res.get("entryType")
        reviewed_bool, status_txt = parse_reviewed_label(entry_type)

        pub = publication_derived(res.get("references"))

        flat = {
            **row_base,
            "fetch_status": "ok",
            "primaryAccession": res.get("primaryAccession"),
            "uniProtKb_id": res.get("uniProtkbId"),
            "entryType": entry_type,
            "reviewed_bool": reviewed_bool,
            "status_label": status_txt,
            "annotation_score": res.get("annotationScore"),
            "protein_existence": res.get("proteinExistence"),
            **pub,
        }

        extras_attrs = res.get("extraAttributes")
        flat["uniParc_id"] = (
            (extras_attrs or {}).get("uniParcId") if isinstance(extras_attrs, dict) else ""
        )

        mapped_sources = {
            "primaryAccession",
            "uniProtkbId",
            "annotationScore",
            "proteinExistence",
            "entryType",
            "references",
            "extraAttributes",
        }

        for k, v in res.items():
            if k in mapped_sources:
                continue
            flat[f"uniprot_{k}"] = json.dumps(v) if isinstance(v, (dict, list)) else v

        rows_out.append(flat)
    return rows_out


def write_summary_tsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    base_cols = [
        "query_accession",
        "source_protein_id",
        "fetch_status",
        "primaryAccession",
        "uniProtKb_id",
        "status_label",
        "reviewed_bool",
        "annotation_score",
        "protein_existence",
        "pubmed_count",
        "doi_count",
        "publication_count",
        "lit_pubmed_id_flat",
        "lit_doi_id_flat",
        "publication_titles",
        "entryType",
        "uniParc_id",
    ]

    dynamic = set()
    for r in rows:
        for k in r:
            if k not in base_cols:
                dynamic.add(k)

    columns = [c for c in base_cols if any(c in row for row in rows)]
    columns.extend(sorted(dynamic))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(dict(r))


def numbered_hover_titles(titles_join: str, limit_lines: int = 80) -> str:
    if not titles_join.strip():
        return "(no titles)"
    parts = [p.strip() for p in titles_join.split(", ") if p.strip()]
    lines = [f"{j + 1}. {parts[j]}" for j in range(min(len(parts), limit_lines))]
    if len(parts) > limit_lines:
        lines.append(f"... (+{len(parts) - limit_lines} more)")
    return "<br>".join(lines)


def build_dashboard_html(
    rows: Sequence[Mapping[str, Any]],
    report_title: str,
    include_plotlyjs_cdn: bool,
) -> str:
    ok_rows = [r for r in rows if r.get("fetch_status") == "ok"]
    labels = [
        str(r.get("source_protein_id") or r.get("uniProtKb_id") or r.get("query_accession"))
        for r in ok_rows
    ]

    scores = [_safe_float(r.get("annotation_score")) for r in ok_rows]
    pub_counts = [
        int(v) if v is not None else 0 for v in (_safe_float(r.get("publication_count")) for r in ok_rows)
    ]
    statuses = [str(r.get("status_label") or "?") for r in ok_rows]
    existence = [str(r.get("protein_existence") or "?") for r in ok_rows]
    titles_long = [str(r.get("publication_titles") or "") for r in ok_rows]

    reviewed_flag = [1 if r.get("reviewed_bool") is True else 0 for r in ok_rows]

    # Panel A
    fig_a = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Review status", "Protein existence", "Annotation score distribution"),
        specs=[[{"type": "domain"}, {"type": "xy"}, {"type": "xy"}]],
        horizontal_spacing=0.06,
    )

    rev_yes = sum(reviewed_flag)
    rev_no = len(reviewed_flag) - rev_yes
    pie_labels = []
    pie_vals = []
    if rev_yes:
        pie_labels.append("Reviewed")
        pie_vals.append(rev_yes)
    if rev_no:
        pie_labels.append("Unreviewed")
        pie_vals.append(rev_no)
    if not pie_vals:
        pie_labels = ["n/a"]
        pie_vals = [1]

    fig_a.add_trace(
        go.Pie(
            labels=pie_labels,
            values=pie_vals,
            hole=0.55,
            textinfo="label+percent",
            marker=dict(colors=["#2ecc71", "#3498db"]),
        ),
        row=1,
        col=1,
    )

    from collections import Counter

    ex_counts = Counter(existence)
    ex_labels_sorted = sorted(ex_counts.keys(), key=lambda k: (-ex_counts[k], k))
    fig_a.add_trace(
        go.Bar(
            x=[ex_counts[k] for k in ex_labels_sorted],
            y=ex_labels_sorted,
            orientation="h",
            marker_color="#9b59b6",
            hovertemplate="%{y}<br>count=%{x}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    hist_scores = [s for s in scores if s is not None]
    nbins = max(3, min(25, max(4, len(hist_scores) // 2))) if hist_scores else 3

    hist_trace = go.Histogram(
        x=hist_scores if hist_scores else [0.0],
        nbinsx=nbins,
        marker_color="#34495e",
        hovertemplate="bin %{x}<br>count=%{y}<extra></extra>",
    )
    fig_a.add_trace(hist_trace, row=1, col=3)

    fig_a.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=60, r=30, t=60, b=40),
        showlegend=False,
        title_text=f"Cohort overview — {report_title}",
    )

    # Panel B — ladder
    idx_sorted = sorted(range(len(ok_rows)), key=lambda i: (-(scores[i] or -1.0), labels[i]))
    lab_ord = [labels[i] for i in idx_sorted]
    score_ord = [scores[i] if scores[i] is not None else float("nan") for i in idx_sorted]
    hover_parts = []
    custom = []
    for i in idx_sorted:
        r = ok_rows[i]
        acc = str(r.get("primaryAccession") or "")
        st = str(r.get("status_label") or "")
        ex = str(r.get("protein_existence") or "")
        pc_val = _safe_float(r.get("publication_count"))
        pc = int(pc_val) if pc_val is not None else 0
        ht = numbered_hover_titles(str(r.get("publication_titles") or ""))
        line = (
            f"<b>{labels[i]}</b><br>{acc}<br>{st}<br>{ex}<br>publications (with titles): {pc}<br>{ht}"
        )
        hover_parts.append(line)
        row_extra = [acc, st, ex, pc]
        for key in sorted(r.keys()):
            if key.startswith("input_"):
                row_extra.append(f"{key}:{r[key]}")
        custom.append(row_extra)

    fig_b = go.Figure(
        go.Bar(
            orientation="h",
            y=lab_ord,
            x=score_ord,
            marker=dict(
                color=score_ord,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=dict(text="Annotation score")),
            ),
            customdata=custom,
            hovertemplate="%{hovertext}<extra></extra>",
            hovertext=hover_parts,
        )
    )
    fig_b.update_layout(
        template="plotly_white",
        title=f"Annotation score ranking — {report_title}",
        height=max(400, 120 + 28 * len(lab_ord)),
        margin=dict(l=140, r=90, t=60, b=40),
        xaxis_title="Annotation score",
    )

    # Panel C — scatter
    xs_c = []
    ys_c = []
    texts_c = []
    sizes_sc = []
    for i in range(len(ok_rows)):
        if scores[i] is None:
            continue
        xs_c.append(scores[i])
        ys_c.append(pub_counts[i])
        texts_c.append(labels[i])
        pc = pub_counts[i]
        sizes_sc.append(min(48, 10 + 3 * (pc ** 0.5)))

    fig_c = go.Figure(
        go.Scatter(
            x=xs_c,
            y=ys_c,
            mode="markers",
            marker=dict(size=sizes_sc, color="#e67e22", opacity=0.85, line=dict(width=0.5, color="#333")),
            text=texts_c,
            hovertemplate=(
                "<b>%{text}</b><br>annotation_score=%{x}<br>publication_count=%{y}<extra></extra>"
            ),
        )
    )
    fig_c.update_layout(
        template="plotly_white",
        title=f"Literature vs annotation score — {report_title}",
        height=480,
        margin=dict(l=60, r=40, t=60, b=50),
        xaxis_title="Annotation score",
        yaxis_title="Publication count (titled refs)",
    )

    # Panel D — table
    trunc_titles = []
    for t in titles_long:
        cap = 120
        if len(t) <= cap:
            trunc_titles.append(t)
        else:
            trunc_titles.append(t[: cap - 1] + "…")

    header_fill = "#dfe6e9"
    stripe_a = "#ffffff"
    stripe_b = "#f8f9fb"
    n_tbl = len(ok_rows)
    ncol_tbl = 8
    row_colors_tbl = [stripe_a if i % 2 == 0 else stripe_b for i in range(n_tbl)]
    fill_color_cols = [list(row_colors_tbl) for _ in range(ncol_tbl)]

    fig_d = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[
                        "Source protein_id",
                        "Accession",
                        "UniProtKB ID",
                        "Status",
                        "Protein existence",
                        "Ann. score",
                        "Pub count",
                        "Titles (trunc.)",
                    ],
                    fill_color=header_fill,
                    align="left",
                    font=dict(size=12, color="#2d3436"),
                ),
                cells=dict(
                    values=[
                        labels,
                        [str(r.get("primaryAccession") or "") for r in ok_rows],
                        [str(r.get("uniProtKb_id") or "") for r in ok_rows],
                        statuses,
                        existence,
                        [("" if s is None else f"{s:.4g}") for s in scores],
                        [str(int(pc)) for pc in pub_counts],
                        trunc_titles,
                    ],
                    fill_color=fill_color_cols,
                    align="left",
                    font=dict(size=11),
                    height=26,
                ),
            )
        ]
    )
    fig_d.update_layout(title=f"Detail table — {report_title}", height=max(320, 80 + 28 * len(ok_rows)))

    cfg = dict(displayModeBar=True, responsive=True)

    plotly_cdn = "https://cdn.plot.ly/plotly-2.35.2.min.js"
    script_tag = (
        f'<script charset="utf-8" src="{plotly_cdn}"></script>\n' if include_plotlyjs_cdn else ""
    )

    html_panel_a = pio.to_html(fig_a, include_plotlyjs=False, full_html=False, div_id="panel_a", config=cfg)
    html_panel_b = pio.to_html(fig_b, include_plotlyjs=False, full_html=False, div_id="panel_b", config=cfg)
    html_panel_c = pio.to_html(fig_c, include_plotlyjs=False, full_html=False, div_id="panel_c", config=cfg)
    html_panel_d = pio.to_html(fig_d, include_plotlyjs=False, full_html=False, div_id="panel_d", config=cfg)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    html_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{report_title} — UniProt summary</title>
{script_tag}<style>
body {{
  font-family: system-ui, sans-serif;
  margin: 0;
  background: #f0f2f6;
  color: #2d3436;
}}
.wrap {{
  max-width: 1280px;
  margin: 0 auto;
  padding: 20px 16px 48px;
}}
header {{
  background: linear-gradient(90deg, #ecf0f1, #fff);
  border-radius: 8px;
  padding: 16px 20px;
  margin-bottom: 18px;
  border: 1px solid #dfe6e9;
}}
header h1 {{ margin: 0 0 6px; font-size: 1.35rem; }}
header .stats {{ font-size: 0.92rem; color: #636e72; }}
section {{
  background: #fff;
  border-radius: 8px;
  padding: 12px 10px 20px;
  margin-bottom: 22px;
  border: 1px solid #dfe6e9;
}}
section h2 {{
  margin: 4px 8px 10px;
  font-size: 1.05rem;
  border-bottom: 1px solid #b2bec3;
  padding-bottom: 6px;
}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>{report_title}</h1>
    <div class="stats">Proteins plotted (found): {len(ok_rows)} · Generated {ts}</div>
  </header>
  <section><h2>Panel A — Cohort overview</h2><div id="panel_a_wrap">{html_panel_a}</div></section>
  <section><h2>Panel B — Annotation score ranking</h2><div id="panel_b_wrap">{html_panel_b}</div></section>
  <section><h2>Panel C — Publications vs annotation score</h2><div id="panel_c_wrap">{html_panel_c}</div></section>
  <section><h2>Panel D — Detail table</h2><div id="panel_d_wrap">{html_panel_d}</div></section>
</div>
</body>
</html>
"""

    return html_page


def _safe_float(val: Any) -> float | None:
    if val is None or val == "":
        return None
    return float(val)


def pipeline_extra_columns(
    csv_path: Path,
    protein_column: str,
    protein_ids: Iterable[str],
) -> Dict[str, Dict[str, str]]:
    wanted = set(protein_ids)
    agg: Dict[str, Dict[str, str]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        for row in reader:
            pid = row.get(protein_column, "").strip()
            if pid not in wanted:
                continue
            if pid in agg:
                continue
            entry = {k: str(v) for k, v in row.items() if k != protein_column and k in fields}
            agg[pid] = entry
    return agg


def split_field_batches(all_names: Sequence[str], max_chars: int = 2800) -> List[List[str]]:
    batches: List[List[str]] = []
    cur: List[str] = []
    cur_len = 0
    for name in all_names:
        if name == "accession":
            continue
        add = len(name) + 1
        if cur and cur_len + add > max_chars:
            batches.append(cur)
            cur = []
            cur_len = 0
        cur.append(name)
        cur_len += add
    if cur:
        batches.append(cur)
    return batches


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch UniProtKB rows for CSV protein_id column.")
    parser.add_argument("input_csv", type=Path, help="Input CSV path.")
    parser.add_argument(
        "--protein-id-column",
        default="protein_id",
        help="Column with UniProt-like IDs (default protein_id).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for outputs (default: next to input CSV).",
    )
    parser.add_argument("--chunk-size", type=int, default=80, help="Accessions per REST batch.")
    parser.add_argument(
        "--fields",
        default=DEFAULT_FIELDS,
        help="Comma-separated UniProt return fields (default includes lit_pubmed_id for refs).",
    )
    parser.add_argument(
        "--all-fields",
        action="store_true",
        help="Merge all configure/uniprotkb/result-fields tokens (extra batched GETs).",
    )
    parser.add_argument("--plot-html", type=Path, default=None, help="Write interactive Plotly HTML dashboard.")
    parser.add_argument(
        "--merge-input-columns",
        action="store_true",
        help="Prefix extra CSV columns as input_* on summary rows (first row per protein_id).",
    )

    args = parser.parse_args()

    ordered_accs, acc_to_pid = collect_ordered_pairs(args.input_csv, args.protein_id_column)
    if not ordered_accs:
        raise SystemExit("No accessions found in CSV.")

    out_dir = args.output_dir if args.output_dir else args.input_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    fields_csv = args.fields.replace(" ", "")
    log_notes: Dict[str, Any] = {
        "input_csv": str(args.input_csv.resolve()),
        "ordered_unique_accessions": len(ordered_accs),
        "chunk_size": args.chunk_size,
        "fields": fields_csv,
        "all_fields": args.all_fields,
    }

    merged_by_acc: Dict[str, Dict[str, Any]] = {}

    for batch in chunk(ordered_accs, args.chunk_size):
        batch_results, missing_accs = fetch_accessions_batch(batch, fields_csv)
        log_notes.setdefault("missing_batches", [])
        if missing_accs:
            log_notes["missing_batches"].append({"batch": list(batch), "missing": missing_accs})

        for r in batch_results:
            acc = r.get("primaryAccession")
            if acc:
                merged_by_acc[acc] = dict(r)

    if args.all_fields:
        all_names = load_configure_field_names()
        batches = split_field_batches(all_names)
        log_notes["all_fields_batches"] = len(batches)
        for batch in chunk(ordered_accs, args.chunk_size):
            merge_extra_fields_rows(merged_by_acc, batch, batches)

    extras_map = None
    if args.merge_input_columns:
        extras_map = pipeline_extra_columns(
            args.input_csv,
            args.protein_id_column,
            acc_to_pid.values(),
        )

    summary_rows = flatten_summary_rows(
        merged_by_acc,
        ordered_accs,
        acc_to_pid,
        extras_map,
    )

    summary_path = out_dir / "uniprot_summary.tsv"
    write_summary_tsv(summary_path, summary_rows)

    log_notes["summary_path"] = str(summary_path.resolve())
    log_notes["rows_written"] = len(summary_rows)

    log_path = out_dir / "uniprot_fetch_log.json"
    log_path.write_text(json.dumps(log_notes, indent=2), encoding="utf-8")

    if args.plot_html:
        html = build_dashboard_html(
            summary_rows,
            report_title=args.input_csv.name,
            include_plotlyjs_cdn=True,
        )
        args.plot_html.parent.mkdir(parents=True, exist_ok=True)
        args.plot_html.write_text(html, encoding="utf-8")

    print(f"Wrote {summary_path}")
    print(f"Wrote {log_path}")
    if args.plot_html:
        print(f"Wrote {args.plot_html}")


if __name__ == "__main__":
    main()
