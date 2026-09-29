"""Combine the explained-variance curves of several latent evaluations into one plot.

The pipeline never persists the cumulative explained-variance curve as data - it
only writes a PNG and embeds the plotly figure in the run's HTML report. That
embedded figure still carries the full curve, so this script recovers the curves
by parsing the reports under latent_evaluation/output/ and overlays them.

Invoke from the repo root, e.g.:
    python main/combine_explained_variance.py --list
    python main/combine_explained_variance.py --dataset context ae-dim256 ae-dim1024

PNG export needs kaleido's Chrome shim; if it fails, see the note in
latent_evaluation/README or run with
    LD_LIBRARY_PATH=/home/fwichert/miniconda3/envs/diff-train/lib
The interactive HTML is always written regardless.
"""
import argparse
import base64
import fnmatch
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = REPO_ROOT / "latent_evaluation" / "output"

# Reference categorical palette, light surface, in its fixed order. Assigned by
# slot and never re-ordered; past slot 8 the hue repeats with a different dash,
# so identity stays a colour+texture pair rather than a recycled hue.
SERIES_COLORS = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]
SERIES_DASHES = ["solid", "dash", "dot", "dashdot"]
SURFACE = "#ffffff"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"

TITLE_RE = re.compile(r"Explained Variance \((?P<dataset>[^)]+)\) - (?P<encoder>.+)")
NEWPLOT_RE = re.compile(r'Plotly\.newPlot\(\s*"[^"]+",\s*(?=\[)')
CURVE_TRACE_NAME = "Cumulative explained variance"


def _scan_json(text, start):
    """Return the index just past the JSON value that begins at `start`."""
    depth = 0
    in_string = False
    escaped = False
    i = start
    while i < len(text):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unterminated JSON value in report")


def _decode_array(value):
    # Recent plotly versions serialise trace arrays as base64 typed arrays.
    if isinstance(value, dict) and "bdata" in value:
        return np.frombuffer(base64.b64decode(value["bdata"]), dtype=value.get("dtype", "f8"))
    return np.asarray(value)


def _iter_figures(html):
    """Yield (traces, layout) for every plotly figure embedded in a report."""
    for match in NEWPLOT_RE.finditer(html):
        data_start = match.end()
        data_end = _scan_json(html, data_start)
        traces = json.loads(html[data_start:data_end])

        layout_match = re.compile(r"\s*,\s*(?=\{)").match(html, data_end)
        if layout_match is None:
            continue
        layout_start = layout_match.end()
        layout = json.loads(html[layout_start:_scan_json(html, layout_start)])
        yield traces, layout


def _layout_title(layout):
    title = layout.get("title")
    if isinstance(title, dict):
        return title.get("text")
    return title


def collect_curves(report_path):
    """Extract every explained-variance curve from one report."""
    html = report_path.read_text(encoding="utf-8", errors="replace")
    if "Explained Variance (" not in html:
        return []

    curves = []
    for traces, layout in _iter_figures(html):
        title_match = TITLE_RE.fullmatch(str(_layout_title(layout) or ""))
        if title_match is None:
            continue
        for trace in traces:
            if trace.get("name") != CURVE_TRACE_NAME:
                continue
            x = _decode_array(trace.get("x"))
            y = _decode_array(trace.get("y"))
            if len(x) == 0:
                continue
            curves.append({
                "dataset": title_match.group("dataset"),
                "encoder": title_match.group("encoder"),
                "run": report_path.parent.name,
                "date": report_path.parent.name[:10],
                "dims": x,
                "cumulative_pct": y,
                "embedding_dim": int(x[-1]),
            })
    return curves


def _run_matches(run, pattern):
    return fnmatch.fnmatch(run, pattern) or pattern in run


def discover_curves(dataset, patterns):
    """All curves for `dataset`, restricted to runs matching any of `patterns`."""
    curves = []
    for report_path in sorted(OUTPUT_ROOT.glob("*/*.html")):
        for curve in collect_curves(report_path):
            if curve["dataset"] != dataset:
                continue
            if patterns and not any(_run_matches(curve["run"], p) for p in patterns):
                continue
            curves.append(curve)
    return curves


def _shared_prefix(names):
    """The longest prefix common to every name, ending on a separator boundary."""
    names = list(names)
    if len(names) < 2:
        return ""
    prefix = os.path.commonprefix(names)
    # The common prefix can stop mid-token. Extend it over a separator that every
    # longer name agrees on; failing that, cut back to the last boundary inside it.
    tails = {name[len(prefix):len(prefix) + 1] for name in names if len(name) > len(prefix)}
    if len(tails) == 1 and tails <= {"-", "_"}:
        return prefix + tails.pop()
    cut = max(prefix.rfind("-"), prefix.rfind("_"))
    return prefix[:cut + 1] if cut >= 0 else ""


def label_curves(curves):
    """Name each curve by its encoder, disambiguating repeats with the run date.

    Every arm shares a long encoder prefix (`tfree-hat-finetuned-ae-...`), which
    would otherwise take up most of the legend's width; it is dropped, except
    from a name that consists of nothing else.
    """
    prefix = _shared_prefix({curve["encoder"] for curve in curves})
    counts = {}
    for curve in curves:
        counts[curve["encoder"]] = counts.get(curve["encoder"], 0) + 1
    for curve in curves:
        name = curve["encoder"].removeprefix(prefix) or curve["encoder"]
        curve["label"] = f"{name} ({curve['date']})" if counts[curve["encoder"]] > 1 else name
    return curves


def build_figure(curves, dataset):
    fig = go.Figure()
    for index, curve in enumerate(curves):
        fig.add_trace(go.Scatter(
            x=curve["dims"],
            y=curve["cumulative_pct"],
            mode="lines",
            name=curve["label"],
            line=dict(
                width=2,
                color=SERIES_COLORS[index % len(SERIES_COLORS)],
                dash=SERIES_DASHES[(index // len(SERIES_COLORS)) % len(SERIES_DASHES)],
            ),
            hovertemplate="%{y:.2f}% at %{x} dims<extra>%{fullData.name}</extra>",
        ))

    max_dim = max(curve["embedding_dim"] for curve in curves)
    tick_dims = [1]
    while tick_dims[-1] * 2 <= max_dim:
        tick_dims.append(tick_dims[-1] * 2)

    fig.update_layout(
        title=f"Cumulative Explained Variance - {dataset}",
        template="plotly_white",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(color=TEXT_PRIMARY),
        hovermode="x unified",
        legend=dict(font=dict(color=TEXT_SECONDARY)),
        width=1100,
        height=700,
    )
    fig.update_xaxes(
        title_text="Number of dimensions", type="log",
        tickvals=tick_dims, exponentformat="none",
    )
    fig.update_yaxes(
        title_text="Cumulative explained variance (%)", range=[0, 105],
        dtick=10, exponentformat="none",
    )
    return fig


def parse_args():
    p = argparse.ArgumentParser(
        description="Overlay the explained-variance curves of several latent evaluations.",
    )
    p.add_argument("runs", nargs="*",
                   help="Run filters matched against the report directory name, as a "
                        "substring or a glob (e.g. ae-dim256 '*v1-phase2*'). "
                        "Default: every run that has a curve.")
    p.add_argument("-d", "--dataset", default="synonyms", choices=["synonyms", "context"],
                   help="Which evaluation population to plot; the two are not "
                        "comparable point-for-point, so one figure holds one. "
                        "Default: synonyms.")
    p.add_argument("-o", "--output-dir", default=None,
                   help="Where to write the figure. "
                        "Default: latent_evaluation/output/<today>_combined/.")
    p.add_argument("--name", default=None,
                   help="Output basename. Default: explained_variance_<dataset>.")
    p.add_argument("-l", "--list", action="store_true",
                   help="List the available curves and exit without plotting.")
    p.add_argument("--label", action="append", default=[], metavar="FILTER=NAME",
                   help="Legend name for the runs matching FILTER (same matching as the "
                        "run filters), e.g. --label ae-dim4096-v1-phase1=t-free-hat. "
                        "Repeatable.")
    return p.parse_args()


def main():
    args = parse_args()

    curves = discover_curves(args.dataset, args.runs)
    if not curves:
        print(f"No '{args.dataset}' explained-variance curves found"
              + (f" for {args.runs}." if args.runs else " under latent_evaluation/output/."))
        return 1

    # Ascending embedding dim puts the curves in the order the legend reads best.
    curves.sort(key=lambda c: (c["embedding_dim"], c["run"]))
    label_curves(curves)
    for override in args.label:
        pattern, sep, name = override.partition("=")
        if not sep:
            print(f"--label expects FILTER=NAME, got '{override}'")
            return 1
        matched = [curve for curve in curves if _run_matches(curve["run"], pattern)]
        if not matched:
            print(f"WARNING: --label '{pattern}' matches no plotted run.")
        for curve in matched:
            curve["label"] = name

    if args.list:
        for curve in curves:
            print(f"{curve['run']}\n    dataset={curve['dataset']} "
                  f"dim={curve['embedding_dim']} label={curve['label']}")
        return 0

    output_dir = Path(args.output_dir) if args.output_dir else (
        OUTPUT_ROOT / f"{datetime.today().strftime('%Y-%m-%d')}_combined"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = args.name or f"explained_variance_{args.dataset}"

    fig = build_figure(curves, args.dataset)

    html_path = output_dir / f"{basename}.html"
    fig.write_html(html_path)
    print(f"Combined {len(curves)} curves ({args.dataset}):")
    for curve in curves:
        print(f"  {curve['label']}  [dim {curve['embedding_dim']}, {curve['run']}]")
    print(f"Wrote {html_path}")

    png_path = output_dir / f"{basename}.png"
    try:
        fig.write_image(png_path)
        print(f"Wrote {png_path}")
    except Exception as exc:
        print(f"WARNING: PNG export failed ({exc}); the HTML above is complete. "
              f"Retry with LD_LIBRARY_PATH set to the diff-train env lib.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
