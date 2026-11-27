"""Create a concise, evidence-based Member 4 final development summary."""

from pathlib import Path
import pandas as pd
import json
import hashlib

BASE = Path("reports/member4")
AUDIO = Path("reports/member4_audio")
PANN = Path("reports/member4_panns")
out = BASE / "development_summary.md"
lines = [
    "# Member 4 — transfer learning development summary",
    "",
    "Protocol: 7,000 training clips, 816 validation clips, 837 final-test clips reserved. All scores below are validation scores.",
    "",
    "## ImageNet-only comparison",
]
for root, label in [(BASE, "ImageNet"), (AUDIO, "AudioSet EfficientAT")]:
    f = root / "model_comparison.csv"
    if not f.exists():
        lines.append(f"{label}: no completed comparison.")
        continue
    d = pd.read_csv(f)
    if root == AUDIO:
        lines.extend(
            [
                "",
                "## AudioSet-pretrained CNN extension",
                "Additional pretraining data and different frontend; not a controlled architecture-only comparison.",
            ]
        )
    top = d.iloc[0]
    fixed = d[d.seed == 42].iloc[0]
    lines.extend(
        [
            f"- Completed runs: {len(d)}",
            f"- Best individual validation run: {top['name']} — {top.accuracy:.2f}%; macro F1 {top.macro_f1:.2f}%.",
            f"- Best seed-42 run: {fixed['name']} — {fixed.accuracy:.2f}% accuracy.",
            "- Custom CNN validation reference: 86.89% (single seed).",
        ]
    )
    f = root / "seed_statistics.csv"
    if f.exists():
        s = pd.read_csv(f, index_col=0)
        n = len(pd.read_csv(root / "seed_comparison.csv"))
        lines.append(
            f"- Repeated recipe ({n} seeds): accuracy {s.loc['mean', 'accuracy']:.2f}% ± {s.loc['std', 'accuracy']:.2f}% (sample SD)."
        )
    f = root / "validation_analysis.csv"
    if f.exists():
        for r in pd.read_csv(f).itertuples():
            lines.append(
                f"- Exploratory validation analysis: {r.name} — accuracy {r.accuracy:.2f}%, macro F1 {r.macro_f1:.2f}%."
            )
lines.extend(["", "## PANNs independent frozen-embedding challenger"])
f = PANN / "summary.json"
if f.exists():
    s = json.loads(f.read_text())
    lines.append(
        f"- Best classifier: {s['best_recipe']}, validation accuracy {s['accuracy']:.2f}%, macro F1 {s['macro_f1']:.2f}%."
    )
else:
    lines.append("Not completed; inspect local job log.")
lines.extend(
    [
        "",
        "## Limitations and release gates",
        "- 95% is an objective, not a result unless supported by the recorded metrics.",
        "- All model selection uses a single validation fold; repeated seeds do not remove validation-selection optimism.",
        "- AudioSet overlap with target recordings cannot be fully ruled out.",
        "- Final test evaluation and GitHub release require team approval.",
        "- No final test result is claimed.",
    ]
)
out.write_text("\n".join(lines) + "\n")
weights = Path("data/upstream/weights")
records = []
for f in sorted(weights.glob("*")):
    if f.is_file() and f.suffix in (".pt", ".pth"):
        with f.open("rb") as h:
            records.append(
                dict(
                    name=f.name,
                    size_bytes=f.stat().st_size,
                    sha256=hashlib.file_digest(h, "sha256").hexdigest(),
                )
            )
(AUDIO / "weight_manifest.json").write_text(json.dumps(records, indent=2))
print(out.read_text())
