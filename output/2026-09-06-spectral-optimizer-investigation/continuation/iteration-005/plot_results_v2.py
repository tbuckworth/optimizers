"""Presentation-only revision: shorten one crowded axis label; preserve draft."""
import hashlib
import json
from pathlib import Path
from plot_results import HERE, SUMMARY_SHA, render, plt


def main():
    summary = HERE / "summary.json"
    assert hashlib.sha256(summary.read_bytes()).hexdigest() == SUMMARY_SHA
    output, manifest = HERE / "common-stream-diagnostics-v2.png", HERE / "figure-manifest-v2.json"
    if output.exists() or manifest.exists():
        raise SystemExit("Refusing to overwrite figure evidence")
    fig = render(json.loads(summary.read_text()))
    fig.axes[0].set_ylabel("Missed optimal rank-32 energy (%)")
    fig.tight_layout(rect=(0, .09, 1, .955))
    fig.savefig(output, dpi=160)
    plt.close(fig)
    record = {"summary_sha256": SUMMARY_SHA, "source_sha256": {
        name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        for name in ("plot_results.py", "plot_results_v2.py")},
        "output": output.name, "size_bytes": output.stat().st_size,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "revision": "Only a shorter axis label. Same data/transforms/panels; original figure and manifest preserved."}
    with manifest.open("x") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
