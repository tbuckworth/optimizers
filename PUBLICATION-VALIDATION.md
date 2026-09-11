# Public snapshot validation

Validated on Linux with Python 3.12. This is software/publication validation,
not another training run or independent replication of the scientific results.

## Commands

The project tests use Python's standard-library `unittest` runner. The local
environment includes PyTorch, NumPy, SciPy and Matplotlib; the report builder
also uses Python-Markdown. No model, dataset or credential download is needed
for these fixture tests. Some fixtures exercise Linux resource/path guards.

```bash
mkdir -p /tmp/spectral-experiment-artifacts
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  python3 -m unittest discover -s tests -q
python3 -m compileall -q experiments tests scripts output
python3 scripts/lint_knowledge.py
python3 scripts/check_public_snapshot.py
```

Results: **584 tests run, no failures or errors, one skipped**. The skipped
`test_actual_source_pins_once_companion_files_exist` requires the omitted
general-augmentation private preview JSON and three preview images. Its mocked
source-inventory checks and tamper-rejection tests pass. This omission is not
a passing archived-acquisition check. Do not disable the acquisition guards
or reconstruct missing approvals from this snapshot.

Compile checks and knowledge lint pass. The original core optimizer is
byte-identical to the public base. Source pins affected by path redaction are
rebound to the public source versions; original experiment audit hashes remain
historical, as explained in [publication scope](PUBLICATION.md).

The 584 exported scientific Python syntax trees were compared with their original sources,
ignoring string contents affected by redaction and source-pin rebinding. JSON
numeric structures in 125 JSON files were checked against originals after omitting private
metadata fields. This protects scientific arithmetic/results from accidental
redaction changes; it is not a proof of scientific correctness. All 17,800
numeric cells across the five exported CSV tables were checked against the
original text values. This caught and corrected 12 decimal-string redaction
errors before publication. CSV line endings and reviewed trailing whitespace
were normalized in the public derivatives.

## Report

```bash
python3 scripts/build_public_report.py
```

This builds a self-contained HTML reader with the two scientific plots embedded.
The included seven-page PDF was printed from that HTML with Chromium's headers
and footers disabled. Source hyperlinks target the public GitHub repository,
not private local paths. The first page was visually checked, and PNG metadata
and PDF links were checked for private paths. Metadata in all 126 exported PNGs
was checked; no private paths were found.

## Disclosure checks

Gitleaks scanned the public working directory with full redaction enabled and
a 100 MB per-file ceiling: **no secrets found**. Separate new-file checks cover
private machine paths, correspondence identifiers and conflict markers. Private
history is absent: the publication branch starts directly from public `main`.
The original local branches and archival worktrees are preserved.
