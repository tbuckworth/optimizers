"""Saved-output arithmetic audit, only after all six reader responses are committed."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
WORK = Path('/tmp/spectral-experiment-artifacts/j-lens-pilot-20260909-MMxZIs/worktree/output')
NEW = WORK/'2026-09-11-j-lens-independent-content'
OLD = WORK/'2026-09-10-j-lens-fresh-content'
PINS = {
    OLD/'preparation/directions.npz': '47dc955bafde649a86cdd44e2975637dbd24aac2443fdfca83214540149030fa',
    NEW/'forwards/features.npz': 'd6ed700ce1b4f74001097880b5ddc0b41481a31c138395156dfda55544c434b1',
    NEW/'forwards/scores.json': 'e03efc7d22d480b2b7f12db89021e20653db2898a7317b63b6e2b6e9a3e5e704',
    NEW/'forwards/gaps.json': 'b2314550bc97cb867dff391f23ccddd4f8776bc25774cd3ce97953d76d1e6b20',
    NEW/'forwards/inputs.json': '6fb66b6e32cdafa46313a87b4e198b2ff8ca3c0e22c0fd01f8f619b5785cabfe',
    NEW/'token-preflight/tokens.json': 'f0f7e6d099d0ec2b9fe5a941980bbe0490cfba8deb6117f17ae538e5fe93c72c',  # gitleaks:allow -- verified saved input-ID file SHA256, not a credential
    HERE/'excerpts-resumed/dataset.json': '7391a1fa6c02872cdeabb2e8dffc1e63181b6379d80e2042f33b38a61308ff02',
    HERE/'excerpts-resumed/pairs.json': 'b38badeebcfe3c54a4315bd60dbaf8b5ace85975a8554e0074c2bd286d301ecc',
}


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    with path.open('rb') as handle: return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_json(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 1024**2, 'JSON file guard')
    return json.loads(path.read_text())  # Exact trusted input digests are checked before parsing.


def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def committed_readers(commit, digest):
    require(len(commit) == 40 and all(c in '0123456789abcdef' for c in commit), 'immutable commit required')
    lock_path = HERE/'responses/lock.json'
    require(sha(lock_path) == digest, 'lock hash mismatch')
    lock = read_json(lock_path)
    require(lock['schema'] == 'jlens_independent_content_response_lock_v1' and lock['count'] == 288
            and set(lock['responses']) == {f'rater{i}' for i in range(1, 7)}, 'complete six-reader lock required')
    root = HERE.parents[1]
    files = [lock_path]
    for i in range(1, 7):
        record = lock['responses'][f'rater{i}']
        require(record['path'] == f'rater{i}.json', 'response path')
        path = HERE/'responses'/record['path']
        require(sha(path) == record['sha256'] and path.stat().st_size == record['size_bytes'], 'response hash/size')
        require(len(read_json(path)['responses']) == 48, '48 responses required')
        files.append(path)
    for path in files:
        stored = subprocess.run(['git', '-C', str(root), 'cat-file', 'blob', f'{commit}:{path.relative_to(root)}'],
                                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10).stdout
        require(stored == path.read_bytes(), 'response/lock not committed before audit')


def arithmetic(h, u, mean, scores, gaps):
    import numpy as np
    for value, shape, dtype in [(h, (24, 1024), np.float32), (u, (1024, 4), np.float32),
                                (mean, (1024,), np.float64), (scores, (24, 4), np.float64), (gaps, (12, 4), np.float64)]:
        require(value.shape == shape and value.dtype == dtype and np.isfinite(value).all(), 'array contract')
    errors = []
    high_accuracy = [[math.fsum((float(h[i, j])-float(mean[j]))*float(u[j, a]) for j in range(1024))
                      for a in range(4)] for i in range(24)]
    for i in range(24):
        for a in range(4):
            expected = high_accuracy[i][a]
            require(math.isclose(expected, float(scores[i, a]), rel_tol=2e-13, abs_tol=2e-12), 'scalar projection mismatch')
            errors.append(abs(expected-float(scores[i, a])))
    sign = lambda x: (x > 0)-(x < 0)
    for i in range(12):
        for a in range(4):
            stored_gap = float(gaps[i, a])
            require(stored_gap.hex() == (float(scores[2*i, a])-float(scores[2*i+1, a])).hex(), 'exact stored gap mismatch')
            require(sign(stored_gap) == sign(high_accuracy[2*i][a]-high_accuracy[2*i+1][a]), 'projection summation changes ordering')
    return {'scalar_fsum_checks': 96, 'exact_gap_checks': 48, 'ordering_sign_checks': 48,
            'absolute_tolerance': 2e-12, 'relative_tolerance': 2e-13, 'maximum_absolute_difference': max(errors)}


def run(commit, lock_sha):
    target = HERE/'measurement-audit'
    target.mkdir(exist_ok=False)
    source_sha = sha(Path(__file__))
    write(target/'attempt.json', {'source_sha256': source_sha, 'lock_commit': commit, 'lock_sha256': lock_sha})
    try:
        committed_readers(commit, lock_sha)  # No scientific input access before this succeeds.
        for path, digest in PINS.items(): require(sha(path) == digest, 'saved input changed')
        import numpy as np
        with np.load(OLD/'preparation/directions.npz', allow_pickle=False) as archive:
            u, mean = archive['u32'], archive['source_mean64']
        with np.load(NEW/'forwards/features.npz', allow_pickle=False) as archive:
            h, scores, gaps = archive['activation_11'], archive['scores64'], archive['gaps64']
        checks = arithmetic(h, u, mean, scores, gaps)
        data = read_json(HERE/'excerpts-resumed/dataset.json')
        pairs = read_json(HERE/'excerpts-resumed/pairs.json')
        before = read_json(NEW/'token-preflight/tokens.json')['records']
        after = read_json(NEW/'forwards/inputs.json')['records']
        require(before == after and len(after) == 24, 'exact frozen model inputs')
        require([{k: row[k] for k in ['id', 'topic', 'prefix']} for row in after] == data, 'source text identity')
        export = read_json(NEW/'forwards/scores.json')
        require(export['schema'] == 'jlens_independent_content_scores_v1' and len(export['scores']) == 4, 'score export schema')
        for a, row in enumerate(export['scores']):
            require(row['axis'] == f'PC{a+1}' and set(row['values']) == {r['id'] for r in data}, 'score export roster')
            for i, source in enumerate(data):
                require(float(row['values'][source['id']]).hex() == float(scores[i, a]).hex(), 'exact JSON scalar mismatch')
        exported_gaps = read_json(NEW/'forwards/gaps.json')
        require(exported_gaps['axes'] == ['PC1', 'PC2', 'PC3', 'PC4'] and exported_gaps['orientation'] == 'left minus right', 'gap axes/orientation')
        require(len(exported_gaps['pairs']) == 12, 'gap row count')
        for i, row in enumerate(exported_gaps['pairs']):
            require({k: row[k] for k in ['id', 'left', 'right']} == pairs[i] and len(row['gaps64']) == 4, 'gap pairing')
            for a, value in enumerate(row['gaps64']): require(float(value).hex() == float(gaps[i, a]).hex(), 'exact JSON gap mismatch')
        require(sha(Path(__file__)) == source_sha, 'audit source changed')
        for path, digest in PINS.items(): require(sha(path) == digest, 'saved input changed during audit')
        write(target/'receipt.json', {'status': 'PASS', 'source_sha256': source_sha,
              'lock_commit': commit, 'lock_sha256': lock_sha, **checks,
              'exact_json_scalar_checks': 96, 'exact_json_gap_checks': 48, 'exact_input_rows': 24,
              'inputs': [{'path': str(p), 'sha256': s} for p, s in PINS.items()],
              'scope': 'Saved arithmetic/serialization/input identity, not a second model acquisition or independent causal/semantic verification.'})
    except Exception as error:
        write(target/'failure.json', {'status': 'FAILED', 'error_type': type(error).__name__, 'reason': str(error)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock-commit', required=True)
    parser.add_argument('--lock-sha', required=True)
    args = parser.parse_args()
    run(args.lock_commit, args.lock_sha)
