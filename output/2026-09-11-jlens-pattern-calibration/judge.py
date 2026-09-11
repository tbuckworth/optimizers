"""Bounded entry point for the reviewed response seal/grade; no reader calls."""
import argparse
import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = Path('/tmp/spectral-experiment-artifacts/j-lens-pilot-20260909-MMxZIs/worktree/output/2026-09-11-j-lens-pattern-calibration/responses.py')
SOURCE_SHA = '55d659a991965a6d572101a0ec513978ec7b0534113e9a6e9438ffe6172d34b5'
RELEASE_SHA = '5565779ea44b67240c5254219d13e2486c3de840ac30ad9a9ffdc1e6f77160aa'
RELEASE_COMMIT = '866ab6f30feecc6e71ef5fc53ef19491d8a5247a'


def main(args):
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError('reviewed source changed')
    spec = importlib.util.spec_from_file_location('fixed_judging', SOURCE)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    kwargs = dict(release_path=HERE/'judging-release.json', release_sha=RELEASE_SHA, release_commit=RELEASE_COMMIT)
    if args.stage == 'seal':
        if args.raw_commit is None: raise ValueError('raw first-response commit required')
        paths = {}
        for rater in module.RATERS:
            path = HERE/'raw-responses'/(rater+'.json'); raw = path.read_bytes()
            paths[rater] = {'path': str(path), 'size_bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        result = module.seal(**kwargs, raw_commit=args.raw_commit, raw_paths=paths, target=HERE/'responses')
    else:
        if not args.lock_sha or not args.lock_commit: raise ValueError('exact committed lock required')
        result = module.grade(**kwargs, responses=HERE/'responses', lock_sha=args.lock_sha,
                              lock_commit=args.lock_commit, target=HERE/'graded')
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['seal', 'grade'])
    for flag in ('raw-commit', 'lock-sha', 'lock-commit'): parser.add_argument('--'+flag)
    main(parser.parse_args())
