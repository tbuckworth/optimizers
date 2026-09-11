"""Independent table/known-metric readback, not a logit reread or resource certificate."""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    verification = json.loads((HERE/'input-verification.json').read_text())
    assert verification['analysis_source_sha256'] == digest(HERE/'analyze.py')
    assert verification['design_sha256'] == digest(HERE/'design.md')
    assert verification['status'] == 'ALL_INPUT_HASHES_PASS_BEFORE_ARRAY_READ'
    assert len(verification['verified_inputs']) == 39
    old_path = ROOT/'output/2026-09-09-spectral-selectivity-boundary/results/checked-summary.json'
    assert digest(old_path) == '8af7caa3932bdb7a6508e5375e00924d12b279bb3bbf8b1813deb0515b7e29b9'
    old = json.loads(old_path.read_text())
    result = json.loads((HERE/'metrics.json').read_text())
    seeds = (202609111, 202609112, 202609113)
    cells = ('clean', 'diffuse', 'shared', 'sham')
    policies = ('raw', 'native32', 'norm_raw')
    rows = {(r['seed'], r['cell'], r['policy'], r['step'], r['view']): r['metrics'] for r in result['rows']}
    expected = {(s,c,p,t,v) for s in seeds for c in cells for p in policies
                for t in (100,2000) for v in ('original','plus_log11')}
    assert len(result['rows']) == len(rows) == 144 and set(rows) == expected
    checks = 0

    def close(actual, expected):
        nonlocal checks
        assert math.isfinite(actual) and math.isfinite(expected)
        assert abs(actual-expected) <= 1e-11, (actual,expected)
        checks += 1

    mapping = {'rare_accuracy':'rare_accuracy','rare_ce':'rare_ce',
               'common_accuracy':'majority_macro_accuracy','common_ce':'majority_macro_ce',
               'balanced_accuracy':'balanced_total_accuracy','balanced_ce':'balanced_total_ce'}
    for s,c,p,t,v in sorted(rows):
        metrics = rows[s,c,p,t,v]
        assert all(math.isfinite(x) for x in metrics.values())
        assert 0 <= metrics['rare_auroc'] <= 1
        if v == 'original':
            for new_key, old_key in mapping.items():
                wanted = old['per_group'][c+'/'+p][old_key]['values'][seeds.index(s)]
                if t == 100:
                    wanted -= old['change_from_warmup'][c+'/'+p][old_key]['values'][seeds.index(s)]
                close(metrics[new_key], wanted)
            adjusted = rows[s,c,p,t,'plus_log11']
            close(metrics['rare_auroc'], adjusted['rare_auroc'])
            assert adjusted['rare_accuracy'] >= metrics['rare_accuracy']
            assert adjusted['common_accuracy'] <= metrics['common_accuracy']
            assert adjusted['rare_ce'] <= metrics['rare_ce']
            assert adjusted['common_ce'] >= metrics['common_ce']
            if t == 100:
                for key in metrics:
                    close(metrics[key], rows[s,'clean','raw',100,'original'][key])
    for group in result['summaries']:
        for key, summary in group['metrics'].items():
            values = [rows[s,group['cell'],group['policy'],group['step'],group['view']][key] for s in seeds]
            assert summary['seed_values'] == values and summary['n_seeds'] == 3
            mean = math.fsum(values)/3
            close(summary['mean'],mean)
            close(summary['sample_sd'], math.sqrt(math.fsum((v-mean)**2 for v in values)/2))
    for group in result['endpoint_minus_warmup']:
        for key, summary in group['metrics'].items():
            values = [rows[s,group['cell'],group['policy'],2000,group['view']][key]
                      -rows[s,group['cell'],group['policy'],100,group['view']][key] for s in seeds]
            assert summary['seed_values'] == values
            close(summary['mean'], math.fsum(values)/3)
    for group in result['native_minus_controls']:
        control = group['contrast'].removeprefix('native32_minus_')
        for key, summary in group['metrics'].items():
            values = [rows[s,group['cell'],'native32',2000,group['view']][key]
                      -rows[s,group['cell'],control,2000,group['view']][key] for s in seeds]
            assert summary['seed_values'] == values
            close(summary['mean'], math.fsum(values)/3)
    receipt = {'status':'TABLE_READBACK_PASS', 'numeric_comparisons':checks,
               'original_known_metric_comparisons':432, 'logical_rows':144,
               'metrics_sha256':digest(HERE/'metrics.json'), 'source_sha256':digest(Path(__file__)),
               'input_verification_sha256':digest(HERE/'input-verification.json'),
               'scope':'Source/design pins, old checked metrics, table arithmetic and fixed-shift monotonicity; no logits/models reread.',
               'not_certified':'Original analysis terminal resource check; new AUROC is source/fixture-reviewed, not independently remeasured.'}
    with (HERE/'main-readback.json').open('x') as handle:
        json.dump(receipt,handle,indent=2)
        handle.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
