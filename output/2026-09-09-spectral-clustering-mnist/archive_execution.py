"""Archive this job's generated journal and terminal receipts without replay."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
SOURCE = Path('/tmp/spectral-experiment-artifacts/spectral-clustering-mnist-20260909.51ggnu/acquisition-001')
UNIT = 'spectral-clustering-mnist-001.service'


def main():
    target = HERE/'execution'
    if target.exists():
        raise FileExistsError('Execution already archived')
    props = subprocess.check_output(['systemctl', '--user', 'show', UNIT,
        '-p', 'MainPID', '-p', 'ActiveState', '-p', 'Result', '-p', 'InvocationID', '-p', 'LoadState',
        '-p', 'ExecMainStartTimestamp', '-p', 'ExecMainExitTimestamp',
        '-p', 'MemoryMax', '-p', 'MemorySwapMax', '-p', 'RuntimeMaxUSec'], text=True)
    state = dict(x.split('=', 1) for x in props.splitlines())
    assert state['MainPID'] == '0' and state['ActiveState'] in ('inactive', 'failed')
    invocation = 'b86c6fb904cd460d9f1b8d757310780d'
    assert (state['InvocationID'] == invocation or
            (state['LoadState'] == 'not-found' and state['InvocationID'] == ''))
    journal = subprocess.check_output(['journalctl', '--user', '-u', UNIT, '--no-pager', '-o', 'cat'])
    assert len(journal) < 5*1024**2
    guard_rows = [json.loads(line) for line in journal.decode().splitlines()
                  if line.startswith('{"resource_guard":')]
    assert len(guard_rows) == 1 and guard_rows[0]['invocation_id'] == invocation
    state['verified_original_invocation'] = invocation
    state['terminal_note'] = ('Transient unit was garbage-collected; use saved completion and archived journal '
                              'for outcome, not default systemctl properties.' if state['LoadState'] == 'not-found'
                              else 'Transient service properties still available.')
    target.mkdir()
    with (target/'run.log').open('xb') as handle:
        handle.write(journal)
    receipts = {}
    for name in ('manifest.json', 'results.json', 'complete.json', 'failed.json'):
        path = SOURCE/name
        if path.exists():
            data = path.read_bytes()
            with (target/name).open('xb') as handle:
                handle.write(data)
            receipts[name] = {'source': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}
    audit_path = SOURCE.parent/'audit-001/result.json'
    if audit_path.exists():
        payload = audit_path.read_bytes()
        audited = json.loads(payload)
        assert audited['status'] == 'PASS'
        assert audited['completion_sha256'] == receipts['complete.json']['sha256']
        summary = {'audit_path': str(audit_path), 'audit_sha256': hashlib.sha256(payload).hexdigest(),
                   'completion_sha256': audited['completion_sha256'], 'summary': audited['summary'],
                   'curves': audited['curves'], 'limitations': audited['limitations']}
        with (target/'checked-summary.json').open('x') as handle:
            json.dump(summary, handle, indent=2, allow_nan=False)
            handle.write('\n')
    with (target/'service.json').open('x') as handle:
        json.dump({'service': state, 'source_receipts': receipts,
                   'journal_sha256': hashlib.sha256(journal).hexdigest()}, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'archive': str(target), 'state': state, 'receipts': receipts}))


if __name__ == '__main__':
    main()
