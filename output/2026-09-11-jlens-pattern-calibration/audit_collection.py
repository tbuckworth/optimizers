"""Independent reconstruction from saved article responses; no network/model imports."""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
WORK = Path('/tmp/spectral-experiment-artifacts/j-lens-pilot-20260909-MMxZIs/worktree/output/2026-09-11-j-lens-pattern-calibration')
DATA = WORK/'excerpts'
RECEIPT = 'cf86bce1a96099939434fca3c391568db3d802786aed05f04d66f6ad76d82776'
MANIFEST = '8287106e8e3238ccf18d0c60bc0a4b59de894c7a8fd038c6a870a2cd1e94ed63'


def check(ok, label):
    if not ok:
        raise ValueError(label)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def eligibility(page, pid):
    # Reconstructed directly from the prospective technical rule, not imported.
    check(page['pageid'] == pid, 'returned ID')
    for marker in ['missing', 'invalid', 'redirect']:
        if marker in page:
            return None, marker
    if type(page.get('ns')) is not int or page['ns'] != 0:
        return None, 'non-main namespace'
    if 'disambiguation' in page.get('pageprops', {}):
        return None, 'disambiguation'
    check(isinstance(page.get('title'), str) and page['title'].strip(), 'title')
    url = urlsplit(page['fullurl'])
    check(url.scheme == 'https' and url.netloc == 'en.wikipedia.org', 'article URL')
    revisions = page['revisions']
    check(len(revisions) == 1 and type(revisions[0]['revid']) is int and revisions[0]['revid'] > 0
          and isinstance(revisions[0]['timestamp'], str) and revisions[0]['timestamp'], 'revision')
    extract = page.get('extract')
    if not isinstance(extract, str) or len(extract.split()) < 16:
        return None, 'short/missing extract'
    return ' '.join(extract.split()[:16]), None


def run():
    output = HERE/'collection-audit.json'
    check(not output.exists(), 'audit already recorded')
    check(sha(DATA/'receipt.json') == RECEIPT and sha(HERE/'selection/manifest.json') == MANIFEST, 'frozen bindings')
    receipt, manifest = read(DATA/'receipt.json'), read(HERE/'selection/manifest.json')
    check(receipt['status'] == 'complete' and receipt['manifest_sha256'] == MANIFEST, 'receipt status')
    check(receipt['source_sha256'] == sha(WORK/'collect.py') == '4001f31ab9d592f0e1f5a230d79e5f493a64fa9ee8d074d2a03719c1534550b1', 'collector binding')
    for record in receipt['inputs']:
        p = Path(record['path'])
        check(p.is_file() and not p.is_symlink() and sha(p) == record['sha256']
              and p.stat().st_size == record['size_bytes'], 'input binding')
    check({p.name for p in DATA.iterdir()} == {'receipt.json'} | {r['path'] for r in receipt['outputs']}, 'output coverage')
    for record in receipt['outputs']:
        check(Path(record['path']).name == record['path'], 'local output')
        p = DATA/record['path']
        check(sha(p) == record['sha256'] and p.stat().st_size == record['size_bytes'], 'output pin')
    total_bytes = sum(p.stat().st_size for p in DATA.iterdir())
    check(total_bytes == receipt['bytes_before_receipt']+(DATA/'receipt.json').stat().st_size < 32*1024**2, 'byte cap')
    old = set(read(WORK/'old-prefixes.json')['unique_prefixes'])
    excluded = set(read(WORK/'inventory.json')['excluded_pageids'])
    dispositions = read(DATA/'candidate-dispositions.json')
    check(len(dispositions) == len(manifest['candidates']) == 86, 'all candidates')
    datasets = {'calibration': [], 'evaluation': []}
    pairs = {'calibration': [], 'evaluation': []}
    targets = {'calibration': 32, 'evaluation': 16}
    requested, accepted_ids, rejected = [], {}, []
    seconds = []
    for candidate, decision in zip(manifest['candidates'], dispositions, strict=True):
        check(all(decision[k] == v for k, v in candidate.items()), 'frozen candidate')
        cid, role = candidate['candidate_id'], candidate['role']
        if len(pairs[role]) == targets[role]:
            check(decision['status'] == 'capacity_skip_no_requests', 'capacity skip')
            check(not (DATA/(cid+'-left.request.json')).exists() and not (DATA/(cid+'-right.request.json')).exists(), 'no skipped request')
            continue
        texts, pages = [], []
        for side in ['left', 'right']:
            pid = candidate[side+'_pageid']; requested.append(pid)
            record = read(DATA/(cid+'-'+side+'.request.json'))
            check(record['params'] == {'action': 'query', 'format': 'json', 'formatversion': 2, 'maxlag': 5,
                'pageids': pid, 'prop': 'extracts|revisions|info|pageprops', 'explaintext': 1, 'exintro': 1,
                'exchars': 1200, 'rvprop': 'ids|timestamp', 'rvlimit': 1, 'inprop': 'url'}, 'exact request')
            check(record['status'] == 200 and record['new_network_request'] is True
                  and record['content_encoding'] in ['', 'identity'] and record['body_truncated_at_cap'] is False, 'transport')
            raw = DATA/(cid+'-'+side+'.response.json')
            check(record['response'] == {'path': raw.name, 'sha256': sha(raw), 'size_bytes': raw.stat().st_size}, 'response join')
            check(raw.stat().st_size <= 2*1024**2, 'response size')
            value = read(raw)
            check('error' not in value and 'warnings' not in value and 'redirects' not in value['query'], 'API schema')
            if 'continue' in value:
                check(set(value['continue']) == {'rvcontinue', 'continue'} and
                      all(isinstance(s, str) and s for s in value['continue'].values()), 'revision continuation only')
            check(len(value['query']['pages']) == 1, 'one article')
            page = value['query']['pages'][0]; pages.append(page)
            prefix, reason = eligibility(page, pid); texts.append(prefix)
            saved = read(DATA/(cid+'-'+side+'.eligibility.json'))
            check(saved == {'side': side, 'pageid': pid, 'response': record['response'],
                            'eligible': prefix is not None, 'technical_reason': reason}, 'eligibility reconstruction')
            seconds.append((datetime.fromisoformat(record['completed_utc'])-datetime.fromisoformat(record['started_utc'])).total_seconds())
        reason = 'technical member exclusion' if None in texts else None
        if reason is None:
            if texts[0] == texts[1]: reason = 'duplicate prefix within pair'
            elif any(t in old for t in texts): reason = 'duplicate old or accepted prefix'
        check(decision['reason'] == reason and decision['status'] == ('rejected' if reason else 'accepted'), 'whole-pair decision')
        check(read(DATA/(cid+'.decision.json')) == decision, 'decision join')
        if reason:
            rejected.append({'candidate_id': cid, 'role': role, 'reason': reason})
            continue
        pair_id = ('C' if role == 'calibration' else 'T')+f'{len(pairs[role])+1:02}'
        check(decision['accepted_pair_id'] == pair_id, 'accepted order')
        pairs[role].append({'id': pair_id, 'left': pair_id+'-L', 'right': pair_id+'-R',
                            'topic': candidate['topic'], 'candidate_id': cid, 'role': role})
        for side, prefix, page in zip(['L', 'R'], texts, pages, strict=True):
            row_id = pair_id+'-'+side
            datasets[role].append({'id': row_id, 'topic': candidate['topic'], 'prefix': prefix})
            accepted_ids[row_id] = (page, prefix, role, cid)
        old.update(texts)
    check(len(requested) == len(set(requested)) == receipt['article_requests'] == 102, 'unique request count')
    check(not set(requested) & excluded and len(requested)//2 == receipt['requested_pairs'] == 51, 'prior exclusion/pair count')
    for role in targets:
        check(len(pairs[role]) == targets[role] and datasets[role] == read(DATA/(role+'-dataset.json'))
              and pairs[role] == read(DATA/(role+'-pairs.json')), 'exact role output')
    attribution = read(DATA/'attribution.json')['articles']
    check(len(attribution) == len(accepted_ids) == len({a['id'] for a in attribution}) == 96, 'attribution coverage')
    for entry in attribution:
        page, prefix, role, cid = accepted_ids[entry['id']]
        check(entry['pageid'] == page['pageid'] and entry['prefix'] == prefix and entry['title'] == page['title']
              and entry['url'] == page['fullurl'] and entry['role'] == role and entry['candidate_id'] == cid
              and entry['reported_revision_id'] == page['revisions'][0]['revid'], 'source attribution join')
    check(0 <= min(seconds) <= max(seconds) < 20 and receipt['elapsed_seconds'] < 3600, 'observed time bounds')
    result = {'status': 'PASS', 'scope': 'saved-response collection reconstruction; no acquisition or model replay',
        'receipt_sha256': RECEIPT, 'manifest_sha256': MANIFEST, 'audit_source_sha256': sha(Path(__file__)),
        'article_requests': len(requested), 'requested_pairs': 51, 'selected_rows': 96,
        'accepted_pairs': {r: len(p) for r, p in pairs.items()}, 'rejected_pairs': rejected,
        'role_topic_rows': {r: dict(Counter(x['topic'] for x in data)) for r, data in datasets.items()},
        'total_artifact_bytes': total_bytes, 'maximum_observed_request_seconds': max(seconds),
        'source_attribution_publication_review_pending': True, 'new_network_or_model_calls': 0}
    with output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False); handle.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    run()
