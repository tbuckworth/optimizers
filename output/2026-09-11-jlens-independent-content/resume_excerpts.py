"""One bounded continuation of the saved catalogue; never repeats the saved first request."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('original_collector', HERE/'collect_text.py')
c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)
COLLECTOR_SHA = 'd277f27eccdbf85bdbf1907ece40fea40148d59dded23e806d5f4721e1756f7a'
CATALOGUE_SHA = 'b15afe7220a4c823454b550553c3ab8a0181712773813922bce5976ccb8a0a0d'
SAVED = {
    'attempt.json': 'ff3fe5683bbbcfb70ca86365337e4f996b774a78a3f518ff23ca9468faa75a56',
    'failure.json': 'c5822ec04f80b750a7f8acfc90514c6c338653d07c51854b2773844035300bc9',
    'astronomy-01.request.json': '33e8cf58c71243e34b59b65f5b08eeea8e37db0b5179a1bd3fc246c2fe48ef20',
    'astronomy-01.response.json': 'f830789882722bbc4fea082f7109f0649fd2e3ea7d552c5c13b32d441585aa40',
}


def validate_response(status, raw, encoding):
    c.require(status == 200 and encoding in ('identity', ''), 'HTTP/encoding error')
    value = c.decode(raw)
    c.require(type(value) is dict and not value.get('error') and not value.get('warnings'), 'API error/warning')
    if 'continue' in value:
        token = value['continue']
        c.require(type(token) is dict and set(token) == {'rvcontinue', 'continue'}
                  and all(type(v) is str and v for v in token.values()), 'non-revision continuation')
        # rvlimit=1 enumerates the latest revision; older revision history is not requested by this study.
    return value


def params_for(pageid):
    return {'action': 'query', 'format': 'json', 'formatversion': 2, 'maxlag': 5,
            'pageids': pageid, 'prop': 'extracts|revisions|info|pageprops', 'explaintext': 1,
            'exintro': 1, 'exchars': 1200, 'rvprop': 'ids|timestamp', 'rvlimit': 1, 'inprop': 'url'}


def request(stage, name, params, transport, saved=None):
    if saved is not None:
        old_request, raw = saved
        c.require(old_request['params'] == params and not old_request['body_truncated_at_cap']
                  and c.sha(raw) == old_request['response']['sha256']
                  and len(raw) == old_request['response']['size_bytes'], 'saved response binding')
        record = stage.raw(name+'.response.json', raw)
        stage.write(name+'.request.json', {**old_request, 'response': record,
                    'reused_from': '../excerpts/astronomy-01.response.json', 'new_network_request': False})
        return validate_response(old_request['status'], raw, old_request['content_encoding']), record
    started = c.utc()
    cap = min(c.MAX_RESPONSE, c.MAX_OUTPUT-c.RESERVE-stage.used-4096)
    c.require(cap > 0, 'no remaining request budget')
    try:
        status, raw, encoding = transport(params, cap)
    except Exception as error:
        stage.write(name+'.request.json', {'endpoint': c.API, 'params': params,
                    'started_utc': started, 'completed_utc': c.utc(),
                    'transport_error_type': type(error).__name__, 'user_agent': c.USER_AGENT})
        raise
    c.require(type(raw) is bytes, 'transport body must be bytes')
    record = stage.raw(name+'.response.json', raw[:cap])
    stage.write(name+'.request.json', {'endpoint': c.API, 'params': params, 'started_utc': started,
                'completed_utc': c.utc(), 'status': status, 'content_encoding': encoding,
                'user_agent': c.USER_AGENT, 'response': record, 'body_truncated_at_cap': len(raw) > cap,
                'new_network_request': True})
    c.require(len(raw) <= cap, 'response byte cap; retained prefix explicitly incomplete')
    return validate_response(status, raw, encoding), record


def resume(root=HERE, transport=c.fetch):
    stage = c.Stage(root, 'excerpts-resumed')
    source_sha = c.sha(Path(__file__).read_bytes())
    try:
        c.require(c.sha(Path(c.__file__).read_bytes()) == COLLECTOR_SHA, 'original collector changed')
        c.protocol(root)
        groups, prior_bytes, catalogue_sha = c.load_catalogue(root)
        c.require(catalogue_sha == CATALOGUE_SHA, 'catalogue changed')
        saved = {}
        for name, digest in SAVED.items():
            raw = c.bounded_read(root/'excerpts'/name)
            c.require(c.sha(raw) == digest, 'failed attempt changed')
            saved[name] = raw
            prior_bytes += len(raw)
        stage.used += prior_bytes
        c.require(stage.used < c.MAX_OUTPUT-c.RESERVE, 'combined output byte cap')
        stage.write('continuation.json', {'source_sha256': source_sha, 'original_source_sha256': COLLECTOR_SHA,
                    'catalogue_receipt_sha256': catalogue_sha, 'preserved_attempt': SAVED,
                    'correction': 'Accept only revision-history continuation; reuse the first saved response.',
                    'new_catalogue_requests': 0, 'automatic_retry': False})
        selected_ids, selected_prefixes, dataset, attribution, decisions = set(), set(), [], [], []
        network_count = 0
        for group in groups:
            accepted = 0
            for rank, candidate in enumerate(group['candidates'][:20], 1):
                topic, pageid = group['topic'], candidate['pageid']
                name = f'{topic}-{rank:02}'
                reuse = None
                if topic == 'astronomy' and rank == 1:
                    reuse = (c.decode(saved['astronomy-01.request.json']), saved['astronomy-01.response.json'])
                else:
                    network_count += 1
                value, response = request(stage, name, params_for(pageid), transport, reuse)
                article, reason = c.eligible(value, pageid, selected_ids, selected_prefixes)
                decision = {'topic': topic, 'candidate_rank': rank, 'pageid': pageid, 'response': response,
                            'selected': article is not None, 'skip_reason': reason}
                if article is not None:
                    ident = f'{topic}-wiki-{accepted}'
                    selected_ids.add(pageid); selected_prefixes.add(article['prefix'])
                    dataset.append({'id': ident, 'topic': topic, 'prefix': article['prefix']})
                    attribution.append({'id': ident, 'response': response, **article})
                    decision['id'] = ident
                    accepted += 1
                decisions.append(decision)
                stage.write(name+'.decision.json', decision)
                if accepted == 6: break
            c.require(accepted == 6, f'Fewer than six eligible pages within first 20 candidates: {group["topic"]}')
        c.require(len(dataset) == len(selected_ids) == len(selected_prefixes) == 24, 'final source roster')
        pairs = [{'id': f'P{i+1:02}', 'left': dataset[2*i]['id'], 'right': dataset[2*i+1]['id']} for i in range(12)]
        stage.write('dataset.json', dataset)
        stage.write('pairs.json', pairs)
        stage.write('attribution.json', {'schema': 'jlens_independent_attribution_v1', 'articles': attribution,
                    'snapshot_scope': 'Exact response bytes, not an atomic historical revision reconstruction.',
                    'reuse_scope': 'Excerpt-derived content retains applicable Wikipedia source terms; code/research are separate.'})
        c.protocol(root)
        c.require(c.load_catalogue(root)[2] == CATALOGUE_SHA, 'catalogue changed during excerpts')
        c.require(c.sha(Path(__file__).read_bytes()) == source_sha, 'continuation source changed')
        return stage.complete('jlens_independent_excerpts_v1', {'continuation_source_sha256': source_sha,
                    'catalogue_receipt_sha256': catalogue_sha, 'selected_count': 24,
                    'candidate_requests': len(decisions), 'new_network_requests': network_count,
                    'reused_response_count': 1, 'additional_notice_check_before_publication': True})
    except Exception as error:
        stage.fail(error)
        raise


if __name__ == '__main__':
    resume()
