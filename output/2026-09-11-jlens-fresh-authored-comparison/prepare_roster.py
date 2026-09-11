"""Format a frozen author's complete eight-contrast roster; no model or scoring.

This validates syntax/declared exclusions, not English grammar or semantics.
Main's full linguistic review remains explicit and precedes measurement.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pairs_hook(items):
    result = {}
    for key, value in items:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def invalid_number(value):
    raise ValueError('nonfinite JSON number')


def decode(raw):
    require(len(raw) <= 65536, 'JSON too large')
    return json.loads(raw.decode('utf-8'), object_pairs_hook=pairs_hook, parse_constant=invalid_number)


def bound(path, expected):
    require(re.fullmatch('[0-9a-f]{64}', expected) is not None, 'invalid SHA256')
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 65536, 'bounded regular file required')
    raw = path.read_bytes()
    require(len(raw) <= 65536 and sha(raw) == expected, 'input SHA256 mismatch')
    return raw


def word(value):
    return type(value) is str and re.fullmatch('[a-z]{1,40}', value) is not None


def build(proposal, excluded):
    require(type(excluded) is list and all(word(x) for x in excluded) and
            excluded == sorted(set(excluded)), 'sorted unique lexical exclusions required')
    require(type(proposal) is dict and set(proposal) == {'schema','contrasts'} and
            proposal['schema'] == 'jlens_fresh_author_proposal_v1', 'author schema')
    require(type(proposal['contrasts']) is list and len(proposal['contrasts']) == 8, 'exactly eight contrasts required')
    fields = {'id','actor','object','observation_lemma','observation_past','provision_lemma','provision_past'}
    used, rows, edges = set(), [], []
    for n, item in enumerate(proposal['contrasts'], 1):
        ident = f'fa{n:02}'
        require(type(item) is dict and set(item) == fields and item['id'] == ident, 'contrast keys/order')
        require(all(word(item[k]) for k in fields-{'id'}), 'lowercase single ASCII words required')
        for side in ('observation','provision'):
            lemma = item[side+'_lemma']
            require(lemma not in excluded and lemma not in used, 'excluded or repeated action lemma')
            used.add(lemma)
        for template in ('active','passive'):
            pair_id = f'{ident}-{template}'
            for side, pole in [('observation','O'),('provision','P')]:
                verb = item[side+'_past']
                prefix = f"The {item['actor']} " if template == 'active' else f"The {item['object']} were "
                suffix = f" the {item['object']}." if template == 'active' else f" by the {item['actor']}."
                text = prefix+verb+suffix
                rows.append({'id':f'{pair_id}-{pole}','pair_id':ident,'template':template,
                    'pole':pole,'text':text,'verb':verb,'verb_span':[len(prefix),len(prefix)+len(verb)]})
            edges.append({'id':pair_id,'content_pair':ident,'template':template,
                'observation_id':f'{pair_id}-O','provision_id':f'{pair_id}-P'})
    require(len({r['text'] for r in rows}) == 32, 'duplicate exact sentence')
    require(len({r['verb'] for r in rows}) == 16, 'repeated surface verb')
    return {'rows':rows}, {'pairs':edges}


def write(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write('\n')


def prepare(output, *, author, author_sha, exclusions, exclusions_sha, protocol, protocol_sha):
    output.mkdir(exist_ok=False)
    write(output/'attempt.json', {'started_utc':datetime.now(timezone.utc).isoformat(),
        'source_sha256':sha(Path(__file__).read_bytes()),'automatic_retry':False})
    try:
        records = [('author',author,author_sha),('exclusions',exclusions,exclusions_sha),('protocol',protocol,protocol_sha)]
        raw = {name:bound(path,pin) for name,path,pin in records}
        dataset, pair_data = build(decode(raw['author']), decode(raw['exclusions']))
        write(output/'dataset.json',dataset); write(output/'pairs.json',pair_data)
        write(output/'receipt.json', {'status':'complete','inputs':{name:{'path':str(path.resolve()),'sha256':pin} for name,path,pin in records},
            'outputs':{name:sha((output/name).read_bytes()) for name in ['dataset.json','pairs.json']},
            'row_count':32,'pair_count':16,'model_calls':0,'semantic_validation':False})
    except Exception as error:
        write(output/'failure.json',{'status':'failed','error_type':type(error).__name__})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('author','exclusions','protocol'):
        parser.add_argument('--'+name,type=Path,required=True)
        parser.add_argument('--'+name+'-sha',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = vars(parser.parse_args())
    prepare(args.pop('out'),**args)
