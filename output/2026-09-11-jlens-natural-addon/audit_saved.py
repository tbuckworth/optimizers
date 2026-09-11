"""Independent read-only grade audit, with committed-response checks before key I/O."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
AXES = ['PC1', 'PC2', 'PC3', 'PC4']
ARMS = ['EA', 'E']
TOPICS = ['astronomy', 'cooking', 'football', 'programming']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    assert p.stat().st_size <= 1024*1024
    return json.loads(p.read_text())


def pinned(pin):
    p = Path(pin['path'])
    assert p.stat().st_size == pin['size_bytes'] and sha(p) == pin['sha256']
    return p


def committed(p, commit):
    blob = subprocess.check_output(['git', '-C', str(REPO), 'show',
        commit+':'+p.relative_to(REPO).as_posix()], timeout=10)
    assert blob == p.read_bytes()


def run(grade_sha):
    gp = ROOT/'graded/grades.json'
    assert sha(gp) == grade_sha
    grade = read(gp)
    assert grade['schema'] == 'jlens_natural_addon_grades_v1'
    prov = grade['provenance']
    lockp = ROOT/'responses/lock.json'
    assert sha(lockp) == prov['lock_sha256']
    lock = read(lockp)
    for name in ['lock.json', *[f'rater{i}.json' for i in range(1, 5)]]:
        committed(ROOT/'responses'/name, prov['lock_commit'])
    # Only after all five response/lock blobs are checked, inspect scores/maps.
    releasep = pinned(prov['release']['receipt'])
    committed(releasep, prov['release']['commit'])
    release = read(releasep)
    assert release['schema'] == 'jlens_natural_addon_release_v1'
    assert release['artifacts']['scores'] == prov['scores']
    assert release['artifacts']['producer'] == prov['producer']
    export = read(pinned(prov['scores']))
    assert export['schema'] == 'jlens_natural_addon_scores_v1'
    assert list(export['locations']) == [prov['graded_location']] == ['prefix_end']
    scores = {r['axis']:r['values'] for r in export['locations']['prefix_end']}
    assert list(scores) == AXES
    mp = ROOT/'packets/manifest.json'
    assert sha(mp) == prov['packet_manifest_sha256'] == lock['packet_manifest_sha256']
    manifest = read(mp)
    assert manifest['release'] == prov['release'] and manifest['inputs'] == prov['inputs']
    inputs = manifest['inputs']
    assert inputs['dataset']['sha256'] == 'd57300cb8511e409ff78e6ddc9fb6e1cf11b91bc629bf5492db8932cf585b129'
    assert inputs['pairs']['sha256'] == '0ae85708e0b5da2942e31aa96af4ffa642c1294ff4c8b78612ec2dfc0ee54725'
    assert inputs['references']['sha256'] == '0a678ad8d58a6b17fc2f0c3ac197d47be83e163750b2408f52759e23a0e6c44c'
    for key,pin in inputs.items():
        assert release['artifacts'][key] == pin
        pinned(pin)
    data = {r['id']:r for r in read(Path(inputs['dataset']['path']))}
    pairs = {r['id']:r for r in read(Path(inputs['pairs']['path']))}
    refs = {r['axis']:r for r in read(Path(inputs['references']['path']))['axes']}
    tokens = {r['id']:r for r in read(Path(inputs['tokens']['path']))['records']}
    mappingp = ROOT/'packets/private-map.json'
    assert sha(mappingp) == manifest['private_map']['sha256']
    mapping = read(mappingp)['rows']
    public, choices = {}, {}
    for i in range(1,5):
        rater = f'rater{i}'
        pp, rp = ROOT/f'packets/public/{rater}.json', ROOT/f'responses/{rater}.json'
        assert sha(pp) == manifest['public'][rater]['sha256']
        assert sha(rp) == lock['responses'][rater]['sha256']
        packet, response = read(pp), read(rp)
        assert packet['packet_id'] == response['packet_id']
        choices[rater] = {r['item_id']:r['choice'] for r in response['responses']}
        assert len(response['responses']) == len(choices[rater]) == 64
        assert set(choices[rater].values()) <= {'FIRST','SECOND'}
        for block in packet['blocks']:
            for item in block['comparisons']:
                assert item['item_id'] not in public
                public[item['item_id']] = (rater,block,item)
    actual = {r['item_id']:r for r in grade['items']}
    assert len(actual) == len(grade['items']) == len(public) == len(mapping) == 256
    assert {r['item_id'] for r in mapping} == set(public) == set(actual)
    assert {(r['rater'],r['axis'],r['pair_id']) for r in mapping} == {
        (f'rater{i}',axis,pair) for i in range(1,5) for axis in AXES for pair in pairs}
    orientations, calculated = {}, []
    for row in mapping:
        rater, block, item = public[row['item_id']]
        assert rater == row['rater'] and block['block_id'] == row['block_id']
        assert row['arm'] == ('EA' if rater in ['rater1','rater3'] else 'E')
        assert row['cohort'] == (1 if rater in ['rater1','rater2'] else 2)
        pair = pairs[row['pair_id']]
        assert row['topic'] == pair['topic'] and row['candidate_id'] == pair['candidate_id']
        assert {row['first_id'],row['second_id']} == {pair['left'],pair['right']}
        key = (row['axis'],row['pair_id'],row['cohort'])
        orient = (row['first_id'],row['second_id'])
        assert orientations.setdefault(key,orient) == orient
        for pole in ['positive','negative']:
            expected = {'example_prefix':refs[row['axis']]['C'][pole]}
            if row['arm']=='EA': expected['direction_tokens']=refs[row['axis']]['A'][pole]
            assert block[pole+'_reference'] == expected
        for side in ['first','second']:
            ident = row[side+'_id']; token = tokens[ident]
            assert item[side] == data[ident]['prefix'] == token['prefix']
            assert token['captured_positions']=={'prefix_end':len(token['input_ids'])-1}
            assert token['offset_mapping'][-1][1]==len(item[side])
        a,b = (scores[row['axis']][row[s+'_id']] for s in ['first','second'])
        gap = a-b; truth='FIRST' if gap>0 else 'SECOND' if gap<0 else 'TIE'
        choice = choices[rater][row['item_id']]
        expected = dict(row,choice=choice,truth=truth,
            chosen_id=row['first_id' if choice=='FIRST' else 'second_id'],
            score_first=a,score_second=b,gap=gap,absolute_gap=abs(gap),
            credit=0.5 if gap==0 else float(choice==truth))
        assert actual[row['item_id']] == expected
        calculated.append(expected)
    for (axis,pair,cohort),orient in orientations.items():
        assert orient == orientations[axis,pair,3-cohort][::-1]

    def subset(**conditions):
        return [r for r in calculated if all(r[k]==v for k,v in conditions.items())]
    def summary(rows):
        return dict(credit=sum(r['credit'] for r in rows),total=len(rows),
            correct=sum(r['credit']==1 for r in rows),incorrect=sum(r['credit']==0 for r in rows),
            exact_ties=sum(r['truth']=='TIE' for r in rows),
            constant_FIRST_credit=sum(0.5 if r['truth']=='TIE' else r['truth']=='FIRST' for r in rows),
            constant_SECOND_credit=sum(0.5 if r['truth']=='TIE' else r['truth']=='SECOND' for r in rows))
    def difference(**conditions):
        return sum(r['credit'] for r in subset(arm='EA',**conditions))-sum(r['credit'] for r in subset(arm='E',**conditions))
    for arm in ARMS:
        assert grade['arms'][arm] == summary(subset(arm=arm))
        for axis in AXES: assert grade['per_axis'][axis][arm] == summary(subset(arm=arm,axis=axis))
        for field,key,nested,values in [('cohort','per_cohort','cohort_axis',[1,2]),('topic','per_category','category_axis',TOPICS)]:
            for value in values:
                assert grade[key][str(value)][arm] == summary(subset(arm=arm,**{field:value}))
                for axis in AXES:
                    assert grade[nested][str(value)][axis][arm] == summary(subset(arm=arm,axis=axis,**{field:value}))
    controls = {}
    for i in range(1,5):
        rater=f'rater{i}'; s=summary(subset(rater=rater)); result=grade['per_reader'][rater]
        assert result['summary']==s
        assert result['cohort']==(1 if i in [1,2] else 2)
        for axis in AXES:
            assert result['per_axis'][axis]['arm']==('EA' if i in [1,3] else 'E')
            assert {k:v for k,v in result['per_axis'][axis].items() if k!='arm'}==summary(subset(rater=rater,axis=axis))
        stronger=max(s['constant_FIRST_credit'],s['constant_SECOND_credit'])
        controls[rater]=dict(arm='EA' if i in [1,3] else 'E',credit_out_of_64=s['credit'],
            constant_FIRST_credit=s['constant_FIRST_credit'],constant_SECOND_credit=s['constant_SECOND_credit'],
            stronger_constant_credit=stronger,exceeds_stronger_constant=s['credit']>stronger)
    diffs=grade['paired_differences']['EA_minus_E']
    assert diffs['total_credit']==difference()
    assert diffs['per_axis_credit']=={a:difference(axis=a) for a in AXES}
    for field,key,values in [('cohort','per_cohort',[1,2]),('topic','per_category',TOPICS)]:
        for value in values:
            assert diffs[key][str(value)]==dict(total_credit=difference(**{field:value}),
                per_axis_credit={a:difference(axis=a,**{field:value}) for a in AXES})
    cohort_diffs={str(c):difference(cohort=c) for c in [1,2]}
    both=all(v>0 for v in cohort_diffs.values())
    constants=all(controls[r]['exceeds_stronger_constant'] for r in ['rater1','rater3'])
    assert grade['primary']==dict(scope='all_four_axes',EA_minus_E_credit_out_of_128=difference(),
        EA_minus_E_by_cohort_out_of_64=cohort_diffs,reader_controls=controls,
        EA_minus_E_positive_each_cohort=both,each_EA_reader_exceeds_stronger_constant=constants,
        incremental_pilot_criterion_met=both and constants)
    assert grade['secondary_PC4']==dict(per_arm=grade['per_axis']['PC4'],
        EA_minus_E_credit_out_of_32=difference(axis='PC4'),
        EA_minus_E_by_cohort_out_of_16={str(c):difference(axis='PC4',cohort=c) for c in [1,2]})
    assert len(grade['gains_harm'])==128
    assert {(r['cohort'],r['axis'],r['pair_id']) for r in grade['gains_harm']} == {
        (c,a,p) for c in [1,2] for a in AXES for p in pairs}
    for r in grade['gains_harm']:
        ea,e=actual[r['EA_item_id']],actual[r['E_item_id']]
        delta=ea['credit']-e['credit']
        assert r==dict(cohort=ea['cohort'],axis=ea['axis'],pair_id=ea['pair_id'],topic=ea['topic'],
            EA_item_id=ea['item_id'],E_item_id=e['item_id'],EA_credit=ea['credit'],E_credit=e['credit'],
            EA_minus_E_credit=delta,outcome='gain' if delta>0 else 'harm' if delta<0 else 'tie')
        assert (ea['axis'],ea['pair_id'],ea['cohort'])==(e['axis'],e['pair_id'],e['cohort'])
    for arm in ARMS:
        total_agree=0
        for axis in AXES:
            cells=defaultdict(list)
            for row in subset(arm=arm,axis=axis): cells[row['pair_id']].append(row)
            result=grade['agreement']['per_axis'][axis][arm]
            agree=sum(v[0]['chosen_id']==v[1]['chosen_id'] for v in cells.values())
            assert {k:v for k,v in result.items() if k!='items'}==dict(agree=agree,disagree=16-agree,total=16)
            assert len(result['items'])==16
            for item in result['items']:
                a,b=sorted(cells[item['pair_id']],key=lambda x:x['cohort'])
                assert item==dict(pair_id=a['pair_id'],cohort1_rater=a['rater'],cohort2_rater=b['rater'],
                    cohort1_chosen_id=a['chosen_id'],cohort2_chosen_id=b['chosen_id'],agree=a['chosen_id']==b['chosen_id'])
            total_agree+=agree
        assert grade['agreement']['arms'][arm]==dict(agree=total_agree,disagree=64-total_agree,total=64)
    assert grade['category_counts']=={t:dict(pairs=sum(p['topic']==t for p in pairs.values()),
        texts=sum(r['topic']==t for r in data.values())) for t in TOPICS}
    return dict(status='PASS',grade_sha256=grade_sha,auditor_sha256=sha(Path(__file__)),
        exact_item_rows=256,committed_response_blobs=5,all_summaries=True,
        all_reference_target_joins=True,primary=grade['primary'],arms=grade['arms'],
        model_calls=0,grader_imported=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grade-sha',required=True)
    print(json.dumps(run(parser.parse_args().grade_sha),indent=2,allow_nan=False))
