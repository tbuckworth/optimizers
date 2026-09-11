"""Fabricated author-output fixtures only; no scientific input or model."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('roster_fixture',Path(__file__).with_name('prepare_roster.py'))
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)


def proposal():
    return {'schema':'jlens_fresh_author_proposal_v1','contrasts':[
        {'id':f'fa{i+1:02}','actor':'actor'+c,'object':'objects'+c,
         'observation_lemma':'obs'+c,'observation_past':'obs'+c+'ed',
         'provision_lemma':'prep'+c,'provision_past':'prep'+c+'ed'} for i,c in enumerate('abcdefgh')]}


class Tests(unittest.TestCase):
    def test_exact_full_roster_and_spans(self):
        data, pairs = p.build(proposal(),[])
        self.assertEqual(len(data['rows']),32); self.assertEqual(len(pairs['pairs']),16)
        for n,row in enumerate(data['rows']):
            self.assertEqual(row['pole'],'O' if n%2 == 0 else 'P')
            self.assertEqual(row['text'][slice(*row['verb_span'])],row['verb'])
            self.assertEqual(len(row['text'].split()),5 if row['template']=='active' else 7)
            self.assertEqual(len(row['text'][:row['verb_span'][1]].split()),3 if row['template']=='active' else 4)
        self.assertEqual(data['rows'][0]['text'],'The actora obsaed the objectsa.')
        self.assertEqual(data['rows'][2]['text'],'The objectsa were obsaed by the actora.')
        for i,edge in enumerate(pairs['pairs']):
            self.assertEqual(edge['observation_id'],data['rows'][2*i]['id'])
            self.assertEqual(edge['provision_id'],data['rows'][2*i+1]['id'])

    def test_schema_exclusions_duplicates_and_nonwords(self):
        for mutate in [lambda x:x.update(extra=True),lambda x:x['contrasts'].pop(),
            lambda x:x['contrasts'][0].update(id='fa02'),lambda x:x['contrasts'][0].update(actor='two words'),
            lambda x:x['contrasts'][0].update(object='Évents'),lambda x:x['contrasts'][1].update(observation_lemma='obsa'),
            lambda x:x['contrasts'][0].update(observation_lemma='banned'),
            lambda x:x['contrasts'][1].update(observation_past='obsaed')]:
            bad=proposal(); mutate(bad)
            with self.assertRaises(ValueError): p.build(bad,['banned'])
        for excluded in [None,['z','a'],['a','a'],['not alpha']]:
            with self.assertRaises(ValueError): p.build(proposal(),excluded)

    def test_strict_json(self):
        for raw in [b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":Infinity}',b' ' * 65537]:
            with self.assertRaises(ValueError): p.decode(raw)

    def test_file_prepare_exact_pins_and_consumed_attempt(self):
        with tempfile.TemporaryDirectory(prefix='jlens-author-fixture-') as tmp:
            root=Path(tmp); kwargs={}
            for name,raw in [('author',json.dumps(proposal()).encode()),('exclusions',b'[]'),('protocol',b'fixture protocol')]:
                path=root/name; path.write_bytes(raw); kwargs[name]=path; kwargs[name+'_sha']=p.sha(raw)
            out=root/'derived'; p.prepare(out,**kwargs)
            self.assertEqual(json.loads((out/'receipt.json').read_text())['row_count'],32)
            with self.assertRaises(FileExistsError): p.prepare(out,**kwargs)
            kwargs['author_sha']='0'*64
            with self.assertRaises(ValueError): p.prepare(root/'failed',**kwargs)
            self.assertTrue((root/'failed/failure.json').exists())
            with self.assertRaises(FileExistsError): p.prepare(root/'failed',**kwargs)


if __name__ == '__main__': unittest.main()
