"""Fabricated packet transport; no actual public packet, reader or key accessed."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import prepare_transport as m
import reader_packets as helper
from test_reader_packets import fixture


class TransportTests(unittest.TestCase):
    def test_exact_prefix_plus_public_bytes_no_private_open(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'packets').mkdir()
            raw_source = Path(helper.__file__).read_bytes()
            (root/'reader_packets.py').write_bytes(raw_source)
            packets, _ = helper.make_packets(*fixture())
            outputs = []
            for rater, packet in packets.items():
                raw = helper.encode(packet); (root/'packets'/(rater+'.json')).write_bytes(raw)
                outputs.append({'path': rater+'.json', 'sha256': m.sha(raw), 'size_bytes': len(raw)})
            manifest = helper.encode({'schema': 'jlens_pattern_calibration_packets_v1',
                'source_sha256': m.PACKET_SOURCE_SHA, 'outputs': outputs})
            (root/'packets/manifest.json').write_bytes(manifest)
            source_digest = m.sha(Path(m.__file__).read_bytes())
            def git(args, **kwargs):
                if args[1:3] == ['rev-parse', '--show-toplevel']: return str(root)+'\n'
                if args[1] == 'rev-parse': return m.PACKET_COMMIT+'\n'
                self.assertEqual(args[1], 'show')
                return (root/args[2].split(':', 1)[1]).read_bytes()
            original = Path.read_bytes; seen = []
            def guard(path):
                self.assertNotIn(path.name, ('private-map.json', 'scores.json', 'gaps.json', 'features.npz'))
                seen.append(path.name); return original(path)
            with patch.object(m, 'HERE', root), patch.object(m, 'MANIFEST_SHA', m.sha(manifest)), \
                 patch.object(m.subprocess, 'check_output', side_effect=git), \
                 patch.dict(m.os.environ, {'JLENS_PATTERN_TRANSPORT_RELEASE': '1'}), \
                 patch.object(Path, 'read_bytes', guard):
                m.run(source_digest)
                with self.assertRaises(FileExistsError): m.run(source_digest)
            registry = json.loads((root/'transport/registry.json').read_bytes())
            self.assertFalse(registry['private_map_or_scientific_key_opened']); self.assertEqual(registry['reader_calls'], 0)
            self.assertEqual(set(registry['transports']), set(helper.RATERS))
            for rater, row in registry['transports'].items():
                raw = (root/'transport'/row['path']).read_bytes()
                self.assertEqual(raw, m.PREFIX.encode()+(root/'packets'/(rater+'.json')).read_bytes())
                self.assertEqual(m.sha(raw), row['sha256']); self.assertEqual(len(raw), row['size_bytes'])
            self.assertTrue(seen)

    def test_wrong_release_or_source_cannot_prepare(self):
        with patch.dict(m.os.environ, {'JLENS_PATTERN_TRANSPORT_RELEASE': ''}):
            with self.assertRaisesRegex(ValueError, 'admission'): m.run('bad')
        with patch.dict(m.os.environ, {'JLENS_PATTERN_TRANSPORT_RELEASE': '1'}):
            with self.assertRaisesRegex(ValueError, 'source hash'): m.run('bad')


if __name__ == '__main__': unittest.main()
