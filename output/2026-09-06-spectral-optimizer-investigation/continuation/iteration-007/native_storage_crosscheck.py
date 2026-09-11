"""Inert structural-recipe/serializer candidate, NOT a measurement controller.

No CLI option launches work; no M/A writer or experiment adapter is imported.
Full component materialization and one-shot supervised measurement are deferred.
The private serializer primitive is exercised only with tiny CPU unit fixtures.
External wall/RSS enforcement cannot be replaced by its cooperative checks.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import struct
import zipfile

import native_storage_recipe_common as common


def validate_metadata(sources, environment):
    """Validate supplied shape/caps/root, not authenticity or live collection."""
    import native_storage_authority as authority
    import source_environment_schema as schema
    authority.encode_bounded(sources, maximum=32768)
    authority.encode_bounded(environment, maximum=8192)
    schema.validate_sources(sources, profile=schema.SCIENTIFIC)
    schema.validate_environment(environment, profile=schema.SCIENTIFIC)
    common._need(environment['runtime_role'] == 'storage_crosscheck_cpu', 'diagnostic role required')
    common._need(sources['repository_root_realpath'] == environment['repository_root_realpath'],
                 'source/diagnostic environment roots differ')


def symbolic_components(*, sources, environment):
    """Yield one complete symbolic tree at a time; no tensors or serialization."""
    import native_storage_recipe_core as core
    import native_storage_recipe_branch_audit as branch_audit
    import native_tensor_inventory as inventory
    validate_metadata(sources, environment)
    for name in inventory.COMPONENT_COUNTS:
        builder = core if name in core.COMPONENTS else branch_audit
        tree = builder.build_template(name, sources=sources, environment=environment)
        yield name, tree, common.validate_template(name, tree)


class _CappedBuffer(io.BytesIO):
    def __init__(self, limit, on_peak=None):
        common._need(type(limit) is int and 0 < limit <= 64 << 20, 'invalid buffer ceiling')
        common._need(on_peak is None or callable(on_peak), 'invalid buffer observer')
        super().__init__()
        self.limit = limit
        self.peak = 0
        self.on_peak = on_peak
        if on_peak is not None:
            on_peak(0)

    def write(self, data):
        size = memoryview(data).nbytes
        end = self.tell() + size
        common._need(end <= self.limit, 'serializer buffer ceiling')
        result = super().write(data)
        self.peak = max(self.peak, end)
        if self.on_peak is not None:
            self.on_peak(self.peak)
        return result

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def seek(self, offset, whence=0):
        base = 0 if whence == 0 else self.tell() if whence == 1 else self.peak if whence == 2 else None
        common._need(base is not None and 0 <= base + offset <= self.limit, 'buffer seek ceiling')
        return super().seek(offset, whence)

    def truncate(self, size=None):
        size = self.tell() if size is None else size
        common._need(0 <= size <= self.limit, 'buffer truncate ceiling')
        return super().truncate(size)


def _equal_tree(left, right, torch):
    common._need(type(left) is type(right), 'roundtrip type differs')
    if torch is not None and type(left) is torch.Tensor:
        common._need(left.dtype == right.dtype and left.shape == right.shape
                     and left.stride() == right.stride() and right.device.type == 'cpu'
                     and torch.equal(left.reshape(-1).view(torch.uint8),
                                     right.reshape(-1).view(torch.uint8)), 'roundtrip tensor differs')
    elif type(left) is dict:
        common._need(tuple(left) == tuple(right)
                     and all(type(a) is type(b) for a, b in zip(left, right)),
                     'roundtrip ordered keys differ')
        for key in left:
            _equal_tree(left[key], right[key], torch)
    elif type(left) in (list, tuple):
        common._need(len(left) == len(right), 'roundtrip sequence length differs')
        for a, b in zip(left, right):
            _equal_tree(a, b, torch)
    elif type(left) is float:
        common._need(struct.pack('>d', left) == struct.pack('>d', right), 'roundtrip float bits differ')
    else:
        common._need(left == right, 'roundtrip primitive differs')


def _json_roundtrip(tree, *, body_ceiling, on_buffer_peak=None, on_stage=None):
    """Bounded strict JSON-only counterpart; no Torch or file writes."""
    common._need(type(tree) is dict and tree.get('profile') == common.PROFILE,
                 'unregistered diagnostic root required')
    owned = common.copy_primitive(tree)
    common._need(on_stage is None or callable(on_stage), 'invalid stage observer')
    if on_stage is not None:
        on_stage('serialize')
    with _CappedBuffer(body_ceiling, on_buffer_peak) as buffer:
        for chunk in json.JSONEncoder(ensure_ascii=True, allow_nan=False,
                                      separators=(',', ':')).iterencode(owned):
            buffer.write(chunk.encode('ascii'))
        buffer.write(b'\n')
        raw = buffer.getvalue()
        if on_stage is not None:
            on_stage('roundtrip')
        _equal_tree(owned, json.loads(raw), None)
    return dict(pickle_bytes=None, body_bytes=len(raw), storage_count=0,
        raw_storage_bytes=0, serialized_sha256=hashlib.sha256(raw).hexdigest(),
        restricted_cpu_roundtrip=True, exact_tree_roundtrip=True)


def _serialize_roundtrip(tree, *, pickle_ceiling, body_ceiling, tensor_count,
                         on_buffer_peak=None, on_stage=None):
    """Internal CPU serializer primitive, not a full run/admission entrypoint.

    Caller supplies a reviewed complete tree and external resource supervision.
    Actual tree traversal determines storage encounter order, not the inventory.
    """
    import native_storage_authority as authority
    common._need(type(tree) is dict and tree.get('profile') == common.PROFILE,
                 'unregistered diagnostic root required')
    for value in (pickle_ceiling, body_ceiling):
        common._need(type(value) is int and 0 < value <= 64 << 20, 'invalid serializer ceiling')
    common._need(type(tensor_count) is int and 0 <= tensor_count <= 132, 'invalid tensor count')
    common._need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and
                 all(os.environ.get(k) == v for k, v in authority.THREAD_SETTINGS.items()),
                 'hidden CUDA and four single-thread variables required')
    import torch
    import pickle_storage_bound as pickle_bound
    import zip_storage_bound as zip_bound
    common._need(not torch.cuda.is_initialized(), 'CUDA must remain uninitialized')
    runtime = zip_bound.validate_runtime_save_configuration(require_cuda_uninitialized=True)
    pickle_bound.assert_pinned_pickle_runtime()
    before = torch.get_rng_state().clone()
    arguments = dict(tensor_count_cap=tensor_count,
        allowed_dtypes=pickle_bound.SUPPORTED_DTYPES, max_depth=32,
        max_nodes=200000, max_utf8_bytes=4 << 20, max_tensor_rank=4)
    actual = pickle_bound.bound_protocol2_tree(tree, **arguments)
    common._need(actual.tensor_count == tensor_count and actual.pickle_bytes <= pickle_ceiling,
                 'actual tree exceeds reviewed pickle/count ceiling')
    envelope = zip_bound.torch_save_zip_ceiling(actual.pickle_bytes, actual.storage_nbytes)
    common._need(envelope['archive_bytes_upper'] <= body_ceiling, 'actual ZIP bound exceeds ceiling')
    common._need(on_stage is None or callable(on_stage), 'invalid stage observer')
    if on_stage is not None:
        on_stage('serialize')
    with _CappedBuffer(body_ceiling, on_buffer_peak) as buffer:
        torch.save(tree, buffer, pickle_protocol=2, _use_new_zipfile_serialization=True,
                   _disable_byteorder_record=False)
        body_size = buffer.peak
        with buffer.getbuffer() as view:
            common._need(len(view) == body_size <= envelope['archive_bytes_upper'],
                         'actual body exceeds tree-specific ZIP theorem')
        buffer.seek(0)
        with zipfile.ZipFile(buffer) as archive:
            expected = [('data.pkl', None), *zip_bound.FIXED_RECORDS,
                        *((f'data/{i}', size) for i, size in enumerate(actual.storage_nbytes)),
                        *zip_bound.FINAL_RECORDS]
            infos = archive.infolist()
            common._need([row.filename for row in infos] == ['archive/' + name for name, _ in expected],
                         'ZIP exact record order differs')
            for info, (_, size) in zip(infos, expected):
                common._need(info.compress_type == zipfile.ZIP_STORED and
                             (size is None or info.file_size == size), 'ZIP stored record differs')
            pickle_size = infos[0].file_size
            common._need(pickle_size <= actual.pickle_bytes and archive.testzip() is None,
                         'pickle ceiling or CRC check failed')
        buffer.seek(0)
        if on_stage is not None:
            on_stage('roundtrip')
        loaded = torch.load(buffer, map_location='cpu', weights_only=True)
        loaded_bound = pickle_bound.bound_protocol2_tree(loaded, **arguments)
        common._need(loaded_bound.storage_nbytes == actual.storage_nbytes,
                     'roundtrip storage topology differs')
        _equal_tree(tree, loaded, torch)
        with buffer.getbuffer() as view:
            digest = hashlib.sha256(view).hexdigest()
        common._need(torch.equal(before, torch.get_rng_state()) and not torch.cuda.is_initialized(),
                     'serializer changed CPU RNG/CUDA state')
    return dict(pickle_bytes=pickle_size, body_bytes=body_size,
        storage_count=actual.tensor_count, raw_storage_bytes=sum(actual.storage_nbytes),
        serialized_sha256=digest, restricted_cpu_roundtrip=True, exact_tree_roundtrip=True), runtime


def main(argv=None):
    print('native_storage_crosscheck: inert; no measurement controller or M/A writes')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
