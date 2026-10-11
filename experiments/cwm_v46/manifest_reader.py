"""Read every sorted native inventory record without loading its JSON object.

Integrity primitive only: no native execution/replay, model qualification,
scene-support check, statistics, fault test, holdout or deployment permission.
The caller must derive complete expected membership independently.
"""
import json
import sys
from itertools import zip_longest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'cwm_v38'))
from archive_io import no_duplicates, regular, safe_name, sha


def reject_constant(value):
    raise ValueError('Nonfinite manifest value: '+value)


class Reader:
    def __init__(self, stream, chunk_size=65536, max_token=1048576):
        if type(chunk_size) is not int or type(max_token) is not int or not 1 <= chunk_size <= max_token:
            raise ValueError('Positive bounded parser buffers required')
        self.stream, self.chunk_size, self.max_token = stream, chunk_size, max_token
        self.buffer, self.eof = '', False
        self.decoder = json.JSONDecoder(object_pairs_hook=no_duplicates, parse_constant=reject_constant)

    def fill(self):
        if self.eof:
            return False
        text = self.stream.read(self.chunk_size)
        self.eof = not text
        self.buffer += text
        return bool(text)

    def space(self):
        while True:
            self.buffer = self.buffer.lstrip(' \t\r\n')
            if self.buffer or not self.fill():
                return

    def take(self, character):
        self.space()
        if not self.buffer.startswith(character):
            raise ValueError('Malformed inventory delimiter: expected '+character)
        self.buffer = self.buffer[1:]

    def value(self):
        self.space()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer)
            except json.JSONDecodeError as error:
                if len(self.buffer) >= self.max_token or not self.fill():
                    raise ValueError('Malformed or oversized inventory token') from error
                continue
            if end > self.max_token:
                raise ValueError('Oversized inventory token')
            self.buffer = self.buffer[end:]
            return value


def valid_record(name, record):
    safe_name(name)
    if (type(record) is not dict or set(record) != {'bytes', 'sha256'} or
        type(record['bytes']) is not int or record['bytes'] < 0 or
        type(record['sha256']) is not str or len(record['sha256']) != 64 or
        any(c not in '0123456789abcdef' for c in record['sha256'])):
        raise ValueError('Exact length and lowercase SHA256 record required')


def iter_manifest(path, *, chunk_size=65536, max_token=1048576):
    """Consume fully to validate closure/trailing bytes; partial iteration is not an audit.

    Native V42/V45 inventory order is lexical, enabling duplicate rejection with
    one previous key. The token limit bounds individual metadata, not total file
    size or total process memory. No scientific member count is imposed here.
    """
    with Path(path).open('r', encoding='utf8') as stream:
        reader = Reader(stream, chunk_size, max_token)
        reader.take('{'); reader.space()
        previous = None
        if not reader.buffer.startswith('}'):
            while True:
                name = reader.value(); reader.take(':'); record = reader.value()
                valid_record(name, record)
                if previous is not None and name <= previous:
                    raise ValueError('Native inventory must be sorted and duplicate-free')
                previous = name
                yield name, record
                reader.space()
                if reader.buffer.startswith('}'):
                    break
                reader.take(',')
        reader.take('}'); reader.space()
        if reader.buffer:
            raise ValueError('Trailing inventory content')


def verify_members(root, manifest, expected):
    """All records must match an independently derived complete sorted iterator.

    Every member is opened/hashed. Never sample, filter missing members, or accept
    a hash-only inventory as proof of physical/native policy replay. This returns
    only an integrity count; it cannot authorize any scientific stage.
    """
    root = Path(root).resolve()
    manifest = Path(manifest)
    if manifest.resolve() != regular(root, 'evidence_manifest.json').resolve():
        raise ValueError('Fixed native evidence manifest inside its root required')
    before = sha(manifest)
    count = 0
    sentinel = object()
    for actual, assigned in zip_longest(iter_manifest(manifest), expected, fillvalue=sentinel):
        if actual is sentinel or assigned is sentinel:
            raise ValueError('Missing or extra scientific inventory membership')
        if type(assigned) is not tuple or len(assigned) != 2:
            raise ValueError('Independently derived (name, record) tuple required')
        name, record = actual
        valid_record(*assigned)
        if name != assigned[0] or record != assigned[1]:
            raise ValueError('Complete expected inventory differs: '+name)
        file = regular(root, name)
        if file.stat().st_size != record['bytes'] or sha(file) != record['sha256']:
            raise ValueError('Scientific evidence file changed: '+name)
        count += 1
    if sha(manifest) != before:
        raise ValueError('Inventory changed during complete verification')
    return count
