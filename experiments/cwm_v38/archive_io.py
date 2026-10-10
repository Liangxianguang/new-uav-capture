"""Streaming, exclusive archive I/O; hashes alone are not scientific replay."""
import hashlib
import json
import stat
import zipfile
from pathlib import PurePosixPath

SCHEMA = 'cwm_v38.complete_streamed_prefix_archive.v1'
MANIFEST = 'ARTIFACT_MANIFEST.json'


def sha(path):
    with path.open('rb') as stream:
        return stream_digest(stream)


def stream_digest(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1 << 20), b''):
        digest.update(block)
    return digest.hexdigest()


def safe_name(name):
    if (not isinstance(name, str) or not name or '\\' in name or ':' in name or
        any(ord(c) < 32 for c in name)):
        raise ValueError('Unsafe archive evidence name')
    path = PurePosixPath(name)
    if path.is_absolute() or path.as_posix() != name or any(p in ('', '.', '..') for p in name.split('/')):
        raise ValueError('Noncanonical archive evidence name')
    devices = {'con', 'prn', 'aux', 'nul', *(f'com{i}' for i in range(1, 10)),
               *(f'lpt{i}' for i in range(1, 10))}
    if any(p.endswith((' ', '.')) or p.split('.')[0].lower() in devices or
           any(c in p for c in '<>"|?*') for p in path.parts):
        raise ValueError('Unsafe Windows archive evidence name')
    return name


def regular(root, name):
    safe_name(name)
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('Archive evidence must be a regular file inside its root')
    current = path
    while current != root:
        if current.is_symlink():
            raise ValueError('Archive evidence symlink is not permitted')
        current = current.parent
    return path


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON evidence key')
        result[key] = value
    return result


def decode_json(raw):
    def reject(value):
        raise ValueError('Nonfinite JSON evidence value: ' + value)
    return json.loads(raw, object_pairs_hook=no_duplicates, parse_constant=reject)


def read_json(path):
    return decode_json(path.read_bytes())


def write_json(path, value):
    with path.open('x', encoding='utf8') as out:
        json.dump(value, out, indent=2, allow_nan=False)


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [safe_name(info.filename) for info in infos]
        if MANIFEST not in names or len({n.casefold() for n in names}) != len(names):
            raise ValueError('Missing manifest or duplicate/case-aliased archive member')
        if any(info.is_dir() or stat.S_IFMT(info.external_attr >> 16) not in (0, stat.S_IFREG)
               for info in infos):
            raise ValueError('Nonregular archive member')
        manifest = decode_json(archive.read(MANIFEST))
        if (set(manifest) != {'schema', 'members'} or manifest['schema'] != SCHEMA or
            set(manifest['members']) != set(names) - {MANIFEST}):
            raise ValueError('Exact archive manifest inventory differs')
        for info in infos:
            if info.filename == MANIFEST:
                continue
            row = manifest['members'][info.filename]
            if (set(row) != {'bytes', 'sha256'} or type(row['bytes']) is not int or
                row['bytes'] != info.file_size):
                raise ValueError('Archive evidence length differs')
            with archive.open(info) as stream:
                if stream_digest(stream) != row['sha256']:
                    raise ValueError('Archive evidence digest differs: ' + info.filename)
    return {'sha256': sha(path), 'bytes': path.stat().st_size, 'members': len(names) - 1}


def extract_archive(path, destination):
    if destination.exists():
        raise ValueError('Exclusive NEW extraction directory required')
    before = verify_archive(path)
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            target = destination.joinpath(*PurePosixPath(safe_name(info.filename)).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open('xb') as out:
                for block in iter(lambda: source.read(1 << 20), b''):
                    out.write(block)
    if verify_archive(path) != before:
        raise ValueError('Archive changed during extraction')
    return before


def npz_payloads(path):
    # Compare ALL uncompressed NPY bytes, including dtype/shape/signed zeros;
    # ZIP timestamps and compression are not scientific array differences.
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names) or any(not n.endswith('.npy') for n in names):
            raise ValueError('Noncanonical or duplicate NPZ field')
        result = {}
        for name in names:
            with archive.open(name) as stream:
                result[name] = stream_digest(stream)
        return result
