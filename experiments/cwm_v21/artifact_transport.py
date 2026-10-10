"""Lossless bounded GitHub chunks; full ZIP remains the semantic audit target."""
import hashlib
import json
from pathlib import Path

CHUNK_BYTES = 45 * 1024 * 1024


def file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1 << 20),b''):
            digest.update(block)
    return digest.hexdigest()


def split_archive(archive_path, manifest_path, chunk_bytes=CHUNK_BYTES):
    if not 1 <= chunk_bytes <= CHUNK_BYTES or manifest_path.exists():
        raise ValueError('Exclusive manifest and bounded positive chunk size required')
    manifest_path.parent.mkdir(parents=True,exist_ok=True)
    parts = []
    with archive_path.open('rb') as stream:
        for index,chunk in enumerate(iter(lambda:stream.read(chunk_bytes),b''),1):
            name = manifest_path.name.removesuffix('.parts.json')+f'.part{index:03d}'
            path = manifest_path.parent/name
            with path.open('xb') as destination:
                destination.write(chunk)
            parts.append({'name':name,'bytes':len(chunk),'sha256':hashlib.sha256(chunk).hexdigest()})
    if not parts:
        raise ValueError('Cannot transport an empty archive')
    manifest = {'schema':'cwm_v21.lossless_archive_parts.v1','full_archive_sha256':file_digest(archive_path),
                'full_archive_bytes':archive_path.stat().st_size,'parts':parts}
    with manifest_path.open('x') as destination:
        json.dump(manifest,destination,indent=2)
    return manifest


def resolve_artifact(path, cache):
    if not path.name.endswith('.parts.json'):
        return path
    manifest = json.loads(path.read_text())
    if manifest['schema'] != 'cwm_v21.lossless_archive_parts.v1' or not manifest['parts']:
        raise ValueError('Invalid transport manifest')
    digest = manifest['full_archive_sha256']
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Invalid full artifact identity')
    names = [r['name'] for r in manifest['parts']]
    if len(set(names)) != len(names) or sum(r['bytes'] for r in manifest['parts']) != manifest['full_archive_bytes']:
        raise ValueError('Incomplete or duplicate transport parts')
    for row in manifest['parts']:
        name = row['name']
        if Path(name).name != name or '/' in name or '\\' in name or ':' in name:
            raise ValueError('Unsafe part path')
        part = path.parent/name
        if not 1 <= row['bytes'] <= CHUNK_BYTES or part.stat().st_size != row['bytes'] or file_digest(part) != row['sha256']:
            raise ValueError('Transport part hash/length differs')
    cache.mkdir(parents=True,exist_ok=True)
    joined = cache/(digest+'.zip')
    if not joined.exists():
        with joined.open('xb') as destination:
            for row in manifest['parts']:
                with (path.parent/row['name']).open('rb') as source:
                    for block in iter(lambda:source.read(1 << 20),b''):
                        destination.write(block)
    if joined.stat().st_size != manifest['full_archive_bytes'] or file_digest(joined) != digest:
        raise ValueError('Joined artifact hash/length differs')
    return joined
