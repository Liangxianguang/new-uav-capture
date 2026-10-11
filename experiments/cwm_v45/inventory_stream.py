"""Disk-backed full V42 inventory; no model, physics, sampling or timing change."""
import json
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'cwm_v38'))
from archive_io import regular, safe_name, sha, read_json


class DiskInventory:
    """At most one file record in Python; SQLite caches bounded to4MiB.

    The ledger is retained operational evidence, not part of the scientific
    manifest. Canonical manifest bytes equal the old sorted JSON object.
    """
    def __init__(self, root):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError('Existing independently owned native output required')
        self.path = self.root/'evidence_manifest_index.sqlite3'
        with self.path.open('xb'):
            pass
        self.connection = None
        self.count = 0
        self.sealed = False
        try:
            self.connection = sqlite3.connect(str(self.path))
            self.connection.execute('PRAGMA cache_size=-4096')
            self.connection.execute('PRAGMA temp_store=FILE')
            self.connection.execute('PRAGMA synchronous=FULL')
            self.connection.execute('CREATE TABLE evidence (name TEXT PRIMARY KEY, folded TEXT UNIQUE NOT NULL, '
                'size INTEGER NOT NULL CHECK(size>=0), digest TEXT NOT NULL)')
            self.connection.commit()
            self.connection.execute('BEGIN IMMEDIATE')
        except Exception:
            self.close()
            raise

    def add(self, name, expected=None):
        if self.connection is None or self.sealed:
            raise ValueError('Inventory is closed or already sealed')
        name = safe_name(name)
        file = regular(self.root, name)
        size, digest = file.stat().st_size, sha(file)
        if expected is not None and digest != expected:
            raise ValueError('Saved native evidence binding differs: '+name)
        old = self.connection.execute('SELECT name,size,digest FROM evidence WHERE folded=?',
            (name.casefold(),)).fetchone()
        if old is not None:
            if old != (name, size, digest):
                raise ValueError('Aliased or mutated complete inventory evidence: '+name)
            return
        self.connection.execute('INSERT INTO evidence VALUES (?,?,?,?)',
            (name, name.casefold(), size, digest))
        self.count += 1

    def seal(self):
        if self.connection is None or self.sealed:
            raise ValueError('Inventory already closed/sealed')
        self.connection.commit()
        if self.connection.execute('PRAGMA quick_check').fetchall() != [('ok',)]:
            raise ValueError('Complete disk inventory integrity check failed')
        actual = self.connection.execute('SELECT COUNT(*) FROM evidence').fetchone()[0]
        if actual != self.count:
            raise ValueError('Complete disk inventory member count differs')
        self.sealed = True

    def items(self):
        if self.connection is None or not self.sealed:
            raise ValueError('Complete sealed inventory required')
        for name, size, digest in self.connection.execute(
                'SELECT name,size,digest FROM evidence ORDER BY name COLLATE BINARY'):
            yield name, {'bytes': size, 'sha256': digest}

    def write(self, path):
        path = Path(path)
        if path.resolve() != self.root/'evidence_manifest.json':
            raise ValueError('Only the fixed native inventory output may be serialized')
        if self.connection is None or not self.sealed:
            raise ValueError('Complete sealed inventory required before serialization')
        with path.open('x', encoding='utf8') as target:
            if not self.count:
                target.write('{}')
                return
            target.write('{\n')
            first = True
            for name, value in self.items():
                if not first:
                    target.write(',\n')
                # Exact original indent2/ensure_ascii/field-order bytes per member.
                target.write(json.dumps({name: value}, indent=2, allow_nan=False)[2:-2])
                first = False
            target.write('\n}')

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None


def build_inventory(output, rows, sources):
    """Same ALL-file membership as V42; never a sampled or filtered inventory."""
    ledger = DiskInventory(output)
    try:
        for name in ('preregistration.json', 'prefix_certificate.json', 'legacy_certificate.json',
                     'scenes.json', 'benchmark_preload.json'):
            ledger.add(name)
        for name in sources:
            ledger.add('source/'+safe_name(name))
        for cohort, population in rows.items():
            for name in ('episodes.jsonl', 'steps.jsonl'):
                ledger.add(safe_name(cohort)+'/'+name)
            for row in population:
                for field in ('trajectory', 'commands', 'events'):
                    ledger.add(row[field+'_path'], row.get(field+'_sha256'))
                folder = Path(row['events_path']).parent
                for event in read_json(regular(ledger.root, row['events_path'])):
                    for field in ('arrays', 'context'):
                        if field+'_path' in event:
                            # Validate before joining: ../ and absolute references
                            # cannot silently normalize into a different member.
                            name = safe_name(event[field+'_path'])
                            relative = (folder/name).as_posix()
                            ledger.add(relative, event[field+'_sha256'])
        ledger.seal()
        return ledger
    except Exception:
        ledger.close()
        raise
