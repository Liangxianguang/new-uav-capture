"""Verify an explicit local native report and its publication wrapper.

No training, environment execution or qualification is performed. File hash
and ALL typed JSON fields must match; integer0, float0.0 and false differ.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'cwm_v28'))
from repeatability_identity import digest, exact_tree


def verify(wrapper_path, native_path):
    wrapper = json.loads(wrapper_path.read_bytes())
    native = json.loads(native_path.read_bytes())
    claimed = wrapper['native_summary_sha256']
    if (wrapper['native_payload_semantic_copy_not_same_file_bytes'] is not True or
        not isinstance(claimed, str) or len(claimed) != 64 or
        any(c not in '0123456789abcdef' for c in claimed)):
        raise ValueError('Explicit semantic-copy identity contract required')
    if digest(native_path) != claimed:
        raise ValueError('Actual native report file SHA differs')
    if not exact_tree(wrapper['native_summary'], native):
        raise ValueError('Complete typed native report semantics differ')
    return {'status': 'native_report_semantic_copy_verified_not_scientific_qualification',
        'native_summary_sha256': claimed, 'all_typed_native_fields_equal': True,
        'native_reexecuted_by_this_checker': False,
        'source_file_rewritten': False, 'publication_wrapper_rewritten': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wrapper', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.wrapper, args.native), indent=2))
