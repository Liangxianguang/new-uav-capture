"""Unchanged V42 ALL10752 policy episodes with disk-backed final inventory.

Only the post-rollout manifest callbacks are scoped replacements. All native
qualification, models, scenes, policies, deadlines and timing remain V42's.
No holdout, deployment, independent replay or scientific success from tests.
"""
import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent/'cwm_v42')]
import paired_prefix as native
from inventory_stream import DiskInventory, build_inventory

ORIGINAL_INVENTORY = native.save_evidence_manifest
ORIGINAL_WRITER = native.write_json


@contextmanager
def post_rollout_inventory():
    if (native.save_evidence_manifest is not ORIGINAL_INVENTORY or
        native.write_json is not ORIGINAL_WRITER):
        raise ValueError('No nested or previously replaced native manifest callbacks')
    opened = []
    def inventory(output, rows, sources):
        if opened:
            raise ValueError('Exactly one complete native inventory expected')
        ledger = build_inventory(output, rows, sources)
        opened.append(ledger)
        return ledger
    def writer(path, value):
        if isinstance(value, DiskInventory):
            if not opened or value is not opened[0]:
                raise ValueError('Only this native run owns the disk inventory')
            return value.write(path)
        return ORIGINAL_WRITER(path, value)
    native.save_evidence_manifest, native.write_json = inventory, writer
    try:
        yield
    finally:
        native.save_evidence_manifest, native.write_json = ORIGINAL_INVENTORY, ORIGINAL_WRITER
        for ledger in opened:
            ledger.close()


def run(prefix_artifact, prefix_certificate, legacy_artifact, legacy_certificate, output):
    with post_rollout_inventory():
        # V42 publishes/verifies actual imported V45 source before any native
        # outputs. It owns ALL qualification/scenes/controllers/timing/gates.
        return native.run(prefix_artifact, prefix_certificate, legacy_artifact, legacy_certificate, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prefix-artifact', 'prefix-certificate', 'legacy-artifact', 'legacy-certificate', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    native.torch.set_num_threads(1)
    native.torch.set_num_interop_threads(1)
    native.torch.use_deterministic_algorithms(True)
    existed = args.output.exists()
    try:
        run(args.prefix_artifact, args.prefix_certificate, args.legacy_artifact, args.legacy_certificate, args.output)
    except Exception as error:
        if not existed and args.output.is_dir() and not (args.output/'summary.json').exists():
            ORIGINAL_WRITER(args.output/'implementation_failure.json', {
                'status': 'incomplete_prefix_streamed_inventory_development_no_qualification',
                'exception_type': type(error).__name__, 'exception': str(error)[:300],
                'enhanced_control_enabled': False, 'holdout_used': False,
                'holdout_authorized_by_this_report': False})
        raise


if __name__ == '__main__':
    main()
