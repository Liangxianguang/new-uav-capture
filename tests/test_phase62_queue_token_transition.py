from __future__ import annotations

import numpy as np
import pytest

from encirclement3d.execution_dynamics import (
    QueueToken,
    apply_command_authority_with_token,
    commit_delayed_command_with_token,
)


def _queue(length: int) -> list[np.ndarray]:
    return [np.full((2, 3), float(index + 1)) for index in range(length)]


@pytest.mark.parametrize("mode", ["immutable", "replace_nonexecuting", "flush_pending"])
def test_stale_queue_token_is_rejected_without_mutation(mode: str) -> None:
    queue = _queue(4)
    current = QueueToken(generation=3, issued_step=10, pending_length=4)
    stale = QueueToken(generation=2, issued_step=10, pending_length=4)

    updated, token, ack = apply_command_authority_with_token(
        queue,
        {
            "mode": mode,
            "emergency_brake": True,
            "expected_queue_token": stale.as_dict(),
            "max_override_slots": 1,
        },
        allowed_mode=mode,
        current_token=current,
    )

    assert not ack.accepted
    assert ack.reason == "stale_queue_token"
    assert token == current
    assert all(np.array_equal(left, right) for left, right in zip(updated, queue))


def test_replace_nonexecuting_is_bounded_and_preserves_due_slot() -> None:
    queue = _queue(4)
    current = QueueToken(generation=3, issued_step=10, pending_length=4)
    updated, token, ack = apply_command_authority_with_token(
        queue,
        {
            "mode": "replace_nonexecuting",
            "emergency_brake": True,
            "expected_queue_token": current,
            "max_override_slots": 2,
        },
        allowed_mode="replace_nonexecuting",
        current_token=current,
    )

    assert ack.accepted and ack.applied
    assert ack.overridden_slots == 2
    assert np.array_equal(updated[0], queue[0])
    assert np.array_equal(updated[1], np.zeros_like(updated[1]))
    assert np.array_equal(updated[2], np.zeros_like(updated[2]))
    assert np.array_equal(updated[3], queue[3])
    assert token.generation == current.generation + 1


@pytest.mark.parametrize("queue_length", [0, 1, 4])
def test_delayed_commit_pops_exactly_one_command(queue_length: int) -> None:
    queue = _queue(queue_length)
    current = QueueToken(generation=3, issued_step=10, pending_length=queue_length)
    new_action = np.full((2, 3), 99.0)
    updated, due, next_token = commit_delayed_command_with_token(
        queue,
        new_action,
        current,
        next_step=11,
    )

    expected_due = new_action if queue_length == 0 else queue[0]
    assert np.array_equal(due, expected_due)
    assert len(updated) == (queue_length if queue_length > 0 else 0)
    assert next_token == QueueToken(
        generation=4,
        issued_step=11,
        pending_length=len(updated),
    )
