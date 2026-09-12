from __future__ import annotations

import numpy as np

from encirclement3d.uncertainty_calibration import MonotoneRiskCalibration


def test_monotone_calibration_pools_non_monotonic_empirical_rates() -> None:
    calibration = MonotoneRiskCalibration.fit(
        np.asarray([0.1, 0.2, 0.3, 0.4]),
        np.asarray([0.0, 1.0, 0.0, 1.0]),
    )

    assert np.all(np.diff(calibration.risk_values) >= 0.0)
    np.testing.assert_allclose(calibration.predict(np.asarray([0.1, 0.4])), [0.0, 1.0])
    assert calibration.threshold_for_risk(0.5) == 0.2


def test_monotone_calibration_round_trips_mapping() -> None:
    calibration = MonotoneRiskCalibration((0.1, 0.5), (0.0, 0.25), label="timeout")
    restored = MonotoneRiskCalibration.from_mapping(calibration.as_dict())
    assert restored == calibration
