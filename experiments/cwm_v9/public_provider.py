"""Lazy default-off public-input provider; failed qualification cannot activate."""
import numpy as np


class PublicResponseProvider:
    def __init__(self, mode="off", loader=None, qualified=False):
        if mode not in ("off", "shadow", "guarded"):
            raise ValueError("Unknown enhancement mode")
        self.mode = mode
        self.status = "off" if mode == "off" else "qualification_failed" if mode == "guarded" and not qualified else "pending"
        self.loader = loader
        self.predictor = None
        self.load_calls = 0
        self.prediction_calls = 0
        self.control_eligible = mode == "guarded" and qualified

    def load(self):
        if self.status in ("off", "qualification_failed", "load_failed", "prediction_failed"):
            return False
        if self.predictor is not None:
            return True
        self.load_calls += 1
        try:
            self.predictor = self.loader()
            if not callable(self.predictor):
                raise ValueError("Loader must return a public forecast function")
            self.status = "shadow_ready" if self.mode == "shadow" else "guarded_ready"
            return True
        except Exception:
            self.status = "load_failed"
            self.predictor = None
            self.control_eligible = False
            return False

    def forecast(self, history, relative, proposed, anchor, backbone):
        # Default-off does not inspect input, load a checkpoint or call the model.
        if not self.load():
            return None
        try:
            h, r, p, a, b = map(np.asarray, (history, relative, proposed, anchor, backbone))
            if h.shape != (8, 252) or r.shape != (4, 6) or p.ndim != 4 or p.shape[1:] != (8, 4, 3) or a.shape != (8, 4, 3) or b.shape != (8, 3):
                raise ValueError("Public input contract mismatch")
            if any(not np.isfinite(x).all() for x in (h, r, p, a, b)):
                raise ValueError("Nonfinite public input")
            self.prediction_calls += 1
            result = self.predictor(h, r, p, a, b)
            if set(result) != {"prediction", "response"}:
                raise ValueError("Forecast output contract mismatch")
            for key in result:
                if result[key].shape != (len(p), 8, 3) or not np.isfinite(result[key]).all():
                    raise ValueError("Invalid forecast output")
            anchor_rows = (p == a[None]).all(axis=(1, 2, 3))
            if (result["response"][anchor_rows] != 0).any():
                raise ValueError("Reference response must be exactly zero")
            if any(row.dtype != b.dtype or row.tobytes() != b.tobytes() for row in result["prediction"][anchor_rows]):
                raise ValueError("Reference forecast must preserve backbone bytes")
            return result
        except Exception:
            self.status = "prediction_failed"
            self.control_eligible = False
            return None


def delayed_joint_context(observation, known, peer_sequences, agent, own_sequences, own_anchor, reference, velocity):
    """Never fill an unknown peer with privileged current plans or zero actions."""
    peers = set(range(4)) - {agent}
    if not peers.issubset(known) or not peers.issubset(peer_sequences):
        return None
    if any(np.asarray(peer_sequences[p]).shape != (8, 3) for p in peers):
        return None
    own = np.asarray(own_sequences, dtype=np.float64)
    if own.ndim != 3 or own.shape[1:] != (8, 3) or np.asarray(own_anchor).shape != (8, 3):
        raise ValueError("Own command shape mismatch")
    joint = np.empty((len(own), 8, 4, 3), dtype=np.float64)
    anchor = np.empty((8, 4, 3), dtype=np.float64)
    positions = np.empty((4, 3), dtype=np.float64)
    velocities = np.empty((4, 3), dtype=np.float64)
    joint[:, :, agent], anchor[:, agent] = own, own_anchor
    positions[agent] = observation["defender_positions"][agent]
    velocities[agent] = observation["defender_velocities"][agent]
    for peer in peers:
        joint[:, :, peer] = peer_sequences[peer]
        anchor[:, peer] = peer_sequences[peer]
        positions[peer], velocities[peer] = known[peer].position, known[peer].velocity
    relative = np.concatenate([positions - reference, velocities - velocity], -1)
    return joint, anchor, relative
