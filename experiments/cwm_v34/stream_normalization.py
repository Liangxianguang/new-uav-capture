"""Fixed TRAIN history-block Welford algorithm; caller must audit full support."""
import numpy as np


def stream_history_normalization(histories):
    count = 0; mean = np.zeros(252, dtype=np.float64); m2 = np.zeros(252, dtype=np.float64)
    for history in histories:
        block = np.asarray(history, dtype=np.float64)
        if block.shape != (8, 252) or not np.isfinite(block).all():
            raise ValueError('Finite original8x252 history blocks required')
        local = block.mean(0); local_m2 = ((block-local)**2).sum(0)
        next_count = count+8; delta = local-mean
        m2 += local_m2+delta**2*(count*8/next_count)
        mean += delta*(8/next_count); count = next_count
    if not count: raise ValueError('Complete nonempty TRAIN history support required')
    scale = np.maximum(np.sqrt(np.maximum(0., m2/count)), .01)
    return mean.reshape(1, 1, 252), scale.reshape(1, 1, 252), count
