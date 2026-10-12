import numpy as np

INPUT_LENGTH = 32
MAX_WINDOWS = 8192
BLOCK_SIZE = 256


def signal_value(raw_x, parent_x, changed_x, parent_metrics):
    arrays = tuple(np.asarray(x) for x in (raw_x, parent_x, changed_x))
    shape = arrays[0].shape
    if len(shape) != 3 or shape[1] != INPUT_LENGTH:
        raise ValueError("Expected [N, 32, C]")
    if not (1 <= shape[0] <= MAX_WINDOWS) or shape[2] < 1:
        raise ValueError("Invalid sample shape")
    for x in arrays:
        if x.shape != shape or x.dtype != np.float32:
            raise ValueError("Inputs must share shape and have dtype float32")
    error = float(parent_metrics["NMSE"])
    if not np.isfinite(error) or error < 0.0:
        raise ValueError("Parent NMSE must be finite and nonnegative")
    shift_energy = 0.0
    for lo in range(0, shape[0], BLOCK_SIZE):
        hi = min(lo + BLOCK_SIZE, shape[0])
        blocks = tuple(x[lo:hi] for x in arrays)
        if not all(np.all(np.isfinite(x)) for x in blocks):
            raise ValueError("Non-finite input")
        difference = (
            blocks[2].astype(np.float64)
            - blocks[1].astype(np.float64)
        )
        mean_shift = np.mean(difference, axis=1, dtype=np.float64)
        shift_energy += float(
            np.sum(mean_shift * mean_shift, dtype=np.float64)
        )
    dose = shift_energy / float(shape[0] * shape[2])
    if dose == 0.0:
        return 0.0
    if error == 0.0:
        return 1.0
    if dose <= error:
        ratio = dose / error
        return float(ratio / (1.0 + ratio))
    ratio = error / dose
    return float(1.0 / (1.0 + ratio))
