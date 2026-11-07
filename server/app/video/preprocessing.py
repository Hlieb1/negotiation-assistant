import cv2
import numpy as np


class FrameSampler:
    def __init__(self, max_fps: float, jitter: float = 0.2):
        self.gap_ms = 1000.0 / max_fps
        self.tolerance_ms = self.gap_ms * jitter
        self._next_ts: float | None = None

    def accept(self, ts: int) -> bool:
        if self._next_ts is not None and 0 < self._next_ts - ts <= 2 * self.gap_ms:
            if ts < self._next_ts - self.tolerance_ms:
                return False
        if self._next_ts is None or abs(ts - self._next_ts) > 2 * self.gap_ms:
            self._next_ts = ts + self.gap_ms
        else:
            self._next_ts += self.gap_ms
        return True


def prepare_frame(frame: np.ndarray, max_width: int, equalize: bool) -> np.ndarray:
    height, width = frame.shape[:2]
    if width > max_width:
        frame = cv2.resize(frame, (max_width, round(height * max_width / width)), interpolation=cv2.INTER_AREA)
    if equalize:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return frame
