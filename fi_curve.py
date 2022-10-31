from typing import List
import numpy as np
from scipy.signal import find_peaks

def fi_curve(
    v_list: List[float], 
    threshold: float,
) -> int:
    v_np = np.array(v_list)
    peaks, _ = find_peaks(v_np, height=threshold)
    return peaks.size