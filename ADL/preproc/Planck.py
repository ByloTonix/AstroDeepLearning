"""
Module for processing Planck data.

This module deals with the pre-processing of HFI data from Planck maps.
These functions transform all-sky maps into HEALPix tiles, according to which the data is split
into training, validation, and test. After that, the samples themselves are generated.

Planck data can be accessed `here
<https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/>`_.
"""
import numpy as np
import os
from typing import List, Dict, Tuple
from astropy.io import fits


def extract_data_key(path: str, key: str, idx: int = 1) -> np.ndarray:
    """From fits file extract data by key.

    :param path: Input path.
    :type path: str
    :param key: Keyword to exract.
    :type key: str
    :rtype: np.ndarray
    """
    with fits.open(path) as hdul:
        return hdul[idx].data[key]


def match_channels(indir: str, channels: List[str]) -> Dict[str, str]:
    """For each channel find corresponding file.

    :param indir: Input dir.
    :type indir: str
    :param channels: channels.
    :type channels: List[str]
    :rtype: Dict[str, str]
    """
    files = sorted(os.listdir(indir))
    files_by_ch = {}
    for channel in channels:
        file = list(filter(lambda x: channel in x, files))[0]
        files_by_ch[channel] = os.path.join(indir, file)
    return files_by_ch


def normalize_asym(
    i_data: np.ndarray,
    p: Tuple[float] = (10**-3, 0.99),
    n_bins: int = 500,
    outlier_thr: float = 10**4,
) -> np.ndarray:
    """Normalize data with asymmetrical distribution.

    (By fitting Gauss curve to left wing of the distribution).

    :param i_data: Data with asymmetrical distribution.
    :type i_data: np.ndarray
    :param p: Probability range for quantile.
    :type p: Tuple[float]
    :param n_bins: Number of bins for histogram.
    :type n_bins: int
    :param outlier_thr: Threshold for finding the outliers.
    :type outlier_thr: float
    :rtype: np.ndarray
    """
    # Narrow down histogram
    q = np.quantile(i_data, p)
    bins = np.arange(*q, (q[1] - q[0]) / n_bins)
    i_cut = i_data[np.where((i_data < q[1]) & (i_data > q[0]))]
    h, _ = np.histogram(i_data, bins)

    # Find the hill
    mean = bins[np.argmax(h)]

    # Left and right wings are asymmetrical, so we mirror the left wing of distribution
    left_wing = i_cut[i_cut < mean]
    left_mirrored = np.concatenate([left_wing, -left_wing + 2 * mean])

    std = np.std(left_mirrored)
    result = (i_data - mean) / std

    # Find deviation from median and scale it
    med = np.median(result)
    dev = np.abs(result - med)
    dev /= np.median(dev)

    # Replace outliers with the closest appropriate value
    idx = (dev > outlier_thr) & (result < med)
    if np.any(idx):
        mask_val = result[~idx].min()
        result[idx] = mask_val
    idx = (dev > outlier_thr) & (result > med)
    if np.any(idx):
        mask_val = result[~idx].max()
        result[idx] = mask_val
    return result
