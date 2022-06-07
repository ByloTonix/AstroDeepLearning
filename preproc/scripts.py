import os
import healpy as hp
import numpy as np
from tqdm import tqdm
from . import match_channels, fits2df, normalize_asym, one_pixel_fragmentation


def preproc_HFI_Planck(inpath: str, outpath: str) -> None:
    files_by_ch = match_channels(inpath, [100, 143, 217, 353, 545, 857])
    data_by_ch = {ch: fits2df(os.path.join(inpath, file), "I_STOKES")
                  for ch, file in files_by_ch.items()}
    data_by_ch[100] = normalize_asym(data_by_ch[100])
    data_by_ch[143] = normalize_asym(data_by_ch[143])
    data_by_ch[217] = normalize_asym(data_by_ch[217])
    data_by_ch[353] = normalize_asym(data_by_ch[353], p=(10**-4, 0.99))
    data_by_ch[545] = normalize_asym(data_by_ch[545], p=(10**-5, 0.9))
    data_by_ch[857] = normalize_asym(data_by_ch[857], p=(10**-5, 0.9))

    for ipix in tqdm(range(hp.nside2npix(2))):
        pix_matr = one_pixel_fragmentation(2, ipix, 2**11)
        img = np.zeros(pix_matr.shape + (6,), dtype=np.float64)
        for i in range(pix_matr.shape[0]):
            for ch_idx, ch in enumerate(data_by_ch):
                data = data_by_ch[ch]
                img[i, :, ch_idx] = data[pix_matr[i]]
        np.save(os.path.join(outpath, '{}.npy'.format(ipix)), img)
    return
