"""Module for HEALPix functions."""
import numpy as np
import pandas as pd
import os
from astropy.coordinates import SkyCoord
from astropy import units as u
import healpy as hp
from typing import Union, List, Tuple, Dict
from tqdm import tqdm
import ADL.other.metr


def recursive_fill(matr: np.ndarray) -> None:
    """Fill matrix recursively to generate correspondence matrix for HEALPix nested scheme.

    :param matr: Input matrix.
    :type matr: np.ndarray
    :rtype: None
    """
    if matr.shape[0] == 1:
        return

    mid = matr.shape[0] // 2
    np.left_shift(matr, 1, out=matr)
    matr[mid:, :] += 1

    np.left_shift(matr, 1, out=matr)
    matr[:, mid:] += 1

    for i in [0, mid]:
        for j in [0, mid]:
            recursive_fill(matr[i:i+mid, j:j+mid])


def one_pixel_fragmentation(o_nside: int, o_pix: int, f_nside: int) -> np.ndarray:
    """Get correspondence matrix between two nsides for chosen pixel. Nested scheme HEALPix.

    :param o_nside: Larger partition nside.
    :type o_nside: int
    :param o_pix: Pixel index for larger nside.
    :type o_pix: int
    :param f_nside: Smaller partition nside.
    :type f_nside: int
    :rtype: np.ndarray
    """
    depth = int(np.log2(f_nside / o_nside))
    m_len = 2 ** depth
    f_matr = np.full((m_len, m_len), o_pix)
    recursive_fill(f_matr)
    return f_matr


def radec2pix(ra: float, dec: float, nside: int, nest: bool = True) -> np.ndarray:
    """Transform RA, Dec coordinates into HEALPix pixel index.

    :param ra: RA value.
    :type ra: float
    :param dec: Dec value.
    :type dec: float
    :param nside: nside parameter for HEALPix.
    :type nside: int
    :param nest: flag for nested scheme.
    :type nest: bool
    :rtype: np.ndarray
    """
    sc = SkyCoord(ra=np.array(ra)*u.degree, dec=np.array(dec)*u.degree, frame='icrs')
    return hp.ang2pix(nside, sc.galactic.l.degree, sc.galactic.b.degree,
                      nest=nest, lonlat=True)


def pix2radec(ipix: int, nside: int, nest: bool = True) -> Tuple[Union[float, List[float]]]:
    """Transform HEALPix pixel index into RA, Dec coordinates.

    :param ipix: Index of pixel.
    :type ipix: int
    :param nside: nside parameter for HEALPix.
    :type nside: int
    :param nest: flag for nested scheme.
    :type nest: bool
    :rtype: Tuple[Optional[float, List[float]]]
    """
    theta, phi = hp.pix2ang(nside, ipix=np.array(ipix), nest=nest, lonlat=True)

    sc = SkyCoord(l=theta*u.degree, b=phi*u.degree, frame='galactic')
    return sc.icrs.ra.degree, sc.icrs.dec.degree


def flat_arr2matr(h_arr: np.ndarray, pix_matr: np.ndarray) -> np.ndarray:
    """Transform flat HEALPix array into 2d matrix with given correspondence matrix.

    Correspondence matrix should be
    created with one_pixel_fragmentation  method.

    :param h_arr: Flat HEALPix array.
    :type h_arr: np.ndarray
    :param pix_matr: Correspondence matrix.
    :type pix_matr: np.ndarray
    :rtype: np.ndarray
    """
    img = np.zeros_like(pix_matr, dtype=h_arr.dtype)
    for i in range(pix_matr.shape[0]):
        img[i] = h_arr[pix_matr[i]]
    return img


def draw_circles(ras: np.ndarray, decs: np.ndarray, radiuses: Union[np.ndarray, float], nside: int,
                 pix_matr: np.ndarray, centers_in_patch: bool = False) -> np.ndarray:
    """For each pair of RA, Dec coordinates, draw circle with given radius in HEALPix projection.

    :param ras: RA coordinates.
    :type ras: np.ndarray
    :param decs: Dec coordinates.
    :type decs: np.ndarray
    :param radiuses: Radiuses in degrees.
    :type radiuses: Union[np.ndarray, float]
    :param nside: nside for HEALPix.
    :type nside: int
    :param pix_matr: Correspondence matrix for HEALPix.
    :type pix_matr: np.ndarray
    :param centers_in_patch: If this flag is true, circles will only be drawn if their radiuses are
        inside given correspondence matrix.
    :type centers_in_patch: bool
    :rtype: np.ndarray
    """
    h_arr = np.zeros(hp.nside2npix(nside), dtype=np.int8)
    if type(radiuses) != np.ndarray:
        radiuses = [radiuses] * len(ras)
    for ra, dec, radius in zip(ras, decs, radiuses):
        sc = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
        vec = hp.ang2vec(theta=sc.galactic.l.degree, phi=sc.galactic.b.degree, lonlat=True)
        if centers_in_patch:
            cl_pix = hp.vec2pix(nside, *vec, nest=True)
            if not (cl_pix in pix_matr):
                continue
        pix = hp.query_disc(nside, vec, np.radians(radius), nest=True, inclusive=True)
        h_arr[pix] = 1
    pic = flat_arr2matr(h_arr, pix_matr)
    return pic


def draw_dots(ras: np.ndarray, decs: np.ndarray, nside: int, pix_matr: np.ndarray) -> np.ndarray:
    """For each pair of RA, Dec coordinates, draw dot in HEALPix projection.

    :param ras: RA coordinates.
    :type ras: np.ndarray
    :param decs: Dec coordinates.
    :type decs: np.ndarray
    :param nside: nside for HEALPix.
    :type nside: int
    :param pix_matr: Correspondence matrix for HEALPix.
    :type pix_matr: np.ndarray
    :rtype: np.ndarray
    """
    h_arr = np.zeros(hp.nside2npix(nside), dtype=np.int8)
    h_arr[radec2pix(ras, decs, nside)] = 1
    pic = flat_arr2matr(h_arr, pix_matr)
    return pic


def src_on_batch(patch_line: Dict, f_matr: np.ndarray, cats: Dict[str, pd.DataFrame]
                 ) -> Dict[str, pd.DataFrame]:
    """Return cats with objects that are visible in selected patch.

    Also generate x, y coords (within patch) for each object.

    :param patch_line: Line of patch in table.
    :type patch_line: Dict
    :param f_matr: HEALPix correspondence matrix.
    :type f_matr: np.ndarray
    :param cats: Catalogs.
    :type cats: Dict[str, pd.DataFrame]
    :rtype: Dict[str, pd.DataFrame]
    """
    output = {}
    x, y = patch_line["x"], patch_line["y"]
    f_matr = f_matr[x:x+64, y:y+64].copy()
    for name, cat in cats.items():
        cur_cat = cat[cat["pix2"] == patch_line["pix2"]]
        cur_cat = cur_cat[np.in1d(cur_cat["pix11"], f_matr.flatten())]
        cur_cat.index = np.arange(len(cur_cat))
        cur_cat["x"] = 0
        cur_cat["y"] = 0
        for i in range(len(cur_cat)):
            pix = cur_cat.loc[i, "pix11"]
            coords = np.where(f_matr == pix)
            cur_cat.loc[i, "x"] = coords[0][0]
            cur_cat.loc[i, "y"] = coords[1][0]
        output[name] = cur_cat
    return output


def generate_patch_coords(cats_path: str, step: int = 20, o_nside: int = 2, nside: int = 2**11,
                          radius: float = 1.83, patch_size: int = 64,
                          n_patches: int = None, cats_subset: List[str] = None) -> pd.DataFrame:
    """Create list of dots from which patches can be generated.

    Each patch will contain at least one object from chosen catalogs.

    :param cats_path: Directory with catalogs.
    :type cats_path: str
    :param step: Step for coordinates (to lessen the size of output table).
    :type step: int
    :param o_nside: Original nside.
    :type o_nside: int
    :param nside: Final nside.
    :type nside: int
    :param radius: Radius of area for patches.
    :type radius: float
    :param patch_size: Size of a patch.
    :type patch_size: int
    :param n_patches: Approximate amount of patches to generate. Overrides step parameter.
    :type n_patches: int
    :param cats_subset: Subset for cats.
    :type cats_subset: List[str]
    :rtype: pd.DataFrame
    """
    if n_patches is not None:
        step = 1
    cats = ADL.other.metr.cats2dict(cats_path)
    if cats_subset is not None:
        cats = {key: val for key, val in cats.items() if key in cats_subset}

    df = pd.concat(cats.values())
    all_idx = {"x": [], "y": [], "pix2": [], "n_src": []}
    for i in tqdm(range(hp.nside2npix(2))):
        pix_matr = one_pixel_fragmentation(o_nside, i, nside)
        pic = draw_dots(df["RA"], df["DEC"], nside=nside, pix_matr=pix_matr)
        xs, ys = [], []
        n_src = []
        for x in range(0, 1024 - 64, step):
            for y in range(0, 1024 - 64, step):
                patch_pic = pic[x:x+patch_size, y:y+patch_size]
                if patch_pic.any():
                    n_src.append(np.count_nonzero(patch_pic))
                    xs.append(x)
                    ys.append(y)

        all_idx["x"].extend(xs)
        all_idx["y"].extend(ys)
        all_idx["pix2"].extend([i] * len(xs))
        all_idx["n_src"].extend(n_src)

    all_idx = pd.DataFrame(all_idx, index=np.arange(len(all_idx["x"])))

    if n_patches is not None and len(all_idx) > n_patches:
        step = len(all_idx) // n_patches
        all_idx = all_idx.loc[::step]
        all_idx.index = np.arange(len(all_idx))
    return all_idx


def draw_masks_and_save(cats_path: str, outpath: str, o_nside: int = 2, nside: int = 2**11,
                        radius: float = 5/60) -> None:
    """Draw masks for training.

    :param cats_path: Directory with catalogs.
    :type cats_path: str
    :param outpath: Path to save masks.
    :type outpath: str
    :param o_nside: Original nside.
    :type o_nside: int
    :param nside: Final nside.
    :type nside: int
    :param radius: Radius of masks.
    :type radius: float
    :rtype: None
    """
    cats = ADL.other.metr.cats2dict(cats_path)
    df = pd.concat(cats.values())
    for i in tqdm(range(hp.nside2npix(o_nside))):
        pix_matr = one_pixel_fragmentation(o_nside, i, nside)
        pic = draw_circles(df["RA"], df["DEC"], radiuses=radius, nside=nside, pix_matr=pix_matr)
        pic = pic.reshape(pic.shape + (1,))
        np.save(os.path.join(outpath, f"{i}.npy"), pic)


def update_old_dataset(path: str, cats_subset: List[str], patch_size: int = 64, o_nside: int = 2,
                       nside: int = 2**11) -> None:
    """Add n_src to old dataset.

    :param path: Path to dataset.
    :type path: str
    :param cats_subset: Subset of catalogs.
    :type cats_subset: List[str]
    :param patch_size: Size of patch.
    :type patch_size: int
    :param o_nside: Original nside.
    :type o_nside: int
    :param nside: Final nside.
    :type nside: int
    :rtype: None
    """
    cats = ADL.other.metr.cats2dict(os.path.join(path, "cats"))
    if cats_subset is not None:
        cats = {key: val for key, val in cats.items() if key in cats_subset}

    df = pd.concat(cats.values())
    patch_coords = pd.read_csv(os.path.join(path, "pc.csv"))
    patch_coords["n_src"] = None
    for i in tqdm(range(hp.nside2npix(2))):
        cur_patch_coords = patch_coords[patch_coords["pix2"] == i]
        pix_matr = one_pixel_fragmentation(o_nside, i, nside)
        pic = draw_dots(df["RA"], df["DEC"], nside=nside, pix_matr=pix_matr)
        n_src = []
        for x, y in zip(cur_patch_coords["x"], cur_patch_coords["y"]):
            patch_pic = pic[x:x+patch_size, y:y+patch_size]
            if patch_pic.any():
                n_src.append(np.count_nonzero(patch_pic))
        patch_coords.loc[patch_coords["pix2"] == i, "n_src"] = n_src

    patch_coords.to_csv(os.path.join(path, "pc.csv"), index=False)
