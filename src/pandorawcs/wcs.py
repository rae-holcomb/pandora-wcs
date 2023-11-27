# imports
import numpy as np
from astropy.io import fits
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS, Sip



def get_wcs(
    detector,
    # detector: Detector,
    target_ra: u.Quantity,
    target_dec: u.Quantity,
    crpix1: int = None,
    crpix2: int = None,
    theta: u.Quantity = 0 * u.deg,
    distortion_file: str = None,
    order: int = 3,
    xreflect: bool = True,
    yreflect: bool = False,    
) -> WCS.wcs:
    """
    Get the World Coordinate System for a detector

    Parameters:
    -----------
    detector : pandorasim.Detector
        The detector to build the WCS for
    target_ra: astropy.units.Quantity
        The target RA in degrees
    target_dec: astropy.units.Quantity
        The target Dec in degrees
    theta: astropy.units.Quantity
        The observatory angle in degrees
    distortion_file: str
        Optional file path to a distortion CSV file. See `read_distortion_file`

    Returns:
    --------
    wcs : astropy.wcs.WCS.wcs
        X pixel positions in undistorted frame, centered around CRPIX1   
    """
    # xreflect = True
    # yreflect = False
    hdu = fits.PrimaryHDU()
    hdu.header["CTYPE1"] = "RA---TAN"
    hdu.header["CTYPE2"] = "DEC--TAN"
    matrix = np.asarray(
        [
            [np.cos(theta).value, -np.sin(theta).value],
            [np.sin(theta).value, np.cos(theta).value],
        ]
    )
    hdu.header["CRVAL1"] = target_ra.value
    hdu.header["CRVAL2"] = target_dec.value
    for idx in range(2):
        for jdx in range(2):
            hdu.header[f"PC{idx+1}_{jdx+1}"] = matrix[idx, jdx]
    hdu.header["CRPIX1"] = (detector.naxis1.value // 2 if crpix1 is None else crpix1)
    hdu.header["CRPIX2"] = (detector.naxis2.value // 2 if crpix2 is None else crpix2)
    hdu.header["NAXIS1"] = detector.naxis1.value
    hdu.header["NAXIS2"] = detector.naxis2.value
    hdu.header["CDELT1"] = detector.pixel_scale.to(u.deg / u.pixel).value * (-1) ** (int(xreflect))
    hdu.header["CDELT2"] = detector.pixel_scale.to(u.deg / u.pixel).value * (-1) ** (int(yreflect))
    
    if distortion_file is not None:
        wcs = _get_distorted_wcs(
            detector, hdu.header, distortion_file, order=order
        )
    else:
        wcs = WCS(hdu.header)
    return wcs

def read_distortion_file(detector: Detector, distortion_file: str):
    """Helper function to read a distortion file.

    This file must be a CSV file that contains a completely "square" grid of pixels
    "Parax X" and "Parax Y", and a corresponding set of distorted pixel positions
    "Real X" and "Real Y". These should be centered CRPIX1 and CRPIX2.

    Parameters:
    -----------
    distortion_file: str
        File path to a distortion CSV file.

    Returns:
    --------
    X : np.ndarray
        X pixel positions in undistorted frame, centered around CRPIX1
    Y : np.ndarray
        Y pixel positions in undistorted frame, centered around CRPIX2
    Xp : np.ndarray
        X pixel positions in distorted frame, centered around CRPIX1
    Yp : np.ndarray
        Y pixel positions in distorted frame, centered around CRPIX2
    """
    raise NotImplementedError
