"""
Contains the detector object. This is a placeholder for development, in the future the package should instead operate with the detector object from pandora-sat.
"""
import numpy as np
import astropy.units as u
from dataclasses import dataclass

@dataclass
class TessDetector:
    """Holds information on a toy version of the TESS detector.

    Attributes
    ----------
    name: str
        Name of the detector. This will determine which files are loaded. This
        will be `"visda"` for this detector
    pixel_scale: float
        The pixel scale of the detector in arcseconds/pixel
    pixel_size: float
        The pixel size in microns/mm
    """

    def __init__(self, shape=(20,20)):
        """Some detector specific functions to run on initialization"""
        # self.shape = (2048, 2048)
        self.shape = shape # make property


    @property
    def pixel_scale(self):
        """Pixel scale of the detector"""
        return 21 * u.arcsec / u.pixel

    @property
    def pixel_size(self):
        """Size of a pixel"""
        return 6.5 * u.um / u.pixel

    @property
    def naxis1(self):
        """WCS's are COLUMN major, so naxis1 is the number of columns"""
        return self.shape[1] * u.pixel

    @property
    def naxis2(self):
        """WCS's are COLUMN major, so naxis2 is the number of rows"""
        return self.shape[0] * u.pixel

    def flux_to_mag(self, flux, reference_mag=20.44):
        """Converts a TESS magnitude to a flux in e-/s. The TESS reference magnitude is taken to be 20.44. If needed, the Kepler reference flux is 1.74e5 electrons/sec.
        
        Parameters
        ----------
        flux : float
            The total flux of the target on the CCD in electrons/sec.
        reference_mag: int
            The zeropoint reference magnitude for TESS. Typically 20.44 +/-0.05.
        reference_mag: float

        Returns
        -------
        Tmag: float
            TESS magnitude of the target.
        
        """
        # kepler_mag = 12 - 2.5 * np.log10(flux / reference_flux)
        mag = -2.5 * np.log10(flux) + reference_mag
        return mag


    def mag_to_flux(self, Tmag, reference_mag=20.44):
        """Converts a TESS magnitude to a flux in e-/s. The TESS reference magnitude is taken to be 20.44. If needed, the Kepler reference flux is 1.74e5 electrons/sec.
        
        Parameters
        ----------
        Tmag: float
            TESS magnitude of the target.
        reference_mag: int
            The zeropoint reference magnitude for TESS. Typically 20.44 +/-0.05.

        Returns
        -------
        flux : float
            The total flux of the target on the CCD in electrons/sec.
        """
        # fkep = (10.0 ** (-0.4 * (mag - 12.0))) * 
        return 10 ** (-(Tmag - reference_mag)/2.5)


# toy tess detector
tessda = TessDetector()