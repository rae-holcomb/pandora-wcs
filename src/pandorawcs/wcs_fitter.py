"""Contains the class WCS Fitter"""
from scipy.stats import multivariate_normal
from astropy.table import Table
import astropy.units as u
import wcs
import detector

class WCSFitter(object):
    """Fill in docstring in design document...
    
    Parameters:
    -----------
    distortion_file: str
        File path to a distortion CSV file.

    Returns:
    --------
    catalog : np.ndarray
        X pixel positions in undistorted frame, centered around CRPIX1
    """

    def __init__(self, ra:u.Quantity, dec:u.Quantity, roll:u.Quantity, 
                 detector: detector.TessDetector,  #<- Note: change this later to a generic detector class
                 crpix1: int = None, crpix2: int = None, order: int = 3,
                 ):
        ...
        self.dectector = detector
        self.shape = self.detector.shape
        self.wcs = wcs.get_wcs( self.detector, self.ra, self.dec, crpix1, crpix2, roll)



    def _get_gaia_query(self):
        """
        Queries the gaia catalog in a circle centered on the ra and dec, with a radius large enough to enclose the size of the detector object. In the future, this should call the gaia query function that lives in another pandora package (either sat or sim?)

        Parameters:
        -----------
        distortion_file: str
            File path to a distortion CSV file.

        Returns:
        --------
        catalog : np.ndarray
            X pixel positions in undistorted frame, centered around CRPIX1
        """
        raise NotImplementedError


    # build scene model
    def _build_scene_model(self, catalog):
        """
        Docstring

        Parameters:
        -----------
        Catalog: astropy.Table
            File path to a distortion CSV file.

        Returns:
        --------
        model : np.array
            3D array with dimensions of (n_sources, naxis1, naxis2) containing a model with gaussian sources.
        """
        model_grid = [[(i, j) for j in range(max(self.shape))] for i in range(max(self.shape))]
        fluxes = self.detector.mag_to_flux(catalog['gmag'].value)  # use gaia g flux
        pix_x, pix_y = self.wcs.all_world2pix(catalog['ra'], catalog['dec'], 0)

        model = [multivariate_normal(mean=[pix_x[i],pix_y[i]], cov=[[1,0],[0,1]]).pdf(model_grid) * fluxes[i] for i in range(len(catalog))]

        return model


    def fit_sources(self, tpf, catalog, distortion_file=None):
        """
        Docstring

        Parameters:
        -----------
        tpf: (type?)
            A real data image.
        catalog: astropy. Table
            A catalog of source positions.
        distortion_file: str
            FIX: Optional file path to a distortion CSV file. See `read_distortion_file`

        Returns:
        --------

        """
        # get the catalog that corresponds to the tpf
        catalog = self._get_gaia_query()

        # build the scene model
        model = self._build_scene_model(catalog)

        return
    
