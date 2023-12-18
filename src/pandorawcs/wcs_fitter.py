"""Contains the class WCS Fitter"""
import numpy as np
from scipy.stats import multivariate_normal
from astropy.table import Table
import astropy.units as u
import wcs
import detector

import utils

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
        Note: might later need to build in capability to distort the model.

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

        # build the model
        model = [multivariate_normal(mean=[pix_x[i],pix_y[i]], cov=[[1,0],[0,1]]).pdf(model_grid) * fluxes[i] for i in range(len(catalog))]

        return model


    def fit_sources(self, tpf, catalog, distortion_file=None):
        """
        Docstring. This will someday be the full fitting function, with all functionality we want it to include.

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
        # # get the catalog that corresponds to the tpf
        # catalog = self._get_gaia_query()

        # build the scene model
        model = self._build_scene_model(catalog)



        return
    
    def fit_simple_linear_model(self, flux, flux_err, catalog, distortion_file=None):
        """
        Docstring. Only fits for linear shifts in position.

        Parameters:
        -----------
        flux: (type?)
            Flux array of shape (n_time, naxis1, naxis2).
        flux_err: (type?)
            Flux error array of shape (n_time, naxis1, naxis2).
        """
        # build the scene model
        model = self._build_scene_model(catalog)

        # get mean and gradients
        g0, g1 = np.gradient(np.sum(np.asarray(model), axis=0))

        # background subtraciton
        bkg = utils.fit_bkg(tpf, polyorder=2)
        # tpf_clean = tpf.flux.value[0] - bkg[0,:,:]
        tpf_clean = tpf.flux.value - bkg

        # set up the linear alg
        mean = np.sum(np.asarray(model), axis=0) # the model image, np.ndarray shape (nrows, npixels), all sources summed together
        g0, g1 = np.gradient(mean) # the gradient in x and y
        # data = np.asarray(tpf.flux[0]) 
        data = np.asarray(tpf_clean[0]) # the real image, np.ndarray shape (nrows, npixels)... make sure to use the background subtracted version!
        X = np.vstack([mean.ravel(), g0.ravel(), g1.ravel()]).T

        sigma_w_inv = X.T.dot(X)
        B = X.T.dot(data.ravel())

        w = np.linalg.solve(sigma_w_inv, B)   # tells you the offests you have to apply


# Notes
# np.linalg will break if you take lots of pixels to high order polynomial bc of floating point precision
#   fix by normalizing to position on detector [-.5,.5]
# add typing for inputs and outputs on the funciton definition
        
    
