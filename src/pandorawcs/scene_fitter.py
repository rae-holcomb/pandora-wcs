import numpy as np
import astropy
import utils

# import warnings
from copy import deepcopy
# from functools import lru_cache

import astropy.units as u
import pandas as pd
from astropy.coordinates import Distance, SkyCoord, match_coordinates_sky
from astropy.time import Time
from astropy.wcs import WCS
from astroquery.gaia import Gaia
from scipy import sparse
from tqdm import tqdm
from astropy.io import fits

class SceneFitter():

    def __init__(self, hdulist, catalog, wcs=None):
        """
        Temporary init function where I can pass stuff in for ease of use.
        """
        self.data = hdulist[1].data
        self.err = hdulist[2].data
        self.df = catalog
        if wcs is None :
            self.wcs = WCS(hdulist[1].header)
        else:
            self.wcs = wcs

        # useful attributes that won't change
        self.shape = self.data.shape
        self.y = self.data.ravel()  # fluxes in 1D
        self.ye = self.err.ravel()  # flux errors in 1D
        self.R, self.C = np.mgrid[1500:1700, 1501:1700]
        # something to represent the center of the image?

        # do background subtraction
        self._subtract_background()



        # the three big ones
        # self.wcs = None
        # self.psf = None #-> Can be set to our known Pandora PSF either from LLNL or from commissioning
        # self.flux = None

        # other attributes that will change as we fit the psf
        self.ss_mask = np.ones(self.shape)   # the single source mask
        self.source_mask = np.ones(self.shape)    # mask to remove certain sources from the catalog


        # string to record what was done here
        self.record = None

        # # functions to call automatically
        # self.catalog = self._initial_get_catalog(self.ra, self.dec)
        # # get the X0 and Y0 pixel positions from the WCS

    def __future_init__(self, hdulist, ra, dec, roll):
        """
        In the future, this will be the init function. But for now, I'm developing a different one where I pass in the catalog and wcs for ease of development.

        hdulist : astropy.io.fits.hdu.hdulist.HDUList
            Assume that data is coming in from a fits file, formatted (for now) like TESS, but to be changed to the format that Ben is using in pandorsim.
        """
        # the three big ones
        self.wcs = None
        self.psf = None #-> Can be set to our known Pandora PSF either from LLNL or from commissioning
        self.flux = None

        # other attributes that will change as we fit the psf
        self.ss_mask = None   # the single source mask
        self.source_mask = None    # mask to remove certain sources from the catalog

        # useful attributes that won't change
        self.shape = None
        # something to represent the center of the image?

        # string to record what was done here
        self.record = None

        # functions to call automatically
        self.catalog = self._initial_get_catalog(self.ra, self.dec)
        # get the X0 and Y0 pixel positions from the WCS


    def copy(self):
        return deepcopy(self)

    def __repr__(self):
        return (
            f"{type(self).__name__}({', '.join(list(self.arg_names))})[n, {self.width}]"
        )
    
    # These functions are "collected" versions of their processes, fill out later
    def fit_wcs() -> astropy.wcs.WCS:  
        """this is a hold over from before, may delete later"""
        ...

    def fit_psf() -> np.ndarray: # Do we instead want a Pandora-PSF object?
        """Collected function of all PSF fitting functions"""
        ...

    def fit_flux() -> np.ndarray:
        """Collected function of all flux fitting functions"""
        ...


    # FUNCTIONS FOR PSF ESTIMATION
        
    def estimate_initial_gaussian_psf(self, stds) -> np.ndarray:
        """stds - a grid of standard deviations to test on"""
        ...

    def estimate_psf_core() -> np.ndarray:
        """Fits a gaussian to the center 1-sigma of the PSF"""
        raise NotImplementedError

    def estimate_psf_wings() -> np.ndarray:
        """Fits a polynomial to the wings of the PSF"""
        raise NotImplementedError
    
    def estimate_piecewise_psf(self) -> np.ndarray:
        """Fits a gaussian to the core and a polynomial to the wings."""
        core = self.estimate_psf_core()
        wings = self.estimate_psf_wings()
        self.psf = max(core, wings) # <-- fix this line, it's a placeholder

    def estimate_gaussian_psf() -> np.ndarray:
        ...

    def estimate_complex_psf() -> np.ndarray:
        ...

    def _convert_to_radial_coordinates() -> np.ndarray:
        """May move or change this later, but intended to be a helper function which gets the xy coordinates of image into an appropriate formate for radial psf fitting."""
        ...

    def custom_sigma_clip() -> np.ndarray:
        """Still deciding if this lives here or in utils.py"""
        raise NotImplementedError


    # FUNCTIONS FOR REFINING THE SINGLE SOURCE MASK
    def calc_contamination_ratio(self, tolerance=0.99):
        self.cont_ratio = ...
        raise NotImplementedError
    
    def estimate_initial_ss_mask(self, tolerance=0.99) -> np.ndarray:
        """Fits a piecewise guess for the psf, then calculates the contamination ratio. Returns a mask to select single sources."""
        # possibly replace with estimate_piecewise_psf()
        piecewise_psf = self.estimate_piecewise_psf()
        self.ss_mask = self.estimate_ss_mask(self, psf)    # <-- just update, or also return?

    def estimate_ss_mask(self, psf, tolerance=0.99) -> np.ndarray:
        """Calculates the single source mask for a given psf."""
        cont_ratio = self.calc_contamination_ratio()
        self.ss_mask = cont_ratio > tolerance     # <-- just update, or also return?


    # FUNCTIONS FOR FLUX FITTING
        
    def fit_constant_flux_coeff() -> float:
        """Finds a constant flux multiplier for the whole image. Depends on the current PSF"""
        raise NotImplementedError

    def fit_flux_weights() -> np.ndarray:
        """Fits flux weights. Incorporates errors and priors."""
        raise NotImplementedError


    # FUNCTIONS FOR CENTROID FITTING
    def find_centroids() -> np.ndarray:
        # no idea how to do this yet
        raise NotImplementedError
    
    def find_nearest_neighbors() -> astropy.table.Table:
        # not sure what I'm doing with this yet either
        raise NotImplementedError


    # HELPER FUNCTIONS
    def _inital_get_catalog(self) -> astropy.table.Table:
        """calls the get_sky_catalog function from utils.py on self.ra and self.dec"""
        raise NotImplementedError 
    
    def _get_gaussian_scene(self, source_flux, std=1.5, x_col='X0', y_col='Y0', nstddevs=5):
        """Creates a model image with dimensions [n_sources, x, y] where each slice contains the gaussian for a single source in the image."""
        # row and column grids
        gR, gC = np.mgrid[
            np.floor(-nstddevs * std) : np.ceil(nstddevs * std + 1),
            np.floor(-nstddevs * std) : np.ceil(nstddevs * std) + 1,
        ]

        gauss = utils.gaussian_2d(
            gC[:, :, None],
            gR[:, :, None],
            np.asarray(self.df[x_col] % 1),
            np.asarray(self.df[y_col] % 1),
            np.atleast_1d(std),
            np.atleast_1d(std),
        )

        s = utils.SparseWarp3D(
                gauss * source_flux,
                gC[:, :, None] + np.asarray(np.floor(self.df[y_col] - self.C[0, 0])).astype(int),
                gR[:, :, None] + np.asarray(np.floor(self.df[x_col] - self.R[0, 0])).astype(int),
                self.shape,
            )
        
        return s

    def _get_gaussian_gradients(self, source_flux, std=1.5, x_col='X0', y_col='Y0', nstddevs=5):
        """Calculates the x and y gradients of the gaussian scene."""
        # row and column grids
        gR, gC = np.mgrid[
            np.floor(-nstddevs * std) : np.ceil(nstddevs * std + 1),
            np.floor(-nstddevs * std) : np.ceil(nstddevs * std) + 1,
        ]

        gauss = utils.gaussian_2d(
            gC[:, :, None],
            gR[:, :, None],
            np.asarray(self.df[x_col] % 1),
            np.asarray(self.df[y_col] % 1),
            np.atleast_1d(std),
            np.atleast_1d(std),
        )

        dG_x, dG_y = utils.dgaussian_2d(
            gC[:, :, None],
            gR[:, :, None],
            np.asarray(self.df[x_col] % 1),
            np.asarray(self.df['Y0'] % 1),
            np.atleast_1d(std),
            np.atleast_1d(std),
        )

        ds_x = utils.SparseWarp3D(
                dG_x * gauss * source_flux,
                gC[:, :, None] + np.asarray(np.floor(self.df[y_col] - self.C[0, 0])).astype(int),
                gR[:, :, None] + np.asarray(np.floor(self.df[x_col] - self.R[0, 0])).astype(int),
                self.shape,
            ).sum(axis=1)
        
        ds_y = utils.SparseWarp3D(
                dG_y *  gauss * source_flux,
                gC[:, :, None] + np.asarray(np.floor(self.df[y_col] - self.C[0, 0])).astype(int),
                gR[:, :, None] + np.asarray(np.floor(self.df[x_col] - self.R[0, 0])).astype(int),
                self.shape,
            ).sum(axis=1)
        
        return ds_x, ds_y

    def _get_gaussian_design_matrix(self, source_flux, std=1.5, x_col='X0', y_col='Y0', nstddevs=5):
        """Calls get_gaussian_scene and get_gaussian_gradients in order to build the design matrix."""
        s = self._get_gaussian_scene(
            source_flux, std=std, x_col=x_col, y_col=y_col, nstddevs=nstddevs
            )
        
        ds_x, ds_y = self._get_gaussian_gradients(
            source_flux, std=std, x_col=x_col, y_col=y_col, nstddevs=nstddevs
            )
        
        components = [s, ds_x, ds_y]
        return sparse.hstack(components, 'csr')

        


    def _subtract_background(self) -> np.ndarray:
        """Subtracts the background from the data. Should only be called once."""
        # subtract out a polynomial background
        R, C = np.mgrid[0:(self.data.shape[0]), 0:(self.data.shape[1])]
        R0, C0 = R[:, 0].mean(), C[0, :].mean()
        wR, wC = R - R0, C - C0
        k = self.data.ravel() < np.percentile(self.data, 20)
        polyorder = 2
        poly = np.vstack(
            [
                wR.ravel() ** idx * wC.ravel() ** jdx
                for idx in range(polyorder + 1)
                for jdx in range(polyorder + 1)
            ]
        ).T
        bkg_model = poly.dot(
            np.linalg.solve(poly[k].T.dot(poly[k]), poly[k].T.dot(self.data.ravel()[k]))
        ).reshape(self.data.shape)
        self.data -= bkg_model

    def estimate_scene(self):
        """Holdover. Come back to this later."""
        # Gaussian model

        # New PSF shape

        # New WCS


    # PLOTTING FUNCTIONS
    # for now, just a list of plotting functions we want to have
        # plot_data
        # plot_model_scene
        # plot psf-view (source-centered, log-space)
        # plot difference between model and data?
        # plot model psf
        # plot single source mask
        # plot contamination ratio



    # SAVING AND LOADING FUNCTIONS
        # TBD


def apply_X_matrix() -> np.ndarray:
    # will also handle errors and priors for me
    ...

def assemble_X_matrix() -> np.ndarray:
    ...


# def _fit_linear_model()
    
# def _build_linear_model_matrix()
    
# def _build_gaussian_scene()
    
# def _build_psf_scene()





# from pandorawcs import SceneFitter

# sf = SceneFitter(flux, ra, dec, roll)

# sf.estimate_wcs()
# sf.estimate_flux()
# sf.estimate_psf()
# sf.estimate_wcs()
# sf.estimate_psf()

# sf.estimate_scene()