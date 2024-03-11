import numpy as np
import astropy

# import warnings
# from copy import deepcopy
# from functools import lru_cache

# import astropy.units as u
# import numpy as np
# import pandas as pd
# from astropy.coordinates import Distance, SkyCoord, match_coordinates_sky
# from astropy.time import Time
# from astropy.wcs import WCS
# from astroquery.gaia import Gaia
# from scipy import sparse
# from tqdm import tqdm
# from astropy.io import fits

class SceneFitter():

    def __init__(self, data, ra, dec, roll):
        """
        data : np.ndarray
            data is an array, also contains the flux errors (possibly split this up, depending on how the data comes in)
        """
        # the three big ones
        self.wcs = None
        self.psf = None #-> Can be set to our known Pandora PSF either from LLNL or from commissioning
        self.flux = None

        # other helpful attributes to track
        self.ss_mask = None   # the single source mask
        self.source_mask = None    # mask to remove certain sources from the catalog

        # string to record what was done here
        self.record = None

        # functions to call automatically
        self.catalog = self._initial_get_catalog(self.ra, self.dec)


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

    def estimate_psf_core() -> np.ndarray:
        """Fits a gaussian to the center 1-sigma of the PSF"""
        raise NotImplementedError

    def estimate_psf_wings() -> np.ndarray:
        """Fits a polynomial to the wings of the PSF"""
        raise NotImplementedError
    
    def estimate_piecewise_psf() -> np.ndarray:
        """Fits a gaussian to the core and a polynomial to the wings."""
        core = self.estimate_psf_core()
        wings = self.estimate_psf_wings()
        self.psf = max(core, wings) # <-- fix this line, it's a placeholder

    def estimate_gaussian_psf() -> np.ndarray:
        ...

    def estimate_complex_psf() -> np.ndarray:
        ...


    # FUNCTIONS FOR REFINING THE SINGLE SOURCE MASK
    def calc_contamination_ratio(self, tolerance=0.99):
        self.cont_ratio = ...
        raise NotImplementedError
    
    def estimate_initial_ss_mask(self, ..., tolerance=0.99) -> np.ndarray:
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
    def find_centroids() -> np.ndarray():
        # no idea how to do this yet
        raise NotImplementedError
    
    def find_nearest_neighbors() -> astropy.table.Table:
        # not sure what I'm doing with this yet either
        raise NotImplementedError


    # HELPER FUNCTIONS
    def _inital_get_catalog(self) -> astropy.table.Table:
        """calls the get_sky_catalog function from utils.py on self.ra and self.dec"""
        ...

    def _subtract_background() -> np.ndarray:
        """Subtracts the background from the data."""
        ...

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
        




def apply_X_matrix() -> np.ndarray:
    # will also handle errors and priors for me
    ...

def assemble_X_matrix() -> np.ndarray:
    ...


def _fit_linear_model()
    
def _build_linear_model_matrix()
    
def _build_gaussian_scene()
    
def _build_psf_scene()





# from pandorawcs import SceneFitter

# sf = SceneFitter(flux, ra, dec, roll)

# sf.estimate_wcs()
# sf.estimate_flux()
# sf.estimate_psf()
# sf.estimate_wcs()
# sf.estimate_psf()

# sf.estimate_scene()