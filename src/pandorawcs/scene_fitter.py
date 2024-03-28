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
import matplotlib.pyplot as plt
import lamatrix as la
from typing import Tuple, Union

"""
NOTES ON TERMINOLOGY:
Throughout this class, I will be using the following terminology for an image that is (nx, ny) pixels and contains N sources.

scene - an (nx, ny, N) data cube, where each slice along the N dimension contains the flux of only a single source. These should always be SparseWarp3D matrices.

"""



class SceneFitter():
    def __init__(self, hdulist, catalog, cutout=None, wcs=None):
        """
        Temporary init function where I can pass stuff in for ease of use.
        """
        if cutout is None :
            # Note: For many data types (including TESS FFIs), this default cutout will be wrong since it doesn't take into account junk and science columns.
            self.cutout = [[0, hdulist[1].header['NAXIS1']],
                            [0, hdulist[1].header['NAXIS2']]]
        else :
            self.cutout = cutout

        # get the cutout of the data
        self.R, self.C = np.mgrid[self.cutout[0][0]:self.cutout[0][1], 
                                  self.cutout[1][0]:self.cutout[1][1]]
        self.shape = self.R.shape
        self.shapeT = self.R.T.shape
        self.naxis1 = cutout[0][1] - cutout[0][0]
        self.naxis2 = cutout[1][1] - cutout[1][0]
        self.data = hdulist[1].data[self.R, self.C]
        self.err = hdulist[2].data[self.R, self.C]
        self._subtract_background() # do background subtraction
        self.y = self.data.ravel()  # fluxes in 1D
        self.yerr = self.err.ravel()  # flux errors in 1D

       # catalog and fluxes
        self.df = catalog

        # later, may want to make the source flux selection more flexible
        self.gaia_flux = catalog['phot_rp_mean_flux'].values

        # wcs
        if wcs is None :
            self.wcs = WCS(hdulist[1].header)
        else:
            # TODO: add check that it is a valid wcs object
            self.wcs = wcs

        # center of the cutout, NOT the same as the CRPIX of the wcs
        self.R0, self.C0 = self.R[:, 0].mean(), self.C[0].mean()
        c = self.wcs.pixel_to_world(self.C0, self.R0)
        self.ra0, self.dec0 = c.ra.deg, c.dec.deg



        # the PSF
        # self.psf = None #-> Can be set to our known Pandora PSF either from LLNL or from commissioning

        # other attributes that will change as we fit the psf
        self.initial_std = 1.   # initial estimate of the std for a purely gaussian psf
        self.gaia_flux_coeff = 1.
        self.source_weights = np.ones(len(self.df)) # weights for individual sources

        self.ss_mask = np.ones(self.shape)   # the single source mask
        self.max_contributor_ind = ...   # the index of the source that dominates each pixel
        self.max_contributor_flux = ...   # the flux of the source that dominates each pixel
        
        self.source_mask = np.ones(self.shape)    # mask to remove certain sources from the catalog

        self.scene = ...   # do we want to track this?


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

    @property
    def source_flux(self):
        return self.gaia_flux * self.gaia_flux_coeff * self.source_weights


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

    # # UPDATE FUNCTIONS
    # def _update_source_flux(self) -> None:
    #     self.source_flux = self.gaia_flux * self.gaia_flux_coeff *self.source_weights
    #     pass

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
    def calc_contamination_ratio(self, scene) -> Tuple[np.ndarray, np.ndarray]:
        """For a given scene, calculates the contamination ratio and the source that is providing the maximum flux contribution in each pixel across the image."""
        # calculate the contamination ratio and identify which source dominates each pixel
        cont_ratio = scene.max(axis=1) / scene.sum(axis=1)
        max_contributor_ind = np.asarray(scene.argmax(axis=1))[:,0]
        max_contributor_flux = self.source_flux[max_contributor_ind]
        
        return cont_ratio, max_contributor_ind, max_contributor_flux
    
    def estimate_initial_ss_mask(self, tolerance=0.99, min_flux=25, update=False) -> np.ndarray:
        """Fits a piecewise guess for the psf, then calculates the contamination ratio. Returns a mask to select single sources."""
        # possibly replace with estimate_piecewise_psf()
        # piecewise_psf = self.estimate_piecewise_psf()
        # self.ss_mask = self.estimate_ss_mask(self, psf)    # <-- just update, or also return?

        # generate the scene
        scene, _, _ = self._get_gaussian_scene(std=self.initial_std)

        # calculate the contamination ratio, etc. 
        cont_ratio, max_contributor_ind, max_contributor_flux = self.calc_contamination_ratio(scene)

        # apply the tolerance and minimum pixel flux to get a single source mask
        m1 = np.array(cont_ratio).squeeze() > tolerance
        m2 = self.y > min_flux
        ss_mask = (m1&m2)
        single_source_mask = np.ma.masked_where(~ss_mask.reshape(self.shape), self.data)

        if update:
            self.ss_mask = ss_mask
            self.single_source_mask = single_source_mask
            self.cont_ratio = cont_ratio   
            self.max_contributor_ind = max_contributor_ind   # the index of the source that dominates each pixel
            self.max_contributor_flux = max_contributor_flux   # the flux of the source that dominates 

        return ss_mask, single_source_mask

        

    # def estimate_ss_mask(self, psf, tolerance=0.99, min_flux=25) -> np.ndarray:
    #     """Calculates the single source mask for a given psf."""
    #     cont_ratio = self.calc_contamination_ratio()
    #     self.ss_mask = cont_ratio > tolerance     # <-- just update, or also return?


    # FUNCTIONS FOR FLUX FITTING
        
    def _fit_flux_coeff(self, std=None, source_flux=None) -> Tuple[float, float, float, float]:
        """For a given standard deviation, finds a constant flux multiplier for the whole image. Depends on the current PSF. (Currently restricted to a pure gaussian psf.)
        
        Output: flux_coeff, dx, dy, rmse"""
        if std is None:
            std = self.initial_std
        if source_flux is None:
            source_flux = self.gaia_flux

        scene, dfdx, dfdy = self._get_gaussian_scene(std=std)
        flux = np.asarray(scene.sum(axis=1)).flatten()
        dfdx = np.array(dfdx).flatten()
        dfdy = np.array(dfdy).flatten()

        # build generator, first order polynomials in flux and shifts
        g1 = la.Polynomial1DGenerator('flux', polyorder=1)
        g2 = la.Polynomial1DGenerator('dfdx', polyorder=1)
        g3 = la.Polynomial1DGenerator('dfdy', polyorder=1)
        g = g1 + g2 + g3

        # fit
        g.fit(flux=flux, dfdx=dfdx, dfdy=dfdy, data=self.y, errors=self.yerr)

        # recover the variables of interest
        flux_coeff = g.mu[1]
        dx, dy = g.mu[3], g.mu[5]
        rmse = np.sqrt(np.mean((self.y - g.evaluate(flux=flux, dfdx=dfdx, dfdy=dfdy))**2))

        return flux_coeff, dx, dy, rmse

    def update_initial_flux_coeff(self, stds: list = np.arange(0.8, 2., 0.1), source_flux=None, plot: bool=False,) -> Tuple[float, float]:
        """For a grid of standard deviations, finds the std and associated flux coefficient that best fits the data. Updates the class variables with this information."""
        if source_flux is None:
            source_flux = self.gaia_flux

        # set up arrays
        flux_coeffs = np.zeros_like(stds)
        dxs = np.zeros_like(stds)
        dys = np.zeros_like(stds)
        rmses = np.zeros_like(stds)

        # loop through
        for ind, std in enumerate(stds):
            flux_coeff, dx, dy, rmse = self._fit_flux_coeff(std=std, source_flux=source_flux)
            flux_coeffs[ind] = flux_coeff
            dxs[ind], dys[ind] = dx, dy
            rmses[ind] = rmse

        # take the STD and the flux_coeff that minimizes the resids
        ind = np.argmin(rmses)
        std = stds[ind]
        flux_coeff = flux_coeffs[ind]
        dx, dy = dxs[ind], dys[ind]
        rmse = rmses[ind]

        # update object values
        self.initial_std = std   # initial estimate of the std for a purely gaussian psf
        self.gaia_flux_coeff = flux_coeff

        # plot
        if plot:
            fig, ax = plt.subplots(1,2, figsize=[8,3])
            ax[0].plot(stds,rmses)
            ax[0].set_title('RMSE')
            ax[0].axvline(std, c='r', linestyle='--')
            ax[0].set_xlabel('STD')

            ax[1].plot(stds,flux_coeffs)
            ax[1].set_title('Flux coefficient')
            ax[1].axvline(std, c='r', linestyle='--')
            ax[1].set_xlabel('STD')
            fig.tight_layout()

        return std, flux_coeff
     

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

    def _recover_cutout(self):
        cutout = [[self.R0 - (self.naxis1-1)/2, 
                self.R0 + (self.naxis1+1)/2],
                [self.C0 - (self.naxis2-1)/2,
                self.C0 + (self.naxis2+1)/2]]
        return cutout


    def _get_gaussian_scene(self, source_flux=None, std=None, x_col='X0', y_col='Y0', nstddevs=5):
        """Creates a model image with dimensions [n_sources, x, y] where each slice contains the gaussian for a single source in the image. Also calculates the x and y gradients of the gaussian scene."""
        if std is None:
            std = self.initial_std
        if source_flux is None:
            source_flux = self.source_flux

        # row and column grids
        gR, gC = np.mgrid[
            np.floor(-nstddevs * std) : np.ceil(nstddevs * std + 1),
            np.floor(-nstddevs * std) : np.ceil(nstddevs * std) + 1,
        ]

        gauss = utils.gaussian_2d(
            gR[:, :, None],
            gC[:, :, None],
            np.asarray(self.df[x_col] % 1),
            np.asarray(self.df[y_col] % 1),
            np.atleast_1d(std),
            np.atleast_1d(std),
        )

        s = utils.SparseWarp3D(
                gauss * source_flux,
                gR[:, :, None] + np.asarray(np.floor(self.df[x_col] - self.R[0, 0])).astype(int),
                gC[:, :, None] + np.asarray(np.floor(self.df[y_col] - self.C[0, 0])).astype(int),
                self.shape,
            )
        
        dG_x, dG_y = utils.dgaussian_2d(
            gR[:, :, None],
            gC[:, :, None],
            np.asarray(self.df[x_col] % 1),
            np.asarray(self.df[y_col] % 1),
            np.atleast_1d(std),
            np.atleast_1d(std),
        )

        ds_x = utils.SparseWarp3D(
                dG_x * gauss * source_flux,
                gR[:, :, None] + np.asarray(np.floor(self.df[x_col] - self.R[0, 0])).astype(int),
                gC[:, :, None] + np.asarray(np.floor(self.df[y_col] - self.C[0, 0])).astype(int),
                self.shape,
            ).sum(axis=1)
        
        ds_y = utils.SparseWarp3D(
                dG_y *  gauss * source_flux,
                gR[:, :, None] + np.asarray(np.floor(self.df[x_col] - self.R[0, 0])).astype(int),
                gC[:, :, None] + np.asarray(np.floor(self.df[y_col] - self.C[0, 0])).astype(int),
                self.shape,
            ).sum(axis=1)
        
        return s, ds_x, ds_y
    
    def _get_gaussian_design_matrix(self, source_flux=None, std=None, x_col='X0', y_col='Y0', nstddevs=5):
        """Calls get_gaussian_scene and get_gaussian_gradients in order to build the design matrix."""
        if std is None:
            std = self.initial_std
        if source_flux is None:
            source_flux = self.source_flux

        s, ds_x, ds_y = self._get_gaussian_scene(
            source_flux, std=std, x_col=x_col, y_col=y_col, nstddevs=nstddevs
            )
        components = [s, ds_x, ds_y]
        return sparse.hstack(components, 'csr')

    def get_flat_gaussian_model(self, source_flux=None, std=None, x_col='X0', y_col='Y0', nstddevs=5):
        """Returns a 2D gaussian scene made with a simple gaussian PSF."""
        if std is None:
            std = self.initial_std
        if source_flux is None:
            source_flux = self.gaia_flux
            
        s, _, _ = self._get_gaussian_scene(source_flux=source_flux, std=std, x_col=x_col, y_col=y_col, nstddevs=nstddevs)

        return s.sum(axis=1)


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

    # DEPRECATED
    def OLD_get_gaussian_scene(self, source_flux=None, std=None, x_col='X0', y_col='Y0', nstddevs=5):
        """Creates a model image with dimensions [n_sources, x, y] where each slice contains the gaussian for a single source in the image."""
        # row and column grids
        if std is None:
            std = self.initial_std
        if source_flux is None:
            source_flux = self.gaia_flux
        
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