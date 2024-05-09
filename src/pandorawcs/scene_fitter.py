import numpy as np
import astropy
import utils

import warnings
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

model_image - (x,y) 

"""



class SceneFitter():
    def __init__(self, hdulist, catalog, cutout=None, wcs=None, initial_std=1., initial_std_prior=1., gaia_flux_coeff=1.,):
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

        # do operations to the data in order to clean it up
        self._subtract_background() # do background subtraction
        self.saturation_mask = self._make_saturation_mask(saturation_limit=1e5, mask_adjacent=True, inplace=True)

        # masked data
        # might change these to a property later
        self.y = np.ma.masked_where(self.saturation_mask.ravel(), self.data.ravel())  # fluxes in 1D
        self.yerr = np.ma.masked_where(self.saturation_mask.ravel(), self.err.ravel())  # flux errors in 1D

        # self.y = self.data.ravel()  # fluxes in 1D
        # self.yerr = self.err.ravel()  # flux errors in 1D

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

        # NOTE: Clean up these stddev values later!
        
        # other attributes that will change as we fit the psf
        self.initial_std = initial_std   # initial estimate of the std for a purely gaussian psf
        self.stddev_x = initial_std   # initial estimate of the std for a purely gaussian psf
        self.stddev_y = initial_std   # initial estimate of the std for a purely gaussian psf
        self.initial_std_prior = initial_std_prior
        self.initial_std_prior = initial_std_prior
        self.xshift = 0
        self.yshift = 0
        self.gaia_flux_coeff = gaia_flux_coeff
        self.source_weights = np.ones(len(self.df)) # weights for individual sources

        # the PSF
        # our very first placeholder psf is gaussian with std (defaults to 1)
        # g0 = la.lnGaussian1DGenerator('r', stddev_prior=(initial_std, initial_std_prior))
        g0 = la.lnGaussian2DGenerator('x', 'y', 
                                      stddev_x_prior=(self.initial_std, self.initial_std_prior), stddev_y_prior=(self.initial_std, self.initial_std_prior))
        self.update_psf(g0)

        # update_psf sets the following variables
            # self.psf
            # self.scene 
            # self.model_image 
            # self.ss_mask 
            # self.single_source_data
            # self.cont_ratio 
            # self.max_contributor_ind 
            # self.max_contributor_flux 
            # self.r, self.th, self.z, self.zerr, self.dx, self.dy 

        # delete later
        if False :
            # # variables associated with single source pixels
            # self.ss_mask = np.zeros(self.shape).ravel()   # the single source mask, unraveled
            # self.single_source_data = np.ma.masked_where(self.ss_mask, self.y).reshape(self.shape)   # data, masked to contain only single sources
            # self.max_contributor_ind = ...   # the index of the source that dominates each pixel
            # self.max_contributor_flux = ...   # the flux of the source that dominates each pixel

            # # radial coordinates
            # self.r, self.th, self.z, self.zerr, self.dx, self.dy = self._convert_to_radial_coordinates()
            
            self.scene = ...   # do we want to track this?
            self.model_image = ... # the scene added into a flat 2D image

        # mask to remove certain sources from the catalog
        self.source_mask = np.ones(self.shape)    

        # string to record what was done here
        self.record = ...

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
            f"{type(self).__name__}"
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
        
    def apply_mask(self, mask) -> None:
        """Allows the user to apply an arbitrary mask to the data. Must have the dimensions specified in self.shape."""
        ...

    # FUNCTIONS FOR PSF ESTIMATION

    def update_psf(self, psf: la.generator.Generator, std=None, nstddevs=5, tolerance=0.99, min_flux=25,) -> None:
        """
        Updates the PSF of the scene. Also automatically update values that derive from the PSF, including contamination variables, single source mass, and radial coordinates.
        
        Inputs:
            psf - an lamatrix generator object
            std, nstddevs - input arguments to _get_psf_scene()
            tolerance, min_flux - cutoffs used to calculate the ignle source mask
        """
        if not isinstance(psf, la.generator.Generator):
            warnings.warn("psf must be an lamatrix.generator.Generator object.")

        # set the psf
        self.psf = psf

        # generate the scene (and later also the gradients?)
        # scene, _, _ = self._get_psf_scene(std=std, nstddevs=nstddevs)
        scene = self._get_psf_scene(std=std, nstddevs=nstddevs)

        # calculate the contamination ratio, etc. 
        cont_ratio, max_contributor_ind, max_contributor_flux = self.calc_contamination_ratio(scene)

        # apply the tolerance and minimum pixel flux to get a single source mask
        m1 = np.array(cont_ratio).squeeze() > tolerance
        m2 = self.y > min_flux
        ss_mask = (m1&m2)
        single_source_data = np.ma.masked_where(~ss_mask.reshape(self.shape), self.data)

        # update class variables
        self.scene = scene
        self.model_image = self._get_model_image()
        self.ss_mask = ss_mask
        self.single_source_data = single_source_data
        self.cont_ratio = cont_ratio   
        self.max_contributor_ind = max_contributor_ind
        self.max_contributor_flux = max_contributor_flux
        self.r, self.th, self.z, self.zerr, self.dx, self.dy = self._convert_to_radial_coordinates()

        pass


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

    def _convert_to_radial_coordinates(self) -> np.ndarray:
        """May move or change this later, but intended to be a helper function which gets the xy coordinates of image into an appropriate formate for radial psf fitting."""
        # z is normalized and in log space
        z = np.log((self.y / self.max_contributor_flux))
        zerr = 2.5 * self.yerr/self.y * np.log(10)

        # get dx, dy, r, and phi
        dx = np.hstack(self.R.ravel() - self.df.iloc[self.max_contributor_ind]['X0'].to_numpy())
        dy = np.hstack(self.C.ravel() - self.df.iloc[self.max_contributor_ind]['Y0'].to_numpy())
        z, zerr, dx, dy, = z[self.ss_mask], zerr[self.ss_mask], dx[self.ss_mask], dy[self.ss_mask] 
        r, phi = np.hypot(dx, dy), np.arctan2(dy, dx)

        return r, phi, z, zerr, dx, dy

    def model_rmse(self):
        """Calculates the RMSE between the data and the model_image."""
        rmse = np.sqrt(np.mean((self.y - self.model_image.ravel())**2))
        return rmse

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
    
    def estimate_initial_ss_mask(self, tolerance=0.99, min_flux=25, update=True) -> np.ndarray:
        """Current fits a gaussian for the psf, then calculates the contamination ratio. Returns a mask to select single sources. In the future may switch to piecewise function."""
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
        single_source_data = np.ma.masked_where(~ss_mask.reshape(self.shape), self.data)

        if update:
            self.ss_mask = ss_mask
            self.single_source_data = single_source_data
            self.cont_ratio = cont_ratio   
            self.max_contributor_ind = max_contributor_ind   # the index of the source that dominates each pixel
            self.max_contributor_flux = max_contributor_flux   # the flux of the source that dominates 

        return ss_mask, single_source_data

        

    # FUNCTIONS FOR FLUX FITTING
    def do_initial_fit(self, std=None, source_flux=None, update=False):
        if std is None:
                std = self.initial_std
        if source_flux is None:
            source_flux = self.gaia_flux

        # set up 2d gaussian with a set stddev
        g1 = la.lnGaussian2DGenerator('x', 'y', stddev_x_prior=(std, 1), stddev_y_prior=(std, 1))
        g1.fit(x=self.dx, y=self.dy, data=self.z, errors=self.zerr)#, mask=data_mask)

        # now add in the gradient and fit
        dg1 = g1.gradient
        model = g1 + dg1
        model.fit(x=self.dx, y=self.dy, data=self.z, errors=self.zerr)#, mask=data_mask)

        if update:
            self.update_psf(model)
            self.gaia_flux_coeff = np.exp(model.mu[0])
            self.stddev_x = model[0].stddev_x
            self.stddev_y = model[0].stddev_y
            _, self.xshift, self.yshift = model[1].fit_mu
            
        # return the psf
        return model
   

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

    def _get_psf_scene(self, source_flux=None, std=None, x_col='X0', y_col='Y0', xshift=0, yshift=0, nstddevs=5) -> utils.SparseWarp3D:
        """Generates a scene from the current psf. Currently does NOT generate gradients of the scene."""    
        if std is None:
            std = self.initial_std
        if source_flux is None:
            source_flux = self.source_flux

        xval = self.df[x_col] + xshift
        yval = self.df[y_col] + yshift

        # row and column grids
        gR, gC = np.mgrid[
            np.floor(-nstddevs * std) : np.ceil(nstddevs * std + 1),
            np.floor(-nstddevs * std) : np.ceil(nstddevs * std) + 1,
        ]

        # for just one slice of the scene
        ggR = gR[:,:,None] - np.asarray(xval % 1)
        ggC = gC[:,:,None] - np.asarray(yval % 1)

        # rads = np.hypot(ggR, ggC)
        rads, thetas = np.hypot(ggR, ggC), np.arctan2(ggC, ggR)
        source = np.exp(self.psf.evaluate(x=ggR, y=ggC, r=rads.ravel(), theta=thetas).reshape(ggR.shape))

        scene = utils.SparseWarp3D(
                        source * source_flux,
                        gR[:, :, None] + np.asarray(np.floor(xval - self.R[0, 0])).astype(int),
                        gC[:, :, None] + np.asarray(np.floor(yval - self.C[0, 0])).astype(int),
                        self.shape,
                    )
        
        return scene

    def _get_psf_scene_OLD(self, source_flux=None, std=None, x_col='X0', y_col='Y0', nstddevs=5) -> utils.SparseWarp3D:
        """
        Doesn't incorporate dx and dy shifts, delete later once you do.

        Generates a scene from the current psf. Currently does NOT generate gradients of the scene."""    
        if std is None:
            std = self.initial_std
        if source_flux is None:
            source_flux = self.source_flux

        # row and column grids
        gR, gC = np.mgrid[
            np.floor(-nstddevs * std) : np.ceil(nstddevs * std + 1),
            np.floor(-nstddevs * std) : np.ceil(nstddevs * std) + 1,
        ]

        # for just one slice of the scene
        ggR = gR[:,:,None] - np.asarray(self.df[x_col] % 1)
        ggC = gC[:,:,None] - np.asarray(self.df[y_col] % 1)

        # rads = np.hypot(ggR, ggC)
        rads, thetas = np.hypot(ggR, ggC), np.arctan2(ggC, ggR)
        source = np.exp(self.psf.evaluate(r=rads.ravel(), theta=thetas).reshape(ggR.shape))

        scene = utils.SparseWarp3D(
                        source * source_flux,
                        gR[:, :, None] + np.asarray(np.floor(self.df[x_col] - self.R[0, 0])).astype(int),
                        gC[:, :, None] + np.asarray(np.floor(self.df[y_col] - self.C[0, 0])).astype(int),
                        self.shape,
                    )
        
        return scene

    def _get_model_image(self):
        return np.asarray(self.scene.sum(axis=1).reshape(self.shape))

    def _get_gaussian_scene_OLD(self, source_flux=None, std=None, x_col='X0', y_col='Y0', nstddevs=5):
        """
        Does not contain ability to apply dx and dy shifts. Delete later if the new version works.

        Creates a model image with dimensions [n_sources, x, y] where each slice contains the gaussian for a single source in the image. Also calculates the x and y gradients of the gaussian scene."""
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

    def _get_gaussian_scene(self, source_flux=None, std=None, x_col='X0', y_col='Y0', xshift=0, yshift=0, nstddevs=5):
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

        xval = self.df[x_col] + xshift
        yval = self.df[y_col] + yshift

        gauss = utils.gaussian_2d(
            gR[:, :, None],
            gC[:, :, None],
            np.asarray(xval % 1),
            np.asarray(yval % 1),
            np.atleast_1d(std),
            np.atleast_1d(std),
        )

        s = utils.SparseWarp3D(
                gauss * source_flux,
                gR[:, :, None] + np.asarray(np.floor(xval - self.R[0, 0])).astype(int),
                gC[:, :, None] + np.asarray(np.floor(yval - self.C[0, 0])).astype(int),
                self.shape,
            )
        
        dG_x, dG_y = utils.dgaussian_2d(
            gR[:, :, None],
            gC[:, :, None],
            np.asarray(xval % 1),
            np.asarray(yval % 1),
            np.atleast_1d(std),
            np.atleast_1d(std),
        )

        ds_x = utils.SparseWarp3D(
                dG_x * gauss * source_flux,
                gR[:, :, None] + np.asarray(np.floor(xval - self.R[0, 0])).astype(int),
                gC[:, :, None] + np.asarray(np.floor(yval - self.C[0, 0])).astype(int),
                self.shape,
            ).sum(axis=1)
        
        ds_y = utils.SparseWarp3D(
                dG_y *  gauss * source_flux,
                gR[:, :, None] + np.asarray(np.floor(xval - self.R[0, 0])).astype(int),
                gC[:, :, None] + np.asarray(np.floor(yval - self.C[0, 0])).astype(int),
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
        """Returns a 2D gaussian scene made with a simple gaussian psf."""
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

    def _make_saturation_mask(self, saturation_limit=1e5, mask_adjacent=True, inplace=True):
        """Creates a mask of pixels that may be affected by saturation."""
        initial_mask = self.data > saturation_limit
        pixel_mask = initial_mask.copy()
        
        # If requested, check pixels above and below those with data > 1e5
        if mask_adjacent:
            for i in range(1, self.data.shape[0]):
                above_mask = initial_mask[i-1, :]
                below_mask = initial_mask[i, :]
                
                # Update pixel mask to include pixels above and below
                pixel_mask[i, :] |= above_mask
                pixel_mask[i-1, :] |= below_mask
        
        if inplace:
            self.saturation_mask = pixel_mask

        return pixel_mask


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

    def plot_single_source_data(self, vmin=0, vmax=300):
        """Shows which pixels are included in the current single source mask."""
        temp = np.ma.masked_where(~self.ss_mask.reshape(self.shape), self.data)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        print(type(ax))
        pc = ax.pcolormesh(self.C, self.R, temp, vmin=vmin, vmax=vmax) 
        fig.colorbar(pc, label='Flux')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_title('Pixels Dominated by a Single Source')
        return fig

    def plot_saturation_mask(self):
        "Shows which pixels are being masked for saturation in red."
        fig, ax = plt.subplots()

        # Plot the data using plt.pcolormesh
        mesh = ax.pcolormesh(self.C, self.R, self.data, cmap='viridis')

        # Overlay the saturated pixel mask
        masked_data = np.ma.masked_where(self.saturation_mask == False, np.ones_like(self.data))
        ax.pcolormesh(self.C, self.R, masked_data, cmap='bwr_r')

        # Add colorbar
        cbar = plt.colorbar(mesh)
        cbar.set_label('Data')

        # Set labels and title
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_title('Data with Saturated Pixel Mask')

        return fig

    def plot_radial_psf(self, vmin=None, vmax=None, rad_max=5):
        """Assumes that the psf is a generator object which takes in either just variable `r`, or both `r` and `th` for radius and theta.
        
        NOTE: PSFs with r and theta as inputs still need to be tested."""
        thetas = np.arange(0,2*np.pi,.1)
        rads = np.arange(0,rad_max,.1)
        X,Y = np.meshgrid(rads, thetas) #rectangular plot of polar data
        # X = rad?
        # Y = theta?

        try:
            # this part may need fixing later
            data2D = self.psf.evaluate(r=X.ravel(), theta=Y.ravel()).reshape(X.shape)
        except:
            data2D = self.psf.evaluate(r=rads)[:, None] * np.ones_like(X)

        print(data2D.shape)
        fig = plt.figure()
        ax = fig.add_subplot(111, polar='True')
        pc = ax.pcolormesh(Y, X, data2D, vmin=vmin, vmax=vmax) #X,Y & data2D must all be same dimensions
        fig.colorbar(pc, label='Normalized ln(flux)')
        return fig

    def plot_radial_data(self, sigma_mask=None, **kwargs):
        """
        Plots the normalized data in radial format.
        """
        if sigma_mask is None:
            sigma_mask = [True] * len(self.dx)
        fig, ax = plt.subplots(figsize=(5, 5))
        #ax.errorbar(rad, y, ye, color='k', ls='', lw=0.3)
        im = ax.scatter(self.dx[sigma_mask], self.dy[sigma_mask], c=self.z[sigma_mask], s=4)
        fig.colorbar(im, label='$\ln(Flux_{norm}$)')
        ax.set(xlabel='$\delta x$', ylabel='$\delta y$', title='')
        return fig


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


# def apply_X_matrix() -> np.ndarray:
#     # will also handle errors and priors for me
#     ...

# def assemble_X_matrix() -> np.ndarray:
#     ...






# def _fit_linear_model()
    
# def _build_linear_model_matrix()
    
# def _build_gaussian_scene()
    
# def _build_psf_scene()





# from pandorawcs import SceneFitter

# sf = SceneFitter(flux, ra, dec, roll)

# self.estimate_wcs()
# self.estimate_flux()
# self.estimate_psf()
# self.estimate_wcs()
# self.estimate_psf()

# self.estimate_scene()