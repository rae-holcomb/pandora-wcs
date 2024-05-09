"""
This file is made to hold functions that have been deprecated from use. 
This will help get the clutter out of other files, but hopefully preserve old pieces of working code in case I want to refer back to it in the future!
All functions should be dates to when they were removed from their file.
"""

##############################
### Scene Fitter Functions ###
##############################


    def _fit_gaussian_flux_coeff(self, std=None, source_flux=None) -> Tuple[float, float, float, float]:
        """
        Removed from SceneFitter() on 5/9/24

        New version, does a better job fitting for the shifts.
        For a given standard deviation, finds a constant flux multiplier for the whole image. Depends on the current psf. (Currently restricted to a pure gaussian psf.)
        
        Output: flux_coeff, x_shift, y_shift, rmse"""
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

        print(model.mu[0])

        # recover the variables of interest
        flux_coeff = np.exp(model.mu[0])
        xshift, yshift = model.mu[-2], model.mu[-1]
        rmse = np.sqrt(np.mean((self.z - model.evaluate(x=self.dx, y=self.dy))**2))

        return model, flux_coeff, xshift, yshift, rmse
  

    def _fit_gaussian_flux_coeff_OLD(self, std=None, source_flux=None) -> Tuple[float, float, float, float]:
        """
        Removed from SceneFitter() on 5/9/24

                Older version. Delete later.
        May move to general PSF version.
        For a given standard deviation, finds a constant flux multiplier for the whole image. Depends on the current psf. (Currently restricted to a pure gaussian psf.)
        
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

    def update_initial_flux_coeff_OLD(self, stds: list = np.arange(0.8, 2., 0.1), source_flux=None, plot: bool=False,) -> Tuple[float, float]:
        """
        Removed from SceneFitter() on 5/9/24

        For a grid of standard deviations, finds the std and associated flux coefficient that best fits the data. Updates the class variables with this information."""
        if source_flux is None:
            source_flux = self.gaia_flux

        # set up arrays
        psfs = [None] * len(stds)
        flux_coeffs = np.zeros_like(stds)
        xshifts = np.zeros_like(stds)
        yshifts = np.zeros_like(stds)
        rmses = np.zeros_like(stds)

        # loop through
        for ind, std in enumerate(stds):
            new_psf, flux_coeff, xshift, yshift, rmse = self._fit_gaussian_flux_coeff(std=std, source_flux=source_flux)
            print(new_psf)
            psfs[ind] = new_psf
            flux_coeffs[ind] = flux_coeff
            xshifts[ind], yshifts[ind] = xshift, yshift
            rmses[ind] = rmse

        # take the STD and the flux_coeff that minimizes the resids
        ind = np.argmin(rmses)
        std = stds[ind]
        flux_coeff = flux_coeffs[ind]
        xshift, yshift = xshifts[ind], yshifts[ind]
        rmse = rmses[ind]

        # update object values
        self.initial_std = std   # initial estimate of the std for a purely gaussian psf
        self.gaia_flux_coeff = flux_coeff
        self.xshift = xshift
        self.yshift = yshift

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

        return std, flux_coeff, psfs

