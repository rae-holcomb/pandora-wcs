class SceneFitter():

    def __init__(self, flux, ra, dec, roll):
        """
        flux : np.ndarray
            flux is an array of shape BLAH and can be 2d
        """
        self.wcs = None
        self.psf = None #--> Can be set to our known Pandora PSF either from LLNL or from commissioning
        self.flux = None


    def fit_wcs() -> astropy.wcs.WCS:      

    def fit_psf() -> np.ndarray # Do we instead want a Pandora-PSF object?
        # 2d

    def fit_flux() -> np.ndarray
        # 1d

    def _inital_get_catalog() -> astropy.table.Table:
        self.ra, self.dec

    def get_catalog() -> astropy.table.Table:
        self._initial_catalog
        ...

    def estimate_scene(self):
        # Gaussian model

        # New PSF shape

        # New WCS


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