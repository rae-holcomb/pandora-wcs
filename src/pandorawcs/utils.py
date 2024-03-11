# Standard library
import numpy as np
import warnings

# Third-party
from astropy.stats import sigma_clip
from astropy.time import Time
from astroquery.gaia import Gaia
import astropy.units as u
from astropy.coordinates import Distance, SkyCoord

from scipy import sparse
from typing import Tuple
from copy import deepcopy



def fit_bkg(
        tpf, polyorder: int = 1  # : lk.TessTargetPixelFile
) -> np.ndarray:
    """Fit a simple 2d polynomial background to a TPF

    Parameters
    ----------
    tpf: lightkurve.TessTargetPixelFile
        Target pixel file object. NOTE: need to make this not depend on lightkurve.
    polyorder: int
        Polynomial order for the model fit.

    Returns
    -------
    model : np.ndarray
        Model for background with same shape as tpf.shape
    """
    # Notes for understanding this function
    # All arrays in this func will have dimensions drawn from one of the following: [ntimes, ncols, nrows, npix, ncomp]
    #   ntimes = number of cadences
    #   ncols, nrows = shape of tpf
    #   npix = ncols*nrows, is the length of the unraveled vectors
    #   ncomp = num of components in the polynomial

    # Error catching
    if not isinstance(tpf, lk.TessTargetPixelFile):
        raise ValueError("Input a TESS Target Pixel File")

    if (np.product(tpf.shape[1:]) < 100) | np.any(np.asarray(tpf.shape[1:]) < 6):
        raise ValueError("TPF too small. Use a bigger cut out.")

    # Grid for calculating polynomial
    R, C = np.mgrid[: tpf.shape[1], : tpf.shape[2]].astype(float)
    R -= tpf.shape[1] / 2
    C -= tpf.shape[2] / 2

    # nested b/c we run twice, once on each orbit
    def func(tpf):
        # Design matrix
        A = np.vstack(
            [
                R.ravel() ** idx * C.ravel() ** jdx
                for idx in range(polyorder + 1)
                for jdx in range(polyorder + 1)
            ]
        ).T

        # Median star image
        m = np.median(tpf.flux.value, axis=0)
        # Remove background from median star image
        mask = ~sigma_clip(m, sigma=3).mask.ravel()
        # plt.imshow(mask.reshape(m.shape))
        bkg0 = A.dot(
            np.linalg.solve(A[mask].T.dot(A[mask]), A[mask].T.dot(m.ravel()[mask]))
        ).reshape(m.shape)

        # m is the median frame
        m -= bkg0

        # Include in design matrix
        A = np.hstack([A, m.ravel()[:, None]])

        # Fit model to data, including a model for the stars in the last column
        f = np.vstack(tpf.flux.value.transpose([1, 2, 0]))
        ws = np.linalg.solve(A.T.dot(A), A.T.dot(f))
        # shape of ws is (num of times, num of components)
        # A . ws gives shape (npix, ntimes)

        # Build a model that is just the polynomial
        model = (
            (A[:, :-1].dot(ws[:-1]))
            .reshape((tpf.shape[1], tpf.shape[2], tpf.shape[0]))
            .transpose([2, 0, 1])
        )
        # model += bkg0
        return model

    # Break point for TESS orbit
    # currently selects where the biggest gap in cadences is
    # could cause problems in certain cases with lots of quality masking! Think about how to handle bit masking
    b = np.where(np.diff(tpf.cadenceno) == np.diff(tpf.cadenceno).max())[0][0] + 1

    # Calculate the model for each orbit, then join them
    model = np.vstack([func(tpf) for tpf in [tpf[:b], tpf[b:]]])

    return model

def make_affine_matrix(
        xreflect: bool = False,
        yreflect: bool = False,
        scale: tuple = (1,1),
        rotate: float = 0,
        shear: tuple = (0,0),
        translate: tuple = (0,0),
) -> np.array :
    """
    Given some affine transformations, constructs the 3x3 matrix needed to perform them.
    
    Parameters
    ----------
    xreflect: bool
        Reflect image over the x axis.
    yreflect: bool
        Reflect image over the y axis.
    scale: tuple
        Scale factor in the (x, y) direction.
    rotate: bool
        Rotate image in units of radians.
    shear: float
        Shear factor in the (x, y) direction.
    translate: tuple
        Shift factor in the (x, y) direction.

    Returns
    -------
    M : np.ndarray
        3x3 matrix that defines the affine transform.
    """

    # identity matrix
    M = np.array([[1,0,0],[0,1,0],[0,0,1]])

    # reflection
    if xreflect:
        M = M.dot(np.array([[1,0,0],[0,-1,0],[0,0,1]]))
    if yreflect:
        M = M.dot(np.array([[-1,0,0],[0,1,0],[0,0,1]]))

    # scale
    M = M.dot(np.array([[scale[0],0,0],[0,scale[1],0],[0,0,1]]))

    # rotate
    M = M.dot(np.array([[np.cos(rotate),-np.sin(rotate),0],[np.sin(rotate),np.cos(rotate),0],[0,0,1]]))

    # shear
    M = M.dot(np.array([[1,shear[0],0],[shear[1],1,0],[0,0,1]]))

    # translate
    M = M.dot(np.array([[1,0,translate[0]],[0,1,translate[1]],[0,0,1]]))

    return M

def apply_affine_transform(
        x: np.array, y: np.array,
        crpix1: int, crpix2: int,
        M: np.array,
) -> (np.array, np.array) :
    """Docstring.
    Rotation is in RADIANS.
    Scale should always be positive.
    If the affine matrix M is provided, then it overrides all the other keywords."""
    if len(x) != len(y):
        raise ValueError('x and y need to be the same length')
        return
    # center the coordinate matrix on the crpix
    coord_mat = np.array([x - crpix1, y - crpix2, np.ones_like(x)])

    # calculate
    output = M.dot(coord_mat)

    # undo the translation to the crpix
    x_new = output[0,:] + crpix1
    y_new = output[1,:] + crpix2

    return x_new, y_new

def gaussian_2d(x, y, mu_x, mu_y, sigma_x=2, sigma_y=2) -> np.array:
    """
    Compute the value of a 2D Gaussian function.

    Parameters
    ----------
    x: float
        x-coordinate.
    y: float
        y-coordinate.
    mu_x: tuple
        Mean of the Gaussian in the x-direction.
    mu_y: float
        Mean of the Gaussian in the y-direction.
    sigma_x: float
        Standard deviation of the Gaussian in the x-direction.
    sigma_y: float
        Standard deviation of the Gaussian in the y-direction.

    Returns
    -------
    output : np.ndarray
        Value of the 2D Gaussian function at (x, y).
    """
    part1 = 1 / (2 * np.pi * sigma_x * sigma_y)
    part2 = np.exp(
        -((x - mu_x) ** 2 / (2 * sigma_x**2) + (y - mu_y) ** 2 / (2 * sigma_y**2))
    )
    return part1 * part2

def dgaussian_2d(x, y, mu_x, mu_y, sigma_x=2, sigma_y=2):
    """
    Compute the value of a 2D Gaussian function. (This returns the amplitude you need to multiply by a Gaussian to get the actual gradient.)

    Parameters:
    x (float): x-coordinate.
    y (float): y-coordinate.
    mu_x (float): Mean of the Gaussian in the x-direction.
    mu_y (float): Mean of the Gaussian in the y-direction.
    sigma_x (float): Standard deviation of the Gaussian in the x-direction.
    sigma_y (float): Standard deviation of the Gaussian in the y-direction.

    """
    
    dG_x = -(x - mu_x)/sigma_x**2
    dG_y = -(y - mu_y)/sigma_y**2
    return dG_x, dG_y

def get_sky_catalog(
    ra=210.8023,
    dec=54.349,
    radius=0.155,
    gbpmagnitude_range=(-3, 20),
    limit=None,
    gaia_keys=[],
    time: Time =Time.now()
) -> dict :
    """Gets a catalog of coordinates on the sky based on an input ra, dec and radius
    
    Gaia keys will add in additional keywords to be grabbed from Gaia catalog."""

    base_keys = ["source_id",
        "ra",
        "dec",
        "parallax",
        "pmra",
        "pmdec",
        "radial_velocity",
        "ruwe",
        "phot_bp_mean_mag",
        "teff_gspphot",
        "logg_gspphot",
        "phot_g_mean_flux", 
        "phot_g_mean_mag",]

    all_keys = base_keys + gaia_keys

    query_str = f"""
    SELECT {f'TOP {limit} ' if limit is not None else ''}* FROM (
        SELECT gaia.{', gaia.'.join(all_keys)}, dr2.teff_val AS dr2_teff_val,
        dr2.rv_template_logg AS dr2_logg, tmass.j_m, tmass.j_msigcom, tmass.ph_qual, DISTANCE(
        POINT({u.Quantity(ra, u.deg).value}, {u.Quantity(dec, u.deg).value}),
        POINT(gaia.ra, gaia.dec)) AS ang_sep,
        EPOCH_PROP_POS(gaia.ra, gaia.dec, gaia.parallax, gaia.pmra, gaia.pmdec,
        gaia.radial_velocity, gaia.ref_epoch, 2000) AS propagated_position_vector
        FROM gaiadr3.gaia_source AS gaia
        JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS xmatch USING (source_id)
        JOIN gaiadr3.dr2_neighbourhood AS xmatch2 ON gaia.source_id = xmatch2.dr3_source_id
        JOIN gaiadr2.gaia_source AS dr2 ON xmatch2.dr2_source_id = dr2.source_id
        JOIN gaiadr3.tmass_psc_xsc_join AS xjoin USING (clean_tmass_psc_xsc_oid)
        JOIN gaiadr1.tmass_original_valid AS tmass ON
        xjoin.original_psc_source_id = tmass.designation
        WHERE 1 = CONTAINS(
        POINT({u.Quantity(ra, u.deg).value}, {u.Quantity(dec, u.deg).value}),
        CIRCLE(gaia.ra, gaia.dec, {(u.Quantity(radius, u.deg) + 50*u.arcsecond).value}))
        AND gaia.parallax IS NOT NULL
        AND gaia.phot_bp_mean_mag > {gbpmagnitude_range[0]}
        AND gaia.phot_bp_mean_mag < {gbpmagnitude_range[1]}) AS subquery
    WHERE 1 = CONTAINS(
    POINT({u.Quantity(ra, u.deg).value}, {u.Quantity(dec, u.deg).value}),
    CIRCLE(COORD1(subquery.propagated_position_vector), COORD2(subquery.propagated_position_vector), {u.Quantity(radius, u.deg).value}))
    ORDER BY ang_sep ASC
    """
    job = Gaia.launch_job_async(query_str, verbose=False)
    tbl = job.get_results()
    if len(tbl) == 0:
        raise ValueError("Could not find matches.")
    plx = tbl["parallax"].value.filled(fill_value=0)
    plx[plx < 0] = 0
    cat = {
        "jmag": tbl["j_m"].data.filled(np.nan),
        "bmag": tbl["phot_bp_mean_mag"].data.filled(np.nan),
        "gmag": tbl["phot_g_mean_mag"].data.filled(np.nan),
        "gflux": tbl["phot_g_mean_flux"].data.filled(np.nan),
        "ang_sep": tbl["ang_sep"].data.filled(np.nan) * u.deg,
    }
    cat["teff"] = (
        tbl["teff_gspphot"].data.filled(
            tbl["dr2_teff_val"].data.filled(np.nan)
        )
        * u.K
    )
    cat["logg"] = tbl["logg_gspphot"].data.filled(
        tbl["dr2_logg"].data.filled(np.nan)
    )
    cat["RUWE"] = tbl["ruwe"].data.filled(99)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cat["coords"] = SkyCoord(
            ra=tbl["ra"].value.data * u.deg,
            dec=tbl["dec"].value.data * u.deg,
            pm_ra_cosdec=tbl["pmra"].value.filled(fill_value=0)
            * u.mas
            / u.year,
            pm_dec=tbl["pmdec"].value.filled(fill_value=0) * u.mas / u.year,
            obstime=Time.strptime("2016", "%Y"),
            distance=Distance(parallax=plx * u.mas, allow_negative=True),
            radial_velocity=tbl["radial_velocity"].value.filled(fill_value=0)
            * u.km
            / u.s,
        ).apply_space_motion(time)
    cat["source_id"] = np.asarray(
        [f"Gaia DR3 {i}" for i in tbl["source_id"].value.data]
    )
    for key in gaia_keys:
        cat[key] = tbl[key].data.filled(np.nan)
    return cat




# Class for managing sparse 3D matrices, originally copied from pandorapsf to avoid a package dependency

class SparseWarp3D(sparse.coo_matrix):
    """Special class for working with stacks of sparse 3D images"""

    def __init__(self, data, row, col, imshape):
        if not np.all([row.ndim == 3, col.ndim == 3, data.ndim == 3]):
            raise ValueError("Pass a 3D array (nrow, ncol, nvecs)")
        self.nvecs = data.shape[-1]
        if not np.all(
            [
                row.shape[-1] == self.nvecs,
                col.shape[-1] == self.nvecs,
            ]
        ):
            raise ValueError("Must have the same 3rd dimension (nvecs).")
        self.subrow = row.astype(int)
        self.subcol = col.astype(int)
        self.subdepth = (
            np.arange(row.shape[-1], dtype=int)[None, None, :]
            * np.ones(row.shape, dtype=int)[:, :, None]
        )
        self.subdata = data
        self._kz = self.subdata != 0

        self.imshape = imshape
        self.subshape = row.shape
        self.cooshape = (np.prod([*self.imshape[:2]]), self.nvecs)
        self.coord = (0, 0)
        super().__init__(self.cooshape)
        index0 = (np.vstack(self.subrow)) * self.imshape[1] + (np.vstack(self.subcol))
        index1 = np.vstack(self.subdepth).ravel()
        self._index_no_offset = np.vstack([index0.ravel(), index1.ravel()])
        self._submask_no_offset = np.vstack(self._get_submask(offset=(0, 0))).ravel()
        self._subrow_v = deepcopy(np.vstack(self.subrow).ravel())
        self._subcol_v = deepcopy(np.vstack(self.subcol).ravel())
        self._subdata_v = deepcopy(np.vstack(deepcopy(self.subdata)).ravel())
        self._index1 = np.vstack(self.subdepth).ravel()

        self._set_data()

    def __add__(self, other):
        if isinstance(other, SparseWarp3D):
            data = deepcopy(self.subdata + other.subdata)
            if (
                (self.subcol != other.subcol)
                | (self.subrow != other.subrow)
                | (self.imshape != other.imshape)
                | (self.subshape != other.subshape)
            ):
                raise ValueError("Must have same base indicies.")
            return SparseWarp3D(
                data=data, row=self.subrow, col=self.subcol, imshape=self.imshape
            )
        else:
            return super(sparse.coo_matrix, self).__add__(other)

    def tocoo(self):
        return sparse.coo_matrix((self.data, (self.row, self.col)), shape=self.cooshape)

    def index(self, offset=(0, 0)):
        """Get the 2D positions of the data"""
        if offset == (0, 0):
            return self._index_no_offset
        index0 = (self._subrow_v + offset[0]) * self.imshape[1] + (
            self._subcol_v + offset[1]
        )
        #        index1 = np.vstack(self.subdepth).ravel()
        #        return np.vstack([index0.ravel(), index1.ravel()])
        # index0 = (self._subrow_v + offset[0]) * self.imshape[1] + (
        #     self._subcol_v * offset[1]
        # )
        return index0, self._index1
        # index0 = (self._subrow_v + offset[0]) * self.imshape[1] + (
        #     self._subcol_v * offset[1]
        # )
        # return index0, self._index1

    def _get_submask(self, offset=(0, 0)):
        # find where the data is within the array bounds
        kr = ((self.subrow + offset[0]) < self.imshape[0]) & (
            (self.subrow + offset[0]) >= 0
        )
        kc = ((self.subcol + offset[1]) < self.imshape[1]) & (
            (self.subcol + offset[1]) >= 0
        )
        return kr & kc & self._kz

    def _set_data(self, offset=(0, 0)):
        if offset == (0, 0):
            index0, index1 = self.index((0, 0))
            self.row, self.col = (
                index0[self._submask_no_offset],
                index1[self._submask_no_offset],
            )
            self.data = self._subdata_v[self._submask_no_offset]
        else:
            # find where the data is within the array bounds
            k = self._get_submask(offset=offset)
            k = np.vstack(k).ravel()
            new_row, new_col = self.index(offset=offset)
            self.row, self.col = new_row[k], new_col[k]
            self.data = self._subdata_v[k]
        self.coord = offset

    def __repr__(self):
        return (
            f"<{(*self.imshape, self.nvecs)} SparseWarp3D array of type {self.dtype}>"
        )

    def dot(self, other):
        if other.ndim == 1:
            other = other[:, None]
        nt = other.shape[1]
        return super().dot(other).reshape((*self.imshape, nt)).transpose([2, 0, 1])

    def reset(self):
        """Reset any translation back to the original data"""
        self._set_data(offset=(0, 0))
        self.coord = (0, 0)
        return

    def clear(self):
        """Clear data in the array"""
        self.data = np.asarray([])
        self.row = np.asarray([])
        self.col = np.asarray([])
        self.coord = (0, 0)
        return

    def translate(self, position: Tuple):
        """Translate the data in the array by `position` in (row, column)"""
        self.reset()
        # If translating to (0, 0), do nothing
        if position == (0, 0):
            return
        self.clear()
        self._set_data(position)
        return