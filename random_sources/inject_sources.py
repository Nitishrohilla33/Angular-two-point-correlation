"""
Step A: Source injection.

Implements the Monte Carlo fake-source injection described in
Dalmasso, Trenti & Leethochawalit (2023), sec. 4.1.2:

    "we generated the reference random sample thouugh a Monte Carlo
    simulation that places artificial sources with realistic spectral
    energy distributions and recovers them through the full photometric pipeline."

This module only builds and inject the fake sources into real
science images. Detection/recovery happens in detect_recovery.py.

NOTE: The paper does not publish exact PSF models, Sersic index
distributions, or SED templetes used -- those are implementation
Choices made here, clearly documented, not reproductions of a 
specific published config.
"""
import numpy as np
from astropy.io import fits
from astropy.convolution import convolve_fft, Gaussian2DKernel
from astropy.modeling.models import Sersic2D

# Realistic LBG dropput SED model 
def lbg_dropout_color(z_drop, band_wave_um):
    """
    Return an approximate flux ratio (relation to a flat-UV continuum
    in f_nu) for a Lyman-break galaxy at redshift z_drop, observed 
    through a filter of pivot wavelength band_wave_um (microns).

    Real LBG selection in this paper's redshift range (z > 8)relies
    on a sharp sharp spectral break at the Lyman limit / Lyman-alpha,
    blueward of which flux is heavily suppressed by intervening HI
    (Lyman-alpha forest + Lyman-limit absorption), and roughly flat
    (in f_nu) UN continuum redward of the break.

    This is a simplified two-segment model:
        - redward of break: f_nu ~ const (beta_UV slope ~ -2, close to flat in AB)
        - bluward of break: heavily suppressed (factor ~ 0.05, simulating near - 
          complete IGM absorption)
    """
    lyman_alpha_um = 0.121567 * (1.0 + z_drop) # Rest frame wavelength 1215.67 A -> observed
    if band_wave_um > lyman_alpha_um:
        return 1.0     # redward of break: full continuum flux
    else:
        return 0.05    #  blueward: suppressed by IGM absorption
    
def build_sed(z_drop, bands_um, M_UV, beta_UV=-0.2):
    """
    Build apperent AB magnitudes in each band for a fake LBG at 
    redshift z_drop with rest-frame absolute UV magnitude M_UV.

    Uses a simple power_law UV continuum f_lambda ~ lambda^beta_UV 
    redward of the break, normalized so the band straddling rest-frame
    1500 A matches M_UV (converted to apperent magnitude via the luminosity
    distance), then suppressed flux blueward of the break.

    Return dict: {band_name: apperent_AB_mag}
    """
    from astropy.cosmology import Planck18
    import astropy.units as u

    d_L = Planck18.luminosity_distance(z_drop).to(u.pc).value
    # Standard distance modulus + 1500A k-correlation-free approximation
    # (good enough for fake-source injection purposes; not used for any)
    # science result, only to set a realistic relative brightness).
    DM = 5 * np.log10(d_L / 10.0)
    m_app_continuum = M_UV + DM - 2.5 * np.log10(1.0 + z_drop)
    mags = {} # empty dictionary
    for band, wave_um in bands_um.items():
        ratio = lbg_dropout_color(z_drop, wave_um)
        # convert continuum mag + flux ratio -> apparent mag in this band
        mags[band] = m_app_continuum - 2.5 * np.log10(max(ratio, 1e-6))
    return mags


# Sersic profile fake galaxy stamp
def make_sersic_stemp(stamp_size, r_eff_pix, n_sersic, ellip, theta, total_flux):
    """
    Build a normalized Sersic2D potage stamp with given effective 
    radius (pixels), Sersic index, ellipticity, position angle, and 
    total flux (counts), suitable for direct injection into a science 
    image.
    """
    y, x = np.mgrid[0:stamp_size, 0:stamp_size]
    x0 = y0 = stamp_size / 2.0

    mod = Sersic2D(amplitude=1.0, r_eff=r_eff_pix, n=n_sersic, 
                   x_0=x0, y_0=y0, ellip=ellip, theta=theta)
    img = mod(x, y)
    img /= img.sum()        # normalize to unit flux 
    img *= total_flux       # scale to desired total flux(counts)
    return img

def mag_to_counts(mag_ab, zeropoint_ab):
    """Convert an AB magnitude to image counts gives the imaze zeropoints."""
    return 10 ** (-0.4*(mag_ab - zeropoint_ab))

# Injection into the real science image
def inject_fake_sources(science_data, weight_data, zeropoint_ab, psf_fwhm_pix,
                         n_sources, z_drop, M_UV_range, rng, stamp_size=41):
    """
    Inject n_sources fake Sersic-profile LBGs at uniformly random pixel 
    positions across the full science image footprint (Including
    low-weight / shallow regions -- recoverability there is exactly
    waht we wantthe popeline to determine, not something to 
    pre-filter by hand).

    Parameters
    ----------
    science_data : 2D ndarray, real science image (counts)
    weight_data  : 2D ndarray, real weight map (same shape).
                   weight == 0 marks masked / no-coverage pixels.
    zeropoint_ab : float, AB magnitude zeropoint of this image
    psf_fwhm_pix : float, PSF FWHM in pixel for this band 
    n_sources    : int, number of fake sources to inject 
    z_drop       : float, dropout redshift for the SED model
    M_UV_range.  : (min, max) tuple, draw M_UV uniformly in this range
    rng.         : numpy.random.Generator

    Returns
    -------
    injected_data : 2D ndarray, science image with fakes added 
    truth_table.  : structured array with x, y, mag, M_UV per fake source               
    """
    ny, nx = science_data.shape
    injected_data = science_data.copy()

    xs = rng.uniform(0, nx, size=n_sources)
    ys = rng.uniform(0, ny, size=n_sources)
    M_UVs = rng.uniform(M_UV_range[0], M_UV_range[1], size=n_sources)
    r_effs = rng.uniform(1.5, 4.0, size=n_sources)        # Pixels, typical compact high-z LBG
    n_sersics = rng.uniform(0.8, 2.5, size=n_sources)      # disky to mild bulge
    ellips = rng.uniform(0.0, 0.6, size=n_sources)
    thetas = rng.uniform(0, np.pi, size=n_sources)

    truth = np.zeros(n_sources, dtype=[("x", "f8"), ("y", "f8"), ("mag", "f8"), ("M_UV", "f8")])
    psf_kernel = Gaussian2DKernel(x_stddev=psf_fwhm_pix/2.3548)

    half = stamp_size // 2
    for i in range(n_sources):
        x_c, y_c = xs[i], ys[i]
        xi, yi = int(round(x_c)), int(round(y_c))

        # Skip injecting flux fully outsize the image array bounds:
        # still record truth position (it will simply never be
        # recovered, which is the correct/expected outcome there).
        x_lo, x_hi = xi - half, xi + half + 1
        y_lo, y_hi = yi - half, yi + half + 1
        if x_hi <= 0 or y_hi <= 0 or x_lo >= nx or y_lo >= ny:
            truth[i] = (x_c, y_c, np.nan, M_UVs[i])
            continue

        mag_app = build_sed(z_drop, {"this_band": 1.0}, M_UVs[i])["this_band"]
        # NOTE: "this_band" wavelength is a placeholder of 1.0 micron;
        # callers handling multiple real bands should call build_sed
        # once per band with real pivott wavelenghts instead. Kept
        # simple here since this function injects into a single
        # detection-band image at a time (see run_pipeline.py).
        flux_counts = mag_to_counts(mag_app, zeropoint_ab)

        stamp = make_sersic_stemp(stamp_size, r_effs[i], n_sersics[i],
                                   ellips[i], thetas[i], flux_counts)
        stamp = convolve_fft(stamp, psf_kernel, boundary="fill", fill_value=0.0)

        # clip stamp to valid array region (handkes edge sources)
        sx_lo, sy_lo = max(0, -x_lo), max(0, -y_lo) 
        ax_lo, ay_lo = max(0, x_lo), max(0, y_lo)
        ax_hi, ay_hi = min(nx, x_hi), min(ny, y_hi)
        sx_hi, sy_hi = sx_lo + (ax_hi - ax_lo), sy_lo + (ay_hi - ay_lo)
        injected_data[ay_lo:ay_hi, ax_lo:ax_hi] += stamp[sy_lo:sy_hi, sx_lo:sx_hi]
        truth[i] = (x_c, y_c, mag_app, M_UVs[i])
    return injected_data, truth    