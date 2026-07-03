"""
run_pipeline.py

End-to-end driver implementing the algorithm described in
Dalmasso, Trenti & Leethochawalit (2023), Sec. 4.1.2, for generating a
depth-aware random-point catalog via Monte Carlo source injection and
recovery, then using it (together with a real LBG catalog) to compute
the angular two-point correlation function.

Algorithm
---------
 1. Load real science + weight (RMS) FITS images for the detection band.
 2. Build N_inject fake Sersic-profile LBGs with a dropout SED, at
    uniformly random positions across the FULL image footprint
    (including shallow/edge regions -- let the data decide what's
    recoverable, don't pre-mask by hand).
 3. Inject the fake sources directly into the real science image.
 4. Run source detection (photutils image segmentation) on the
    injected image -- the SAME detection settings used for the real
    data.
 5. Match detections back to the injected truth positions.
 6. Apply the same M_UV magnitude-limited selection cut used for real
    LBG candidates.
 7. Keep only fake sources that are BOTH recovered AND pass the cut --
    their (RA, Dec) positions become the random-point catalog, with
    spatial density automatically suppressed in low-completeness
    regions exactly as the survey itself would suppress real source
    counts there.
 8. Repeat injection until the random catalog reaches the target size
    N_r = 20 * N_d (Sec. 4, as in the paper).
 9. Combine with the real data catalog to compute DD, DR, RR pair
    counts and the Landy-Szalay w(theta) estimator, with bootstrap
    errors, and fit the power-law amplitude A_w with beta fixed = 0.6.

This is a complete, runnable implementation of the ALGORITHM described
in the paper. It is not a reproduction of the authors' actual source
code (which is not published in the paper) -- PSF model, Sersic
parameter ranges, SED template, and detection-threshold choices below
are reasonable implementation choices, documented inline, standing in
for details the paper does not specify numerically.
"""

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from inject_sources import inject_fake_sources
from detect_recover import detect_in_image, match_recovered, apply_selection_cut
from acf_estimator import (
    pair_counts,
    landy_szalay,
    bootstrap_errors,
    compute_ic_ratio,
    fit_power_law_mle,
    limber_transform_Aw_to_r0,
    galaxy_bias,
)

# Load real images
def load_field(science_path, weight_path):
    with fits.open(science_path) as hdul:
        hdul.info()
        sci_hdu = hdul["SCI"]
        science_data = sci_hdu.data.astype(float)
        wcs = WCS(sci_hdu.header)
        zeropoint_ab = sci_hdu.header.get("ZP_AB", 28.9) # For F277W 

    with fits.open(weight_path) as hdul:
        hdul.info()
        try:
            wht_hdu = hdul["SCI"]
        except:    
            wht_hdu = hdul[1] # In case there is different survey
        weight_data = wht_hdu.data.astype(float)
    return science_data, weight_data, wcs, zeropoint_ab

# One injection-and-recovery round
def one_injection_round(science_data, weight_data, zeropoint_ab, psf_fwhm_pix, 
                        n_inject, z_drop, M_UV_range, M_UV_cut, rng):
    injected_data, truth = inject_fake_sources(science_data, weight_data, zeropoint_ab,
                                               psf_fwhm_pix, n_inject, z_drop, M_UV_range, rng)

    cat, _ = detect_in_image(injected_data, weight_data)
    recovered, _ = match_recovered(cat, truth, match_radius_pix=2.0)
    keep = apply_selection_cut(truth, recovered, M_UV_cut=M_UV_cut)

    return truth["x"][keep], truth["y"][keep]


# Step 8: Build the full random catalog to the target size
def build_random_catalog(science_data, weight_data, wcs, zeropoint_ab, 
                         psf_fwhm_pix, n_target, z_drop, M_UV_range, 
                         M_UV_cut, rng, n_inject_per_round=500, max_rounds=200):
    """
    Repeatedly injects and recovers fake sources until n_target random
    points have survived detection + selection, then returns their sky
    coordinates (RA, Dec).
    """
    xs_kept, ys_kept = [], []
    n_have = 0

    for round_i in range(max_rounds):
        x_round, y_round = one_injection_round(science_data, weight_data, zeropoint_ab,
                                               psf_fwhm_pix, n_inject_per_round, z_drop,
                                               M_UV_range, M_UV_cut, rng)
        xs_kept.append(x_round)
        ys_kept.append(y_round)
        n_have += len(x_round)

        print(f"  injection round {round_i+1}: "
              f"+{len(x_round)} recovered (running total {n_have}/{n_target})")

        if n_have >= n_target:
            break

    x_all = np.concatenate(xs_kept)[:n_target]
    y_all = np.concatenate(ys_kept)[:n_target]

    ra, dec = wcs.all_pix2world(x_all, y_all, 0)
    return ra, dec


# Step 9: Compute the ACF, fit A_w via MLE (with proper IC), then
#         derive r_0 (Limber transform) and galaxy bias -- Eq. 1-7
def compute_acf_and_bias(ra_data, dec_data, ra_rand, dec_rand, z_central, N_z_func,
                         cosmo, h=0.678,  sigma8_0=0.828, theta_min_arcsec=12.5,
                         theta_max_arcsec=250.0, bin_width_arcsec=12.5, beta=0.6, n_boot=200,
                         rng=None, z_integration_range=None):
    """
    Full clustering pipeline, Eq. 1-7:

      1. Linear binning, 12.5" wide, theta_max=250" (paper's Sec. 3 choice).
      2. Landy-Szalay w_obs(theta)  [Eq. 1]
      3. Bootstrap errors on w_obs  [Ling, Frenk & Barrow 1986]
      4. IC/A_w from the random catalog's own RR(theta)  [Eq. 2-3]
      5. Maximum-likelihood fit for A_w  [Eq. 4-5]
      6. Limber transform A_w -> r_0  [Eq. 6, Adelberger et al. 2005]
      7. Galaxy bias b = sigma_8,g / sigma_8(z)  [Eq. 7]

    Returns a dict with every intermediate quantity, not just the
    final bias, so each step can be inspected/sanity-checked.
    """
    theta_bins = np.arange(theta_min_arcsec, theta_max_arcsec + bin_width_arcsec, bin_width_arcsec)
    theta_centers = 0.5 * (theta_bins[:-1] + theta_bins[1:])

    n_data = len(ra_data)
    n_rand = len(ra_rand)

    DD = pair_counts(ra_data, dec_data, ra_data, dec_data, theta_bins, same_catalog=True)
    DR = pair_counts(ra_data, dec_data, ra_rand, dec_rand, theta_bins)
    RR = pair_counts(ra_rand, dec_rand, ra_rand, dec_rand, theta_bins, same_catalog=True)

    w_obs = landy_szalay(DD, DR, RR, n_data, n_rand)
    w_err = bootstrap_errors(ra_data, dec_data, ra_rand, dec_rand, theta_bins, n_boot=n_boot, rng=rng)

    A_w, A_w_err, ic_over_Aw = fit_power_law_mle(theta_centers, w_obs, w_err, RR, beta=beta)

    if z_integration_range is None:
        z_integration_range = (max(0.0, z_central - 1.5), z_central + 1.5)
    z_grid = np.linspace(*z_integration_range, 300)

    r0_h_inv_mpc = limber_transform_Aw_to_r0(A_w, beta, N_z_func, z_grid, cosmo, h=h)

    gamma = beta + 1.0
    sigma8_g, sigma8_z, bias = galaxy_bias(
        r0_h_inv_mpc, gamma, z_central, cosmo, sigma8_0=sigma8_0
    )

    return {
        "theta_centers": theta_centers, "theta_bins": theta_bins,
        "DD": DD, "DR": DR, "RR": RR,
        "w_obs": w_obs, "w_err": w_err,
        "ic_over_Aw": ic_over_Aw, "A_w": A_w, "A_w_err": A_w_err,
        "r0_h_inv_mpc": r0_h_inv_mpc,
        "gamma": gamma,
        "sigma8_g": sigma8_g, "sigma8_z": sigma8_z,
        "bias": bias,
    }


# ---------------------------------------------------------------------
# Example end-to-end usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # --- user-editable paths / parameters ---
    SCIENCE_FITS = "hlsp_ceers_jwst_nircam_fullceers_f277w_v1_sci.fits.gz"
    WEIGHT_FITS = "hlsp_ceers_jwst_nircam_fullceers_f277w_v1_wht.fits.gz"
    PSF_FWHM_PIX = 3.0          # Approximate JWST/NIRCam F277W Gaussian PSF FWHM (pixels)
    Z_DROP = 8.5
    M_UV_RANGE = (-24.0, -15.0)
    M_UV_CUT = np.inf
    N_RANDOM_TARGET = 20 * 44  # N_r = 20 * N_d, per Sec. 4

    # Real LBG catalog positions (RA, Dec in degrees) -- load your own
    # selected sample here. Placeholder arrays shown for structure only.
    ra_data, dec_data = np.loadtxt("CEERS_z8.5_selected.csv", delimiter=",", skiprows=1, usecols=(0, 1), unpack=True)

    science_data, weight_data, wcs, zeropoint_ab = load_field(SCIENCE_FITS, WEIGHT_FITS)
    nx, ny = science_data.shape
    n_random = 3000
    x_rand = rng.integers(0, nx, size=n_random)
    y_rand = rng.integers(0, ny, size=n_random)

    print("Building depth-aware random catalog via injection-recovery...")
    ra_rand, dec_rand = build_random_catalog(
        science_data,
        weight_data,
        wcs,
        zeropoint_ab,
        PSF_FWHM_PIX,
        N_RANDOM_TARGET,
        Z_DROP,
        M_UV_RANGE,
        M_UV_CUT,
        rng,
    )
    random_catalog = np.column_stack((ra_rand, dec_rand))
    np.savetxt("random_catalog.txt", random_catalog, fmt="%.8f", header="RA(deg)    DEC(deg)", comments="")
    print("Random catalog saved as random_catalog.txt")
    print(f"Random catalog complete: {len(ra_rand)} points")

    # --- Cosmology and N(z), needed for the Limber transform (Eq. 6) ---
    from astropy.cosmology import FlatLambdaCDM

    cosmo = FlatLambdaCDM(H0=67.8, Om0=0.308)

    def N_z(z, z0=Z_DROP, sigma_z=0.3):
        # Placeholder Gaussian dropout selection window. Replace with
        # the actual completeness-weighted N(z) from your own
        # injection-recovery results (i.e. the redshift distribution
        # of RECOVERED fake sources, not an assumed analytic shape) --
        # the paper builds N(z) from the same Monte Carlo recovery
        # process used for the random catalog, per Sec. 4.1.2.
        return np.exp(-0.5 * ((z - z0) / sigma_z) ** 2)

    # Uncomment once a real catalog is loaded:
    results = compute_acf_and_bias(
        ra_data, dec_data, ra_rand, dec_rand,
        z_central=Z_DROP, N_z_func=N_z, cosmo=cosmo, h=0.678,
    )
    print("A_w =", results["A_w"], "+/-", results["A_w_err"])
    print("IC/A_w =", results["ic_over_Aw"])
    print("r_0 =", results["r0_h_inv_mpc"], "h^-1 Mpc")
    print("sigma_8,g =", results["sigma8_g"])
    print("sigma_8(z) =", results["sigma8_z"])
    print("galaxy bias b =", results["bias"])



    # Plotting
    import matplotlib.pyplot as plt
    import numpy as np
    theta, w, err, beta = results["theta_centers"], results["w_obs"], results["w_err"], 0.6
    theta_fit = np.linspace(theta.min(), theta.max(), 300)
    w_fit = results["A_w"] * theta_fit**(-beta)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    ax[0].scatter(ra_rand, dec_rand, s=2, color="royalblue")
    ax[0].set_xlabel("RA (deg)", fontsize=12)
    ax[0].set_ylabel("DEC (deg)", fontsize=12)
    ax[0].set_title("Random Catalog", fontsize=14)
    ax[0].grid(alpha=0.3)
    ax[0].invert_xaxis()    # Astronomical convention 

    ax[1].errorbar(theta, w, yerr=err, fmt='o', color='black', 
                   markersize=5, capsize=3,label='Measured $w(\\theta)$')

    # Plot fitted power law only if positive
    if results["A_w"] > 0:
        ax[1].plot(theta_fit, w_fit, color='red', linewidth=2,
                    label=r'Best fit: $A_w\theta^{-0.6}$')
    # ax[1].set_xscale("log")
    # Use logarithmic y-axis only if all values are positive
    if np.all(w > 0):
        ax[1].set_yscale("log")
    ax[1].set_xlabel("Angular Separation (arcsec)", fontsize=12)
    ax[1].set_ylabel(r"$w(\theta)$", fontsize=12)
    ax[1].set_title("Angular Two-Point Correlation Function", fontsize=14)
    ax[1].grid(True, which="both", alpha=0.3)
    ax[1].legend()
    # Display fitted amplitude
    ax[1].text(
        0.05,
        0.95,
        rf"$A_w = {results['A_w']:.4f}$" "\n"
        rf"$\sigma(A_w) = {results['A_w_err']:.4f}$" "\n"
        rf"$\beta = {beta}$",
        transform=ax[1].transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(facecolor="white", edgecolor="black")
    )

    plt.savefig("results.png", dpi=300, bbox_inches="tight")
    plt.show()