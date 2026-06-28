"""
run_pipeline.py
End-toend driver implementing the algorithm described in Dalmasso, 
Trenti & Leethochawalit (2023), section 4.1.2, for generating a 
depth-aware random-point catalog via Monte Carlo source injection 
and recovery, then using it (together with areal LBG catalog) to 
compute the angular two-point correlation function.

Algorithm
---------
 1. Load real science + weight (RMS) FITS images for the detection band.
 2. Build N_inject fake Sersic-profile LBGs with a dropout SED, at 
    uniformly random positions across the FULL image footprint (including
    shallow/edge regions -- let the data decide what's recoverable, don't
    pre-mask by hand).
 3. Inject the fake sources directly into the real science image.
 4. Run sources detection (photutils image segmentation) on the real data.
 5. Match detection back to the injected truth positions.
 6. Apply the same M_UV magnitude-limited selection cut used for real LBG 
    candidates.
 7. Keep only fake sources that are BOTH recovered AND pass the cut --
    their (RA, Dec) positions become the random-point catalog, with 
    spatial density autometically suppressed in low-completeness regions
    exactly as the survey itself would suppress real source counts there.
 8. Repeat injection untill the random catalog reaches the target size 
    N_r = 20 * N_d (sec. 4, as in the paper).
 9. Combine with the real data catalog to compute DD, DR, RR pair counts
    and the Landy-Szalay w(theta) estimator, with bootstrap errors, and 
    fot the power-law amplitude A_w with beta fixed = 0.6.

This is a complete, runable implimentation of the ALGORITHM described in the
paper. It is not a reproduction of the auther's actual source code (which is not 
published in the paper) -- PSF model, Sersic parameter ranges, SED template, and 
detection-threshold choices below are reasonable implementation choices, documented
inline, standing in for details the paper does not specify numerically.

For CEERS (JWST/NIRCam), the zeropoint depends on the filter.
| Filter | Typical AB Zeropoint |
| ------ | -------------------: |
| F115W  |                ~28.0 |
| F150W  |                ~28.3 |
| F200W  |                ~28.6 |
| F277W  |           ~28.8-29.0 |
| F356W  |                ~28.8 |
| F444W  |                ~28.8 |
"""
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from inject_sources import inject_fake_sources
from detect_recover import detect_in_image, match_recovered, apply_selection_cut
from acf_estimator import pair_counts, landy_szalay, bootstrap_errors, fit_power_law

# Load real images
def load_feild(science_path, weight_path):
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

# One injection and recovery round 
def one_injection_round(science_data, weight_data, zeropoint_ab, psf_fwhm_pix, 
                        n_inject, z_drop, M_UV_range, M_UV_cut, rng):
    
    injected_data, truth = inject_fake_sources(science_data, weight_data, zeropoint_ab, 
                                               psf_fwhm_pix, n_inject, z_drop, M_UV_range, rng)
    cat, _ = detect_in_image(injected_data, weight_data)
    recovered, _ = match_recovered(cat, truth, match_radius_pix=2.0)
    keep = apply_selection_cut(truth, recovered, M_UV_cut=M_UV_cut)

    return truth["x"][keep], truth["y"][keep]

# Build the full random catalog to the target size
def build_random_catalog(science_data, weight_data, wcs, zeropoint_ab, psf_fwhm_pix, n_target, 
                         z_drop, M_UV_range, M_UV_cut, rng, n_inject_per_round=2000, max_rounds=200):
    """
    Repeatedly inject and recovers fake sources untill n_target random
    points have survived detection + selection, then returns their sky
    coordinates (RA, Dec).
    """
    xs_kept, ys_kept = [], []
    n_have = 0

    for round_i in range(max_rounds):
        x_round, y_round = one_injection_round(science_data, weight_data, zeropoint_ab, psf_fwhm_pix, 
                                                n_inject_per_round, z_drop, M_UV_range, M_UV_cut, rng)
        xs_kept.append(x_round)
        ys_kept.append(y_round)
        n_have += len(x_round)

        print(f"injection round {round_i+1}:"
              f"+{len(x_round)} recovered (running total {n_have}/{n_target})")
        
        if n_have >= n_target:
            break

    x_all = np.concatenate(xs_kept)[:n_target]
    y_all = np.concatenate(ys_kept)[:n_target]

    ra, dec = wcs.all_pix2world(x_all, y_all, 0)
    return ra, dec

# Compute the ACF using the random catalog
def compute_acf(ra_data, dec_data, ra_rand, dec_rand, theta_min_arcsec=12.5, theta_max_arcsec=250.0, 
                n_bins=20, beta=0.6, integral_constraint_ratio=0.0, n_boot=200, rng=None):
    theta_bins = np.linspace(theta_min_arcsec, theta_max_arcsec, n_bins+1)
    theta_center = 0.5 * (theta_bins[:-1] + theta_bins[1:])
    n_data, n_rand = len(ra_data), len(ra_rand)

    DD = pair_counts(ra_data, dec_data, ra_data, dec_data, theta_bins, same_catalog=True)
    DR = pair_counts(ra_data, dec_data, ra_rand, dec_rand, theta_bins)
    RR = pair_counts(ra_rand, dec_rand, ra_rand, dec_rand, theta_bins, same_catalog=True)

    w_obs = landy_szalay(DD, DR, RR, n_data, n_rand)
    w_err = bootstrap_errors(ra_data, dec_data, ra_rand, dec_rand, theta_bins, n_boot=n_boot, rng=rng)

    A_w, A_w_err = fit_power_law(theta_center, w_obs, w_err, beta=beta, 
                                 integral_constraint_ratio=integral_constraint_ratio)
    
    return {"theta_centers": theta_center, "w_obs": w_obs,
            "w_err": w_err, "A_w": A_w, "A_w_err": A_w_err}

# Example end-to-end usage
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    SCIENCE_FITS = "hlsp_ceers_jwst_nircam_fullceers_f277w_v1_sci.fits.gz"
    WEIGHT_FITS = "hlsp_ceers_jwst_nircam_fullceers_f277w_v1_wht.fits.gz"
    PSF_FWHM_PIX = 2.0          # H160 PSF FWHM in WFC3/IR pixels (~0.18" / 0.06"/pix)
    Z_DROP = 8.7
    M_UV_RANGE = (-22.0, -18.0)
    M_UV_CUT = -20.0
    N_RANDOM_TARGET = 20 * 160  # N_r = 20 * N_d, per Sec. 4

    # Real LBG catalog positions (RA, Dec in degrees) -- load your own
    # selected sample here, Placeholder array shown for structure only.
    # ra_data, dec_data = np.loadtxt("real_lbg_catalog.txt", unpack=True)

    science_data, weight_data, wcs, zeropoint_ab = load_feild(SCIENCE_FITS, WEIGHT_FITS)
    nx, ny = science_data.shape
    n_random = 3000
    x_rand = rng.integers(0, nx, size=n_random)
    y_rand = rng.integers(0, ny, size=n_random)

    print("Building depth-aware random catalog via injection-recovery ...")
    ra_rand, dec_rand = build_random_catalog(science_data, weight_data, wcs, zeropoint_ab, PSF_FWHM_PIX, 
                                             N_RANDOM_TARGET, Z_DROP, M_UV_RANGE, M_UV_CUT, rng)
    random_catalog = np.column_stack((ra_rand, dec_rand))
    np.savetxt("random_catalog.txt", random_catalog, fmt="%.8f", header="RA(deg)    DEC(deg)", comments="")
    print("Random catalog saved as random_catalog.txt")
    print(f"Random catalog complete: {len(ra_rand)} points")

    # Uncomment once a real catalog is loaded:
    # results = compute_acf(ra_data, dec_data, ra_rand, dec_rand)
    # print("A_w =", results["A_w"], "+/-", results["A_w_err"])

    import matplotlib.pyplot as plt
    import numpy as np

    # Results from compute_acf()
    # theta, w, err, beta = results["theta_centers"], results["w_obs"], results["w_err"], 0.6
    # theta_fit = np.linspace(theta.min(), theta.max(), 300)
    # w_fit = results["A_w"] * theta_fit**(-beta)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    ax[0].scatter(ra_rand, dec_rand, s=2, color="royalblue")
    ax[0].set_xlabel("RA (deg)", fontsize=12)
    ax[0].set_ylabel("DEC (deg)", fontsize=12)
    ax[0].set_title("Random Catalog", fontsize=14)
    ax[0].grid(alpha=0.3)
    ax[0].invert_xaxis()    # Astronomical convention 

    # ax[1].errorbar(theta, w, yerr=err, fmt='o', color='black', 
    #                markersize=5, capsize=3,label='Measured $w(\\theta)$')

    # # Plot fitted power law only if positive
    # if results["A_w"] > 0:
    #     ax[1].plot(theta_fit, w_fit, color='red', linewidth=2,
    #                 label=r'Best fit: $A_w\theta^{-0.6}$')
    # ax[1].set_xscale("log")
    # # Use logarithmic y-axis only if all values are positive
    # if np.all(w > 0):
    #     ax[1].set_yscale("log")
    # ax[1].set_xlabel("Angular Separation (arcsec)", fontsize=12)
    # ax[1].set_ylabel(r"$w(\theta)$", fontsize=12)
    # ax[1].set_title("Angular Two-Point Correlation Function", fontsize=14)
    # ax[1].grid(True, which="both", alpha=0.3)
    # ax[1].legend()
    # # Display fitted amplitude
    # ax[1].text(
    #     0.05,
    #     0.95,
    #     rf"$A_w = {results['A_w']:.4f}$" "\n"
    #     rf"$\sigma(A_w) = {results['A_w_err']:.4f}$" "\n"
    #     rf"$\beta = {beta}$",
    #     transform=ax[1].transAxes,
    #     fontsize=11,
    #     verticalalignment="top",
    #     bbox=dict(facecolor="white", edgecolor="black")
    # )

    plt.savefig("results.png", dpi=300, bbox_inches="tight")
    plt.show()