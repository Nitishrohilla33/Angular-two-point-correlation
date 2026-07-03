"""
test_pipeline_synthetic.py

Test harness for the injection-recovery / ACF pipeline, using the
synthetic wedding-cake field from make_test_field.py instead of the
real CEERS mosaic.

Five stages, each testing a different layer of the pipeline in
isolation, so a failure tells you WHICH module broke rather than just
"something is wrong":

  A. Detection mechanics    -> detect_in_image() only
  B. Matching mechanics     -> match_recovered() / apply_selection_cut() only
  C. Completeness vs depth  -> full inject+detect+match loop, per depth zone
  D. ACF math (no detection)-> pair_counts()/landy_szalay()/fit_power_law_mle()
                                on KNOWN synthetic catalogs (positive control)
  E. End-to-end smoke test  -> the full pipeline, just checking it runs
                                and produces sane, non-crashing output

Run: python test_pipeline_synthetic.py
Requires make_test_field.py's science/weight FITS to exist (or run it
first -- see main() below).
"""
import numpy as np
from astropy.io import fits
from astropy.table import Table

# --- import your actual pipeline modules -----------------------------------
# Adjust these imports to match your real module names/paths.
from detect_recover import detect_in_image, match_recovered, apply_selection_cut
from acf_estimator import pair_counts, landy_szalay, fit_power_law_mle, bootstrap_errors

rng = np.random.default_rng(42)


# =============================================================================
# Helper: minimal Gaussian-blob injector for Stages A-C.
# This is deliberately NOT your real Sersic injector -- it's a stripped-down
# stand-in so Stages A-C test the DETECTION side in isolation, without also
# depending on your Sersic/SED/flux-conversion code being bug-free. Once
# A-C pass, swap this for your real inject_sources.inject_galaxy() to test
# the injector itself against the same field.
# =============================================================================
def inject_gaussian_source(image, x, y, flux, fwhm_pix=3.0, stamp_half=15):
    sigma = fwhm_pix / 2.3548
    y0, x0 = int(round(y)), int(round(x))
    yy, xx = np.mgrid[y0 - stamp_half:y0 + stamp_half + 1,
                       x0 - stamp_half:x0 + stamp_half + 1]
    valid = (yy >= 0) & (yy < image.shape[0]) & (xx >= 0) & (xx < image.shape[1])
    profile = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    profile /= profile.sum()
    image[yy[valid], xx[valid]] += flux * profile[valid]
    return image


def load_field():
    sci = fits.getdata("ceers_f277w_sci.fits").astype(np.float64)
    wht = fits.getdata("ceers_f277w_wht.fits").astype(np.float64)
    return sci, wht


def zone_masks(ny, nx):
    top, bottom = ny // 3, 2 * ny // 3
    return {
        "deep_center": (slice(top, bottom), slice(0, nx)),
        "shallow_top": (slice(0, top), slice(0, nx)),
        "shallow_bottom": (slice(bottom, ny), slice(0, nx)),
    }


# =============================================================================
# Stage A: detection mechanics only
# =============================================================================
def stage_a_detection_mechanics():
    print("\n=== Stage A: detection mechanics ===")
    sci, wht = load_field()
    ny, nx = sci.shape
    zones = zone_masks(ny, nx)

    # Inject one very bright source dead-center in each zone, plus one
    # inside the star mask (should NOT be detected).
    truth_xy = {
        "deep_center": (300, (zones["deep_center"][0].start + zones["deep_center"][0].stop) // 2),
        "shallow_top": (300, zones["shallow_top"][0].stop // 2),
        "shallow_bottom": (300, (zones["shallow_bottom"][0].start + ny) // 2),
        "inside_star_mask": (450, 300),  # matches star_mask center from make_test_field.py
    }
    bright_flux = 500.0  # very bright, should be trivially detectable anywhere except the mask
    injected = sci.copy()
    for name, (x, y) in truth_xy.items():
        injected = inject_gaussian_source(injected, x, y, bright_flux)

    cat, coverage_mask = detect_in_image(injected, wht, nsigma=3.0)
    assert cat is not None, "FAIL: detect_in_image returned None on an image with bright injected sources"

    det_x, det_y = np.asarray(cat.xcentroid), np.asarray(cat.ycentroid)
    print(f"  {len(det_x)} sources detected total")

    for name, (x, y) in truth_xy.items():
        dist = np.sqrt((det_x - x) ** 2 + (det_y - y) ** 2)
        found = dist.min() < 3.0 if len(dist) else False
        if name == "inside_star_mask":
            assert not found, f"FAIL: a source inside the star mask was detected at {(x, y)} -- coverage_mask is not excluding it"
            print(f"  OK: star-masked bright source correctly NOT detected")
        else:
            assert found, f"FAIL: bright source in {name} at {(x, y)} was NOT detected -- threshold/weight-map bug"
            print(f"  OK: bright source in {name} detected")

    assert coverage_mask[300, 450], "FAIL: coverage_mask does not flag the star-masked region"
    print("  OK: coverage_mask correctly flags the star-masked region")
    print("Stage A: PASSED")


# =============================================================================
# Stage B: matching / selection-cut mechanics, using a hand-built fake catalog
# (no real detection involved -- isolates match_recovered/apply_selection_cut)
# =============================================================================
class _FakeCat:
    def __init__(self, x, y, flux):
        self.xcentroid = np.asarray(x)
        self.ycentroid = np.asarray(y)
        self.segment_flux = np.asarray(flux)


def stage_b_matching_mechanics():
    print("\n=== Stage B: matching / selection-cut mechanics ===")
    truth = Table({
        "x": [100.0, 200.0, 300.0, 400.0],
        "y": [100.0, 200.0, 300.0, 400.0],
        "mag": [25.0, 26.0, 27.0, 28.0],
        "M_UV": [-22.0, -21.0, -20.0, -18.0],
    })

    # Detections exist for truth[0] (exact match) and truth[2] (1 px off);
    # truth[1] and truth[3] have nothing nearby.
    fake_det = _FakeCat(x=[100.0, 301.0], y=[100.0, 300.5], flux=[150.0, 20.0])

    recovered, meas_mag = match_recovered(fake_det, truth, match_radius_pix=2.0)
    expected = np.array([True, False, True, False])
    assert np.array_equal(recovered, expected), (
        f"FAIL: match_recovered gave {recovered}, expected {expected}"
    )
    print(f"  OK: match_recovered -> {recovered} matches expectation")

    # M_UV_CUT = -inf should recover NOTHING (this is the bug we found earlier)
    cut_neg_inf = apply_selection_cut(truth, recovered, M_UV_cut=-np.inf)
    assert not cut_neg_inf.any(), "FAIL: M_UV_cut=-inf should zero out everything but didn't"
    print("  OK: M_UV_cut=-inf correctly zeroes all recoveries (regression check for the earlier bug)")

    # M_UV_CUT = +inf should pass everything recovered through unchanged
    cut_pos_inf = apply_selection_cut(truth, recovered, M_UV_cut=np.inf)
    assert np.array_equal(cut_pos_inf, recovered), "FAIL: M_UV_cut=+inf should be a no-op but changed the result"
    print("  OK: M_UV_cut=+inf is a correct no-op")

    # A real magnitude cut should remove the two faintest (M_UV=-20 passes, -18 doesn't)
    cut_m20 = apply_selection_cut(truth, recovered, M_UV_cut=-20.0)
    assert np.array_equal(cut_m20, [True, False, False, False]), (
        f"FAIL: M_UV_cut=-20 gave {cut_m20}, expected [True, False, False, False]"
    )
    print("  OK: M_UV_cut=-20 correctly drops the M_UV=-20 and -18 truth[2] boundary case")
    print("Stage B: PASSED")


# =============================================================================
# Stage C: completeness should track the depth structure of the field
# =============================================================================
def stage_c_completeness_vs_depth(n_per_zone=150, flux_test=8.0):
    print(f"\n=== Stage C: completeness vs depth (flux={flux_test} counts) ===")
    sci, wht = load_field()
    ny, nx = sci.shape
    zones = zone_masks(ny, nx)

    results = {}
    for zname, (yslice, xslice) in zones.items():
        injected = sci.copy()
        xs = rng.uniform(xslice.start + 20, xslice.stop - 20, n_per_zone)
        ys = rng.uniform(yslice.start + 20, yslice.stop - 20, n_per_zone)
        for x, y in zip(xs, ys):
            injected = inject_gaussian_source(injected, x, y, flux_test)

        cat, _ = detect_in_image(injected, wht, nsigma=3.0)
        truth = Table({"x": xs, "y": ys, "mag": np.full(n_per_zone, 27.0),
                        "M_UV": np.full(n_per_zone, -18.0)})
        recovered, _ = match_recovered(cat, truth, match_radius_pix=3.0)
        completeness = recovered.mean()
        results[zname] = completeness
        print(f"  {zname:16s}: completeness = {completeness:.2f}  (n={n_per_zone})")

    assert results["deep_center"] > results["shallow_top"], (
        "FAIL: deep center should have HIGHER completeness than the shallow top strip at fixed flux"
    )
    assert results["deep_center"] > results["shallow_bottom"], (
        "FAIL: deep center should have HIGHER completeness than the shallow bottom strip at fixed flux"
    )
    print("  OK: completeness correctly tracks the wedding-cake depth structure")
    print("Stage C: PASSED")


# =============================================================================
# Stage D: ACF math sanity checks with KNOWN catalogs (positive control)
# This is the most important stage -- it validates DD/DR/RR/landy_szalay/
# the MLE fit against inputs where you KNOW the answer in advance,
# completely independent of detection/injection.
# =============================================================================
def stage_d_acf_positive_control():
    print("\n=== Stage D: ACF math positive control ===")
    theta_bins = np.linspace(2.0, 60.0, 8)  # arcsec
    n_rand = 4000
    ra_r = rng.uniform(214.90, 214.94, n_rand)
    dec_r = rng.uniform(52.86, 52.90, n_rand)

    # --- D1: null test -- data drawn from the SAME distribution as randoms
    #     should give w(theta) consistent with zero everywhere.
    n_data = 200
    ra_null = rng.uniform(214.90, 214.94, n_data)
    dec_null = rng.uniform(52.86, 52.90, n_data)

    DD = pair_counts(ra_null, dec_null, ra_null, dec_null, theta_bins, same_catalog=True)
    DR = pair_counts(ra_null, dec_null, ra_r, dec_r, theta_bins)
    RR = pair_counts(ra_r, dec_r, ra_r, dec_r, theta_bins, same_catalog=True)
    w_null = landy_szalay(DD, DR, RR, n_data, n_rand)
    print(f"  D1 (null test) w(theta) = {np.round(w_null, 3)}")
    assert np.abs(np.nanmean(w_null)) < 0.3, (
        f"FAIL: null test (uniform data vs uniform randoms) should average near zero, got mean={np.nanmean(w_null):.3f}"
    )
    print("  OK: null test is consistent with zero clustering, as expected")

    # --- D2: positive control -- inject REAL clumped clustering and check
    #     the pipeline recovers a positive, significant A_w.
    n_clumps = 15
    pts_per_clump = 14
    clump_centers_ra = rng.uniform(214.905, 214.935, n_clumps)
    clump_centers_dec = rng.uniform(52.865, 52.895, n_clumps)
    clump_scatter_deg = 3.0 / 3600.0  # ~3 arcsec clump radius

    ra_clustered, dec_clustered = [], []
    for cra, cdec in zip(clump_centers_ra, clump_centers_dec):
        ra_clustered.append(rng.normal(cra, clump_scatter_deg, pts_per_clump))
        dec_clustered.append(rng.normal(cdec, clump_scatter_deg, pts_per_clump))
    ra_clustered = np.concatenate(ra_clustered)
    dec_clustered = np.concatenate(dec_clustered)
    n_data_c = len(ra_clustered)

    DD_c = pair_counts(ra_clustered, dec_clustered, ra_clustered, dec_clustered, theta_bins, same_catalog=True)
    DR_c = pair_counts(ra_clustered, dec_clustered, ra_r, dec_r, theta_bins)
    RR_c = pair_counts(ra_r, dec_r, ra_r, dec_r, theta_bins, same_catalog=True)
    w_clustered = landy_szalay(DD_c, DR_c, RR_c, n_data_c, n_rand)
    print(f"  D2 (clustered) w(theta) = {np.round(w_clustered, 3)}")

    theta_centers = 0.5 * (theta_bins[:-1] + theta_bins[1:])
    w_err_dummy = np.full(len(theta_centers), 0.15)  # rough placeholder, not bootstrapped here
    A_w, A_w_err, ic_over_Aw = fit_power_law_mle(theta_centers, w_clustered, w_err_dummy, RR_c,
                                                   beta=0.6, min_RR_counts=5)
    print(f"  D2 fit: A_w = {A_w:.3f} +/- {A_w_err:.3f}")
    assert A_w > 0, f"FAIL: known clustered catalog gave NEGATIVE A_w={A_w:.3f} -- something in DD/DR/RR/MLE chain is broken"
    assert A_w > 3 * A_w_err, f"FAIL: known strong clustering signal not significant (A_w={A_w:.3f}, err={A_w_err:.3f})"
    print("  OK: known clustered catalog recovers a significant POSITIVE A_w -- ACF math chain is correct")
    print("Stage D: PASSED")


# =============================================================================
# Stage E: end-to-end smoke test (just checks it runs without crashing)
# =============================================================================
def stage_e_smoke_test():
    print("\n=== Stage E: end-to-end smoke test ===")
    # Wire this up to call your actual run_pipeline() entry point on the
    # synthetic field with a small N_RANDOM_TARGET, e.g.:
    #
    # from run_pipeline import run_pipeline
    # run_pipeline(sci_path="ceers_f277w_sci.fits", wht_path="ceers_f277w_wht.fits",
    #              n_random_target=2000, z_lo=8.0, z_hi=9.0)
    #
    # For now this is a placeholder -- fill in once Stages A-D pass.
    print("  (placeholder -- wire up your real run_pipeline() here once A-D pass)")


if __name__ == "__main__":
    import subprocess, os
    if not os.path.exists("ceers_f277w_sci.fits"):
        print("Synthetic field not found -- generating it first...")
        subprocess.run(["python3", "make_test_field.py"], check=True)

    stage_a_detection_mechanics()
    stage_b_matching_mechanics()
    stage_c_completeness_vs_depth()
    stage_d_acf_positive_control()
    stage_e_smoke_test()

    print("\n" + "=" * 50)
    print("ALL STAGES PASSED")
    print("=" * 50)