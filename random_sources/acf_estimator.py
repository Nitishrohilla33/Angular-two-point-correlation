"""
Step C: Angular correlation function (ACF) estimation.

Implements the Landy & Szalay (1993) estimater used in the paper (eq. 1):
    w_obs(theta) = (DD-2*DR + RR) / RR   [normalized pair counts]
with bootstrap-resampling errors (Ling, Frenk & Barrow 1986), and a 
power-law fit w(theta) = A_w * theta^-beta with beta fixed = 0.6 as 
described in section 3.    
"""
import numpy as np
from scipy.spatial import cKDTree

# Pair counts in angular separation bins
def pair_counts(ra1, dec1, ra2, dec2, theta_bins_arcsec, same_catalog=False):
    """
    Conut pairs between two catalogs (or within one catalog if 
    same_catalog=Ture) in angular separation bins.

    Uses a flat-sky small-angle approximation (valid for the few-
    arcminute survey areas considered here): converts RA/Dec offsets
    to arcsec assuming a local tangent plane, which is adeuate since
    theta_max = 250" << field curvature scale.
    """
    dec0 = np.mean(np.concatenate([dec1, dec2]))
    cos_dec0 = np.cos(np.radians(dec0))

    x1, y1 = ra1 * cos_dec0 * 3600.0, dec1 * 3600.0
    x2, y2 = ra2 * cos_dec0 * 3600.0, dec2 * 3600.0

    tree2 = cKDTree(np.column_stack([x2, y2]))
    max_theta = theta_bins_arcsec[-1]

    counts = np.zeros(len(theta_bins_arcsec)-1)
    pts1 = np.column_stack([x1, y1])

    for i, (x0,y0) in enumerate(pts1):
        if same_catalog:
            # avoid double counting / self-pais: only query points
            # with index > i by excluding self afterwards
            idxs = tree2.query_ball_point([x0, y0], max_theta)
            idxs = [j for j in idxs if j>1]
        else:
            idxs = tree2.query_ball_point([x0, y0], max_theta)

        if not idxs:
            continue

        dx, dy = x2[idxs] - x0, y2[idxs] - y0
        sep = np.sqrt(dx ** 2 + dy ** 2)
        hist, _ = np.histogram(sep, bins=theta_bins_arcsec)
        counts += hist
    return counts

# Landy_Szalay estimator
def landy_szalay(DD, DR, RR, n_data, n_rand):
    """
    Standard Landy & Szalay (1993) estimator, with the customary 
    normalization by the relative catalog sizes so DD, DR, RR can be 
    raw pair counts:
        w(theta) = (DD_norm - 2*DR_nprm + RR_norm) / RR_norm
    where DD_norm = DD / (n_data*(n_data-1)/2),
          DR_norm = DR / (n_data*n_rand),    
          RR_norm = RR / (n_rand*(n_rand-1)/2)
    """
    DD_norm = DD / (n_data * (n_data - 1) / 2.0)
    DR_norm = DR / (n_data * n_rand)
    RR_norm = RR / (n_rand * (n_rand - 1) / 2.0)

    with np.errstate(divide='ignore', invalid='ignore'):
        w = (DD_norm - 2 * DR_norm + RR_norm) / RR_norm
    return w

# Bootstrap error estomation (Ling, Frenk & Barrow 1986)
def bootstrap_errors(ra, dec, ra_r, dec_r, theta_bins_arcsec, n_boot=200, rng=None):
    """
    Bootstrap-resample the data catalog (with replacement) n_boot times,
    recomputing w(theta) each time, and return the standard deviation
    across resamples as the per-bin error estimate.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_data, n_rand = len(ra), len(ra_r)
    w_samples = np.zeros((n_boot, len(theta_bins_arcsec) - 1))

    for b in range(n_boot):
        idx = rng.integers(0, n_data, size=n_data)
        ra_b, dec_b = ra[idx], dec[idx]

        DD = pair_counts(ra_b, dec_b, ra_b, dec_b, theta_bins_arcsec, same_catalog=True)
        DR = pair_counts(ra_b, dec_b, ra_r, dec_r, theta_bins_arcsec) 
        RR = pair_counts(ra_r, dec_r, ra_r, dec_r, theta_bins_arcsec, same_catalog=True)   

        w_samples[b] = landy_szalay(DD, DR, RR, n_data, n_rand)
    return np.nanstd(w_samples, axis=0)   

# Power-law fit, w(theta) = A_w * theta^-beta, beta fixed
def fit_power_law(theta_centers_arcsec, w_obs, w_err, beta=0.6, integral_constraint_ratio=0.0):
    """
    Fit the single free parameter A_w by weighted least sqaures of:
        w_model(theta) = A_w * (theta^-beta - IC_over_Aw)
    against w_obs. The integral constraint ratio IC/A_w (Peacock & 
    Nicholson 1991) is geometry-dependent (depends only on survey
    area/shape once beta is fixed), should be precomputed externally,
    and is passed in here as 'integral_constraint_ratio'.
    """ 
    theta = np.asarray(theta_centers_arcsec)
    valid = np.isfinite(w_obs) & np.isfinite(w_err) & (w_err >0)

    model_shape = theta[valid] ** (-beta) - integral_constraint_ratio

    weights = 1.0 / w_err[valid] ** 2
    A_w = np.sum(weights * model_shape * w_obs[valid]) / np.sum(weights * model_shape ** 2)
    A_w_err = 1.0 / np.sqrt(np.sum(weights * model_shape ** 2))
    return A_w, A_w_err