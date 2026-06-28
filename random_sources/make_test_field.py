"""
make_test_field.py

Generates a small synthetic science + weight FITS pair that mimics the 
key structural feature of the real survey: a 'wedding-cake' depth 
pattern (a deeper centeral strip, shallower top/bottom strips), plus a 
masked circular region (simulating a bright forground star), so the 
pipeline's depth-dependent recovery behaviour can be sanity-checked
without needing the real CEERS data on disk.
"""
import numpy as np 
from astropy.io import fits 
from astropy.wcs import WCS

rng = np.random.default_rng(0)

NY, NX = 600, 600
pixel_scale, ZP_AB = 0.063, 25.94

# Calibrate background noise to the zeropoint: a typical HLF-depth
# H160 image reaches ~5 sigma at m_Ab ! 27.5 in a few-pixel aperture.
# For a single-pixel proxy here, set sigma so m_5sigma(centeral region)
# corresponds to a counts-level threshold a few times below the 
# faintest sources we plan to inject (~m=27).
m_5sigma_centeral = 30.5
sigma_central = (10 ** (-0.4 * (m_5sigma_centeral - ZP_AB))) / 5.0

science = rng.normal(0.0, sigma_central, size=(NY, NX))

# wedding-cake depth: top/bottom thirds shallower (noisier) then center
top = NY // 3 
bottom = 2 * NY // 3
shallow_factor = 2.5    # shallower regions have this much higher noise
science[:top, :] = rng.normal(0, sigma_central * shallow_factor, size=(top, NX))
science[bottom:, :] = rng.normal(0, sigma_central * shallow_factor, size=(NY - bottom, NX))

weight = np.ones((NY, NX)) / sigma_central ** 2 
weight[:top, :] = 1.0 / (sigma_central * shallow_factor) **2
weight[bottom:, :] = 1.0 / (sigma_central * shallow_factor) ** 2

# Masked circular region simulation a bright star 
yy, xx = np.mgrid[0:NY, 0:NX]
star_mask = (xx - 450) ** 2 + (yy - 300) ** 2 < 40 ** 2
weight[star_mask] = 0.0
science[star_mask] = 0.0

wcs = WCS(naxis=2)
wcs.wcs.crpix = [NX/2, NY/2]
wcs.wcs.cdelt = [-pixel_scale/3600.0, pixel_scale/3600.0]     # 0.06"/pix, like WFC3/IR
wcs.wcs.crval = [214.92, 52.88]                               # near CEERS coords
wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

hdu = fits.PrimaryHDU(data=science.astype(np.float32), header=wcs.to_header())
hdu.header["ZP_AB"] = ZP_AB
hdu.writeto("ceers_f277w_sci.fits", overwrite=True)

wht_hdu = fits.PrimaryHDU(data=weight.astype(np.float32), header=wcs.to_header())
wht_hdu.writeto("ceers_f277w_wht.fits", overwrite=True)

print("synthetic test field written: ceers_f277w_sci.fits, ceers_f277w_wht.fits")
print(f"Shape: {science.shape}, masked star pixels: {star_mask.sum()}")