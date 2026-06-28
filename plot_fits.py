import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# --------------------------------------------------
# Read CEERS catalog
# --------------------------------------------------

fits_path = "/Users/mac/Desktop/JupyterNoteBooks/astro_env/fits/ceers_cat_v1.0.fits"

with fits.open(fits_path) as hdul:
    data = hdul[1].data

# --------------------------------------------------
# Print columns to find RA and DEC
# --------------------------------------------------

print("\nPossible coordinate columns:\n")

for col in data.names:
    name = col.upper()
    if ("RA" in name) or ("DEC" in name) or ("ALPHA" in name) or ("DELTA" in name):
        print(col)

# --------------------------------------------------
# CHANGE THESE AFTER CHECKING COLUMN NAMES
# --------------------------------------------------

RA_COL = "RA"      # replace with actual column name
DEC_COL = "DEC"    # replace with actual column name

# --------------------------------------------------
# Extract coordinates
# --------------------------------------------------

ra = np.array(data[RA_COL])
dec = np.array(data[DEC_COL])

# Remove invalid values
mask = np.isfinite(ra) & np.isfinite(dec)

ra = ra[mask]
dec = dec[mask]

print(f"\nNumber of sources = {len(ra)}")
print(f"RA range  : {ra.min():.4f} - {ra.max():.4f}")
print(f"DEC range : {dec.min():.4f} - {dec.max():.4f}")

# --------------------------------------------------
# Plot
# --------------------------------------------------

plt.figure(figsize=(10,6))

plt.scatter(
    ra,
    dec,
    s=0.1,
    alpha=1
)

plt.xlabel("RA (deg)")
plt.ylabel("DEC (deg)")
plt.title("CEERS Source Density")

plt.colorbar(label="Number of Sources")

plt.gca().invert_xaxis()

plt.tight_layout()
plt.savefig("test.png", dpi=800)
plt.show()