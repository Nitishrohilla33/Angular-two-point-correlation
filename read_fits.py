# import os
# from astropy.io import fits

# fits_path = "/Users/mac/Desktop/JupyterNoteBooks/astro_env/fits/ceers_cat_v1.0.fits"
# output_file = "ceers_catalog_v1.0_info.txt"

# with open(output_file, "w") as fout:

#     with fits.open(fits_path) as hdul:

#         fout.write("=== FITS FILE STRUCTURE ===\n")
#         hdul.info(output=fout)

#         fout.write("\n\n=== PRIMARY HEADER (FIRST 15 KEYWORDS) ===\n")
#         for i, (key, value) in enumerate(hdul[0].header.items()):
#             if i < 15:
#                 fout.write(f"{key}: {value}\n")
#             else:
#                 break

#         data = hdul[1].data

#         fout.write("\n\n====================================\n")
#         fout.write("CATALOG INFORMATION\n")
#         fout.write("====================================\n")

#         fout.write(f"Number of sources : {len(data)}\n")
#         fout.write(f"Number of columns : {len(data.names)}\n")

#         fout.write("\n=== COLUMN NAMES ===\n")
#         for i, col in enumerate(data.names):
#             fout.write(f"{i+1:3d}. {col}\n")

#         fout.write("\n\n=== FIRST SOURCE ===\n")
#         fout.write(str(data[0]))
#         fout.write("\n")

#         keywords = ["RA", "DEC", "ALPHA", "DELTA", "Z", "REDSHIFT", "MAG", "FLUX"]

#         fout.write("\n\n=== POSSIBLE USEFUL COLUMNS ===\n")

#         for key in keywords:
#             matches = [col for col in data.names if key.upper() in col.upper()]

#             if matches:
#                 fout.write(f"\nColumns containing '{key}':\n")
#                 for m in matches:
#                     fout.write(f"   {m}\n")

#         fout.write("\n\n=== FIRST 5 SOURCES ===\n")

#         for i in range(min(5, len(data))):
#             fout.write(f"\nSource {i+1}\n")
#             fout.write(str(data[i]))
#             fout.write("\n")

# print(f"\nResults saved to: {os.path.abspath(output_file)}")








from astropy.table import Table
import numpy as np

# ==========================================================
# Read the CEERS catalog
# ==========================================================

catalog = Table.read("ceers_cat_v1.0.fits")

print("Total number of sources:", len(catalog))

# ==========================================================
# Column names
# ==========================================================

RA_COL = "RA"
DEC_COL = "DEC"
Z_COL = "LP_Z_BEST"

# ==========================================================
# Select the redshift range
# Example: 7.5 <= z < 8.5
# Change these values as needed
# ==========================================================

z_min = 8.4
z_max = 8.6

mask = (
    (catalog[Z_COL] >= z_min) &
    (catalog[Z_COL] < z_max)
)

selected = catalog[mask]

print(f"\nNumber of galaxies between z = {z_min} and {z_max}: {len(selected)}")

# ==========================================================
# Extract RA, Dec and redshift
# ==========================================================

ra = selected[RA_COL]
dec = selected[DEC_COL]
z = selected[Z_COL]

# ==========================================================
# Display the first few sources
# ==========================================================

print("\nFirst 10 selected galaxies:\n")

for i in range(min(10, len(selected))):
    print(f"{i+1:2d}  RA = {ra[i]:10.6f}   Dec = {dec[i]:10.6f}   z = {z[i]:6.3f}")

# ==========================================================
# Save the selected catalog
# ==========================================================

output = Table()

output["RA"] = ra
output["DEC"] = dec
output["LP_Z_BEST"] = z

output.write("CEERS_z_selected.fits", overwrite=True)
output.write("CEERS_z_selected.csv", overwrite=True)

print("\nFiles saved successfully:")
print("  CEERS_z_selected.fits")
print("  CEERS_z_selected.csv")

