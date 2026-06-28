import os
from astropy.io import fits

fits_path = "/Users/mac/Desktop/JupyterNoteBooks/astro_env/fits/ceers_cat_v1.0.fits"
output_file = "ceers_catalog_v1.0_info.txt"

with open(output_file, "w") as fout:

    with fits.open(fits_path) as hdul:

        fout.write("=== FITS FILE STRUCTURE ===\n")
        hdul.info(output=fout)

        fout.write("\n\n=== PRIMARY HEADER (FIRST 15 KEYWORDS) ===\n")
        for i, (key, value) in enumerate(hdul[0].header.items()):
            if i < 15:
                fout.write(f"{key}: {value}\n")
            else:
                break

        data = hdul[1].data

        fout.write("\n\n====================================\n")
        fout.write("CATALOG INFORMATION\n")
        fout.write("====================================\n")

        fout.write(f"Number of sources : {len(data)}\n")
        fout.write(f"Number of columns : {len(data.names)}\n")

        fout.write("\n=== COLUMN NAMES ===\n")
        for i, col in enumerate(data.names):
            fout.write(f"{i+1:3d}. {col}\n")

        fout.write("\n\n=== FIRST SOURCE ===\n")
        fout.write(str(data[0]))
        fout.write("\n")

        keywords = ["RA", "DEC", "ALPHA", "DELTA", "Z", "REDSHIFT", "MAG", "FLUX"]

        fout.write("\n\n=== POSSIBLE USEFUL COLUMNS ===\n")

        for key in keywords:
            matches = [col for col in data.names if key.upper() in col.upper()]

            if matches:
                fout.write(f"\nColumns containing '{key}':\n")
                for m in matches:
                    fout.write(f"   {m}\n")

        fout.write("\n\n=== FIRST 5 SOURCES ===\n")

        for i in range(min(5, len(data))):
            fout.write(f"\nSource {i+1}\n")
            fout.write(str(data[i]))
            fout.write("\n")

print(f"\nResults saved to: {os.path.abspath(output_file)}")