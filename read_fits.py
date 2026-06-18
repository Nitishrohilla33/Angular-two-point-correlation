import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from photutils.background import Background2D, MedianBackground
from astropy.stats import SigmaClip

# USE THIS EXACT IMPORT LINE FOR FINDERS:
from photutils.detection import DAOStarFinder

def read_fits_file(file_path):
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
    
    # Open the FITS file safely using a context manager
    with fits.open(file_path) as hdul:
        print("\n=== FITS File Structure (HDU List) ===")
        hdul.info()  # Prints the extensions available (Images, Tables, etc.)
        
        # 1. Read the Primary Header
        primary_header = hdul[0].header
        print("\n=== Primary Header (First 10 Keywords) ===")
        # Print a few metadata keys like NAXIS, TELESCOP, OBJECT if present
        for i, (key, value) in enumerate(primary_header.items()):
            if i < 15: 
                print(f"{key}: {value}")
            else:
                break
                
        # 2. Extract Data (Usually in HDU 0 for simple files, or HDU 1 for tables/extensions)
        # Let's check what kind of data it contains
        data_hdu = hdul[0] if hdul[0].data is not None else hdul[1]
        data = data_hdu.data
        header = data_hdu.header
        
        print(f"\nData Shape: {data.shape if hasattr(data, 'shape') else 'Table structure'}")
        return data, header

# === Example Usage ===
# Replace 'your_file.fits' with your actual file path
fits_path = '/Users/mac/Downloads/assoc_f150w_segmentation.fits' 

data, header = read_fits_file(fits_path)

if data is not None:
    print(f"\nAnalyzing image array of shape {data.shape}...")

    # 1. Clean the data of NaNs or Infs if they exist
    clean_data = np.nan_to_num(data, nan=np.median(data))

    # 2. Estimate and subtract the background sky noise
    # This ensures faint objects are caught while bright background variations are ignored
    sigma_clip = SigmaClip(sigma=3.0)
    bkg_estimator = MedianBackground()
    bkg = Background2D(clean_data, box_size=(50, 50), filter_size=(3, 3),
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    
    # Subtract background so objects stand out cleanly from zero
    data_subtracted = clean_data - bkg.background
    
    # Define a detection threshold (e.g., 5-sigma above the background noise)
    threshold = 5.0 * bkg.background_rms

    # 3. Find the objects
    # fwhm: Full Width at Half Maximum (typical size of a star profile in pixels)
    # Adjust 'fwhm' and 'threshold' based on your image's resolution and noise level
    finder = DAOStarFinder(fwhm=3.0, threshold=np.mean(threshold))
    sources = finder(data_subtracted)

    # 4. Count and display results
    if sources is not None:
        num_objects = len(sources)
        print(f"\n==============================================")
        print(f" SUCCESS: Found {num_objects} objects in the FITS image!")
        print(f"==============================================")
        
        # Look at the coordinates and fluxes of the first few objects found
        print("\nFirst 5 detected sources properties:")
        sources['id', 'xcentroid', 'ycentroid', 'peak'].pprint(max_lines=6)
    else:
        print("\nNo sources detected. Try lowering the fwhm or threshold sigma value.")