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
fits_path = '/Users/mac/Downloads/assoc_f150w_catalog.fit' 

data, header = read_fits_file(fits_path)

if data is not None:
    print(f"\nProcessing catalog table with {data.shape[0]} sources...")
    
    # 1. Extract the coordinates and magnitudes from the SExtractor catalog columns
    # (Using the column names we saw in your earlier error trace)
    ra = data['ALPHA_J2000']
    dec = data['DELTA_J2000']
    mag = data['MAG_AUTO']
    
    # 2. Initialize the plot
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.figure(figsize=(9, 7))
    
    # 3. Create a spatial scatter plot color-coded by brightness (magnitude)
    # 'cmap=viridis_r' is inverted so smaller numbers (brighter objects) stand out sharply
    scatter = plt.scatter(ra, dec, c=mag, cmap='viridis_r', s=80, edgecolor='black', alpha=0.8)
    
    # Add a colorbar to show the magnitude scale
    cbar = plt.colorbar(scatter)
    cbar.set_label('Apparent Magnitude (MAG_AUTO)', fontsize=12)
    
    # 4. Format labels and titles for an astronomical sky map
    plt.xlabel('Right Ascension (J2000 deg) - ALPHA_J2000', fontsize=12)
    plt.ylabel('Declination (J2000 deg) - DELTA_J2000', fontsize=12)
    plt.title('Spatial Distribution of Detected Sources (CEERS JWST)', fontsize=14, fontweight='bold')
    
    # CRITICAL FOR ASTRONOMY: Invert the RA axis so East points to the left
    plt.gca().invert_xaxis()
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Display the plot window
    plt.show()