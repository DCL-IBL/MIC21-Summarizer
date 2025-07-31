#!/bin/bash

# Set the base URL
BASE_URL="https://dcl.bas.bg/MIC-21/dataset/"

# Create a directory to store the downloaded images
mkdir -p downloaded_images

# Use wget to download images recursively while maintaining directory structure
wget -r -np -P downloaded_images -A jpg,jpeg,png,gif \
     --accept-regex "/MIC-21/dataset/" \
     --reject-regex "index.html" \
     "$BASE_URL"

# Remove the base part of the directory structure
mv downloaded_images/dcl.bas.bg/MIC-21/dataset/* downloaded_images/
rm -rf downloaded_images/dcl.bas.bg

echo "Download complete. Images saved in the 'downloaded_images' directory with subdirectory structure maintained."
