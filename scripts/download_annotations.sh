#!/bin/bash

# Set the base URL
BASE_URL="https://dcl.bas.bg/MIC-21/dataset/"

# Create a directory to store the downloaded JSON files
mkdir -p downloaded_json

# Use wget to download JSON files recursively
wget -r -np -nd -P downloaded_json -A json \
     --accept-regex "/MIC-21/dataset/" \
     "$BASE_URL"

echo "Download complete. JSON files saved in the 'downloaded_json' directory."
