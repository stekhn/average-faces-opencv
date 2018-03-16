#!/usr/bin/env python

# Download multiple images using a CSV list as reference

import csv
import sys
import os
import time
import requests
import unidecode

if len(sys.argv) != 2:
    print(
        "Missing argument. Please provide a path to where the images should be downloaded.\n"
        "Usage example: python download.py ./images\n"
    )
    exit()

LIST_PATH = "data/politicians.csv"
BASE_URL = "https://www.bayern.landtag.de/images/abgeordnete/"

IMG_DIR = sys.argv[2]

# Load the data from a CSV file
def main():

    # Create output folder, if not there
    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)

    # Open and parse CSV file
    with open(LIST_PATH, "rb") as f:
        reader = csv.reader(f)
        # Skip the header
        next(reader, None)

        download(list(reader))

# Construct URL and download images
def download(lst):

    for row in lst:
        # Set encoding to UTF-8
        row[:] = [cell.decode('utf-8').strip() for cell in row]

        # Build file name file path
        img_name = dashcase(row[4] + ' ' + row[5] + ' ' + row[1] + ' ' + row[2]) + ".jpg"
        img_path = os.path.join(IMG_DIR, img_name)

        # Construct URL for download
        img_url = BASE_URL + row[0] + ".jpg"
        img_data = requests.get(img_url).content

        # Save image
        with open(img_path, "wb") as handler:
            handler.write(img_data)
            print("Saved " + img_path)

        time.sleep(1)

# Convert a string to dashcase and strip funny characters
def dashcase(string):

    string = unidecode.unidecode(string)
    string = string.strip().lower().replace(" ", "-")
    return string

# Main sentinel
if __name__ == "__main__":
    main()
