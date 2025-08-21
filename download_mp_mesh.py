#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
Modified script to download specific files from the Matterport3D dataset.
Currently configured for 'matterport_mesh' and 'house_segmentations'.
Updated to Python 3.
Original script by Angel Chang, Manolis Savva.
"""

import argparse
import os
import tempfile
import urllib.request
import sys

BASE_URL = 'http://kaldir.vc.in.tum.de/matterport/'
RELEASE = 'v1/scans'
TOS_URL = BASE_URL + 'MP_TOS.pdf'

# -- List of file types to download for each scan --
# You can add or remove items from this list as needed.
FILE_TYPES_TO_DOWNLOAD = [
    'matterport_mesh',
    'house_segmentations'
]


def get_release_scans(release_file):
    """Fetches the list of scan IDs from the release file."""
    try:
        with urllib.request.urlopen(release_file) as response:
            scan_lines = response.read().decode('utf-8').splitlines()
    except urllib.error.URLError as e:
        print(f"ERROR: Could not fetch scan list from {release_file}")
        print(f"Reason: {e.reason}")
        sys.exit(1)
        
    return [line.strip() for line in scan_lines]


def download_file(url, out_file):
    """Downloads a single file, skipping if it already exists."""
    out_dir = os.path.dirname(out_file)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    if not os.path.isfile(out_file):
        print(f"\tDownloading {url} -> {out_file}")
        
        # --- FIX IS HERE ---
        # Use a temporary file to avoid partial downloads.
        # mkstemp returns a file descriptor and the path. We need the path.
        fd, tmp_path = -1, ""
        try:
            fd, tmp_path = tempfile.mkstemp(dir=out_dir)
            urllib.request.urlretrieve(url, tmp_path)
            os.rename(tmp_path, out_file)
        except Exception as e:
            print(f"ERROR downloading {url}: {e}")
            # Clean up the partially downloaded temp file if it exists
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        finally:
            # Ensure the file descriptor is always closed
            if fd != -1:
                os.close(fd)
        # --- END FIX ---

    else:
        print(f"WARNING: Skipping download of existing file {out_file}")


def download_scan(scan_id, base_out_dir):
    """Downloads the required files for a single scan."""
    print(f"Processing scan '{scan_id}'...")
    scan_out_dir = os.path.join(base_out_dir, RELEASE, scan_id)
    
    for file_type in FILE_TYPES_TO_DOWNLOAD:
        url = f"{BASE_URL}{RELEASE}/{scan_id}/{file_type}.zip"
        out_file = os.path.join(scan_out_dir, f"{file_type}.zip")
        download_file(url, out_file)
        
    print(f"Finished scan '{scan_id}'.")


def download_all_scans(release_scans, base_out_dir):
    """Downloads the required files for all scans in the release."""
    file_list_str = "', '".join(FILE_TYPES_TO_DOWNLOAD)
    print(f"Downloading '{file_list_str}' files for all scans to {base_out_dir}...")
    for scan_id in release_scans:
        download_scan(scan_id, base_out_dir)
    print("Completed download of all requested files.")


def main():
    file_list_str = "', '".join(FILE_TYPES_TO_DOWNLOAD)
    parser = argparse.ArgumentParser(
        description=
        f'''
        Downloads specific Matterport3D files: '{file_list_str}'.
        Example invocation:
          python3 download_mp_custom.py -o /path/to/downloads --id ALL
        The -o argument is required and specifies the base output directory.
        The --id argument can be 'ALL' to download all scans, or a specific scan ID.
        ''',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-o', '--out_dir', required=True, help='Base directory in which to download')
    parser.add_argument('--id', default='ALL', help="Specific scan id to download or 'ALL' to download the entire dataset's meshes")
    args = parser.parse_args()

    print('='*80)
    print('You are about to download data from the Matterport3D dataset.')
    print('By continuing, you confirm that you have agreed to the MP terms of use:')
    print(TOS_URL)
    print('='*80)
    try:
        input("Press Enter to continue, or CTRL-C to exit.")
    except KeyboardInterrupt:
        print("\nExiting.")
        sys.exit()

    release_file_url = BASE_URL + RELEASE + '.txt'
    release_scans = get_release_scans(release_file_url)

    if args.id.upper() == 'ALL':
        download_all_scans(release_scans, args.out_dir)
    else:  # Download a single scan
        scan_id = args.id
        if scan_id not in release_scans:
            print(f"ERROR: Invalid scan id: '{scan_id}'")
            print("Please choose from the list of available scans or use 'ALL'.")
        else:
            download_scan(scan_id, args.out_dir)

if __name__ == "__main__":
    main()