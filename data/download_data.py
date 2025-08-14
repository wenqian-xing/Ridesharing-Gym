#!/usr/bin/env python3
"""
Download script for large data files.

This script downloads the required data files for the NYC rideshare simulation.
Run this after cloning the repository to get the necessary data files.

Usage:
    python download_data.py
"""

import os
import sys
from pathlib import Path
import urllib.request
import hashlib

# Data file specifications
DATA_FILES = {
    "manhattan-distances.npy": {
        "url": "https://drive.google.com/uc?export=download&id=1HeBfRdeNwjB3UKLPlQnVdTPWlEFwiBub",  # Replace FILE_ID_1
        "size": "143MB",
        "sha256": "9e8455932b43e5399dfa498f92dfd2b2a2a62ce8bfe70d7855424279f98ea6a4",
        "description": "Distance matrix between Manhattan nodes"
    },
    "manhattan-trips.parquet": {
        "url": "https://drive.google.com/uc?export=download&id=1YEENtqqZl3mGFgzflilTjNrfgPEbTg4u",  # Replace FILE_ID_2
        "size": "68MB", 
        "sha256": "cd008e572fe53a17ddefb3e7df3816afbcbebf8012164499448af536f66a73c1",
        "description": "Historical NYC taxi trip data"
    },
    "manhattan-nodes.parquet": {
        "url": "https://drive.google.com/uc?export=download&id=1DocHvblRWo2X0mAIFgl3AUN4w1K91qls",  # Replace FILE_ID_3
        "size": "164KB",
        "sha256": "b3a2d272b1f6fd6a49dc50332086f079830193f81aedd195c68c885b4547f631",
        "description": "Manhattan street network nodes"
    },
    "taxi-zones.parquet": {
        "url": "https://drive.google.com/uc?export=download&id=1H-McV4AtPssXx79ytwmj7DzgQitF-7Uz",  # Replace FILE_ID_4
        "size": "40KB",
        "sha256": "e365e4f0025bf127107cbccf77c369b5c06d5faa8bf130a5f32b92fea76e88ea",
        "description": "NYC taxi zone boundaries"
    }
}

def verify_file_hash(filepath, expected_hash):
    """Verify file integrity using SHA256 hash."""
    if expected_hash == "placeholder_hash":
        print(f"⚠️  Skipping hash verification for {filepath} (placeholder hash)")
        return True
        
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    
    actual_hash = sha256_hash.hexdigest()
    if actual_hash == expected_hash:
        print(f"✅ Hash verification passed for {filepath}")
        return True
    else:
        print(f"❌ Hash verification failed for {filepath}")
        print(f"   Expected: {expected_hash}")
        print(f"   Actual:   {actual_hash}")
        return False

def download_file(url, filepath, description):
    """Download a file with progress indication."""
    print(f"📥 Downloading {description}...")
    print(f"   URL: {url}")
    print(f"   Target: {filepath}")
    
    try:
        import urllib.request
        import urllib.parse
        from urllib.request import Request, urlopen
        
        # Special handling for Google Drive URLs
        if 'drive.google.com' in url:
            print("   🔄 Handling Google Drive download...")
            
            # Extract file ID from URL
            file_id = url.split('id=')[1] if 'id=' in url else None
            if not file_id:
                raise ValueError("Could not extract file ID from Google Drive URL")
            
            # Use the direct download URL that bypasses virus scan
            download_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
            
        else:
            download_url = url
        
        # Create request with headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size / total_size) * 100)
                sys.stdout.write(f"\r   Progress: {percent:.1f}%")
                sys.stdout.flush()
            else:
                # For Google Drive, total_size might be -1
                downloaded = block_num * block_size
                if downloaded > 1024*1024:  # Show MB if over 1MB
                    sys.stdout.write(f"\r   Downloaded: {downloaded // (1024*1024):.1f}MB")
                else:
                    sys.stdout.write(f"\r   Downloaded: {downloaded // 1024:.1f}KB")
                sys.stdout.flush()
        
        # Create opener with custom headers
        opener = urllib.request.build_opener()
        opener.addheaders = [(k, v) for k, v in headers.items()]
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(download_url, filepath, progress_hook)
        print("\n✅ Download completed")
        
        # Check if we got an HTML error page (common with Google Drive)
        file_size = Path(filepath).stat().st_size
        if file_size < 10000:  # Less than 10KB is suspicious for these files
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(1000)
                if '<html' in content.lower() or 'google' in content.lower():
                    print("⚠️  Warning: Downloaded file appears to be an HTML page, not the data file")
                    print("   This usually means the Google Drive link needs to be made public")
                    print("   Please ensure the file is shared with 'Anyone with the link' access")
                    return False
        
        return True
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("   Note: For Google Drive files, make sure the sharing is set to 'Anyone with the link'")
        return False

def main():
    """Main download function."""
    print("🚀 NYC Rideshare Data Downloader")
    print("=" * 50)
    
    # Use current directory (where download_data.py is located)
    data_dir = Path(".")
    
    # Check which files need downloading
    missing_files = []
    for filename, info in DATA_FILES.items():
        filepath = data_dir / filename
        if not filepath.exists():
            missing_files.append((filename, info))
        else:
            print(f"✅ {filename} already exists")
    
    if not missing_files:
        print("\n🎉 All data files are already present!")
        return
    
    print(f"\n📋 Need to download {len(missing_files)} files:")
    
    for filename, info in missing_files:
        print(f"   • {filename} ({info['size']}) - {info['description']}")
    
    # Confirm download
    response = input(f"\n📥 Download {len(missing_files)} files? [y/N]: ").lower()
    if response not in ['y', 'yes']:
        print("❌ Download cancelled")
        return
    
    # Download files
    success_count = 0
    for filename, info in missing_files:
        filepath = data_dir / filename
        
        if info['url'].startswith('https://example.com'):
            print(f"\n⚠️  {filename}: Placeholder URL detected")
            print("   Please update the URL in download_data.py with the actual data source")
            continue
            
        if download_file(info['url'], filepath, info['description']):
            if verify_file_hash(filepath, info['sha256']):
                success_count += 1
            else:
                print(f"⚠️  File downloaded but hash verification failed: {filename}")
        else:
            print(f"❌ Failed to download: {filename}")
    
    # Summary
    print(f"\n📊 Download Summary:")
    print(f"   ✅ Successful: {success_count}/{len(missing_files)}")
    
    if success_count == len(missing_files):
        print("\n🎉 All data files downloaded successfully!")
        print("\n🚀 You can now run the simulation:")
        print("   python run_rideshare_demo.py")
    else:
        failed = len(missing_files) - success_count
        print(f"\n⚠️  {failed} downloads failed. Check URLs and network connection.")

def list_data_sources():
    """Print information about data sources."""
    print("📋 Data Sources Information")
    print("=" * 50)
    
    for filename, info in DATA_FILES.items():
        print(f"\n📁 {filename}")
        print(f"   Size: {info['size']}")
        print(f"   Description: {info['description']}")
        print(f"   URL: {info['url']}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        list_data_sources()
    else:
        main()