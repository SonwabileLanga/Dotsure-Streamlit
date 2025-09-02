#!/usr/bin/env python3
"""
CSV File Splitter for Large Datasets
====================================

This script helps split large CSV files into smaller, manageable chunks
for easier upload and processing in the DOTSURE STREAMLIT dashboard.

Usage:
    python csv_splitter.py input_file.csv [chunk_size] [output_prefix]

Example:
    python csv_splitter.py large_data.csv 100000 chunks_
"""

import pandas as pd
import sys
import os
from pathlib import Path

def split_csv(input_file, chunk_size=100000, output_prefix="chunk_"):
    """
    Split a large CSV file into smaller chunks
    
    Args:
        input_file (str): Path to the input CSV file
        chunk_size (int): Number of rows per chunk (default: 100,000)
        output_prefix (str): Prefix for output files (default: "chunk_")
    """
    
    print(f"🔄 Splitting {input_file} into chunks of {chunk_size:,} rows...")
    
    # Get file info
    file_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
    print(f"📁 File size: {file_size:.2f} MB")
    
    # Create output directory
    output_dir = Path("split_files")
    output_dir.mkdir(exist_ok=True)
    
    chunk_count = 0
    total_rows = 0
    
    try:
        # Read and split the CSV file
        for chunk in pd.read_csv(input_file, chunksize=chunk_size):
            chunk_count += 1
            total_rows += len(chunk)
            
            # Create output filename
            output_file = output_dir / f"{output_prefix}{chunk_count:03d}.csv"
            
            # Save chunk
            chunk.to_csv(output_file, index=False)
            
            print(f"✅ Created {output_file} ({len(chunk):,} rows)")
        
        print(f"\n🎉 Successfully split {input_file}!")
        print(f"📊 Total chunks created: {chunk_count}")
        print(f"📊 Total rows processed: {total_rows:,}")
        print(f"📁 Output directory: {output_dir.absolute()}")
        
        # Create a manifest file
        manifest_file = output_dir / "manifest.txt"
        with open(manifest_file, 'w') as f:
            f.write(f"CSV Split Manifest\n")
            f.write(f"==================\n")
            f.write(f"Original file: {input_file}\n")
            f.write(f"Chunk size: {chunk_size:,} rows\n")
            f.write(f"Total chunks: {chunk_count}\n")
            f.write(f"Total rows: {total_rows:,}\n")
            f.write(f"Output prefix: {output_prefix}\n\n")
            f.write(f"Files created:\n")
            for i in range(1, chunk_count + 1):
                f.write(f"  {output_prefix}{i:03d}.csv\n")
        
        print(f"📋 Manifest created: {manifest_file}")
        
    except Exception as e:
        print(f"❌ Error splitting file: {str(e)}")
        return False
    
    return True

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python csv_splitter.py input_file.csv [chunk_size] [output_prefix]")
        print("\nExample:")
        print("  python csv_splitter.py large_data.csv 50000 chunks_")
        sys.exit(1)
    
    input_file = sys.argv[1]
    chunk_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
    output_prefix = sys.argv[3] if len(sys.argv) > 3 else "chunk_"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: File '{input_file}' not found!")
        sys.exit(1)
    
    # Split the file
    success = split_csv(input_file, chunk_size, output_prefix)
    
    if success:
        print("\n🚀 Next steps:")
        print("1. Upload the chunk files to your cloud storage (Google Drive, Dropbox, etc.)")
        print("2. Use the 'Load from URL' option in the dashboard")
        print("3. Or upload individual chunks using 'Upload CSV Files'")
        print("4. Use the 'Large File Solutions' option for chunked processing")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
