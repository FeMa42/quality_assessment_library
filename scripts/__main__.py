#!/usr/bin/env python3
"""
Main entry point for scripts package.
"""

import sys
from scripts.download_models import download_models

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "download_models":
        output_dir = "./models"
        if len(sys.argv) > 2:
            output_dir = sys.argv[2]
        success = download_models(output_dir)
        sys.exit(0 if success else 1)
    else:
        print("Available commands:")
        print("  download_models [output_dir] - Download model files")
        sys.exit(1)
