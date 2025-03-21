#!/usr/bin/env python3
"""
Installation script for car_quality_estimator package.
This script installs the package and downloads the required model files.
"""

import os
import sys
import subprocess
import shutil
import argparse


def run_command(command):
    """Run a shell command and print output"""
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    while True:
        output = process.stdout.readline()
        if output == b'' and process.poll() is not None:
            break
        if output:
            print(output.decode('utf-8').strip())

    return process.poll()


def install_package(develop=False):
    """Install the package"""
    command = f"{sys.executable} -m pip install {'--editable' if develop else ''} . --use-pep517"
    print(f"Installing package with command: {command}")
    return run_command(command) == 0


def download_models(package_dir=None):
    """Download model files"""
    try:
        # Add the current directory to the path so scripts can be imported
        sys.path.insert(0, os.path.abspath('.'))
        from scripts.download_models import download_models as dl_models

        if package_dir:
            # Install models to package directory
            # Check whether to use car_quality_estimator or quality_assessment_library
            # based on what's actually installed
            potential_paths = [
                os.path.join(package_dir, "car_quality_estimator"),
                os.path.join(package_dir, "quality_assessment_library")
            ]

            for path in potential_paths:
                if os.path.exists(path):
                    models_dir = os.path.join(path, "models")
                    break
            else:  # If no directory found
                models_dir = os.path.join(
                    package_dir, "car_quality_estimator", "models")
                print(
                    f"Warning: Could not find installed package directory. Using {models_dir}")
        else:
            # Install models to current directory
            models_dir = os.path.join(".", "models")

        os.makedirs(models_dir, exist_ok=True)
        return dl_models(models_dir)
    except ImportError as e:
        print(f"Error importing download_models module: {e}")
        print("Running script directly...")
        command = f"{sys.executable} scripts/download_models.py"
        return run_command(command) == 0


def get_site_packages_dir():
    """Get the site-packages directory where the package is installed"""
    import site
    site_packages = site.getsitepackages()[0]
    return site_packages


def main():
    parser = argparse.ArgumentParser(
        description="Install car_quality_estimator package and download model files")
    parser.add_argument("--develop", "-d", action="store_true",
                        help="Install in development mode")
    parser.add_argument("--models-only", "-m", action="store_true",
                        help="Only download model files, don't install package")
    parser.add_argument("--package-models", "-p", action="store_true",
                        help="Install models to package directory")
    args = parser.parse_args()

    if args.models_only:
        print("Downloading model files only...")
        if download_models(get_site_packages_dir() if args.package_models else None):
            print("Model files downloaded successfully")
            return 0
        else:
            print("Failed to download model files")
            return 1

    print("Installing car_quality_estimator package...")
    if install_package(args.develop):
        print("Package installed successfully")
        if args.develop:
            # get current directory
            current_dir = os.path.abspath('.')
            print(f"Installing models to current directory: {current_dir}")

            # Download models to the current directory
            if download_models(current_dir):
                print("Model files downloaded successfully to current directory")
                return 0
        elif args.package_models:
            # Get the site-packages directory
            site_packages = get_site_packages_dir()
            print(f"Installing models to package directory: {site_packages}")

            # Download models to the package directory
            if download_models(site_packages):
                print("Model files downloaded successfully to package directory")
                return 0
            else:
                print("Failed to download model files to package directory")
                return 1
        else:
            # Download models to local directory
            if download_models():
                print("Model files downloaded successfully to local directory")
                print(
                    "Note: You'll need to specify model_dir when using load_car_quality_score()")
                return 0
            else:
                print("Failed to download model files")
                return 1
    else:
        print("Failed to install package")
        return 1


if __name__ == "__main__":
    sys.exit(main())
