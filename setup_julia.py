"""
Setup script to initialize the Julia environment for Symbolic Regression.
Run this once before starting the main pipeline to ensure all dependencies 
are downloaded and configured.
"""
import os
import shutil
import stat
import juliapkg

print("Initializing Julia environment for Symbolic Regression...")
print("This may take a few minutes as it downloads and installs Julia and necessary packages.")

# Clean up any corrupted or partial Julia installations in the venv
julia_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "julia_env")

def remove_readonly(func, path, _):
    "Clear the readonly bit and reattempt the removal"
    os.chmod(path, stat.S_IWRITE)
    func(path)

if os.path.exists(julia_env_path):
    print(f"Cleaning up existing Julia environment at {julia_env_path}...")
    shutil.rmtree(julia_env_path, onerror=remove_readonly)

# Force resolution of dependencies to fix missing package errors (like PythonCall)
print("Resolving Julia dependencies...")
juliapkg.resolve(force=True)

# Importing PySR triggers the Julia setup via juliapkg
import pysr

print("\nSUCCESS: Julia environment is ready. You can now run the pipeline.")