"""
setup.py
========
Run this ONCE before running app.py.
It installs all packages and downloads NLTK data.

    python setup.py
"""
import subprocess, sys

print("\n" + "="*55)
print("  Legal NLP Project — Setup")
print("="*55)

# 1. Install packages
print("\n[1/2] Installing packages from requirements.txt …")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# 2. NLTK data
print("\n[2/2] Downloading NLTK data …")
import nltk
for pkg in ["punkt", "punkt_tab", "stopwords"]:
    nltk.download(pkg, quiet=True)

print("\n" + "="*55)
print("  Setup complete!")
print("  Now run:  python app.py")
print("  Then open:  http://localhost:7860")
print("="*55 + "\n")