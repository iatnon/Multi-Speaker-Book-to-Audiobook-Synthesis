# src/main.py
import os
import subprocess

def setup_zonos():
    """
    Sets up the Zonos model
    """    

    # Clone Zonos if not already present
    if not os.path.exists("Zonos"):
        subprocess.run(["git", "clone", "https://github.com/Zyphra/Zonos.git"], check=True)
    
    # Change to Zonos directory and install dependencies
    os.chdir("Zonos")
    subprocess.run(["uv", "sync"], check=True)
    subprocess.run(["uv", "sync", "--extra", "compile"], check=True)

if __name__ == "__main__":
    setup_zonos()
