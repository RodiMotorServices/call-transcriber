#!/usr/bin/env python3
"""
Quick installation script for Call Transcriber
For testing purposes - installs core dependencies
"""

import subprocess
import sys
import os

def run_command(cmd):
    """Run a command and return success status"""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False

def install_minimal_dependencies():
    """Install minimal dependencies for testing"""
    print("🚀 Installing minimal dependencies for testing...")
    
    # Try installing with --break-system-packages for testing
    basic_deps = [
        "openai-whisper",
        "torch",
        "torchaudio", 
        "pydub",
        "click",
        "rich"
    ]
    
    for dep in basic_deps:
        print(f"\n📦 Installing {dep}...")
        success = run_command(f"pip3 install --break-system-packages {dep}")
        if not success:
            print(f"❌ Failed to install {dep}")
            return False
    
    print("✅ Basic dependencies installed!")
    return True

def test_imports():
    """Test if core dependencies can be imported"""
    print("\n🧪 Testing imports...")
    
    try:
        import whisper
        print("✅ whisper imported successfully")
        
        import torch
        print("✅ torch imported successfully")
        
        import pydub
        print("✅ pydub imported successfully")
        
        import click
        print("✅ click imported successfully")
        
        import rich
        print("✅ rich imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def main():
    print("🎙️ Call Transcriber - Quick Install for Testing")
    print("=" * 50)
    print("⚠️  This will install packages with --break-system-packages")
    print("Only use this for testing purposes!")
    
    response = input("\nDo you want to continue? (y/n): ")
    if response.lower() != 'y':
        print("Installation cancelled.")
        sys.exit(0)
    
    if install_minimal_dependencies():
        if test_imports():
            print("\n✅ Installation successful!")
            print("\n🚀 Now you can test with:")
            print("python3 main.py your_audio.mp3 --language es")
            print("python3 main.py your_audio.mp3 --language auto")
        else:
            print("\n❌ Installation completed but imports failed")
    else:
        print("\n❌ Installation failed")

if __name__ == "__main__":
    main() 