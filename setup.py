#!/usr/bin/env python3
"""
Setup script for Call Transcriber
Helps with installation and environment setup
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def run_command(cmd, check=True):
    """Run a command and return the result"""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_ffmpeg():
    """Check if FFmpeg is installed"""
    if run_command("ffmpeg -version", check=False):
        print("✅ FFmpeg is installed")
        return True
    else:
        print("❌ FFmpeg is not installed or not in PATH")
        print_ffmpeg_instructions()
        return False

def print_ffmpeg_instructions():
    """Print installation instructions for FFmpeg"""
    system = platform.system().lower()
    
    print("\n📦 FFmpeg Installation Instructions:")
    print("=" * 40)
    
    if system == "linux":
        print("Ubuntu/Debian:")
        print("  sudo apt update")
        print("  sudo apt install ffmpeg")
        print("\nCentOS/RHEL/Fedora:")
        print("  sudo dnf install ffmpeg")
        print("  # or: sudo yum install ffmpeg")
    
    elif system == "darwin":  # macOS
        print("macOS (using Homebrew):")
        print("  brew install ffmpeg")
        print("\nIf you don't have Homebrew:")
        print("  /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
    
    elif system == "windows":
        print("Windows:")
        print("  1. Download from: https://ffmpeg.org/download.html")
        print("  2. Extract to a folder (e.g., C:\\ffmpeg)")
        print("  3. Add C:\\ffmpeg\\bin to your PATH environment variable")
        print("  4. Restart your command prompt/terminal")
    
    print("\nAlternatively, you can use the static builds from:")
    print("https://github.com/BtbN/FFmpeg-Builds/releases")

def setup_virtual_environment():
    """Set up a virtual environment"""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    print("📦 Creating virtual environment...")
    if run_command(f"{sys.executable} -m venv venv"):
        print("✅ Virtual environment created")
        
        # Determine activation script path
        if platform.system().lower() == "windows":
            activate_script = "venv\\Scripts\\activate"
            pip_path = "venv\\Scripts\\pip"
        else:
            activate_script = "venv/bin/activate"
            pip_path = "venv/bin/pip"
        
        print(f"\n📝 To activate the virtual environment:")
        print(f"   source {activate_script}")
        print(f"   # On Windows: {activate_script}")
        
        return True
    else:
        print("❌ Failed to create virtual environment")
        return False

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    
    # Determine pip path
    if platform.system().lower() == "windows":
        pip_path = "venv\\Scripts\\pip" if Path("venv").exists() else "pip"
    else:
        pip_path = "venv/bin/pip" if Path("venv").exists() else "pip"
    
    # Upgrade pip first
    if run_command(f"{pip_path} install --upgrade pip"):
        print("✅ Pip upgraded")
    
    # Install requirements
    if run_command(f"{pip_path} install -r requirements.txt"):
        print("✅ Dependencies installed successfully")
        return True
    else:
        print("❌ Failed to install dependencies")
        print("\n💡 You can try installing manually:")
        print(f"   {pip_path} install -r requirements.txt")
        return False

def test_installation():
    """Test if the installation works"""
    print("🧪 Testing installation...")
    
    # Determine python path
    if platform.system().lower() == "windows":
        python_path = "venv\\Scripts\\python" if Path("venv").exists() else "python"
    else:
        python_path = "venv/bin/python" if Path("venv").exists() else "python"
    
    # Test import
    test_cmd = f'{python_path} -c "import whisper, torch, pydub; print(\'✅ All core dependencies imported successfully\')"'
    
    if run_command(test_cmd, check=False):
        print("✅ Installation test passed!")
        return True
    else:
        print("❌ Installation test failed")
        print("Some dependencies might not be installed correctly")
        return False

def create_sample_script():
    """Create a sample usage script"""
    sample_script = """#!/usr/bin/env python3
# Sample usage script for Call Transcriber

import os
from main import CallTranscriber

def test_transcriber():
    # Initialize the transcriber
    transcriber = CallTranscriber(whisper_model="tiny", device="cpu")
    
    print("Call Transcriber is ready!")
    print("Usage examples:")
    print("1. python main.py your_audio_file.mp3")
    print("2. python main.py audio.mp3 --model base --preview")
    print("3. python batch_process.py --directory ./audio_files")

if __name__ == "__main__":
    test_transcriber()
"""
    
    with open("test_installation.py", "w") as f:
        f.write(sample_script)
    
    print("✅ Created test_installation.py")

def main():
    """Main setup function"""
    print("🎙️ Call Transcriber Setup")
    print("=" * 30)
    
    # Check requirements
    if not check_python_version():
        sys.exit(1)
    
    if not check_ffmpeg():
        print("\n⚠️  FFmpeg is required but not found.")
        print("Please install FFmpeg and run this setup again.")
        response = input("\nDo you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Setup virtual environment
    print("\n" + "=" * 30)
    setup_success = setup_virtual_environment()
    
    # Install dependencies
    print("\n" + "=" * 30)
    deps_success = install_dependencies()
    
    # Test installation
    print("\n" + "=" * 30)
    test_success = test_installation()
    
    # Create sample script
    print("\n" + "=" * 30)
    create_sample_script()
    
    # Final summary
    print("\n" + "=" * 50)
    print("🎯 Setup Summary:")
    print("=" * 50)
    
    if setup_success and deps_success and test_success:
        print("✅ Setup completed successfully!")
        print("\n🚀 Next steps:")
        print("1. Activate the virtual environment:")
        if platform.system().lower() == "windows":
            print("   venv\\Scripts\\activate")
        else:
            print("   source venv/bin/activate")
        print("\n2. Test with a sample file:")
        print("   python main.py your_audio_file.mp3")
        print("\n3. See the README.md for more usage examples")
    else:
        print("⚠️  Setup completed with some issues")
        print("Please check the error messages above and resolve them")
        print("You can run this setup script again after fixing the issues")
    
    print("\n📚 Documentation:")
    print("- README.md: Comprehensive usage guide")
    print("- config.py: Configuration options")
    print("- sample_output.json: Example output format")

if __name__ == "__main__":
    main() 