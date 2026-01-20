#!/usr/bin/env python
"""
VoxLens Installation Checker
Verifies that all required dependencies are installed correctly
"""
import sys


def check_import(module_name, package_name=None):
    """Check if a module can be imported"""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"✅ {package_name}: OK")
        return True
    except ImportError as e:
        print(f"❌ {package_name}: NOT FOUND ({str(e)})")
        return False


def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA: Available (Device: {torch.cuda.get_device_name(0)})")
            return True
        else:
            print("⚠️  CUDA: Not available (CPU mode will be used)")
            return False
    except ImportError:
        print("❌ PyTorch: NOT FOUND")
        return False


def check_ollama():
    """Check if Ollama is running"""
    try:
        import requests
        response = requests.get("http://localhost:11434", timeout=2)
        if response.status_code == 200:
            print("✅ Ollama: Running")
            return True
        else:
            print("⚠️  Ollama: Server responded but may not be ready")
            return False
    except Exception as e:
        print(f"❌ Ollama: Not running ({str(e)})")
        print("   Please start Ollama with: ollama serve")
        return False


def main():
    """Main function to check all requirements"""
    print("=" * 60)
    print("VoxLens Installation Checker")
    print("=" * 60)
    print()
    
    print("Checking Python dependencies...")
    print("-" * 60)
    
    all_ok = True
    
    # Check core dependencies
    all_ok &= check_import("streamlit", "Streamlit")
    all_ok &= check_import("torch", "PyTorch")
    all_ok &= check_import("pyannote.audio", "pyannote.audio")
    all_ok &= check_import("faster_whisper", "faster-whisper")
    all_ok &= check_import("langchain", "LangChain")
    all_ok &= check_import("langchain_community", "LangChain Community")
    
    print()
    print("Checking system configuration...")
    print("-" * 60)
    
    # Check CUDA
    check_cuda()
    
    # Check Ollama
    check_ollama()
    
    print()
    print("=" * 60)
    
    if all_ok:
        print("✅ All required dependencies are installed!")
        print()
        print("Next steps:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Pull the model: ollama pull llama3.2:8b")
        print("3. Get HuggingFace token from: https://huggingface.co/settings/tokens")
        print("4. Start the app: streamlit run app.py")
    else:
        print("❌ Some dependencies are missing. Please run:")
        print("   pip install -r requirements.txt")
        return 1
    
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
