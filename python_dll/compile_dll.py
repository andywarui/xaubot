"""
Compile Python DLL for MT5 using PyInstaller
Converts lightgbm_predictor.py to lightgbm_predictor.dll

Requirements:
- Python 3.11+
- PyInstaller: pip install pyinstaller
- LightGBM: pip install lightgbm
- NumPy: pip install numpy

Usage:
    python compile_dll.py
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_step(step_num, text):
    print(f"\n[Step {step_num}] {text}")


def check_requirements():
    """Check that all required packages are installed."""
    print_step(1, "Checking requirements...")
    
    requirements = ['pyinstaller', 'lightgbm', 'numpy']
    missing = []
    
    for pkg in requirements:
        try:
            __import__(pkg.replace('-', '_'))
            print(f"  ✓ {pkg} installed")
        except ImportError:
            print(f"  ✗ {pkg} NOT installed")
            missing.append(pkg)
    
    if missing:
        print(f"\n  Missing packages: {', '.join(missing)}")
        print(f"  Install with: pip install {' '.join(missing)}")
        return False
    
    return True


def run_local_tests():
    """Run local tests before compiling."""
    print_step(2, "Running local tests...")
    
    script_dir = Path(__file__).parent
    test_script = script_dir / "test_local.py"
    
    if not test_script.exists():
        print("  ⚠️  test_local.py not found, skipping tests")
        return True
    
    result = subprocess.run([sys.executable, str(test_script)], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print("  ✓ All local tests passed")
        return True
    else:
        print("  ✗ Some tests failed:")
        print(result.stdout)
        print(result.stderr)
        return False


def compile_dll():
    """Compile the predictor to DLL using PyInstaller."""
    print_step(3, "Compiling with PyInstaller...")
    
    script_dir = Path(__file__).parent
    predictor_script = script_dir / "lightgbm_predictor.py"
    
    if not predictor_script.exists():
        print(f"  ✗ lightgbm_predictor.py not found at {predictor_script}")
        return False
    
    # Clean previous builds
    for folder in ['build', 'dist', '__pycache__']:
        folder_path = script_dir / folder
        if folder_path.exists():
            print(f"  Cleaning {folder}/...")
            shutil.rmtree(folder_path)
    
    spec_file = script_dir / "lightgbm_predictor.spec"
    if spec_file.exists():
        spec_file.unlink()
    
    # PyInstaller command
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--onefile',                    # Single file output
        '--clean',                      # Clean cache
        '--name', 'lightgbm_predictor', # Output name
        '--distpath', str(script_dir / 'dist'),
        '--workpath', str(script_dir / 'build'),
        '--specpath', str(script_dir),
        '--hidden-import', 'lightgbm',
        '--hidden-import', 'numpy',
        '--hidden-import', 'sklearn',
        '--hidden-import', 'sklearn.ensemble',
        str(predictor_script)
    ]
    
    print(f"  Running: {' '.join(cmd[:6])}...")
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(script_dir))
    
    if result.returncode == 0:
        print("  ✓ PyInstaller compilation successful")
        return True
    else:
        print("  ✗ PyInstaller compilation failed:")
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        return False


def rename_to_dll():
    """Rename .exe to .dll for MT5 compatibility."""
    print_step(4, "Renaming to DLL...")
    
    script_dir = Path(__file__).parent
    exe_path = script_dir / 'dist' / 'lightgbm_predictor.exe'
    dll_path = script_dir / 'dist' / 'lightgbm_predictor.dll'
    
    if not exe_path.exists():
        print(f"  ✗ Compiled file not found at {exe_path}")
        return False
    
    if dll_path.exists():
        dll_path.unlink()
    
    exe_path.rename(dll_path)
    
    file_size = dll_path.stat().st_size / 1024 / 1024
    print(f"  ✓ Created: {dll_path}")
    print(f"  ✓ Size: {file_size:.1f} MB")
    
    return True


def print_next_steps(dll_path):
    """Print deployment instructions."""
    print_header("Compilation Complete!")
    
    print(f"""
  DLL created: {dll_path}
  
  Next Steps:
  -----------
  1. Copy DLL to MT5:
     Copy "{dll_path}"
     To:  C:\\Program Files\\MetaTrader 5\\MQL5\\Libraries\\
     
  2. Copy model file:
     Copy "python_training\\models\\lightgbm_xauusd.pkl"
     To:  C:\\Program Files\\MetaTrader 5\\MQL5\\Files\\
     
  3. Enable DLL imports in MT5:
     Tools → Options → Expert Advisors → Allow DLL imports
     
  4. Compile the EA in MetaEditor (coming next)
  
  5. Run Strategy Tester backtest
""")


def main():
    print_header("LightGBM DLL Compiler for MT5")
    
    # Step 1: Check requirements
    if not check_requirements():
        print("\n  ✗ Missing requirements. Install them and try again.")
        return 1
    
    # Step 2: Run local tests
    if not run_local_tests():
        response = input("\n  Tests failed. Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return 1
    
    # Step 3: Compile
    if not compile_dll():
        print("\n  ✗ Compilation failed.")
        return 1
    
    # Step 4: Rename to DLL
    script_dir = Path(__file__).parent
    dll_path = script_dir / 'dist' / 'lightgbm_predictor.dll'
    
    if not rename_to_dll():
        print("\n  ✗ Failed to rename to DLL.")
        return 1
    
    # Success!
    print_next_steps(dll_path)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
