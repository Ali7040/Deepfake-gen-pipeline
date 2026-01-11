"""
Verification Script for DeepTrace Migration
Checks that all files have been properly updated
"""

import os
import re
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists"""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {filepath}")
    return exists

def check_import_in_file(filepath, old_import, new_import):
    """Check if file has been updated from old import to new import"""
    if not os.path.exists(filepath):
        print(f"⚠️  File not found: {filepath}")
        return False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        old_found = old_import in content
        new_found = new_import in content
        
        if old_found:
            print(f"❌ Old import '{old_import}' still found in {filepath}")
            return False
        elif new_found:
            print(f"✅ New import '{new_import}' correctly used in {filepath}")
            return True
        else:
            print(f"⚠️  Neither old nor new import found in {filepath}")
            return True  # May not have the import at all
    except Exception as e:
        print(f"❌ Error checking {filepath}: {e}")
        return False

def main():
    print("="*70)
    print("DeepTrace Migration Verification")
    print("="*70)
    print()
    
    base_path = Path(__file__).parent
    
    # Check renamed files
    print("📁 Checking Renamed Files:")
    print("-" * 70)
    check_file_exists(base_path / "deeptrace.py", "Main entry point")
    check_file_exists(base_path / "deeptrace.ini", "Configuration file")
    
    # Check for old files that should be renamed
    if os.path.exists(base_path / "facefusion.py"):
        print("❌ Old file 'facefusion.py' still exists - should be removed or renamed")
    else:
        print("✅ Old file 'facefusion.py' has been properly renamed")
    
    if os.path.exists(base_path / "facefusion.ini"):
        print("❌ Old file 'facefusion.ini' still exists - should be removed or renamed")
    else:
        print("✅ Old file 'facefusion.ini' has been properly renamed")
    
    print()
    
    # Check key files for proper imports
    print("📝 Checking Import Updates:")
    print("-" * 70)
    
    files_to_check = [
        "deeptrace.py",
        "install.py",
        "simple_app.py",
        "test_optimizations.py",
        "deeptrace/args.py",
        "deeptrace/audio.py",
        "deeptrace/benchmarker.py",
        "deeptrace/app_context.py",
    ]
    
    for file in files_to_check:
        filepath = base_path / file
        check_import_in_file(filepath, "from facefusion", "from deeptrace")
    
    print()
    
    # Check configuration files
    print("⚙️  Checking Configuration Files:")
    print("-" * 70)
    
    # Check .flake8
    flake8_path = base_path / ".flake8"
    if os.path.exists(flake8_path):
        with open(flake8_path, 'r') as f:
            content = f.read()
        if "deeptrace.py" in content and "application_import_names = deeptrace" in content:
            print("✅ .flake8 properly updated")
        else:
            print("❌ .flake8 needs updating")
    
    # Check CI workflow
    ci_path = base_path / ".github" / "workflows" / "ci.yml"
    if os.path.exists(ci_path):
        with open(ci_path, 'r') as f:
            content = f.read()
        if "deeptrace.py" in content and "deeptrace tests" in content:
            print("✅ CI workflow properly updated")
        else:
            print("❌ CI workflow needs updating")
    
    print()
    print("="*70)
    print("Verification Complete!")
    print("="*70)
    print()
    print("Next steps:")
    print("1. Review any ❌ items above")
    print("2. Test the application: python deeptrace.py")
    print("3. Run tests: pytest tests/")
    print("4. Start web app: python simple_app.py")

if __name__ == "__main__":
    main()
