#!/usr/bin/env python3
"""
Fix all references to 'facefusion' and replace with 'deeptrace'
"""

import os
import re

def fix_file(file_path):
    """Replace all facefusion references with deeptrace in a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if file contains facefusion
        if 'facefusion' not in content:
            return False, 0
        
        # Count replacements
        count = content.count('facefusion')
        
        # Replace all occurrences
        new_content = content.replace('facefusion', 'deeptrace')
        
        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        return True, count
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False, 0

def main():
    root_dir = r'c:\Users\DELL\FYP\deeptrace\deeptrace'
    
    total_files = 0
    total_replacements = 0
    
    # Walk through all Python files
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.py'):
                file_path = os.path.join(dirpath, filename)
                modified, count = fix_file(file_path)
                if modified:
                    print(f"Fixed: {file_path} ({count} replacements)")
                    total_files += 1
                    total_replacements += count
    
    print(f"\n{'='*50}")
    print(f"Total files modified: {total_files}")
    print(f"Total replacements: {total_replacements}")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()
