#!/usr/bin/env python3
"""
Script to check for Chinese characters in all .sh and .py files in the current directory.
"""

import os
import re

def has_chinese(text):
    """Check if text contains Chinese characters."""
    chinese_pattern = re.compile(r'[\u4e00-\u9fa5]')
    return bool(chinese_pattern.search(text))

def check_files_for_chinese(directory):
    """Check all .sh and .py files in directory for Chinese characters."""
    files_with_chinese = []
    files_without_chinese = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.sh', '.py')):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        chinese_lines = []
                        found_chinese = False
                        for line_num, line in enumerate(lines, 1):
                            if has_chinese(line):
                                chinese_lines.append((line_num, line.rstrip('\n')))
                                found_chinese = True
                        if found_chinese:
                            files_with_chinese.append({'filepath': filepath, 'lines': chinese_lines})
                        else:
                            files_without_chinese.append(filepath)
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
    
    return files_with_chinese, files_without_chinese

def main():
    current_dir = os.getcwd()
    print(f"Checking for Chinese characters in .sh and .py files in: {current_dir}")
    print("-" * 60)
    
    files_with_chinese, files_without_chinese = check_files_for_chinese(current_dir)
    
    print(f"Files WITH Chinese characters ({len(files_with_chinese)}):")
    for file_info in files_with_chinese:
        print(f"  {file_info['filepath']}")
        for line_num, line_content in file_info['lines']:
            print(f"    Line {line_num}: {line_content}")
    
    print(f"\nFiles WITHOUT Chinese characters ({len(files_without_chinese)}):")
    for filepath in files_without_chinese:
        print(f"  {filepath}")
    
    print(f"\nSummary: {len(files_with_chinese)} files with Chinese, {len(files_without_chinese)} files without Chinese.")

if __name__ == "__main__":
    main()