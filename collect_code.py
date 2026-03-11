#!/usr/bin/env python3
"""
Collects all project source code into a single all_code.txt file.
Scans: scripts/, models/, src/, configs/
Usage: python collect_code.py
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "all_code.txt")

# Directories and extensions to scan
SCAN_DIRS = ["scripts", "models", "src", "configs"]
EXTENSIONS = {".py", ".yaml", ".yml"}

# Files to skip
SKIP_FILES = {"__pycache__", ".pyc", "__init__.py"}


def collect_files():
    """Walk scan dirs and return sorted list of (relative_path, absolute_path)."""
    files = []
    for scan_dir in SCAN_DIRS:
        abs_dir = os.path.join(PROJECT_ROOT, scan_dir)
        if not os.path.isdir(abs_dir):
            continue
        for root, _, filenames in os.walk(abs_dir):
            for fname in filenames:
                if fname in SKIP_FILES or any(fname.endswith(s) for s in [".pyc"]):
                    continue
                ext = os.path.splitext(fname)[1]
                if ext not in EXTENSIONS:
                    continue
                abs_path = os.path.join(root, fname)
                rel_path = os.path.relpath(abs_path, PROJECT_ROOT)
                files.append((rel_path, abs_path))
    
    # Sort by directory depth then alphabetically
    files.sort(key=lambda x: (x[0].count(os.sep), x[0]))
    return files


def main():
    files = collect_files()
    
    with open(OUTPUT_FILE, "w") as out:
        out.write("=" * 80 + "\n")
        out.write("AISE-Pollution — Full Project Source Code\n")
        out.write("=" * 80 + "\n\n")
        
        # Table of contents
        out.write("TABLE OF CONTENTS\n")
        out.write("-" * 40 + "\n")
        for i, (rel_path, _) in enumerate(files, 1):
            out.write(f"  {i:2d}. {rel_path}\n")
        out.write("\n" + "=" * 80 + "\n\n")
        
        # File contents
        for rel_path, abs_path in files:
            out.write(f"{'#' * 80}\n")
            out.write(f"# FILE: {rel_path}\n")
            out.write(f"{'#' * 80}\n\n")
            
            try:
                with open(abs_path, "r") as f:
                    content = f.read()
                out.write(content)
                if not content.endswith("\n"):
                    out.write("\n")
            except Exception as e:
                out.write(f"# ERROR reading file: {e}\n")
            
            out.write("\n")
    
    print(f"✅ Collected {len(files)} files into: {OUTPUT_FILE}")
    print(f"   Total size: {os.path.getsize(OUTPUT_FILE):,} bytes")
    for rel_path, _ in files:
        print(f"   • {rel_path}")


if __name__ == "__main__":
    main()
