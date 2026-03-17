#!/usr/bin/env python3
import os

def gather_code():
    output_file = "all_code.txt"
    # Directories to exclude
    exclude_dirs = {'.git', 'venv', '__pycache__', '.pytest_cache', '.ipynb_checkpoints'}
    # Files to exclude to avoid recursion/duplicates
    exclude_files = {'all_code.txt', 'combined_code.txt', 'gather_code.py'}
    
    # Text-based extensions to include
    include_extensions = {
        '.py', '.yaml', '.yml', '.md', '.txt', '.sh', 
        '.json', '.js', '.ts', '.tsx', '.htm', '.html', 
        '.css', '.sql', '.c', '.cpp', '.h', '.hpp'
    }
    
    found_any = False
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Walk from the current directory (workspace root)
        for root, dirs, files in os.walk('.'):
            # Prune excluded directories in-place for os.walk
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            # Sort to ensure consistent output order
            files.sort()
            
            for file in files:
                if file in exclude_files:
                    continue
                    
                ext = os.path.splitext(file)[1].lower()
                if ext in include_extensions:
                    filepath = os.path.join(root, file)
                    # Normalize path for readability (remove leading ./)
                    rel_path = os.path.normpath(filepath)
                    
                    outfile.write("=" * 80 + "\n")
                    outfile.write(f"FILE: {rel_path}\n")
                    outfile.write("=" * 80 + "\n\n")
                    
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
                            outfile.write(infile.read())
                    except Exception as e:
                        outfile.write(f"ERROR READING FILE: {e}\n")
                    
                    outfile.write("\n\n")
                    found_any = True
                    print(f"Added: {rel_path}")

    if found_any:
        print(f"\nSuccessfully created {output_file}")
    else:
        print("\nNo files found to add.")

if __name__ == "__main__":
    gather_code()
