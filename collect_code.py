import os
import glob

def collect_code():
    output_file = 'all_code.txt'
    
    # List of files or glob patterns to include
    patterns = [
        'configs/*.yaml',
        'models/*.py',
        'scripts/*.py',
        'src/utils/*.py'
    ]
    
    files_to_collect = []
    for pattern in patterns:
        files_to_collect.extend(glob.glob(pattern, recursive=True))
        
    # Sort for consistent output
    files_to_collect.sort()
    
    total_size = 0
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_path in files_to_collect:
            if not os.path.isfile(file_path):
                continue
            
            total_size += os.path.getsize(file_path)
            
            outfile.write(f"{'='*80}\n")
            outfile.write(f"FILE: {file_path}\n")
            outfile.write(f"{'='*80}\n\n")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
            except Exception as e:
                outfile.write(f"Error reading file: {e}\n")
            
            outfile.write("\n\n")

    print(f"✅ Collected {len(files_to_collect)} files into: {os.path.abspath(output_file)}")
    print(f"Total size: {total_size:,} bytes")
    for f in files_to_collect:
        print(f"  • {f}")

if __name__ == '__main__':
    collect_code()
