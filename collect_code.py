import os
import glob

output_file = 'all_code.txt'

# List of directories to search
directories = ['configs', 'models', 'scripts', 'src']
extensions = ['.py', '.yaml', '.txt', '.md']

with open(output_file, 'w', encoding='utf-8') as outfile:
    for directory in directories:
        for ext in extensions:
            # Recursively find files
            for filepath in glob.glob(f'{directory}/**/*{ext}', recursive=True):
                if os.path.isfile(filepath):
                    outfile.write(f"{'='*80}\n")
                    outfile.write(f"FILE: {filepath}\n")
                    outfile.write(f"{'='*80}\n\n")
                    
                    try:
                        with open(filepath, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())
                    except Exception as e:
                        outfile.write(f"[Error reading file: {e}]\n")
                    
                    outfile.write("\n\n")

print(f"Successfully collected all code into {output_file}")
