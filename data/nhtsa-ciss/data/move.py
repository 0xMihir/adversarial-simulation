import json
import shutil
import sys
from pathlib import Path

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'search_deltav_2023.json'
    
    with open(path, 'r') as f:
        data = json.load(f)

    case_ids = [int(case['caseId']) for case in data]
    print(f"Found {len(case_ids)} case IDs.")

    source_dir = Path('output_all')
    dest_dir = Path('output')
    
    # Create output directory if it doesn't exist
    dest_dir.mkdir(exist_ok=True)
    
    moved_count = 0
    not_found_count = 0
    skipped_count = 0
    
    for case_id in case_ids:
        # Find files matching the pattern {case_id}_*.far in output_all
        matching_files = list(source_dir.glob(f'{case_id}_*.far'))
        
        if not matching_files:
            print(f"No .far file found for case ID {case_id}")
            not_found_count += 1
            continue
        
        for source_file in matching_files:
            dest_file = dest_dir / source_file.name
            
            if dest_file.exists():
                print(f"File {source_file.name} already exists in output. Skipping.")
                skipped_count += 1
                continue
            
            try:
                shutil.move(str(source_file), str(dest_file))
                print(f"Moved {source_file.name}")
                moved_count += 1
            except Exception as e:
                print(f"Error moving {source_file.name}: {e}")
    
    print("\nSummary:")
    print(f"  Moved: {moved_count}")
    print(f"  Skipped (already exists): {skipped_count}")
    print(f"  Not found: {not_found_count}")

if __name__ == '__main__':
    main()
