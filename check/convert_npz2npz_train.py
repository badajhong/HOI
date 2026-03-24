import os
import subprocess
from pathlib import Path

# Configuration
INPUT_DIR = "/home/rllab/haechan/text2HOI/src/holosoma_retargeting/holosoma_retargeting/demo_results_parallel/g1/object_interaction/omomo_suitcase"
OUTPUT_DIR = "/home/rllab/haechan/text2HOI/train"
DATA_FORMAT = "smplh"
OBJECT_NAME = "suitcase"
OUTPUT_FPS = 50

# Base command template
COMMAND_TEMPLATE = (
    "python data_conversion/convert_data_format_mj.py "
    "--input_file {input_file} "
    "--output_fps {output_fps} "
    "--output_name {output_name} "
    "--data_format {data_format} "
    "--object_name {object_name} "
    "--has_dynamic_object "
    "--once"
)

def main():
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all .npz files in input directory
    npz_files = sorted(input_path.glob("*.npz"))
    
    if not npz_files:
        print(f"❌ No .npz files found in {INPUT_DIR}")
        return
    
    print(f"✓ Found {len(npz_files)} .npz files")
    print(f"✓ Input directory: {INPUT_DIR}")
    print(f"✓ Output directory: {OUTPUT_DIR}\n")
    
    # Process each file
    success_count = 0
    failed_count = 0
    failed_files = []
    
    for i, npz_file in enumerate(npz_files, 1):
        filename = npz_file.name
        stem = npz_file.stem  # filename without extension
        
        input_file = str(npz_file)
        output_file = str(output_path / f"{stem}.npz")
        
        # Build command
        command = COMMAND_TEMPLATE.format(
            input_file=input_file,
            output_fps=OUTPUT_FPS,
            output_name=output_file,
            data_format=DATA_FORMAT,
            object_name=OBJECT_NAME
        )
        
        print(f"[{i}/{len(npz_files)}] Processing: {filename}")
        
        try:
            # Run command from the holosoma_retargeting directory
            result = subprocess.run(
                command,
                shell=True,
                cwd="/home/rllab/haechan/text2HOI/src/holosoma_retargeting/holosoma_retargeting",
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout per file
            )
            
            if result.returncode == 0:
                print(f"  ✓ Success → {output_file}\n")
                success_count += 1
            else:
                print(f"  ❌ Failed with return code {result.returncode}")
                print(f"  Error: {result.stderr[:200]}\n")
                failed_count += 1
                failed_files.append(filename)
                
        except subprocess.TimeoutExpired:
            print(f"  ❌ Timeout (>10 min)\n")
            failed_count += 1
            failed_files.append(filename)
        except Exception as e:
            print(f"  ❌ Error: {str(e)}\n")
            failed_count += 1
            failed_files.append(filename)
    
    # Summary
    print("\n" + "="*60)
    print(f"✓ Completed: {success_count}/{len(npz_files)} files processed successfully")
    if failed_count > 0:
        print(f"❌ Failed: {failed_count} files")
        print("\nFailed files:")
        for f in failed_files:
            print(f"  - {f}")
    print("="*60)

if __name__ == "__main__":
    main()