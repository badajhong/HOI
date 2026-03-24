import os
import json
from pathlib import Path
from collections import defaultdict
import re

# Configuration
DATA_DIR = "/home/rllab/haechan/InterAct/data/omomo/sequences_canonical/"
OUTPUT_DIR = "/home/rllab/haechan/holosoma_prev/check/"

def extract_object_name(subfolder_name):
    """
    Extract object_name from subfolder format: sub{number}_{object_name}_{number}
    Example: sub4_whitechair_003 -> whitechair
    """
    parts = subfolder_name.split("_")
    if len(parts) >= 2:
        # object_name is everything between first and last underscore
        return "_".join(parts[1:-1])
    return "unknown"

def extract_unique_texts():
    """
    Extract unique texts from text.txt files in subdirectories.
    Organize by text, object_name, and count subfolders.
    """
    text_to_objects = defaultdict(lambda: defaultdict(list))  # text -> object_name -> [folders]
    
    print(f"Scanning directory: {DATA_DIR}")
    
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Directory does not exist: {DATA_DIR}")
        return None
    
    # Walk through all subdirectories
    for subfolder in os.listdir(DATA_DIR):
        subfolder_path = os.path.join(DATA_DIR, subfolder)
        
        if not os.path.isdir(subfolder_path):
            continue
        
        # Look for text.txt files
        text_file = os.path.join(subfolder_path, "text.txt")
        
        if os.path.exists(text_file):
            try:
                with open(text_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                    # Extract text before first "#"
                    text_before_hash = content.split("#")[0].strip()
                    
                    if text_before_hash:
                        # Extract object name from subfolder
                        object_name = extract_object_name(subfolder)
                        text_to_objects[text_before_hash][object_name].append(subfolder)
                        
            except Exception as e:
                print(f"Error reading {text_file}: {e}")
    
    print(f"\nFound {len(text_to_objects)} unique texts")
    
    return text_to_objects


def save_summary_report(text_to_objects):
    """Save summary report organized by object_name (primary split), then texts within each object."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Restructure: text_to_objects -> objects_to_texts
    objects_to_texts = defaultdict(lambda: defaultdict(list))
    
    for text in text_to_objects.keys():
        for object_name, folders in text_to_objects[text].items():
            objects_to_texts[object_name][text] = sorted(folders)
    
    # Build objects section with subfolders
    objects_section = {}
    objects_summary = {}
    
    for object_name in sorted(objects_to_texts.keys()):
        texts_dict = objects_to_texts[object_name]
        total_subfolders = sum(len(folders) for folders in texts_dict.values())
        texts_list = sorted(texts_dict.keys())
        
        # Objects section with full details
        objects_section[object_name] = {
            "total_texts": len(texts_dict),
            "total_subfolders": total_subfolders,
            "texts": {}
        }
        
        # Add each text and its subfolders
        for text in texts_list:
            folders = texts_dict[text]
            objects_section[object_name]["texts"][text] = {
                "subfolder_count": len(folders),
                "subfolders": folders
            }
        
        # Objects summary (without subfolders but with texts list)
        objects_summary[object_name] = {
            "total_texts": len(texts_dict),
            "total_subfolders": total_subfolders,
            "texts": texts_list
        }
    
    # Write summary report
    summary_file = os.path.join(OUTPUT_DIR, "summary_report.json")
    
    report = {
        "total_objects": len(objects_section),
        "total_unique_texts": len(text_to_objects),
        "objects": objects_section,
        "objects_summary": objects_summary
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved: {summary_file}")
    print(f"✓ Total objects: {len(objects_section)}")
    print(f"✓ Total unique texts: {len(text_to_objects)}")


if __name__ == "__main__":
    print("Starting text extraction process...\n")
    
    text_to_objects = extract_unique_texts()
    
    if text_to_objects:
        save_summary_report(text_to_objects)
        print("\n✓ Process completed successfully!")
    else:
        print("\nNo text.txt files found.")
