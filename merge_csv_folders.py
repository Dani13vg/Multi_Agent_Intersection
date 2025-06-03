import os
import shutil
import argparse

def merge_folders(source_dirs, target_name):
    # assert target_name in ["full_train", "full_val"], "Target name must be 'full_train' or 'full_val'"
    
    target_dir = os.path.join("./csv/", target_name)
    os.makedirs(target_dir, exist_ok=True)

    for folder in source_dirs:
        if not os.path.isdir(folder):
            print(f"Skipping '{folder}' (not a directory)")
            continue
        
        for filename in os.listdir(folder):
            if filename.endswith(".csv"):
                src_path = os.path.join(folder, filename)
                dst_path = os.path.join(target_dir, filename)
                
                # Avoid overwriting existing files
                if os.path.exists(dst_path):
                    base, ext = os.path.splitext(filename)
                    i = 1
                    while os.path.exists(os.path.join(target_dir, f"{base}_{i}{ext}")):
                        i += 1
                    dst_path = os.path.join(target_dir, f"{base}_{i}{ext}")

                shutil.copy2(src_path, dst_path)
                print(f"Copied {src_path} → {dst_path}")

    print(f"\n✅ Merged into: {target_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge CSV files from multiple folders.")
    parser.add_argument("folders", nargs="+", help="Paths to folders containing CSV files.")
    parser.add_argument("--target", choices=["full_train", "full_val"], required=True, help="Name of the output folder")

    args = parser.parse_args()
    merge_folders(args.folders, args.target)
