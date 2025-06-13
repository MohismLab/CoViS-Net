import os
from PIL import Image
from pathlib import Path
import concurrent.futures
from tqdm import tqdm

def convert_image(image_path):
    try:
        img = Image.open(image_path)
        if img.mode == 'RGBA':
            print(f"Converting image {image_path} from RGBA to RGB")
            img = img.convert('RGB')
        # elif img.mode == 'RGB':
        #     print(f"Image {image_path} is already in RGB mode")
        # else:
        #     print(f"Image {image_path} is not RGB or RGBA mode, skipping processing")
        #     try:
        #         img = img.convert('RGB')
        #         print(f"Converting image {image_path} to RGB mode")
        #     except Exception as e:
        #         print(f"Unable to convert image {image_path} to RGB mode: {str(e)}")
        
        img = img.resize((448, 448), Image.LANCZOS)
        
        new_path = image_path.parent / f"{image_path.stem}.jpg"
        
        img.save(new_path, format='JPEG', quality=95)
        
        if image_path.suffix.lower() != '.jpg':
            image_path.unlink()
            
        return True
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return False

def process_directory(rgb_dir):
    image_files = list(rgb_dir.glob("*.png")) + list(rgb_dir.glob("*.jpg"))
    
    if not image_files:
        return 0, 0
    
    print(f"\nProcessing directory {rgb_dir}: found {len(image_files)} image files")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(
            executor.map(convert_image, image_files),
            total=len(image_files),
            desc=f"Converting images in {rgb_dir}"
        ))
    
    successful = sum(1 for r in results if r)
    failed = len(image_files) - successful
    
    return successful, failed

def main():
    current_dir = Path(".")
    
    rgb_dirs = []
    for item in current_dir.iterdir():
        if item.is_dir() and (item / "rgb").exists():
            rgb_dirs.append(item / "rgb")
    
    if not rgb_dirs:
        print("Error: No folders containing rgb directory found")
        return
    
    print(f"Found {len(rgb_dirs)} folders containing rgb directory")
    
    total_successful = 0
    total_failed = 0
    
    for rgb_dir in rgb_dirs:
        successful, failed = process_directory(rgb_dir)
        total_successful += successful
        total_failed += failed
    
    print(f"\nAll directories processing completed:")
    print(f"Total successful: {total_successful}")
    print(f"Total failed: {total_failed}")

if __name__ == "__main__":
    main()