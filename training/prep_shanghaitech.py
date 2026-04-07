import os
import glob
import cv2
import scipy.io
import shutil
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def convert_shanghaitech_to_yolo(source_dirs, target_base_dir, box_size=20):
    """
    Convert ShanghaiTech points to YOLO bounding boxes.
    source_dirs: list of paths to Part A and Part B (train/test directories)
    target_base_dir: where the YOLO formatted dataset will be created
    box_size: fixed bounding box size for faces/heads.
    """
    os.makedirs(os.path.join(target_base_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(target_base_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(target_base_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(target_base_dir, 'labels', 'val'), exist_ok=True)

    all_images = []
    
    # Collect all images across part A and part B, test and train
    for s_dir in source_dirs:
        for split in ['train_data', 'test_data']:
            img_dir = os.path.join(s_dir, split, 'images')
            if not os.path.exists(img_dir): continue
            
            for img_path in glob.glob(os.path.join(img_dir, '*.jpg')):
                all_images.append(img_path)

    # We will just do a standard 80/20 train/val split over all shangaitech data
    train_imgs, val_imgs = train_test_split(all_images, test_size=0.2, random_state=42)
    
    def process_split(img_list, split_name):
        print(f"Processing {split_name} split with {len(img_list)} images...")
        for img_path in img_list:
            base_name = os.path.basename(img_path)
            # The ground truth is prefixed with 'GT_' and ends with '.mat'
            gt_name = 'GT_' + base_name.replace('.jpg', '.mat')
            
            gt_dir = os.path.join(os.path.dirname(os.path.dirname(img_path)), 'ground_truth')
            gt_path = os.path.join(gt_dir, gt_name)
            
            if not os.path.exists(gt_path):
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                continue
            height, width = img.shape[:2]
            
            mat = scipy.io.loadmat(gt_path)
            points = mat['image_info'][0][0][0][0][0]
            
            label_name = base_name.replace('.jpg', '.txt')
            # Handle naming collision if Part A and Part B have identical names
            part = 'A' if 'part_A' in img_path else 'B'
            new_base_name = f"part{part}_{base_name}"
            new_label_name = f"part{part}_{label_name}"
            
            out_img_path = os.path.join(target_base_dir, 'images', split_name, new_base_name)
            out_lbl_path = os.path.join(target_base_dir, 'labels', split_name, new_label_name)
            
            shutil.copy(img_path, out_img_path)
            
            with open(out_lbl_path, 'w') as f:
                for pt in points:
                    x, y = pt[0], pt[1]
                    
                    # Convert to YOLO format: class x_center y_center width height
                    # Box size is fixed, clamp it to image boundaries just in case
                    # We normalize coordinates
                    x_center = x / width
                    y_center = y / height
                    w_norm = box_size / width
                    h_norm = box_size / height
                    
                    # Ensure coordinates are within [0, 1]
                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    w_norm = max(0.0, min(1.0, w_norm))
                    h_norm = max(0.0, min(1.0, h_norm))
                    
                    # class 0 = person/head
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                    
    process_split(train_imgs, 'train')
    process_split(val_imgs, 'val')
    
    # Create the config file
    yaml_path = os.path.join(target_base_dir, 'crowd_dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"path: {target_base_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write("names:\n")
        f.write("  0: person\n")
    print(f"Data generation complete! Dataset ready at {target_base_dir}")

if __name__ == "__main__":
    source_directories = [
        r"c:\C Model\ShanghaiTech_Crowd_Counting_Dataset\part_A_final",
        r"c:\C Model\ShanghaiTech_Crowd_Counting_Dataset\part_B_final"
    ]
    target_dir = r"c:\C Model\datasets\crowd"
    convert_shanghaitech_to_yolo(source_directories, target_dir)
