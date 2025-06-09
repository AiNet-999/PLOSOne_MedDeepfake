import os
import random
import shutil
import argparse

def collect_images_from_subfolders(root_dir):

    image_paths = []
    subfolders = ['KL01_160K', 'KL234_160K']

    for sub in subfolders:
        sub_path = os.path.join(root_dir, sub)
        if not os.path.exists(sub_path):
            print(f"Warning: Subfolder not found: {sub_path}")
            continue

        for img_file in os.listdir(sub_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_img_path = os.path.join(sub_path, img_file)
                image_paths.append(full_img_path)

    return image_paths


def main():
    parser = argparse.ArgumentParser(description="Randomly select images from DeepFake KOA folders.")
    parser.add_argument('--source_root', required=True, help='Root folder containing KL01_160K and KL234_160K')
    parser.add_argument('--output_dir', required=True, help='Folder where selected images will be saved')
    parser.add_argument('--num_images', type=int, default=3000, help='Total number of images to select')

    args = parser.parse_args()


    all_images = collect_images_from_subfolders(args.source_root)

    print(f"Total images found: {len(all_images)}")

    if len(all_images) < args.num_images:
        raise ValueError(f"Not enough images to sample {args.num_images}. Found only {len(all_images)}.")

    selected_images = random.sample(all_images, args.num_images)


    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Copying {args.num_images} images to {args.output_dir} ...")
    for i, img_path in enumerate(selected_images):
        filename = f"deepfake_{i}_{os.path.basename(img_path)}"
        dest_path = os.path.join(args.output_dir, filename)
        shutil.copy2(img_path, dest_path)

    print("Done!")


if __name__ == "__main__":
    main()
