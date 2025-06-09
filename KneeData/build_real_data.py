import os
import random
import shutil
import argparse
from collections import defaultdict
def collect_enhanced_image_paths(root_dir):

    image_paths = defaultdict(list)

    for dataset in ['kneeKL224', 'kneeKL299']:
        for split in ['train', 'test']:
            base_path = os.path.join(root_dir, dataset, split)
            if not os.path.exists(base_path):
                continue

            for folder_name in os.listdir(base_path):
                if folder_name.endswith("_enhanced"):
                    full_path = os.path.join(base_path, folder_name)
                    if os.path.isdir(full_path):
                        for img_file in os.listdir(full_path):
                            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                image_paths[folder_name].append(os.path.join(full_path, img_file))

    return image_paths


def select_random_images(image_paths_dict, total_count):

    all_images = []
    folder_lists = list(image_paths_dict.values())

    while len(all_images) < total_count:
        for lst in folder_lists:
            if lst and len(all_images) < total_count:
                all_images.append(lst.pop(random.randint(0, len(lst) - 1)))
            if len(all_images) >= total_count:
                break

        if all(not lst for lst in folder_lists):  # If all lists are empty, break
            break

    return all_images


def main():
    parser = argparse.ArgumentParser(description="Create a 'real' folder with randomly sampled enhanced knee images.")
    parser.add_argument('--source_root', required=True, help='Root folder of enhanced datasets')
    parser.add_argument('--output_dir', required=True, help='Target folder where 2500 random images will be stored')
    parser.add_argument('--num_images', type=int, default=2500, help='Number of images to copy')

    args = parser.parse_args()

    print("Collecting images...")
    image_paths_dict = collect_enhanced_image_paths(args.source_root)

    print(f"Total folders found: {len(image_paths_dict)}")
    print("Selecting random images...")
    selected_images = select_random_images(image_paths_dict, args.num_images)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Copying {len(selected_images)} images to {args.output_dir}...")
    for i, src_path in enumerate(selected_images):
        filename = f"real_{i}_{os.path.basename(src_path)}"
        dst_path = os.path.join(args.output_dir, filename)
        shutil.copy2(src_path, dst_path)

    print("Done!")


if __name__ == "__main__":
    main()
