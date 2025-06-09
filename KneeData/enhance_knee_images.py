import os
import cv2
import numpy as np
import argparse


def increase_contrast(image):
    """
    Enhance image contrast using CLAHE + Histogram Equalization.
    """
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    contrast_image = clahe.apply(np.uint8(image))
    contrast_image = cv2.equalizeHist(contrast_image)
    return contrast_image


def process_directory(input_directory, output_directory):

    os.makedirs(output_directory, exist_ok=True)
    for file_name in os.listdir(input_directory):
        input_path = os.path.join(input_directory, file_name)
        if not os.path.isfile(input_path):
            continue
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            enhanced_image = increase_contrast(image)
            output_path = os.path.join(output_directory, file_name)
            cv2.imwrite(output_path, enhanced_image)


def main():
    parser = argparse.ArgumentParser(description="Enhance contrast of knee X-ray images using CLAHE + Histogram Equalization.")
    parser.add_argument('--input_root', required=True, help='Root directory of the original dataset (e.g. D:/.../ClsKLData)')
    parser.add_argument('--output_root', required=True, help='Root directory to save enhanced images (e.g. D:/KneePreprocessed)')

    args = parser.parse_args()

    for dataset_name in ['kneeKL224', 'kneeKL299']:
        dataset_path = os.path.join(args.input_root, dataset_name)
        if not os.path.exists(dataset_path):
            print(f"Dataset {dataset_path} does not exist. Skipping.")
            continue

        for split in ['train', 'test']:  # Only process train and test
            max_class = 4 if dataset_name.endswith('224') else 4
            for label in range(0, max_class + 1):
                input_dir = os.path.join(dataset_path, split, str(label))
                output_dir = os.path.join(args.output_root, dataset_name, split, f"{label}_enhanced")

                if os.path.exists(input_dir):
                    print(f"Processing: {input_dir} -> {output_dir}")
                    process_directory(input_dir, output_dir)
                else:
                    print(f"Skipping: {input_dir} (does not exist)")


if __name__ == "__main__":
    main()
