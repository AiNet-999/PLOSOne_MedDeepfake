import pandas as pd
import os
import cv2
import argparse


def process_dataset(csv_path, processed_dir, output_base, class_mode, offset_start, offset_end):
    data = pd.read_csv(csv_path)
    processed_uuids = set()
    count = 0

    for index, row in data.iterrows():
        img_type = row['type']
        uuid = row['uuid']
        slice_no = row['slice']
        x = row['x']
        y = row['y']

        if uuid in processed_uuids:
            continue

        for offset in range(offset_start, offset_end + 1):
            current_slice_no = slice_no + offset
            slice_filename = f'slice_{current_slice_no}.jpg'
            slice_path = os.path.join(processed_dir, str(uuid), slice_filename)

            if os.path.exists(slice_path):
                img = cv2.imread(slice_path)
                if img is not None:
                    img = cv2.resize(img, (224, 224))

                    if class_mode == 'multi':
                        if img_type == 'FM':
                            output_dir = os.path.join(output_base, 'MultiClass', 'FM')
                        elif img_type == 'FB':
                            output_dir = os.path.join(output_base, 'MultiClass', 'FB')
                        elif img_type in ['TM', 'TB']:
                            output_dir = os.path.join(output_base, 'MultiClass', 'Real')
                        else:
                            continue
                    elif class_mode == 'binary':
                        if img_type in ['FM', 'FB']:
                            output_dir = os.path.join(output_base, 'BinaryClass', 'Deepfake')
                        elif img_type in ['TM', 'TB']:
                            output_dir = os.path.join(output_base, 'BinaryClass', 'Real')
                        else:
                            continue
                    else:
                        raise ValueError("Invalid class mode. Use 'multi' or 'binary'.")

                    os.makedirs(output_dir, exist_ok=True)
                    output_filename = f'{str(uuid)}_slice{current_slice_no}_{count}.png'

                    output_path = os.path.join(output_dir, output_filename)

                    if cv2.imwrite(output_path, img):
                        print(f'Saved {output_filename}')
                        count += 1
                    else:
                        print(f'Failed to save {output_filename}')

        processed_uuids.add(uuid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', required=True, help='Path to the root dataset folder (e.g. D:/archive)')
    parser.add_argument('--processed_dir', required=True, help='Path to the preprocessed output folder (e.g. D:/processed_output)')
    parser.add_argument('--output_dir', required=True, help='Base output directory to save classified slices (e.g. D:/Dataset)')
    parser.add_argument('--class_mode', required=True, choices=['multi', 'binary'], help='Choose between multi or binary classification folders')
    parser.add_argument('--offset_start', type=int, default=-10, help='Start of slice offset range (inclusive)')
    parser.add_argument('--offset_end', type=int, default=10, help='End of slice offset range (inclusive)')
    args = parser.parse_args()

    csv_paths = [
        os.path.join(args.dataset_dir, 'labels_exp1.csv'),
        os.path.join(args.dataset_dir, 'labels_exp2.csv'),
    ]

    for csv_path in csv_paths:
        print(f"\nProcessing CSV: {csv_path}")
        process_dataset(csv_path, args.processed_dir, args.output_dir, args.class_mode, args.offset_start, args.offset_end)


if __name__ == '__main__':
    main()
