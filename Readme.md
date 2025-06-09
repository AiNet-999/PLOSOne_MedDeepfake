The datasets are publicly available

1. CT-GAN dataset Available at https://www.kaggle.com/datasets/ymirsky/medical-deepfakes-lung-cancer

2. Synthetic Knee Deepfake images Dataset Available at https://data.mendeley.com/datasets/fyybnjkw7v/3
   
3. Knee X-ray Real Images Dataset Available at https://data.mendeley.com/datasets/t9ndx37v5h/1 

### 📚 **Citations to Datasets**

1. **Mirsky, Yisroel; Mahler, Tom; Shelef, Ilan; Elovici, Yuval** (2019).  
   *CT-GAN: Malicious tampering of 3D medical imagery using deep learning*.  
   In *28th USENIX Security Symposium (USENIX Security 19)*, pp. 461–478.  
   [`Paper Link`](https://www.usenix.org/conference/usenixsecurity19/presentation/mirsky)

2. **Prezja, Fabi; Paloneva, Juha; Pölönen, Ilkka; Niinimäki, Esko; Äyrämö, Sami** (2022).  
   *Synthetic (DeepFake) Knee Osteoarthritis X-ray Images from Generative Adversarial Neural Networks*.  
   Mendeley Data, V3. DOI: [10.17632/fyybnjkw7v.3](https://doi.org/10.17632/fyybnjkw7v.3)

3. **Gornale, Shivanand; Patravali, Pooja** (2020).  
   *Digital Knee X-ray Images*.  
   Mendeley Data, V1. DOI: [10.17632/t9ndx37v5h.1](https://doi.org/10.17632/t9ndx37v5h.1)


### 🛠️ **Instructions**

```bash
# 📥 Download the Datasets
# Download datasets from the sources provided in the 📚 Citations to Datasets section.

# 🧪 Preprocess CT-GAN Dataset
# This script converts .DICOM files into .jpg slices and organizes them.

# Without annotation:
python preprocessing.py --dataset_dir D:/archive --output_dir D:/processed_output

# With tumor annotation:
python preprocessing.py --dataset_dir D:/archive --output_dir D:/processed_output --annotate

# 🗂️ Create Class-wise Folders
# Organize the dataset into folders based on classification type.

# Multi-class:
python extract_samples.py --dataset_dir D:/archive --processed_dir D:/processed_output --output_dir D:/Dataset --class_mode multi --offset_start -10 --offset_end 10

# Binary class:
python extract_samples.py --dataset_dir D:/archive --processed_dir D:/processed_output --output_dir D:/Dataset --class_mode binary --offset_start -10 --offset_end 10

# 🦵 Preprocess Knee X-ray Dataset
# Preprocess and generate real and DeepFake Knee X-ray data samples.

python enhance_knee_images.py --input_root "D:/56rmx5bjcr-1/KneeXrayData/ClsKLData" --output_root "D:/KneePreprocessed"

python build_real_data.py --source_root "D:/KneePreprocessed" --output_dir "D:/KneePreprocessed/real" --num_images 2500

python build_deepfake_data.py --source_root "D:/Data" --output_dir "D:/KneeMedDataset/deepfake" --num_images 3000

# 🧠 Train and Evaluate Models
# Navigate to the Models folder and run the training scripts.
# Ensure paths to the dataset are set correctly inside the scripts.
# These scripts will train, test, and save the ROC curves, confusion matrix, and training curves.


   
 
   
**For any issue, please contact us at research.ainet@gmail.com.**




