The datasets are publicly available

1. CT-GAN dataset Available at https://www.kaggle.com/datasets/ymirsky/medical-deepfakes-lung-cancer

2. Synthetic Knee Deepfake images Dataset Available at https://data.mendeley.com/datasets/fyybnjkw7v/3
   
3. Knee X-ray Real Images Dataset Available at https://data.mendeley.com/datasets/56rmx5bjcr/1

### 📚 **Citations to Datasets**

1. **Mirsky, Yisroel; Mahler, Tom; Shelef, Ilan; Elovici, Yuval** (2019).  
   *CT-GAN: Malicious tampering of 3D medical imagery using deep learning*.  
   In *28th USENIX Security Symposium (USENIX Security 19)*, pp. 461–478.  
   [`Paper Link`](https://www.usenix.org/conference/usenixsecurity19/presentation/mirsky)

2. **Prezja, Fabi; Paloneva, Juha; Pölönen, Ilkka; Niinimäki, Esko; Äyrämö, Sami** (2022).  
   *Synthetic (DeepFake) Knee Osteoarthritis X-ray Images from Generative Adversarial Neural Networks*.  
   Mendeley Data, V3. DOI: [10.17632/fyybnjkw7v.3](https://doi.org/10.17632/fyybnjkw7v.3)

3. **Chen, Pingju**(2018).  
   *Knee Osteoarthritis Severity Grading Dataset*.  
   Mendeley Data, V1. DOI: [10.17632/56rmx5bjcr.1](https://data.mendeley.com/datasets/56rmx5bjcr/1)


### 🛠️ **Instructions**

```bash
# 📥 Download the Datasets
# Download datasets from the sources provided in the 📚 Citations to Datasets section.

# 🧪 Preprocess CT-GAN Dataset
# This script converts .DICOM files into .jpg slices and organizes them.

# Without annotation:
python preprocessing.py --dataset_dir ./dataset --output_dir ./processed_output

# With tumor annotation:
python preprocessing.py --dataset_dir ./dataset --output_dir ./processed_output --annotate

# 🗂️ Create Class-wise Folders
# Organize the dataset into folders based on classification type.

# Multi-class:
python extract_samples.py --dataset_dir ./dataset --processed_dir ./processed_output --output_dir ./DatasetLungs --class_mode multi --offset_start -10 --offset_end 10

# Binary class:
python extract_samples.py --dataset_dir ./dataset --processed_dir ./processed_output --output_dir ./DatasetLungs --class_mode binary --offset_start -10 --offset_end 10

# 🦵 Preprocess Knee X-ray Dataset
# Preprocess and generate real and DeepFake Knee X-ray data samples.

python enhance_knee_images.py --input_root ./KneeXrayData/ClsKLData --output_root ./KneePreprocessed

python build_real_data.py --source_root ./KneePreprocessed --output_dir ./KneePreprocessed/real --num_images 2500

python build_deepfake_data.py --source_root ./Dataset_Knee --output_dir ./KneeMedDataset/deepfake --num_images 3000

# 🧠 Train and Evaluate Models
# Navigate to the Models folder and run the training scripts.
# Ensure paths to the dataset are set correctly inside the scripts.
# These scripts will train, test, and save the ROC curves, confusion matrix, and training curves.

Additional Test:
you can consider seperate division i.e., EXP1_Blind and EXP2_OPEN of CT-GAN dataset as train and test sets:
python generate_patient_slices.py --dataset_dir ./dataset --output_dir ./preprocessed_output1
and afterthat run:
python extract_samples.py --dataset_dir ./dataset --processed_dir ./processed_output1/exp1_preprocessed --output_dir ./DatasetLungs/Train/ --class_mode binary --offset_start -10 --offset_end 10
python extract_samples.py --dataset_dir ./dataset --processed_dir ./processed_output1/exp2_preprocessed --output_dir ./DatasetLungs/Test/ --class_mode binary --offset_start -10 --offset_end 10
After that, run file model_exp1-exp2.py.


**NOte**: you can install dependencies as pip install -r requirements.txt

```



   
 
   
**For any issue, please contact us at research.ainet@gmail.com.**




