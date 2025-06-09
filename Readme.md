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


**Instructions:**
1. First, download the datasets from the corresponding links.
2. For CT-GAN dataset, first run preprocessing.py. It will convert .DICOM files into corresponding .jpg slices for each scan and save them into corresponding folders.
   !python preprocessing.py --dataset_dir D:/archive --output_dir D:/processed_output
   if you need annotation around tumor, then pass annotate i.e., !python preprocessing.py --dataset_dir D:/archive --output_dir D:/processed_output

   After that, run create_folders.py. It will create folders class wise accordingly.
   For Multi-class
   !python extract_samples.py --dataset_dir D:/archive --processed_dir D:/processed_output --output_dir D:/Dataset --class_mode multi --offset_start -10 --offset_end 10
   For Binary classes:
   !python extract_samples.py --dataset_dir D:/archive --processed_dir D:/processed_output --output_dir D:/Dataset --class_mode binary --offset_start -10 --offset_end 10
   
3. For Knee-X ray dataset, first run, preprocessing.py on Knee-X ray real data and after that run build_real_data.py and build_deepfake_data.py.
   !python enhance_knee_images.py --input_root "D:/56rmx5bjcr-1/KneeXrayData/ClsKLData" --output_root "D:/KneePreprocessed"
   !python build_real_data.py --source_root "D:/KneePreprocessed" --output_dir "D:/KneePreprocessed/real" --num_images 2500
   !python build_deepfake_data.py --source_root "D:/Data" --output_dir "D:/KneeMedDataset/deepfake" --num_images 3000

   
   
4. After than run corresponding files from Models folder by setting paths of folder you obtained at step 2 or 3. This model files load the processed data, trains and test the model followed by 
   saving ROC, Confusin matrix and training curves.
   
**For any issue, please contact us at research.ainet@gmail.com.**




