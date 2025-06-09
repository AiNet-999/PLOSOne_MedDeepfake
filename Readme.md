The datasets are publicly available

1. CT-GAN dataset Available at https://www.kaggle.com/datasets/ymirsky/medical-deepfakes-lung-cancer

2. Synthetic Knee Deepfake images Dataset Available at https://data.mendeley.com/datasets/fyybnjkw7v/3
   
3. Knee X-ray Real Images Dataset Available at https://data.mendeley.com/datasets/t9ndx37v5h/1 

Citations to Datasets:
1. @inproceedings{mirsky2019ct,
  title={CT-GAN: Malicious tampering of 3D medical imagery using deep learning},
  author={Mirsky, Yisroel and Mahler, Tom and Shelef, Ilan and Elovici, Yuval},
  booktitle={28th $\{$USENIX$\}$ Security Symposium ($\{$USENIX$\}$ Security 19)},
  pages={461--478},
  year={2019}
}
2. Prezja, Fabi; Paloneva, Juha; Pölönen, Ilkka; Niinimäki, Esko; Äyrämö, Sami (2022), 
“Synthetic (DeepFake) Knee Osteoarthritis X-ray Images from Generative Adversarial Neural Networks”,
Mendeley Data, V3, doi: 10.17632/fyybnjkw7v.3

3.Gornale, Shivanand; Patravali, Pooja (2020), “Digital Knee X-ray Images”, Mendeley Data, V1, doi: 10.17632/t9ndx37v5h.1


Steps To Run:
1. First, download the datasets from the corresponding links.
2. Then run preprocessing_CTGAN.py. It will convert .DICOM files into corresponding .jpg slices for each scan and save them into corresponding folders.
3. After that, run create_folders.py. It will create folders class wise according. 


