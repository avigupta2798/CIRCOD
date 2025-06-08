<!-- # CIRCOD - Official Pytorch Implementation (WACV 2025) -->
<div align="center">
<h1>Official Pytorch Implementation of CIRCOD: 

Co-Saliency Inspired Referring Camouflaged Object Discovery </h1>
Avi Gupta, Koteswar Rao Jerripothula, Tammam Tillo <br />
Indraprastha Institute of Information Technology, Delhi, India</sub><br />

[![Conference](https://img.shields.io/badge/WACV-2025-blue)](https://openaccess.thecvf.com/content/WACV2025/papers/Gupta_CIRCOD_Co-Saliency_Inspired_Referring_Camouflaged_Object_Discovery_WACV_2025_paper.pdf)
[![Project](https://img.shields.io/badge/Project-2025-red)](https://www.iiitd.edu.in/~avig/project/CIRCOD/index.html)<br />

<!--[![Paper]()]() -->

<img src = "Figures/Architecture.png" width="100%" height="100%">
</div>

## Abtract
Camouflaged object detection (COD), the task of identifying objects concealed within their surroundings, is often quite challenging due to the similarity that exists between the foreground and background. By incorporating an additional referring image where the target object is clearly visible, we can leverage the similarities between the two images to detect the camouflaged object. In this paper, we propose a novel problem setup: referring camouflaged object discovery (RCOD). In RCOD, segmentation occurs only when the object in the referring image is also present in the camouflaged image; otherwise, a blank mask is returned. This setup is particularly valuable when searching for specific camouflaged objects. Current COD methods are often generic, leading to numerous false positives in applications focused on specific objects. To address this, we introduce a new framework called Co-Saliency Inspired Referring Camouflaged Object Discovery (CIRCOD). Our approach consists of two main components: Co-Saliency-Aware Image Transformation (CAIT) and Co-Salient Object Discovery (CSOD). The CAIT module reduces the appearance and structural variations between the camouflaged and referring images, while the CSOD module utilizes the similarities between them to segment the camouflaged object, provided the images are semantically similar. Covering all semantic categories in current COD benchmark datasets, we collected over 1,000 referring images to validate our approach. Our extensive experiments demonstrate the effectiveness of our method and show that it achieves superior results compared to existing methods.

## Preparation

### Requirements
Conda environment settings:
```
conda env create -f environment.yml
conda activate circod
```

### Datasets

We use the [data](https://drive.google.com/drive/folders/16pzODVztI8ea0BRxJC0ZSobG7b56iXb-?usp=sharing) in below format for evaluation:

```
data_root/
   ├── COD10K/
   │   ├── Images/
   │   ├── GT
   └── NC4K
   │   ├── Images/
   │   ├── GT
   ├── CAMO/
   │   ├── Images/
   │   ├── GT
   ├── R2C7K/
   │   ├── Camo/
   │   ├── Ref
   ├── Ref-1K/
   │   ├── Images/
   │   ├── GT
```

### Pre-trained Models
Download the pre-trained models from [here](https://drive.google.com/drive/folders/13dIAgv27Cu0FJdAEf4g1QZtzjwiz7HM_?usp=sharing) and place it in the ``pre-trained`` folder.

### Training
To train the networks, run the below commands for saliency enhancement network (SEN)
```
python train_sen.py --lr 5e-5 --wd 0.0001 --gpu_main 0 --train_batch_size 8 --test_batch_size 8 --num_worker 4 --image_size 512 --epoches 10 --train_dataset cod10k_train --test_dataset camo --model_file checkpoints/sen_cod10k.pkl --task cod
```
```
python train_sen.py --lr 5e-5 --wd 0.0001 --gpu_main 0 --train_batch_size 8 --test_batch_size 8 --num_worker 4 --image_size 512 --epoches 10 --train_dataset r2c7k_train --test_dataset r2c7k_test --model_file checkpoints/sen_r2c7k.pkl --task cod
```
and final CIRCOD:
```
CUDA_VISIBLE_DEVICES=0 python train_main.py --lr 5e-5 --wd 0.0001 --gpu_main 0 --num_worker 4 --train_batch_size 8 --test_batch_size 8 --image_size 512 --task rcod --epoches 60 --cod_train_dataset cod10k_train --cod_test_dataset camo --search_dataset si1k_ref --checkpoint checkpoints/circod_cod10k.pkl --sen_path checkpoints/sen_cod10k.pkl
```
```
CUDA_VISIBLE_DEVICES=0 python train_main.py --lr 5e-5 --wd 0.0001 --gpu_main 0 --num_worker 4 --train_batch_size 8 --test_batch_size 8 --image_size 512 --task ref-cod --epoches 60 --cod_train_dataset r2c7k_train --cod_test_dataset r2c7k_test --search_dataset r2ck_ref --checkpoint checkpoints/circod_cod10k.pkl --sen_path checkpoints/sen_r2c7k.pkl
```
### Testing
Download the checkpoints from [here](https://drive.google.com/drive/folders/15WOWaYTOtRWmvye9GIsJZ3BU8q7OkUZo?usp=sharing) and place it in the ``checkpoints`` folder. Run the below commands for saliency enhancement network (SEN):
```
python test_sen.py --gpu_main 0 --image_size 512 --dataset camo --snapshot checkpoints/sen_cod10k.pkl --task cod
```
```
python test_sen.py --gpu_main 0 --image_size 512 --dataset r2c7k_test --snapshot checkpoints/sen_r2c7k.pkl --task cod
```
and final CIRCOD:
```
CUDA_VISIBLE_DEVICES=0 python test_main.py --gpu 0 --image_size 512 --cod_dataset camo --search_dataset si1k_ref --snapshot checkpoints/sen_cod10k.pkl --sen_path checkpoints/sen_cod10k.pkl --task rcod
```
```
CUDA_VISIBLE_DEVICES=0 python test_main.py --gpu 0 --image_size 512 --cod_dataset r2c7k_test --search_dataset r2ck_ref --snapshot checkpoints/ --sen_path checkpoints/sen_r2c7k.pkl --task ref-cod
```
**Remaining checkpoints will be uploaded soon.**
## Citation
If you find the repository or the paper useful or you use the proposed data, please use the following entry for citation.
````BibTeX
@InProceedings{Gupta_2025_WACV,
    author    = {Gupta, Avi and Jerripothula, Koteswar Rao and Tillo, Tammam},
    title     = {CIRCOD: Co-Saliency Inspired Referring Camouflaged Object Discovery},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {8302-8312}
}
````
## Contributors and Contact
If there are any questions, feel free to contact the authors: Avi Gupta (avig@iiitd.ac.in).
