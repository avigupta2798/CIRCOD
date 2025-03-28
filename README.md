<!-- # CIRCOD - Official Pytorch Implementation (WACV 2025) -->
<div align="center">
<h1>Official Pytorch Implementation of CIRCOD: 

Co-Saliency Inspired Referring Camouflaged Object Discovery </h1>
Avi Gupta, Koteswar Rao Jerripothula, Tammam Tillo <br />
Indraprastha Institute of Information Technology, Delhi, India</sub><br />

[![Conference](https://img.shields.io/badge/WACV-2025-blue)](https://openaccess.thecvf.com/content/WACV2025/papers/Gupta_CIRCOD_Co-Saliency_Inspired_Referring_Camouflaged_Object_Discovery_WACV_2025_paper.pdf)<br />

<!--[![Paper]()]() -->

<img src = "Figures/Architecture.png" width="100%" height="100%">
</div>

## Abtract
Camouflaged object detection (COD), the task of identifying objects concealed within their surroundings, is often quite challenging due to the similarity that exists between the foreground and background. By incorporating an additional referring image where the target object is clearly visible, we can leverage the similarities between the two images to detect the camouflaged object. In this paper, we propose a novel problem setup: referring camouflaged object discovery (RCOD). In RCOD, segmentation occurs only when the object in the referring image is also present in the camouflaged image; otherwise, a blank mask is returned. This setup is particularly valuable when searching for specific camouflaged objects. Current COD methods are often generic, leading to numerous false positives in applications focused on specific objects. To address this, we introduce a new framework called Co-Saliency Inspired Referring Camouflaged Object Discovery (CIRCOD). Our approach consists of two main components: Co-Saliency-Aware Image Transformation (CAIT) and Co-Salient Object Discovery (CSOD). The CAIT module reduces the appearance and structural variations between the camouflaged and referring images, while the CSOD module utilizes the similarities between them to segment the camouflaged object, provided the images are semantically similar. Covering all semantic categories in current COD benchmark datasets, we collected over 1,000 referring images to validate our approach. Our extensive experiments demonstrate the effectiveness of our method and show that it achieves superior results compared to existing methods.

## Preparation

### Requirements
Conda environment settings:
```
conda create -n CIRCOD python=3.8
conda activate CIRCOD
```

### Datasets

We use the [COD10K](), [NC4K](), [CAMO](), [R2C7K]() and proposed [Ref-1K](https://drive.google.com/file/d/15lPH9-ueSLx90cCeVFJhh-8N9M3om74S/view?usp=sharing) for evaluation.

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

## Citation
If you find the repository or the paper useful or you use the data, please use the following entry for citation.
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
