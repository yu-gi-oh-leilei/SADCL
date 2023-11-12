# SADCL
Official PyTorch implementation of the paper "[Semantic-Aware Dual Contrastive Learning for Multi-label Image Classification](https://arxiv.org/format/2307.09715)"


Our paper was accepted by ECAI2023(European Conference on Artificial Intelligence).


## Results on MS-COCO2014
![coco](./image/coco.png)


## Visualization
### prototypes
![prototypes](./image/prototypes.png)

### category-related attention maps
![cam](./image/cam.png)


### 2000 label-level visual representations

#### without dual contrastive learning
![vis_embed_list_base](./image/vis_embed_list_base.svg)
#### with dual contrastive learning
![vis_embed_list_sadcl](./image/vis_embed_list_sadcl.svg)

#### with dual contrastive learning (Projector head)
![vis_embed_list_sadclv2](./image/vis_embed_list_sadclv2.svg)

# BibTex
```
@article{ma2023semantic,
  title={Semantic-Aware Dual Contrastive Learning for Multi-label Image Classification},
  author={Ma, Leilei and Sun, Dengdi and Wang, Lei and Zhao, Haifang and Luo, Bin},
  journal={arXiv preprint arXiv:2307.09715},
  year={2023}
}
or
@inproceedings{DBLP:conf/ecai/MaSWZ023,
  author       = {Leilei Ma and Dengdi Sun and Lei Wang and Haifeng Zhao and Bin Luo},
  editor       = {Kobi Gal and Ann Now{\'{e}} and Grzegorz J. Nalepa and Roy Fairstein and Roxana Radulescu},
  title        = {Semantic-Aware Dual Contrastive Learning for Multi-Label Image Classification},
  booktitle    = {{ECAI} 2023 - 26th European Conference on Artificial Intelligence,
                  September 30 - October 4, 2023, Krak{\'{o}}w, Poland - Including
                  12th Conference on Prestigious Applications of Intelligent Systems
                  {(PAIS} 2023)},
  series       = {Frontiers in Artificial Intelligence and Applications},
  volume       = {372},
  pages        = {1656--1663},
  publisher    = {{IOS} Press},
  year         = {2023},
  url          = {https://doi.org/10.3233/FAIA230449},
  doi          = {10.3233/FAIA230449},
  timestamp    = {Wed, 18 Oct 2023 09:31:16 +0200},
  biburl       = {https://dblp.org/rec/conf/ecai/MaSWZ023.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
# Contact US
In case of any queries, please feel free to contact us for assistance.
E-mail: xiaoleilei1990@gmail.com

# Acknowledgement
We thank the authors of Query2label, detr, for their great works and codes. Thanks to @SlongLiu for providing a useful script for training.