# SADCL
 Official PyTorch implementation of the paper "Semantic-Aware Dual Contrastive Learning for Multi-label Image Classification"

The corresponding code will be released.


## Results on MS-COCO2014
![coco](/image/coco.png)


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