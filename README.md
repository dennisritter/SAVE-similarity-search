# similarity-search



## Getting started

0. Clone this repo and move into project root directory
```bash
cd similarity-search
```
1. Create a conda environment with python 3.9 or higher and activate it
```bash
conda env create --name py39-similaritysearch python=3.9
conda activate py39-similaritysearch
```
2. get pytorch following install instructions from [pytorch.org](https://pytorch.org/)
3. install other requirements into your conda environment from requirements.txt
```bash
pip install -r requirements.txt
```
## Description
This project implements methods for similarity search.

## Usage

### Building a Search index
To build an index we provide the `build_index_vit_in21k_shapenet.py` script. It creates a FAISS index using the Huggingface ViT pretrained with ImageNet21K as feature extractor for ShapenetSemV0 screenshots of predefined classes.

### Similarity Search

#### vit_in21k_shapenet
requirements:
1. Get the ShapeNetSemV0 dataset (screenshots, metadata.csv)
2. Build the index for ViT_in21k features of shapenet images
3. make sure these files were created after building the index: features_vit_in21k_shapenet, index_vit_in21k_shapenet, imfn_vit_in21k_shapenet (if imfn_vit_in21k_shapenet.npy is missing, but im_paths_vit_in21k_shapenet.npy exists, rename file)
4. Import the simiarity_search module and run ss.get_meshes_from_images()
```python
import similarity_search as ss

fns = ['q_img_1.png', 'q_img_2.png'] # Query images
k = 5 # How many nearest neighbours to retrieve
index_name = 'vit_in21k_shapenet'

shapenet_ids = ss.get_meshes_from_images(fns, k , index_name)

# example output -> shapened ids of most similar objects
# [['5893038d979ce1bb725c7e2164996f48', 'a42e7daa5ec4b3f0e213bbda0587ff6e'], ['3fc6ab5d3c52c128d810b14a81e12eca', 'cacaca67988f6686f91663a74ccd2338', '47b32bdc02e4025780d6227ff9b21190', 'a42e7daa5ec4b3f0e213bbda0587ff6e', 'd794f296dbe579101e046801e2748f1a']]
```

