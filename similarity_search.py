import os
import numpy as np
import faiss
import torch
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image



def get_meshes_from_images(ims_q_fn: list[str], k: int=5, index_name:str ='vit_in21k_shapenet'):
    """ Returns meshes for given query images. The number of items that is returned varies as
        nearest neighbour images may belong to the same mesh.
    
        Args:
            ims_fn (list[str]): A list of filenames of query images.
            k (int): The number of neighbours to find.
            index_name (str): The index_name specifies the index/method that is used for retrieval. 
    """
    if index_name == 'vit_in21k_shapenet':
        ims_fn = np.load(f'./data/{index_name}/imfn_{index_name}.npy')
        distances, indices = get_neighbours(ims_q_fn, k, index_name)
        
        # Get neighbour indices for each query image
        mesh_ids_for_qs = []
        for ims_q_i, neighbours_i in enumerate(indices):
            mesh_ids_for_q = []
            for im_i in neighbours_i:
                im_fn = ims_fn[im_i]
                shapenet_id = im_fn.split('-')[-2] 
                mesh_ids_for_q.append(shapenet_id)
            # remove duplicates
            mesh_ids_for_qs.append(list(set(mesh_ids_for_q)))
        print(mesh_ids_for_qs)
        return mesh_ids_for_qs
    else:
        raise ValueError(f'Argument: index_name={index_name} is not a valid feature.')
    



def get_neighbours(ims_q_fn: list[str], k: int, index_name:str ='vit_in21k_shapenet'):
    """ Returns list of tuples of distances and indices of the k nearest neighbours 
        for each given image filename from a search index of specified features.

        Args:
            ims_fn (list[str]): A list of filenames of query images.
            k (int): The number of neighbours to find.
            index_name (str): The index_name specifies the index/method that is used for retrieval. 
    """

    if not all([os.path.isfile(fn) for fn in ims_q_fn]):
        raise ValueError(f'Argument: ims_fn contains filenames that are not valid')
    
    if index_name == 'vit_in21k_shapenet':
        return get_neighbours_vit_in21k(ims_q_fn, k)
    # elif features == '':
    #     return ...
    else:
        raise ValueError(f'Argument: index_name={index_name} is not a valid feature.')

def get_neighbours_vit_in21k(ims_q_fn: list[str], k: int):
    """ Returns list of tuples of distances and indices of the k nearest neighbours for each 
        given image filename from an index that contains ImageNet 21k pretrained ViT features of ShapeNetSemV0 rendered images.

        Args:
            ims_fn (list[str]): A list of filenames of query images.
            k (int): The number of neighbours to find.
            features (str): The features the search index contains. Specifies the index/method. 
    """
    index = faiss.read_index('./data/vit_in21k_shapenet/index_vit_in21k_shapenet.faiss')
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

    ims_q = [Image.open(fn).convert('RGB') for fn in ims_q_fn]
    inputs = feature_extractor(ims_q, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs).pooler_output
    outputs = np.array(outputs)

    distances, indices = index.search(outputs, k)
    return distances, indices