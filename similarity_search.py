import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import faiss
import torch
import torchvision
from transformers import ViTFeatureExtractor, ViTModel, FeatureExtractionPipeline
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans
import cv2
import json
import pickle

def get_neighbours(ims_q_fn: list[str], k: int=5, index_name:str ='vit_in21k_shapenet'):
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