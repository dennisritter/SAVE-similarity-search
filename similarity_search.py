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

with open('./out/features.pickle', 'rb') as f:
    loaded_features = pickle.load(f)

im_paths, im_features = zip(*loaded_features.items())
im_paths = list(im_paths)
im_features = list(im_features)
im_features = [feat.detach().cpu().numpy().astype('float32').squeeze() for feat in im_features]
im_features = np.array(im_features)
## Using a flat index
index_flat = faiss.IndexFlatL2(768)  # build a flat (CPU) index
faiss.write_index(index_flat, './out/shapenetsem_index.faiss')



#### SEARCH
print(im_paths[:3])

k = 4
index = faiss.read_index('./out/shapenetsem_index.faiss')
D, I = index.search(im_features[:3], k)
print(I)
print(D)