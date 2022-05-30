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

print(len(loaded_features.keys()))

res = faiss.StandardGpuResources()  # use a single GPU
## Using a flat index
index_flat = faiss.IndexFlatL2(768)  # build a flat (CPU) index
# make it a flat GPU index
gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
gpu_index_flat.add_with_ids()