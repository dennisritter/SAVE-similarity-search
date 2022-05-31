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

## Using a flat index



## Sanity Check
#### SEARCH
k = 4
index = faiss.read_index('./out/shapenetsem_index.faiss')
D, I = index.search(im_features[:3], k)
print(index.is_trained)
print(index.ntotal)
print(I)
print(D)