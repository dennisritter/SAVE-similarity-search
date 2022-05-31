import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import faiss
import torch
import torchvision
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
import cv2
import json
import pickle

DATA_DIR = "./data/ShapeNetSemV0"
METADATA_FILE = f'{DATA_DIR}/metadata.csv'
MATERIALS_FILE = f'{DATA_DIR}/materials.csv'
CATEGORY_SYNYSET_FILE = f'{DATA_DIR}/categories.synset.csv'
# CLASSES = ["Chair"]
CLASSES = ["Couch", "Chair", "OfficeSideChair"]
# CLASSES = ["Couch", "Sectional", "Sleeper", "Futon", "Loveseat", "Bench", "Stool", "Barstool", "Chair", "AccentChair", "OfficeSideChair", "SideChair", "Recliner", "Chaise", "ChairWithOttoman", "KneelingChair", "BeanBag",
# "Ottoman","Counter", "BarCounter","Desk",	"LDesk","Table",   "AccentTable", "CoffeeTable","DiningTable"," BarTable", "RoundTable", "DraftingTable", "OutdoorTable","SupportFurniture","Pedestal"]
N_CLUSTERS = 128

device = "cuda:0" if torch.cuda.is_available() else "cpu"

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
model = model.to(device)

# get metadata for specified classes
meta = pd.read_csv(METADATA_FILE)
meta = meta[meta["category"].str.contains('|'.join(CLASSES), na=False)]

features = {}
# Sample threee example ids
# sample_ids = [sid[4:] for sid in meta.sample(n=3)["fullId"].values]
sample_ids = [sid[4:] for sid in meta["fullId"].values]
for sample_id in tqdm(sample_ids):
    sample_id_dir = f'{DATA_DIR}/screenshots/{sample_id}'
    if os.path.isdir(sample_id_dir):
        samples_fn = [f'{sample_id_dir}/{imfn}' for imfn in os.listdir(sample_id_dir) if imfn.endswith('.png')]
        for fn in samples_fn:
            im = Image.open(fn).convert('RGB')
            inputs = feature_extractor(im, return_tensors="pt")
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = model(**inputs).pooler_output
                features[fn] = outputs.to('cpu')


# Save he features
with open('./out/features.pickle', 'wb') as f:
    print(f'SAVING FEATURES TO: {f}')
    pickle.dump(features, f)

with open('./out/features.pickle', 'rb') as f:
    print(f'LOADING FEATURES FROM: {f}')
    features = pickle.load(f)

im_paths, im_features = zip(*features.items())
im_paths = list(im_paths)
im_features = list(im_features)
im_features = [feat.numpy().astype('float32').squeeze() for feat in im_features]
im_features = np.array(im_features)
## Using a flat index
index_flat = faiss.IndexFlatL2(768)  # build a flat (CPU) index
index_flat.add(im_features)
faiss.write_index(index_flat, './out/shapenetsem_index.faiss')

## Sanity Check
#### SEARCH
k = 4
index = faiss.read_index('./out/shapenetsem_index.faiss')
D, I = index.search(im_features[:3], k)
print(index.is_trained)
print(index.ntotal)
print(I)
print(D)
