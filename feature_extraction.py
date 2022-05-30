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
                features[fn] = outputs


with open('./out/features.pickle', 'wb') as f:
    pickle.dump(features, f)

