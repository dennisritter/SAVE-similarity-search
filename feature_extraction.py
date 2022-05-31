import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
from tqdm import tqdm
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

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
inputs = feature_extractor(img, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs).pooler_output


