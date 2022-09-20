import os
import numpy as np
import pandas as pd
import faiss
import torch
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
from tqdm import tqdm
import pickle

INDEX_NAME="vit_in21k_shapenet"
DATA_DIR = "./data/ShapeNetSemV0"
OUT_DIR = f"./data/{INDEX_NAME}"
METADATA_FILE = f'{DATA_DIR}/metadata.csv'
CLASSES = ["Couch", "Chair", "OfficeSideChair"]
# CLASSES = ["Couch", "Sectional", "Sleeper", "Futon", "Loveseat", "Bench", "Stool", "Barstool", "Chair", "AccentChair", "OfficeSideChair", "SideChair", "Recliner", "Chaise", "ChairWithOttoman", "KneelingChair", "BeanBag",
# "Ottoman","Counter", "BarCounter","Desk",	"LDesk","Table",   "AccentTable", "CoffeeTable","DiningTable"," BarTable", "RoundTable", "DraftingTable", "OutdoorTable","SupportFurniture","Pedestal"]
N_CLUSTERS = 128

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# define model
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
        samples_fn = [f'{imfn}' for imfn in os.listdir(sample_id_dir) if imfn.endswith('.png')]
        for fn in samples_fn:
            im = Image.open(f'{sample_id_dir}/{fn}').convert('RGB')
            inputs = feature_extractor(im, return_tensors="pt")
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = model(**inputs).pooler_output
                features[fn] = outputs.to('cpu')


# Save the features
with open(f'{OUT_DIR}/features_{INDEX_NAME}.pickle', 'wb+') as f:
    print(f'SAVING FEATURES TO: {f}')
    pickle.dump(features, f)


im_paths, im_features = zip(*features.items())
# Save image paths in numpy array separately for similarity search
im_paths = np.array(im_paths)
np.save(f'{OUT_DIR}/im_paths_{INDEX_NAME}.npy', im_paths)

# Save image features 
im_features = list(im_features)
im_features = [feat.numpy().astype('float32').squeeze() for feat in im_features]
im_features = np.array(im_features)

## Using a flat index
index_flat = faiss.IndexFlatL2(768)  # build a flat (CPU) index
index_flat.add(im_features)
faiss.write_index(index_flat, f'{OUT_DIR}/index_{INDEX_NAME}.faiss')


## Sanity Check
print('-' * 10)
print('SANITY CHECK')
k = 5
index = faiss.read_index(f'{OUT_DIR}/index_{INDEX_NAME}.faiss')
distances, indices = index.search(im_features[:3], k)
print(f'Index is trained?       {index.is_trained}')
print(f'Index n_total           {index.ntotal}')
print(f'Retrieval Indices       {indices}')
print(f'Retrieval Distances     {distances}')
