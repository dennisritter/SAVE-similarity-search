import similarity_search as ss



if __name__ == "__main__":
    fns = ['./data/jhoster_chair/10.png', './data/jhoster_chair/20.png']
    k = 5
    features = 'vit_in21k_shapenet'

    ss.get_meshes_from_images(fns, k , features)