import similarity_search as ss


if __name__ == "__main__":
    fns = ['./data/jhoster_chair/10.png', './data/jhoster_chair/20.png']
    k = 5
    index_name = 'vit_in21k_shapenet'

    shapenet_ids = ss.get_meshes_from_images(fns, k , index_name)