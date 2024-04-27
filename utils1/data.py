import numpy as np
import h5py  # Folosit pentru data storage


def load_patch_data(h5_filename='h5_data/Patches_noHole_and_collected.h5', skip_rate=1, points=2048,
                    random_input=True, norm=False):
    if random_input:
        print(f"Random input: {h5_filename}")
        f = h5py.File(h5_filename)
        input1 = f['poisson_4096'][:]
        gt = f['poisson_4096'][:]
    else:
        print(f"Not random input {h5_filename}")
        f = h5py.File(h5_filename)
        input1 = f['montecarlo_1024'][:]
        gt = f['poisson_4096'][:]
    name = f['name'][:]
    assert len(input1) == len(gt)

    if norm:
        print("Normalize the data")
        data_radius = np.ones(shape=(len(input1)))
        centroid = np.mean(gt[:, :, 0:3], axis=1, keepdims=True)
        gt[:, :, 0:3] = gt[:, :, 0:3] - centroid
        furthest_distance = np.amax(np.sqrt(np.sum(gt[:, :, 0:3] ** 2, axis=-1)), axis=1, keepdims=True)
        gt[:, :, 0:3] = gt[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)
        input1[:, :, 0:3] = input1[:, :, 0:3] - centroid
        input1[:, :, 0:3] = input1[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)
    else:
        print("Do not normalize the data")
        centroid = np.mean(gt[:, :, 0:3], axis=1, keepdims=True)
        furthest_distance = np.amax(np.sqrt(np.sum((gt[:, :, 0:3] - centroid) ** 2, axis=-1)), axis=1, keepdims=True)
        data_radius = furthest_distance[0, :]

    input1 = input1[::skip_rate]
    gt = gt[::skip_rate]
    data_radius = data_radius[::skip_rate]
    name = name[::skip_rate]
    object_name = list(set([str(item).split('/')[-1].split('_')[0] for item in name]))
    print(object_name)
    object_name.sort()
    return input1, gt, data_radius, name
