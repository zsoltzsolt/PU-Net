import numpy as np
import h5py


def load_patch_data(h5_filename='h5_data/Patches_noHole_and_collected.h5', skip_rate=1, points=2048,
                    random_input=True, norm=False):

    data = h5py.File(h5_filename)

    input_ = data['poisson_4096'][:] if random_input else data['montecarlo_1024'][:]
    ground_truth = data['poisson_4096'][:]
    name = data['name'][:]
    
    assert len(input_) == len(ground_truth), "Input and gt do not have the same size"

    if norm:
        data_radius = np.ones(shape=(len(input_)))
        centroid = np.mean(ground_truth[:, :, :3], axis=1, keepdims=True)
        ground_truth[:, :, :3] = ground_truth[:, :, :3] - centroid
        furthest_distance = np.amax(np.sqrt(np.sum(ground_truth[:, :, :3] ** 2, axis=-1)), axis=1, keepdims=True)
        ground_truth[:, :, :3] = ground_truth[:, :, :3] / np.expand_dims(furthest_distance, axis=-1)
        input_[:, :, :3] = input_[:, :, :3] - centroid
        input_[:, :, :3] = input_[:, :, :3] / np.expand_dims(furthest_distance, axis=-1)
    else:
        centroid = np.mean(ground_truth[:, :, :3], axis=1, keepdims=True)
        furthest_distance = np.amax(np.sqrt(np.sum((ground_truth[:, :, :3] - centroid) ** 2, axis=-1)), axis=1, keepdims=True)
        data_radius = furthest_distance[0, :]

    input_ = input_[::skip_rate]
    ground_truth = ground_truth[::skip_rate]
    data_radius = data_radius[::skip_rate]
    name = name[::skip_rate]
    object_name = list(set([str(item).split('/')[-1].split('_')[0].upper() for item in name]))
    object_name.sort()

    return input_, ground_truth, data_radius, object_name
