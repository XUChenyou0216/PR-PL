# -*- coding: utf-8 -*-
"""
Prepare SEED-IV features in the same file layout used by data_prepare_seed.m.

Input:
    D:/EGG_dataset/SEED-IV/eeg_feature_smooth/{session}/*.mat

Output:
    D:/EGG_dataset/SEED-IV/feature_seediv/sub_{subject}_session_{session}.mat

Each output file contains:
    dataset_session{session}.feature  -> (samples, 310), float32, normalized to [-1, 1]
    dataset_session{session}.label    -> (samples, 1), int16, labels in 1..4
"""

import os

import numpy as np
import scipy.io as scio
from sklearn import preprocessing


RAW_DATA_PATH = 'D:/EGG_dataset/SEED-IV/eeg_feature_smooth'
SAVE_PATH = 'D:/EGG_dataset/SEED-IV/feature_seediv'
NUM_SUBJECTS = 15
NUM_TRIALS = 24
NUM_FEATURES = 310

SESSION_LABELS = {
    1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    2: [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    3: [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
}


def find_subject_file(data_path, subject_id):
    prefix = '{}_'.format(subject_id)
    matches = sorted(
        file_name for file_name in os.listdir(data_path)
        if file_name.startswith(prefix) and file_name.endswith('.mat')
    )
    if not matches:
        raise FileNotFoundError('Cannot find subject {} in {}'.format(subject_id, data_path))
    return os.path.join(data_path, matches[0])


def flatten_trial(trial_data):
    return trial_data.transpose(1, 0, 2).reshape(trial_data.shape[1], NUM_FEATURES)


def load_subject_session(file_path, trial_labels):
    trial_keys = ['de_LDS{}'.format(i) for i in range(1, NUM_TRIALS + 1)]
    mat = scio.loadmat(file_path, variable_names=trial_keys)

    features = []
    labels = []
    for trial_idx, key in enumerate(trial_keys):
        trial_feature = flatten_trial(mat[key])
        features.append(trial_feature)
        labels.append(np.full((trial_feature.shape[0], 1), trial_labels[trial_idx] + 1, dtype=np.int16))

    feature_all = np.vstack(features).astype(np.float32)
    label_all = np.vstack(labels)
    feature_all = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(feature_all).astype(np.float32)
    return feature_all, label_all


def save_subject_session(save_file, session, feature, label):
    dataset_key = 'dataset_session{}'.format(session)
    dataset = np.zeros((1, 1), dtype=[('feature', object), ('label', object)])
    dataset['feature'][0, 0] = feature
    dataset['label'][0, 0] = label

    # Uncompressed .mat files are larger, but much faster to write and load during training.
    scio.savemat(save_file, {dataset_key: dataset}, do_compression=False)


def main():
    os.makedirs(SAVE_PATH, exist_ok=True)

    for session, trial_labels in SESSION_LABELS.items():
        data_path = os.path.join(RAW_DATA_PATH, str(session))
        print('Session {}'.format(session))

        for subject_id in range(1, NUM_SUBJECTS + 1):
            file_path = find_subject_file(data_path, subject_id)
            feature, label = load_subject_session(file_path, trial_labels)

            save_file = os.path.join(
                SAVE_PATH,
                'sub_{}_session_{}.mat'.format(subject_id, session)
            )
            save_subject_session(save_file, session, feature, label)
            print('  subject {:2d}: {} samples'.format(subject_id, feature.shape[0]))

    print('Done. Output: {}'.format(SAVE_PATH))


if __name__ == '__main__':
    main()
