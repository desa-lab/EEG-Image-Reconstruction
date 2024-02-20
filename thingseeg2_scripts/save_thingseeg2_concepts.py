import os
import numpy as np
from PIL import Image
from tqdm import tqdm

img_metadata = np.load('data/things-eeg2_preproc/image_metadata.npy',allow_pickle=True).item()
n_train_img = len(img_metadata['train_img_concepts'])
n_test_img = len(img_metadata['test_img_concepts'])

train_concepts = []
test_concepts = []

for train_img_idx in tqdm(range(n_train_img), total=n_train_img, desc='Loading train images'):
    train_concepts.append(' '.join(img_metadata['train_img_concepts'][train_img_idx].split('_')[1:]))
train_concepts = np.array(train_concepts)

np.save('data/things-eeg2_preproc/train_concepts.npy', train_concepts)

for test_img_idx in tqdm(range(n_test_img), total=n_test_img, desc='Loading test images'):
    test_concepts.append(' '.join(img_metadata['test_img_concepts'][test_img_idx].split('_')[1:]))
test_concepts = np.array(test_concepts)

np.save('data/things-eeg2_preproc/test_concepts.npy', test_concepts)