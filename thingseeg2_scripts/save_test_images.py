import numpy as np
import os
from PIL import Image


images = np.load('data/things-eeg2_preproc/test_images.npy', mmap_mode='r')

test_images_dir = 'data/things-eeg2_preproc/test_images_direct/'

if not os.path.exists(test_images_dir):
   os.makedirs(test_images_dir)
for i in range(len(images)):
    im = Image.fromarray(images[i].astype(np.uint8))
    im.save('{}/{}.png'.format(test_images_dir,i))


