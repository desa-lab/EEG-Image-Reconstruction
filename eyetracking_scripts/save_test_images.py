import numpy as np
import os
from PIL import Image


# images = np.load('data/eyetracking_data/simonPilotResults/test_img.npy', mmap_mode='r')

# test_images_dir = 'data/eyetracking_data/simonPilotResults/test_images/'

images = np.load('data/eyetracking_data/simonPilotResults/test_img_ts.npy', mmap_mode='r')

test_images_dir = 'data/eyetracking_data/simonPilotResults/test_images_ts/'

if not os.path.exists(test_images_dir):
   os.makedirs(test_images_dir)
for i in range(len(images)):
    im = Image.fromarray(images[i].astype(np.uint8))
    im.save('{}/{}.png'.format(test_images_dir,i))


