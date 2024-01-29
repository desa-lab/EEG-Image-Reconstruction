# %%
import mne
# mne.viz.set_browser_backend("matplotlib")
# mne.viz.set_3d_backend("notebook")
from matplotlib import pyplot as plt
# %config InlineBackend.figure_format='retina'
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids
from itertools import product
from tqdm import tqdm
import os

# %%
data_dir = '/mnt/sphere/projects/image_decoding_from_brain/THINGS-MEG-raw/download'
subject_file = data_dir + '/participants.tsv'
subjects = pd.read_csv(subject_file, sep="\t")
def get_subject_id(x):
    return x.split("-")[1]  # noqa
subjects = subjects.participant_id.apply(get_subject_id).values
subject = subjects[0]
tasks = ['main']
sessions = ['{:02d}'.format(x) for x in range(1,13)]  # 2 recording sessions
runs = ['{:02d}'.format(x) for x in range(1,11)]

# %%
print('Loading AutoKL...')

# %%
autokl = np.load('/mnt/sphere/projects/image_decoding_from_brain/data/extracted_features/things_autokl.npy', mmap_mode='r')

# %%
print('AutoKL shape: ', autokl.shape)

# %%
train_autokl = None
test_autokl = None
for session, task, run in tqdm(product(sessions, tasks, runs), total=len(sessions)*len(tasks)*len(runs), desc=f'Processing subject {subject}'):
    bids_path = BIDSPath(
        subject=subject,
        session=session,
        task=task,
        root=data_dir,
        run = run,
        datatype="meg",
    )
    raw = read_raw_bids(bids_path, verbose='ERROR')
    events = list()
    for annot in raw.annotations:
        # event = eval(annot.pop("description"))
        event = dict()
        event['kind'] = annot['description'].split('/')[0]
        if annot['description'].split('/')[0] != 'catch':
            event['image_id'] = annot['description'].split('/')[1]
        else:
            event['image_id'] = '-1' # catch trials, maybe we should change this
        event['start'] = annot['onset']
        event['duration'] = annot['duration']
        events.append(event)
    events_df = pd.DataFrame(events)
    train_events_df = events_df[events_df['kind'] == 'exp']
    test_events_df = events_df[events_df['kind'] == 'test']
    train_image_ids = pd.to_numeric(train_events_df['image_id']).astype(int).to_numpy() # +1 if we need one-based indexing
    test_image_ids = pd.to_numeric(test_events_df['image_id']).astype(int).to_numpy() # +1 if we need one-based indexing
    if train_autokl is None:
        train_autokl = autokl[train_image_ids]
        test_autokl = autokl[test_image_ids]
    else:
        train_autokl = np.concatenate((train_autokl, autokl[train_image_ids]), axis=0)
        test_autokl = np.concatenate((test_autokl, autokl[test_image_ids]), axis=0)

print('train_autokl shape: ', train_autokl.shape) # (22248, 4, 64, 64)
print('test_autokl shape: ', test_autokl.shape) # (2400, 4, 64, 64)

# %%
save_dir = 'cache/extracted_embeddings/' + subject + '/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
np.save(save_dir + f'train_autokl_sub-{subject}.npy', train_autokl)
np.save(save_dir + f'test_autokl_sub-{subject}.npy', test_autokl)
