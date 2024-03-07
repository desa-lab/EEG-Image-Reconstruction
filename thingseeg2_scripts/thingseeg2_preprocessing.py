from scipy import io
import numpy as np
import mne
from mne import concatenate_raws
from mne_bids import BIDSPath, read_raw_bids
from tqdm import tqdm
from itertools import product

data_dir = 'data/thingseeg2/raw_bids/'
subjects = ['{:02d}'.format(x) for x in range(1,11)]
sessions = ['{:02d}'.format(x) for x in range(1,5)]
runs = ['{:02d}'.format(x) for x in range(1,6)]

raw_list = []
for session, run in tqdm(product(sessions, runs), total=len(sessions)*len(runs)):
    beh_path = BIDSPath(
        subject=subjects[0],
        session=session,
        task='train',
        root=data_dir,
        run = run,
        datatype="beh",
        suffix='beh'
    )
    beh_data = io.loadmat(str(beh_path.fpath) + '.mat')['data']
    events_beh = np.asarray(beh_data[0][0][2]['tot_img_number'][0], dtype=int) - 1
    # events_beh = np.array(['stim/' + item for item in events_beh.astype(str)])
    events_beh = np.array(['stim/' + item for item in np.char.zfill(events_beh.astype(str), 5)]) # Outputs: ['04000' '00500' '00060' '00007']

    eeg_path = BIDSPath(
        subject=subjects[0],
        session=session,
        task='train',
        root=data_dir,
        run = run,
        datatype="eeg",
    )
    raw = read_raw_bids(eeg_path, verbose='ERROR')
    raw.annotations.description[1:] = events_beh
    # raw.annotations.rename({'stim/-1':'catch'})
    raw.annotations.rename({'stim/-0001':'catch'})
    raw.annotations.delete(0) # delete the "new segment" event
    raw.resample(100)

    raw_list.append(raw)
raws = concatenate_raws(raw_list, on_mismatch="ignore", preload=True)
events, event_dict = mne.events_from_annotations(raws)
epochs = mne.Epochs(raws, events, event_id=event_dict, tmin=-0.2, tmax=1.0, baseline=(-0.2, 0), preload=True)
epochs.equalize_event_counts(method='truncate')

epochs_np = epochs.get_data()
i_event_dict = {v: k for k, v in event_dict.items()} # invert key value pairs
events_list = [i_event_dict[i] for i in epochs.events[:, 2]]
# Sort the list and get the indices
sorted_list = sorted(enumerate(events_list), key=lambda x: x[1])
# Extract the sorted indices and values
sorted_indices, sorted_values = zip(*sorted_list)

epochs_np = epochs_np[np.array(sorted_indices)]
epochs_np = epochs_np[4:]
epochs_np = epochs_np.reshape((16540, 4, 63, 121))
epochs_np = epochs_np[:,:,:,:120]
epochs_np_avg = epochs_np.mean(1)

train_thingseeg2_avg = epochs_np_avg[:,:,20:]
np.save('data/thingseeg2/train_thingseeg2_avg1.npy', train_thingseeg2_avg)
np.save('data/thingseeg2/train_thingseeg2_avg1_200ms.npy', train_thingseeg2_avg[:,:,:20])
np.save('data/thingseeg2/train_thingseeg2_avg1_400ms.npy', train_thingseeg2_avg[:,:,:40])
np.save('data/thingseeg2/train_thingseeg2_avg1_600ms.npy', train_thingseeg2_avg[:,:,:60])
np.save('data/thingseeg2/train_thingseeg2_avg1_800ms.npy', train_thingseeg2_avg[:,:,:80])


raw_list = []
for session in tqdm(sessions, total=len(sessions)):
    beh_path = BIDSPath(
        subject=subjects[0],
        session=session,
        task='test',
        root=data_dir,
        datatype="beh",
        suffix='beh'
    )
    beh_data = io.loadmat(str(beh_path.fpath) + '.mat')['data']
    events_beh = np.asarray(beh_data[0][0][2]['tot_img_number'][0], dtype=int) - 1
    # events_beh = np.array(['stim/' + item for item in events_beh.astype(str)])
    events_beh = np.array(['stim/' + item for item in np.char.zfill(events_beh.astype(str), 5)]) # Outputs: ['04000' '00500' '00060' '00007']

    eeg_path = BIDSPath(
        subject=subjects[0],
        session=session,
        task='test',
        root=data_dir,
        datatype="eeg",
    )
    raw = read_raw_bids(eeg_path, verbose='ERROR')

    raw.annotations.description[1:] = events_beh
    # raw.annotations.rename({'stim/-1':'catch'})
    raw.annotations.rename({'stim/-0001':'catch'})
    raw.annotations.delete(0) # delete the "new segment" event
    raw.resample(100)

    raw_list.append(raw)
raws = concatenate_raws(raw_list, on_mismatch="ignore", preload=True)
events, event_dict = mne.events_from_annotations(raws)
epochs = mne.Epochs(raws, events, event_id=event_dict, tmin=-0.2, tmax=1.0, baseline=(-0.2, 0), preload=True)
epochs.equalize_event_counts(method='truncate')

epochs_np = epochs.get_data()
i_event_dict = {v: k for k, v in event_dict.items()} # invert key value pairs
events_list = [i_event_dict[i] for i in epochs.events[:, 2]]
# Sort the list and get the indices
sorted_list = sorted(enumerate(events_list), key=lambda x: x[1])
# Extract the sorted indices and values
sorted_indices, sorted_values = zip(*sorted_list)

epochs_np = epochs_np[np.array(sorted_indices)]
epochs_np = epochs_np[80:]
epochs_np = epochs_np.reshape((200, 80, 63, 121))
epochs_np = epochs_np[:,:,:,:120]
epochs_np_avg = epochs_np.mean(1)

test_thingseeg2_avg = epochs_np_avg[:,:,20:]
np.save('data/thingseeg2/test_thingseeg2_avg1.npy', test_thingseeg2_avg)
np.save('data/thingseeg2/test_thingseeg2_avg1_200ms.npy', test_thingseeg2_avg[:,:,:20])
np.save('data/thingseeg2/test_thingseeg2_avg1_400ms.npy', test_thingseeg2_avg[:,:,:40])
np.save('data/thingseeg2/test_thingseeg2_avg1_600ms.npy', test_thingseeg2_avg[:,:,:60])
np.save('data/thingseeg2/test_thingseeg2_avg1_800ms.npy', test_thingseeg2_avg[:,:,:80])