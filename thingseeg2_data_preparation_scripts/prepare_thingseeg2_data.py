import numpy as np

for sub in range(1, 11):
    data_dir = f'data/thingseeg2_preproc/sub-{sub:02d}/'
    eeg_data_train = np.load(data_dir + 'preprocessed_eeg_training.npy', allow_pickle=True).item()
    eeg_data_test = np.load(data_dir + 'preprocessed_eeg_test.npy', allow_pickle=True).item()
    print(f'\nTraining EEG data shape for sub-{sub:02d}:')
    print(eeg_data_train['preprocessed_eeg_data'].shape)
    print('(Training image conditions × Training EEG repetitions × EEG channels × '
        'EEG time points)')

    print(f'\nTest EEG data shape for sub-{sub:02d}:')
    print(eeg_data_test['preprocessed_eeg_data'].shape)
    print('(Test image conditions × Test EEG repetitions × EEG channels × '
        'EEG time points)')

    train_thingseeg2 = eeg_data_train['preprocessed_eeg_data'][:,:,:,20:]
    train_thingseeg2_avg = eeg_data_train['preprocessed_eeg_data'].mean(1)[:,:,20:]
    np.save(data_dir + 'train_thingseeg2.npy', train_thingseeg2)
    np.save(data_dir + 'train_thingseeg2_avg.npy', train_thingseeg2_avg)
    test_thingseeg2 = eeg_data_test['preprocessed_eeg_data'][:,:,:,20:]
    test_thingseeg2_avg = eeg_data_test['preprocessed_eeg_data'].mean(1)[:,:,20:]
    np.save(data_dir + 'test_thingseeg2.npy', test_thingseeg2)
    np.save(data_dir + 'test_thingseeg2_avg.npy', test_thingseeg2_avg)
