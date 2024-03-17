import sys
import os
import numpy as np
import sklearn.linear_model as skl
import pickle
import argparse
from scipy.spatial.distance import correlation
import time
start = time.time()
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,5,7]

# train_path = 'data/things-eeg2_preproc/train_thingseeg2_avg.npy'
# train_path = 'data/things-eeg2_preproc/train_thingseeg2_avg_200ms.npy'
# train_path = 'data/things-eeg2_preproc/train_thingseeg2_avg_400ms.npy'
# train_path = 'data/things-eeg2_preproc/train_thingseeg2_avg_600ms.npy'
train_path = 'data/things-eeg2_preproc/train_thingseeg2_avg_800ms.npy'
# train_path = 'data/things-eeg2_preproc/train_thingseeg2_null.npy'
# train_path = 'data/thingseeg2/train_thingseeg2_avg1_200ms.npy'
# train_path = 'data/thingseeg2/train_thingseeg2_avg1_400ms.npy'
# train_path = 'data/thingseeg2/train_thingseeg2_avg1_600ms.npy'
# train_path = 'data/thingseeg2/train_thingseeg2_avg1_800ms.npy'
train_meg = np.load(train_path)
# train_meg = train_meg[:8000,:,:]
train_meg = train_meg.reshape(train_meg.shape[0], -1)
# test_path = 'data/things-eeg2_preproc/test_thingseeg2_avg.npy'
# test_path = 'data/things-eeg2_preproc/test_thingseeg2_avg_200ms.npy'
# test_path = 'data/things-eeg2_preproc/test_thingseeg2_avg_400ms.npy'
# test_path = 'data/things-eeg2_preproc/test_thingseeg2_avg_600ms.npy'
test_path = 'data/things-eeg2_preproc/test_thingseeg2_avg_800ms.npy'
# test_path = 'data/things-eeg2_preproc/test_thingseeg2_null.npy'
# test_path = 'data/thingseeg2/test_thingseeg2_avg1_200ms.npy'
# test_path = 'data/thingseeg2/test_thingseeg2_avg1_400ms.npy'
# test_path = 'data/thingseeg2/test_thingseeg2_avg1_600ms.npy'
# test_path = 'data/thingseeg2/test_thingseeg2_avg1_800ms.npy'
test_meg = np.load(test_path)
# test_meg = test_meg[:1000,:,:]
test_meg = test_meg.reshape(test_meg.shape[0], -1)
print(train_meg.shape, test_meg.shape)

## Preprocessing fMRI

train_fmri = train_meg
test_fmri = test_meg


norm_mean_train = np.mean(train_fmri, axis=0)
norm_scale_train = np.std(train_fmri, axis=0, ddof=1)
train_fmri = (train_fmri - norm_mean_train) / norm_scale_train
test_fmri = (test_fmri - norm_mean_train) / norm_scale_train

print(np.mean(train_fmri),np.std(train_fmri))
print(np.mean(test_fmri),np.std(test_fmri))

print(np.max(train_fmri),np.min(train_fmri))
print(np.max(test_fmri),np.min(test_fmri))

num_voxels, num_train, num_test = train_fmri.shape[1], len(train_fmri), len(test_fmri)


train_clip = np.load('cache/thingseeg2_preproc/extracted_embeddings/train_clipvision.npy', mmap_mode='r')
test_clip = np.load('cache/thingseeg2_preproc/extracted_embeddings/test_clipvision.npy', mmap_mode='r')
train_clip_flattened = train_clip.reshape(train_clip.shape[0], -1)
test_clip_flattened = test_clip.reshape(test_clip.shape[0], -1)
# train_clip = train_clip[:8000,:,:]
# test_clip = test_clip[:1000,:,:]

#train_clip = train_clip[:,1:,:]
# num_samples,num_embed,num_dim = train_clip.shape

print("Training Regression")
# reg_w = np.zeros((num_embed,num_dim,num_voxels)).astype(np.float32)
# reg_b = np.zeros((num_embed,num_dim)).astype(np.float32)
pred_clip_flattened = np.zeros_like(test_clip_flattened)
# for i in range(num_embed):


reg = skl.Ridge(alpha=60000, max_iter=50000, fit_intercept=True) # old alpha=60000, optimal alpha=300000
reg.fit(train_fmri, train_clip_flattened)
reg_w = reg.coef_
reg_b = reg.intercept_

pred_test_latent = reg.predict(test_fmri)
std_norm_test_latent = (pred_test_latent - np.mean(pred_test_latent,axis=0)) / np.std(pred_test_latent,axis=0)
pred_clip_flattened = std_norm_test_latent * np.std(train_clip_flattened,axis=0) + np.mean(train_clip_flattened,axis=0)
pred_clip = pred_clip_flattened.reshape(test_clip.shape)

for i in range(257):
    # Compute the Euclidean distances
    euclidean_distances = np.array([np.linalg.norm(u - v) for u, v in zip(pred_clip[:,i], test_clip[:,i])])
    correlation_distances = np.array([correlation(u, v) for u, v in zip(pred_clip[:,i], test_clip[:,i])])
    # Compute the average Euclidean distance
    average_euclidean_distance = euclidean_distances.mean()
    correlations = (1 - correlation_distances).mean()

    print(i, average_euclidean_distance, correlations)


# Compute the Euclidean distances
euclidean_distances = np.array([np.linalg.norm(u - v) for u, v in zip(pred_clip_flattened, test_clip_flattened)])
correlation_distances = np.array([correlation(u, v) for u, v in zip(pred_clip_flattened, test_clip_flattened)])
# Compute the average Euclidean distance
average_euclidean_distance = euclidean_distances.mean()
correlations = (1 - correlation_distances).mean()

print(reg.score(test_fmri,test_clip_flattened), average_euclidean_distance, correlations)
    

# np.save('data/predicted_features/subj{:02d}/nsd_clipvision_predtest_nsdgeneral.npy'.format(sub),pred_clip)
    # np.save('data/predicted_features/subj{:02d}/nsd_clipvision_predtest_nsdgeneral_assumehrf.npy'.format(sub),pred_clip)
# subject = 'BIGMEG1'
# save_dir = 'cache/predicted_embeddings/' + subject + '/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# np.save(save_dir + f'thingsmeg_regress_clipvision1b_sub-{subject}.npy', pred_clip)
    
# save_dir = 'cache/thingseeg2_preproc/predicted_embeddings/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# np.save(save_dir + 'thingseeg2_regress_clipvision.npy', pred_clip)
    
save_dir = 'cache/thingseeg2_preproc/predicted_embeddings/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# np.save(save_dir + 'thingseeg2_regress_clipvision_200ms_1regressor.npy', pred_clip)
# np.save(save_dir + 'thingseeg2_regress_clipvision_400ms_1regressor.npy', pred_clip)
# np.save(save_dir + 'thingseeg2_regress_clipvision_600ms_1regressor.npy', pred_clip)
np.save(save_dir + 'thingseeg2_regress_clipvision_800ms_1regressor.npy', pred_clip)

# save_dir = 'cache/thingseeg2_preproc/predicted_embeddings/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# np.save(save_dir + 'thingseeg2_regress_clipvision_null.npy', pred_clip)
    
# save_dir = 'cache/thingseeg2/predicted_embeddings/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# np.save(save_dir + 'thingseeg2_regress_clipvision_avg1_200ms.npy', pred_clip)
    
# save_dir = 'cache/thingseeg2/predicted_embeddings/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# np.save(save_dir + 'thingseeg2_regress_clipvision_avg1_400ms.npy', pred_clip)
    
# save_dir = 'cache/thingseeg2/predicted_embeddings/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# np.save(save_dir + 'thingseeg2_regress_clipvision_avg1_600ms.npy', pred_clip)
    
# save_dir = 'cache/thingseeg2/predicted_embeddings/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# np.save(save_dir + 'thingseeg2_regress_clipvision_avg1_800ms.npy', pred_clip)


datadict = {
    'weight' : reg_w,
    'bias' : reg_b,

}

# with open('data/regression_weights/subj{:02d}/clipvision_regression_weights.pkl'.format(sub),"wb") as f:
# with open('data/regression_weights/subj{:02d}/clipvision_regression_weights_assumehrf.pkl'.format(sub),"wb") as f:
#   pickle.dump(datadict,f)
# subject = 'BIGMEG1'
# save_dir = 'cache/regression_weights/' + subject + '/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# with open(save_dir + f'thingsmeg_regress_clipvision1b_weights_sub-{subject}.pkl', "wb") as f:
#     pickle.dump(datadict,f)

# save_dir = 'cache/thingseeg2_preproc/regression_weights/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# with open(save_dir + 'thingseeg2_regress_clipvision_weights.pkl', "wb") as f:
#     pickle.dump(datadict,f)

save_dir = 'cache/thingseeg2_preproc/regression_weights/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# with open(save_dir + 'thingseeg2_regress_clipvision_weights_200ms_1regressor.pkl', "wb") as f:
#     pickle.dump(datadict,f)
# with open(save_dir + 'thingseeg2_regress_clipvision_weights_400ms_1regressor.pkl', "wb") as f:
#     pickle.dump(datadict,f)
# with open(save_dir + 'thingseeg2_regress_clipvision_weights_600ms_1regressor.pkl', "wb") as f:
#     pickle.dump(datadict,f)
with open(save_dir + 'thingseeg2_regress_clipvision_weights_800ms_1regressor.pkl', "wb") as f:
    pickle.dump(datadict,f)

# save_dir = 'cache/thingseeg2_preproc/regression_weights/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# with open(save_dir + 'thingseeg2_regress_clipvision_weights_null.pkl', "wb") as f:
#     pickle.dump(datadict,f)

# save_dir = 'cache/thingseeg2/regression_weights/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# with open(save_dir + 'thingseeg2_regress_clipvision_weights_avg1_200ms.pkl', "wb") as f:
#     pickle.dump(datadict,f)

# save_dir = 'cache/thingseeg2/regression_weights/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# with open(save_dir + 'thingseeg2_regress_clipvision_weights_avg1_400ms.pkl', "wb") as f:
#     pickle.dump(datadict,f)

# save_dir = 'cache/thingseeg2/regression_weights/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# with open(save_dir + 'thingseeg2_regress_clipvision_weights_avg1_600ms.pkl', "wb") as f:
#     pickle.dump(datadict,f)

# save_dir = 'cache/thingseeg2/regression_weights/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# with open(save_dir + 'thingseeg2_regress_clipvision_weights_avg1_800ms.pkl', "wb") as f:
#     pickle.dump(datadict,f)
    
print("Time taken: ", time.time()-start)