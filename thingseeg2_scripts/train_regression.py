import numpy as np
import scipy
from scipy.spatial.distance import correlation
import random
import sklearn.linear_model as skl
import os
import pickle

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub=int(args.sub)

# Load EEG data
eeg_train = np.load(f'data/thingseeg2_preproc/sub-{sub:02d}/train_thingseeg2_avg.npy')
eeg_train = eeg_train.reshape(eeg_train.shape[0],-1)
eeg_test = np.load(f'data/thingseeg2_preproc/sub-{sub:02d}/test_thingseeg2_avg.npy')
eeg_test = eeg_test.reshape(eeg_test.shape[0],-1)
print(eeg_train.shape, eeg_test.shape)
norm_mean_train = np.mean(eeg_train, axis=0)
norm_scale_train = np.std(eeg_train, axis=0, ddof=1)
eeg_train = (eeg_train - norm_mean_train) / norm_scale_train
eeg_test = (eeg_test - norm_mean_train) / norm_scale_train

# Save Directory
weights_save_dir = f'cache/thingseeg2_preproc/regression_weights/sub-{sub:02d}/'
if not os.path.exists(weights_save_dir):
    os.makedirs(weights_save_dir)
vdvae_weights_filename = 'regress_vdvae_weights.pkl'
clipvision_weights_filename = 'regress_clipvision_weights.pkl'
cliptext_weights_filename = 'regress_cliptext_weights.pkl'
save_dir = f'cache/thingseeg2_preproc/predicted_embeddings/sub-{sub:02d}/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
vdvae_filename = 'regress_vdvae.npy'
clipvision_filename = 'regress_clipvision.npy'
cliptext_filename = 'regress_cliptext.npy'

ids = list(range(len(eeg_train)))
# VDVAE Regression
train_latents= np.load('cache/thingseeg2_extracted_embeddings/train_autokl.npy', mmap_mode='r')[ids]
test_latents = np.load('cache/thingseeg2_extracted_embeddings/test_autokl.npy', mmap_mode='r')
print(train_latents.shape, test_latents.shape)

print("Training VDVAE Regression")
reg = skl.Ridge(alpha=1000, max_iter=50000, fit_intercept=True)
reg.fit(eeg_train, train_latents)
print('VDVAE training complete')

datadict = {
    'weight' : reg.coef_,
    'bias' : reg.intercept_,
}

with open(weights_save_dir + vdvae_weights_filename, "wb") as f:
    pickle.dump(datadict,f)

pred_latent = reg.predict(eeg_test)
pred_latent_mean = np.mean(pred_latent,axis=0)
pred_latent_std = np.std(pred_latent,axis=0)
std_norm_pred_latent = (pred_latent - pred_latent_mean) / pred_latent_std
train_latents_mean = np.mean(train_latents,axis=0)
train_latents_std = np.std(train_latents,axis=0)
pred_latents = std_norm_pred_latent * train_latents_std + train_latents_mean

np.save(save_dir + vdvae_filename, pred_latents)

# Compute the Euclidean distances
euclidean_distances = np.array([np.linalg.norm(u - v) for u, v in zip(pred_latents, test_latents)])
correlation_distances = np.array([correlation(u, v) for u, v in zip(pred_latents, test_latents)])
# Compute the average Euclidean distance
average_euclidean_distance = euclidean_distances.mean()
correlations = (1 - correlation_distances).mean()
print(reg.score(eeg_test,test_latents), average_euclidean_distance, correlations)

# CLIP-Vision Regression
train_clip_all = np.load('cache/thingseeg2_extracted_embeddings/train_clipvision.npy', mmap_mode='r')
test_clip = np.load('cache/thingseeg2_extracted_embeddings/test_clipvision.npy', mmap_mode='r')
train_clip = train_clip_all[ids]
print(train_clip.shape, test_clip.shape)

num_features = eeg_train.shape[1]
num_samples,num_token,num_dim = train_clip.shape
print("Training Regression")
reg_w = np.zeros((num_token,num_dim,num_features)).astype(np.float32)
reg_b = np.zeros((num_token,num_dim)).astype(np.float32)
pred_clip = np.zeros_like(test_clip)
for i in range(num_token):

    reg = skl.Ridge(alpha=1000, max_iter=50000, fit_intercept=True)
    reg.fit(eeg_train, train_clip[:,i])
    reg_w[i] = reg.coef_
    reg_b[i] = reg.intercept_
    
    pred_test_latent = reg.predict(eeg_test)
    std_norm_test_latent = (pred_test_latent - np.mean(pred_test_latent,axis=0)) / np.std(pred_test_latent,axis=0)
    pred_clip[:,i] = std_norm_test_latent * np.std(train_clip_all[:,i],axis=0) + np.mean(train_clip_all[:,i],axis=0)

    # Compute the Euclidean distances
    euclidean_distances = np.array([np.linalg.norm(u - v) for u, v in zip(pred_clip[:,i], test_clip[:,i])])
    correlation_distances = np.array([correlation(u, v) for u, v in zip(pred_clip[:,i], test_clip[:,i])])
    # Compute the average Euclidean distance
    average_euclidean_distance = euclidean_distances.mean()
    correlations = (1 - correlation_distances).mean()
    
    print(i,reg.score(eeg_test,test_clip[:,i]), average_euclidean_distance, correlations)

datadict = {
    'weight' : reg_w,
    'bias' : reg_b,
}
with open(weights_save_dir + clipvision_weights_filename, "wb") as f:
    pickle.dump(datadict,f)
np.save(save_dir + clipvision_filename, pred_clip)

# CLIP-Text Regression
train_clip_all = np.load('cache/thingseeg2_extracted_embeddings/train_cliptext.npy', mmap_mode='r')
test_clip = np.load('cache/thingseeg2_extracted_embeddings/test_cliptext.npy', mmap_mode='r')
train_clip = train_clip_all[ids]

num_features = eeg_train.shape[1]
num_samples,num_token,num_dim = train_clip.shape

print("Training Regression")
reg_w = np.zeros((num_token,num_dim,num_features)).astype(np.float32)
reg_b = np.zeros((num_token,num_dim)).astype(np.float32)
pred_clip = np.zeros_like(test_clip)
for i in range(num_token):
    reg = skl.Ridge(alpha=10000, max_iter=50000, fit_intercept=True)
    reg.fit(eeg_train, train_clip[:,i])

    reg_w[i] = reg.coef_
    reg_b[i] = reg.intercept_
    
    pred_test_latent = reg.predict(eeg_test)
    std_norm_test_latent = (pred_test_latent - np.mean(pred_test_latent,axis=0)) / np.std(pred_test_latent,axis=0)
    pred_clip[:,i] = std_norm_test_latent * np.std(train_clip_all[:,i],axis=0) + np.mean(train_clip_all[:,i],axis=0)

    # Compute the Euclidean distances
    euclidean_distances = np.array([np.linalg.norm(u - v) for u, v in zip(pred_clip[:,i], test_clip[:,i])])
    correlation_distances = np.array([correlation(u, v) for u, v in zip(pred_clip[:,i], test_clip[:,i])])
    # Compute the average Euclidean distance
    average_euclidean_distance = euclidean_distances.mean()
    correlations = (1 - correlation_distances).mean()

    print(i,reg.score(eeg_test,test_clip[:,i]), average_euclidean_distance, correlations)

datadict = {
    'weight' : reg_w,
    'bias' : reg_b,
}
with open(weights_save_dir + cliptext_weights_filename, "wb") as f:
    pickle.dump(datadict,f)
np.save(save_dir + cliptext_filename, pred_clip)