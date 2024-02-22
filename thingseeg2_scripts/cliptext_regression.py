import sys
import os
import numpy as np
import sklearn.linear_model as skl
import pickle
import argparse
from scipy.spatial.distance import correlation
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,5,7]

train_path = 'data/things-eeg2_preproc/train_thingseeg2_avg.npy'
train_meg = np.load(train_path, mmap_mode='r')
# train_meg = train_meg[:8000,:,:]
train_meg = train_meg.reshape(train_meg.shape[0], -1)
test_path = 'data/things-eeg2_preproc/test_thingseeg2_avg.npy'
test_meg = np.load(test_path, mmap_mode='r')
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


train_clip = np.load('cache/thingseeg2_preproc/extracted_embeddings/train_cliptext.npy', mmap_mode='r')
test_clip = np.load('cache/thingseeg2_preproc/extracted_embeddings/test_cliptext.npy', mmap_mode='r')
# train_clip = train_clip[:8000,:,:]
# test_clip = test_clip[:1000,:,:]

## Regression
num_samples,num_embed,num_dim = train_clip.shape

print("Training Regression")
reg_w = np.zeros((num_embed,num_dim,num_voxels)).astype(np.float32)
reg_b = np.zeros((num_embed,num_dim)).astype(np.float32)
pred_clip = np.zeros_like(test_clip)
for i in range(num_embed):
    reg = skl.Ridge(alpha=100000, max_iter=50000, fit_intercept=True) # old alpha=100000, optimal alpha=1200000
    reg.fit(train_fmri, train_clip[:,i])

    reg_w[i] = reg.coef_
    reg_b[i] = reg.intercept_
    
    pred_test_latent = reg.predict(test_fmri)
    std_norm_test_latent = (pred_test_latent - np.mean(pred_test_latent,axis=0)) / np.std(pred_test_latent,axis=0)
    pred_clip[:,i] = std_norm_test_latent * np.std(train_clip[:,i],axis=0) + np.mean(train_clip[:,i],axis=0)

    # Compute the Euclidean distances
    euclidean_distances = np.array([np.linalg.norm(u - v) for u, v in zip(pred_clip[:,i], test_clip[:,i])])
    correlation_distances = np.array([correlation(u, v) for u, v in zip(pred_clip[:,i], test_clip[:,i])])
    # Compute the average Euclidean distance
    average_euclidean_distance = euclidean_distances.mean()
    correlations = (1 - correlation_distances).mean()

    print(i,reg.score(test_fmri,test_clip[:,i]), average_euclidean_distance, correlations)



# np.save('data/predicted_features/subj{:02d}/nsd_cliptext_predtest_nsdgeneral.npy'.format(sub),pred_clip)
# np.save('data/predicted_features/subj{:02d}/nsd_cliptext_predtest_nsdgeneral_assumehrf.npy'.format(sub),pred_clip)
# subject = 'BIGMEG1'
save_dir = 'cache/thingseeg2_preproc/predicted_embeddings/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
np.save(save_dir + 'thingseeg2_regress_cliptext.npy', pred_clip)

datadict = {
    'weight' : reg_w,
    'bias' : reg_b,

}

# with open('data/regression_weights/subj{:02d}/cliptext_regression_weights.pkl'.format(sub),"wb") as f:
#   pickle.dump(datadict,f)
# with open('data/regression_weights/subj{:02d}/cliptext_regression_weights_assumehrf.pkl'.format(sub),"wb") as f:
#   pickle.dump(datadict,f)
# subject = 'BIGMEG1'
save_dir = 'cache/thingseeg2_preproc/regression_weights/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
with open(save_dir + 'thingseeg2_regress_cliptext_weights.pkl', "wb") as f:
    pickle.dump(datadict,f)

# 0 0.0010279285044471904 0.07064513842718463 0.9999999999999997                                                                                                                                                     
# 1 -0.0034537804712458813 0.809644754288077 0.5463085379711279                                                                                                                                                      
# 2 -0.0007449790232825906 0.8398043402157598 0.5629047168635853                                                                                                                                                     
# 3 0.0013552855584596495 0.7815252597332585 0.6822932067150512                                                                                                                                                      
# 4 0.0012589360440632856 0.7572834659170797 0.7124342867568092                                                                                                                                                      
# 5 0.0009476298867733534 0.7518503787559134 0.7149659479186584                                                                                                                                                      
# 6 0.00130062299693553 0.7468960505988419 0.7159786246917852                                                                                                                                                        
# 7 0.0013604163640780133 0.7420138768969836 0.7165186616578134                                                                                                                                                      
# 8 0.001127216468269187 0.7381280905816804 0.7162693517067424                                                                                                                                                       
# 9 0.0010998368179121457 0.7337060496868247 0.7163486027075535                                                                                                                                                      
# 10 0.001246295847492224 0.7299076592836218 0.7168399275231582                                                                                                                                                      
# 11 0.0014074211383366597 0.7256592888532686 0.7173516423687399                                                                                                                                                     
# 12 0.0015532622611352396 0.7209814134155949 0.718754493053012                                                                                                                                                      
# 13 0.0016504287335936962 0.717435090604606 0.7196474406185414                                                                                                                                                      
# 14 0.0017724965782467354 0.7131167909455908 0.7210020801698763                                                                                                                                                     
# 15 0.0018872745831092228 0.7097978047545186 0.7223215923647885                                                                                                                                                     
# 16 0.002016718336099274 0.7064736096519728 0.7231886096242665                                                                                                                                                      
# 17 0.0021266204042647533 0.703282135740187 0.7241111070773281                                                                                                                                                      
# 18 0.00220525881769309 0.7012296338307508 0.7245493186728373                                                                                                                                                       
# 19 0.0022468301974543797 0.6990125214695404 0.7250701024474374                                                                                                                                                     
# 20 0.002284613407632745 0.69741479278194 0.7260045496360403                                                                                                                                                        
# 21 0.0023078232874864575 0.6959232284062057 0.726663321622315                                                                                                                                                      
# 22 0.002319857882647124 0.694009375394287 0.7277049446465811                                                                                                                                                       
# 23 0.002356305899877365 0.6923464745516478 0.7285323110024392                                                                                                                                                      
# 24 0.002362255730070762 0.6905001154530571 0.7289005833013821                                                                                                                                                      
# 25 0.0023878733097408415 0.6890727775127888 0.7291606386368775
# 26 0.0024022343839055073 0.6882172489622215 0.7286246670277297
# 27 0.002392819347077576 0.6871686004845166 0.7282388241016549
# 28 0.0024137282447649045 0.6865245134825604 0.727847749081136
# 29 0.0024028639042511066 0.6856157073156749 0.7273026168290493
# 30 0.0024202852500410255 0.6848247362440092 0.727254137512841
# 31 0.0024381333408716255 0.6844972364298318 0.7268041036992199
# 32 0.0024374238608816418 0.6838780853399231 0.7267752214384098
# 33 0.0024677208702695895 0.6832709122600096 0.7269230166960586
# 34 0.0024653744781806346 0.6821694158689741 0.7268695928558305
# 35 0.0024848079346609015 0.6809922252338151 0.7270812566829732
# 36 0.00250711208029605 0.6805104143747017 0.7263460803001598
# 37 0.0024900841685006705 0.6794308839227255 0.7258112032005863
# 38 0.002509949034350292 0.678798416426991 0.7256408963186467
# 39 0.002506907897641613 0.6781666470485038 0.7251496659746033
# 40 0.002508447037643656 0.6776459331580057 0.7250720197557342
# 41 0.0025240985270252292 0.6771372722621481 0.7254353760234531
# 42 0.0025136208666624964 0.6763180321625595 0.7255105934119976
# 43 0.0025299207912625314 0.6751217838454775 0.7262559643160392
# 44 0.00252204882221992 0.673873002882646 0.7270354478690316
# 45 0.0025385401840471735 0.6728594634018995 0.7277925926309069
# 46 0.0025212876001797987 0.671589013956465 0.7283021503142542
# 47 0.0025293044912613172 0.6705062674337355 0.728462618366105
# 48 0.0025218740700154816 0.6694843282698725 0.7280985627820769
# 49 0.002532975599922826 0.6685871941722471 0.7283032010588372
# 50 0.0025511362035346955 0.6680020730119187 0.7286464648324312
# 51 0.002539335441840393 0.6671782383829558 0.7290683611241124
# 52 0.0025543619086896958 0.6666899416378931 0.7294407323541787
# 53 0.002532853471386248 0.6654534661370289 0.7299651407708819
# 54 0.0025417570713317426 0.6643156298066917 0.730316540560224
# 55 0.002529368901764454 0.662677628483408 0.7313011341570737
# 56 0.0025158876602854987 0.6612217243635742 0.7321488481335845
# 57 0.0025194831221703323 0.6601767333298514 0.732544371546255
# 58 0.002486303353636213 0.6587585017256958 0.732919898995439
# 59 0.0024999805853583063 0.657793263417145 0.7337440474655063
# 60 0.002503201600138206 0.6563900308880877 0.7350485955048306
# 61 0.0024896129745843043 0.6549563931518873 0.7358987805845554
# 62 0.002527229925844392 0.6537878506746264 0.7369988354380496
# 63 0.0025239690365591537 0.6520725744521302 0.7381739152977596
# 64 0.0025280542720397237 0.6509729444492662 0.738835938855449
# 65 0.002525654948428055 0.6494003143761704 0.7398486829003997
# 66 0.002506654168000984 0.6481293599770106 0.7400162402948469
# 67 0.0025203111660616027 0.6467232493167581 0.7403884707999101
# 68 0.002505106868053142 0.645412087407378 0.7405349026367936
# 69 0.0025310885786671827 0.6450861032387545 0.7406070841773519
# 70 0.002541652930827263 0.6445825003253575 0.7407051303705464
# 71 0.002521554372727931 0.6437875172721036 0.7412138071144733
# 72 0.002538305872384881 0.6426502257884225 0.7419112900537015
# 73 0.0025176579993909237 0.6414408929538955 0.7426627811995904
# 74 0.0025468430307936476 0.6404263594695788 0.7436375220016734
# 75 0.00253012507137682 0.6392004770842994 0.7428323313001389
# 76 0.00269384343322149 0.6374430423532321 0.7492651222026033