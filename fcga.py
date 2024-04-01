""" ... """

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from scipy.stats import zscore


def procrustes(source, template):
    u, w, vt = np.linalg.svd(np.dot(source.T, template))
    transformation = np.dot(u, vt)
    source_aligned = np.dot(source, transformation)
    return source_aligned, transformation

def pairwise_correlation(A, B):
    am = A - np.mean(A, axis=0, keepdims=True)
    bm = B - np.mean(B, axis=0, keepdims=True)
    return am.T @ bm /  (np.sqrt(
        np.sum(am**2, axis=0,
               keepdims=True)).T * np.sqrt(
        np.sum(bm**2, axis=0, keepdims=True)))


def get_fc_mat(bold, landmarks=""):
    if len(landmarks)>0:
        fc_mat = pairwise_correlation(bold.T, landmarks.T)
    else:
        fc_mat = pairwise_correlation(bold.T, bold.T)

    return fc_mat

def get_thrsh_mat(fc_mat, thrsh=90):
    n,_ = fc_mat.shape
    perc_thresh = np.percentile(fc_mat,thrsh,axis=1)
    for ii in range(n):
        fc_mat[ii, fc_mat[ii, :] <  perc_thresh[ii] ] = 0
    fc_mat[ fc_mat < 0 ] = 0

    return fc_mat

def get_aff_mat(fc_mat, landmark_fc=""):
    if len(landmark_fc)>0:        
        fc_mat = cosine_similarity(fc_mat, landmark_fc) 
    else:
        fc_mat = cosine_similarity(fc_mat) 
    
    return fc_mat

def do_embedding(fc_mat, n_gradients=25):
    pca = PCA(n_components=n_gradients)
    embedding = pca.fit_transform(fc_mat) 

    return embedding, pca.explained_variance_


def get_regionbased_targets(bold_ts, parcellations, landmarks="", vidx_lh=None, vidx_rh=None):

    nidx_lh = np.sum(vidx_lh)
    nidx_rh = np.sum(vidx_rh)

    landmarks_ts = np.array([])
    hemis = ['lh', 'rh']
    for hemi in hemis:
        if (hemi == 'lh'):
            hemi_bold = bold_ts[:nidx_lh,:]
            vidx = vidx_lh
        elif (hemi == 'rh'):
            hemi_bold = bold_ts[nidx_lh:,:]
            vidx = vidx_rh

        #print(hemi_bold.shape)
        parc = parcellations[(landmarks, hemi)][vidx] 
        #print(parc.shape)
        n_rois = np.unique(parc)
        for roi in n_rois:
            if (roi > 0):
                roix_idx = (parc==roi)
                if (np.sum(roix_idx) > 0):
                    roi_bold = hemi_bold[roix_idx, :]
                    roi_bold_mean = zscore(np.nanmean(roi_bold, axis=0))
                    landmarks_ts = np.vstack((landmarks_ts, roi_bold_mean)) if landmarks_ts.size else roi_bold_mean
                    
    print(landmarks_ts.shape)
    return landmarks_ts

# [252,492,642,1002,1442,1962,2562]
def get_uniform_targets(bold_ts, uniform_vertices, n_landmarks = 252, vidx_lh=None, vidx_rh=None):
        
    nidx_lh = np.sum(vidx_lh)
    nidx_rh = np.sum(vidx_rh)

    ulm_lh = np.zeros(vidx_lh.shape)
    ulm_lh[uniform_vertices[(n_landmarks,'lh')]] = 1
    ulm_rh = np.zeros(vidx_rh.shape)
    ulm_rh[uniform_vertices[(n_landmarks,'rh')]] = 1

    ulm_lh = ulm_lh[vidx_lh]
    ulm_rh = ulm_rh[vidx_rh]

    uniform_indices = np.hstack((np.where(ulm_lh==1)[0], np.where(ulm_rh==1)[0]+nidx_lh))
    #print(uniform_indices.shape)
    
    landmarks_ts = bold_ts[uniform_indices,:]
    landmarks_ts.shape

    return landmarks_ts

    
def get_random_targets(bold_ts, n_landmarks = 1000, vidx_lh=None, vidx_rh=None):
    
    nidx_lh = np.sum(vidx_lh)
    nidx_rh = np.sum(vidx_rh)

    if (nidx_lh==None):
        n_bold,d = bold_ts.shape 
        random_indices = np.random.permutation(n_bold)
        random_indices = random_indices[:n_landmarks]
        #print(random_indices.shape)
    else:
        rand_idx_lh = np.random.permutation(nidx_lh)
        rand_idx_rh = np.random.permutation(nidx_rh)
        random_indices = np.hstack((rand_idx_lh[:n_landmarks], rand_idx_rh[:n_landmarks]+nidx_lh))
        #print(random_indices.shape)

    landmarks_ts = bold_ts[random_indices,:]
    landmarks_ts.shape

    return landmarks_ts

def do_fastgrads(bold_ts, landmarks_ts="", n_gradients=25):
    
    if len(landmarks_ts)==0:
        print(f"#landmarks: {bold_ts.shape[0]}")
    else:
        print(f"#landmarks: {len(landmarks_ts)}")
    print(f"fmri data shape: {bold_ts.shape}")  
    n_bold,d = bold_ts.shape 
    # full gradients
    if len(landmarks_ts)==0:
        # 1) get connectivity matrix
        fc_mat = get_fc_mat(bold_ts)
        
        # 2) threshold and kernel
        fc_mat = get_thrsh_mat(fc_mat)
        
        # affinity matrix
        fc_mat =  get_aff_mat(fc_mat)

    else:
        # 1) get connectivity matrix
        fc_mat = get_fc_mat(bold_ts, landmarks=landmarks_ts)
        lm_mat = get_fc_mat(landmarks_ts)
        
        # 2) threshold and kernel
        fc_mat = get_thrsh_mat(fc_mat)
        lm_mat = get_thrsh_mat(lm_mat)

        # affinity matrix
        fc_mat =  get_aff_mat(fc_mat, lm_mat)
        
    
    print(f"affinity matrix shape: {fc_mat.shape}") 
    fc_mat = np.nan_to_num(fc_mat)
    gradients, lambdas = do_embedding(fc_mat, n_gradients)

    return gradients, lambdas
