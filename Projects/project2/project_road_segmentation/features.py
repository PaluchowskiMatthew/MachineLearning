import numpy as np
def extract_features_edge(img):
    from skimage import feature, color
    feat_m = np.mean(img, axis=(0,1))
    feat_v = np.var(img, axis=(0,1))
    feat_e = np.asarray(feature.canny(color.rgb2gray(img), sigma=5).sum()).reshape(1,)
    feat = np.append(feat_m, feat_v)
    feat = np.hstack([feat, feat_e])
    return feat
