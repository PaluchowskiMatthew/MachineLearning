import numpy as np
# Extract 6-dimensional features consisting of average RGB color as well as variance
def extract_features(img):
    feat_m = np.mean(img, axis=(0,1))
    feat_v = np.var(img, axis=(0,1))
    feat = np.append(feat_m, feat_v)
    return feat

# Extract 2-dimensional features consisting of average gray color as well as variance
def extract_features_2d(img):
    feat_m = np.mean(img)
    feat_v = np.var(img)
    feat = np.append(feat_m, feat_v)
    return feat

# Extract 6-dimensional features as above and add canny edge detector
def extract_features_edge(img):
    from skimage import feature, color
    feat_m = np.mean(img, axis=(0,1))
    feat_v = np.var(img, axis=(0,1))
    feat_e = np.asarray(feature.canny(color.rgb2gray(img), sigma=5, low_threshold=0, high_threshold=0.15).sum()).reshape(1,)
    feat = np.append(feat_m, feat_v)
    feat = np.hstack([feat, feat_e])
    return feat
