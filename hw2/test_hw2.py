from hw2 import * 
from scipy import ndimage 
import matplotlib.pyplot as plt 

image_str = "data/goldengate/goldengate-03.png"
#image_str = "data/checker.png"
image = ndimage.imread(image_str, flatten=True)

def test_interest(max_points=200, scale=1.0): 
	R = find_interest_points(image, max_points, scale)
	#plt.imshow(R, cmap="gray")
	#plt.show()
	return R

def test_feats(max_points=200, scale=1.0): #fails at 98 
	xs, ys, scores = find_interest_points(image, max_points, scale)
	feats = extract_features(image, xs, ys, scale)
	return feats 

def test_match_features(): 
	image_str0 = "data/goldengate/goldengate-01.png"
	image0 = ndimage.imread(image_str0, flatten=True)
	image_str1 = "data/goldengate/goldengate-02.png"
	image1 = ndimage.imread(image_str1, flatten=True)	
	xs0, ys0, scores0 = test_interest(image0)
	xs1, ys1, scores1 = test_interest(image1)
	feats0 = extract_features(image0, xs0, ys0)
	feats1 = extract_features(image1, xs1, ys1) 
	matches, scores = match_features(feats0, feats1, scores0, scores1)
	return matches, scores 


