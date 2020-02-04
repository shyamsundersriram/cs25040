from hw2 import * 
from visualize import * 
from scipy import ndimage 
import matplotlib.pyplot as plt
import PIL, pickle, time, glob
from util import *


#image_str = "data/checker.png"
image_str = 'data/shanghai/shanghai-24.png'
image = ndimage.imread(image_str, flatten=True)

image_str0 = "data/goldengate/goldengate-01.png"
image0 = ndimage.imread(image_str0, flatten=True)
image_str1 = "data/goldengate/goldengate-02.png"
image1 = ndimage.imread(image_str1, flatten=True)	


img0 = load_image('data/shanghai/shanghai-23.png')
img1 = load_image('data/shanghai/shanghai-24.png')

def test_interest(max_points=200, scale=1.0): 
	#R = find_interest_points(image, max_points, scale)
	#plt.imshow(R, cmap="gray")
	#plt.show()
	#return R 
	xs, ys, scores = find_interest_points(image, max_points, scale)
	plot_interest_points(image, xs, ys, scores)
	plt.show()

def test_feats(): 
	N = 200
	xs0, ys0, scores0 = find_interest_points(img0, N, 1.0)
	xs1, ys1, scores1 = find_interest_points(img1, N, 1.0)
	feats0 = extract_features(img0, xs0, ys0, 1.0)
	feats1 = extract_features(img1, xs1, ys1, 1.0)
	return feats0, feats1


##########




