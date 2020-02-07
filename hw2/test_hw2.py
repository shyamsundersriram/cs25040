from hw2 import * 
from visualize import * 
from scipy import ndimage 
import matplotlib.pyplot as plt
import PIL, pickle, time, glob
from util import *
import random 


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
	xs0, ys0, scores0 = find_interest_points(img1, max_points, scale)
	xs1, ys1, scores1 = find_interest_points(img1, max_points, scale)
	#plot_interest_points(image, xs, ys, scores)
	#plt.show()
	return xs0, xs1, ys0, ys1, scores0, scores1 

def test_feats(): 
	N = 200
	xs0, ys0, scores0 = find_interest_points(img0, N, 1.0)
	xs1, ys1, scores1 = find_interest_points(img1, N, 1.0)
	feats0 = extract_features(img0, xs0, ys0, 1.0)
	feats1 = extract_features(img1, xs1, ys1, 1.0)
	return feats0, feats1, scores0, scores1

def test_matches(): 
	f0, f1, s0, s1 = test_feats() 
	m, s = match_features(f0, f1, s0, s1)
	return m, s

def test_kdtree(): 
	img = np.array([[0, 0, 1, 2, 3], 
				   [3, 3, 2, 6, 2], 
				   [3, 9, 5, 8, 0], 
				   [0, 1, 0, 0, 6], 
				   [2, 3, 7, 0, 0]]) 
	depth = 3
	split_indices = random.sample(range(0, 5), depth)
	feat_indices = [i for i in range(0, 5)]  
	t = build_kdtree(img, feat_indices, split_indices, depth)
	return t                       

def test_matches_kd(): 
	f0, f1, s0, s1 = test_feats() 
	m, s = match_features(f0, f1, s0, s1, 'kdtree')
	return m, s

def setup_hough_votes(): 

	xs0, xs1, ys0, ys1, scores0, scores1 = test_interest(200, 1.0)
	print('found interest points')
	m, s = test_matches() 
	print('found matches')
	return xs0, xs1, ys0, ys1, m, s
	#tx, ty, votes = hough_votes(xs0, xs1, ys0, ys1, m, s)
	#return tx, ty, votes


def test_hough_votes(xs0, xs1, ys0, ys1, m, s): 
	tx, ty, votes = hough_votes(xs0, xs1, ys0, ys1, m, s)
	show_overlay(img0, img1, tx, ty)
	plt.show()
##########




