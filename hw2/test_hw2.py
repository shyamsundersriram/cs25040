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
	works= True 
	#for i in range(max_points): 
	#	if len(feats[i]) == 72:
	#		print(feats[i]) 
	#		print('this is i')
	#		print(i)
	print(np.shape(image))
	return feats 

