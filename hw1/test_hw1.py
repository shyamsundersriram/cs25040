import numpy as np 
from hw1 import conv_2d, denoise_gaussian, sobel_gradients, nonmax_suppress
import scipy.signal
import math 
from scipy import ndimage

image1 = np.array([[5, 10, 3, 2, 9], 
                  [6, 4, 2, 3, 8], 
                  [1, 4, 7, 1, 2 ],
                  [6, 5, 2, 3, 9],
                  [7, 3, 5, 1, 0], 
                  [4, 5, 3, 2, 2]])
filt1 = np.array([[1, 0, 3], [0, 3, 1], [-1, 2, 0]])
filt2 = np.array([[1, 0, 3]])
image2 = np.array([[25, 100, 75, 49, 130], 
                  [50, 80, 0, 70, 100], 
                  [5, 10, 10, 30, 0 ],
                  [60, 50, 12, 29, 32],
                  [37, 33, 55, 21, 90], 
                  [140, 17, 0, 23, 222]])
filt3 = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]])
big_filt = np.array([[1, 0, 1, 1, 1], 
                  [0, 1, 0, 1, 1],
                  [0, 0, 1, 1, 1], 
                  [1, 1, 1, 1, 1], 
                  [1, 1, 1, 1, 1]])
image3 = np.array([[16. ,  8. ,  0.5],
                    [ 8.5,  0.5,  0. ],
                    [ 1. ,  1. ,  0.5]])
image4 = np.array([[2 ,  0 ,  0],
                    [ 0,  3,  0 ],
                    [ 0 ,  0 ,  1]])

def test_conv2d(mode='mirror'): 
  if mode == 'mirror': 
  	bdry = 'symm'
  elif mode =='zero': 
  	bdry = 'fill'
  else: 
  	raise ValueError("mode must be either zero or mirror")

  mine = conv_2d(big_image, filt2, mode)
  actual = scipy.signal.convolve2d(big_image, filt2, mode='same', boundary=bdry, fillvalue=0)
  return np.array_equal(mine, actual)

def test_denoise_gaussian():
	sigma1 = 1.0 
	result = denoise_gaussian(image1, sigma1)
	actual = ndimage.gaussian_filter(image1, sigma1)
	return result, actual

def test_denoise_median():
	width1 = 1.0 
	result = denoise_gaussian(image1, width1)
	actual = ndimage.gaussian_filter(image1, sigma1)
	return result, actual

def test_nonmax_suppress(image): 
	df_x, df_y = sobel_gradients(image)
	mag = np.sqrt((df_x ** 2 + df_y ** 2))
	theta = np.arctan2(df_x, df_y) 
	theta = np.where(theta < 0, theta + 2 * math.pi, theta)
	result = nonmax_suppress(mag, theta)
	print(mag)
	print(theta / math.pi * 8)
	print(result) 



if __name__ == "__main__": 
	test_nonmax_suppress(image4)


    
