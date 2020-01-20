import numpy as np 
from hw1 import conv_2d 
import scipy.signal

def test_conv2d(mode='mirror'): 
  if mode == 'mirror': 
  	bdry = 'symm'
  elif mode =='zero': 
  	bdry = 'fill'
  else: 
  	raise ValueError("mode must be either zero or mirror")

  image1 = np.array([[5, 10, 3, 2, 9], 
                    [6, 4, 2, 3, 8], 
                    [1, 4, 7, 1, 2 ],
                    [6, 5, 2, 3, 9],
                    [7, 3, 5, 1, 0], 
                    [4, 5, 3, 2, 2]])

  filt1 = np.array([[1, 0, 3], [0, 3, 1], [-1, 2, 0]])

  filt2 = np.array([[1, 0, 3]])

  big_image = np.array([[25, 100, 75, 49, 130], 
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

  mine = conv_2d(big_image, filt2, mode)
  actual = scipy.signal.convolve2d(big_image, filt2, mode='same', boundary=bdry, fillvalue=0)
  return np.array_equal(mine, actual)


