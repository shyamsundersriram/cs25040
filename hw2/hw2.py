import numpy as np
from canny import *
import operator
import math 
import random 

"""
   INTEREST POINT OPERATOR (12 Points Implementation + 3 Points Write-up)

   Implement an interest point operator of your choice.

   Your operator could be:

   (A) The Harris corner detector (Szeliski 4.1.1)

               OR

   (B) The Difference-of-Gaussians (DoG) operator defined in:
       Lowe, "Distinctive Image Features from Scale-Invariant Keypoints", 2004.
       https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

               OR

   (C) Any of the alternative interest point operators appearing in
       publications referenced in Szeliski or in lecture

              OR

   (D) A custom operator of your own design

   You implementation should return locations of the interest points in the
   form of (x,y) pixel coordinates, as well as a real-valued score for each
   interest point.  Greater scores indicate a stronger detector response.

   In addition, be sure to apply some form of spatial non-maximum suppression
   prior to returning interest points.

   Whichever of these options you choose, there is flexibility in the exact
   implementation, notably in regard to:

   (1) Scale

       At what scale (e.g. over what size of local patch) do you operate?

       You may optionally vary this according to an input scale argument.

       We will test your implementation at the default scale = 1.0, so you
       should make a reasonable choice for how to translate scale value 1.0
       into a size measured in pixels.

   (2) Nonmaximum suppression

       What strategy do you use for nonmaximum suppression?

       A simple (and sufficient) choice is to apply nonmaximum suppression
       over a local region.  In this case, over how large of a local region do
       you suppress?  How does that tie into the scale of your operator?

   For making these, and any other design choices, keep in mind a target of
   obtaining a few hundred interest points on the examples included with
   this assignment, with enough repeatability to have a large number of
   reliable matches between different views.

   If you detect more interest points than the requested maximum (given by
   the max_points argument), return only the max_points highest scoring ones.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

   Arguments:
      image       - a grayscale image in the form of a 2D numpy array
      max_points  - maximum number of interest points to return
      scale       - (optional, for your use only) scale factor at which to
                    detect interest points

   Returns:
      xs          - numpy array of shape (N,) containing x-coordinates of the
                    N detected interest points (N <= max_points)
      ys          - numpy array of shape (N,) containing y-coordinates
      scores      - numpy array of shape (N,) containing a real-valued
                    measurement of the relative strength of each interest point
                    (e.g. corner detector criterion OR DoG operator magnitude)
"""
def find_interest_points(image, max_points =100, scale = 1.0):
   # check that image is grayscale
   assert image.ndim == 2, 'image should be grayscale'
   ##########################################################################
   # calculating Rs 
   Ix, Iy = sobel_gradients(image)
   Sx2 = conv_2d_gaussian(np.multiply(Ix, Ix), 2 * scale)
   Sy2 = conv_2d_gaussian(np.multiply(Iy, Iy), 2 * scale)
   Sxy = conv_2d_gaussian(np.multiply(Ix, Iy), 2 * scale) 
   det = np.multiply(Sx2, Sy2) - np.multiply(Sxy, Sxy)
   alpha = 0.06
   trace = Sx2 + Sy2 
   alpha_trace = alpha * np.multiply(trace, trace)
   R = det - alpha_trace 
   theta = np.arctan2(Iy, Ix)
   R = nonmax_suppress(R, theta) # same dimensions as image

   # finding max Rs 
   X, Y = np.shape(R)   
   sortedR = []
   for i in range(X):
    for j in range(Y):
      sortedR.append((R[i, j], i, j))
   sortedR.sort(key = operator.itemgetter(0), reverse = True)
   sortedR = sortedR[:max_points]
   scores = np.array([tup[0] for tup in sortedR])
   xs = np.array([tup[1] for tup in sortedR])
   ys = np.array([tup[2] for tup in sortedR])
   ##########################################################################
   return xs, ys, scores

"""
   FEATURE DESCRIPTOR (12 Points Implementation + 3 Points Write-up)

   Implement a SIFT-like feature descriptor by binning orientation energy
   in spatial cells surrounding an interest point.

   Unlike SIFT, you do not need to build-in rotation or scale invariance.

   A reasonable default design is to consider a 3 x 3 spatial grid consisting
   of cell of a set width (see below) surrounding an interest point, marked
   by () in the diagram below.  Using 8 orientation bins, spaced evenly in
   [-pi,pi), yields a feature vector with 3 * 3 * 8 = 72 dimensions.

             ____ ____ ____
            |    |    |    |
            |    |    |    |
            |____|____|____|
            |    |    |    |
            |    | () |    |
            |____|____|____|
            |    |    |    |
            |    |    |    |
            |____|____|____|

                 |----|
                  width

   You will need to decide on a default spatial width.  Optionally, this can
   be a multiple of a scale factor, passed as an argument.  We will only test
   your code by calling it with scale = 1.0.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

  Arguments:
      image    - a grayscale image in the form of a 2D numpy
      xs       - numpy array of shape (N,) containing x-coordinates
      ys       - numpy array of shape (N,) containing y-coordinates
      scale    - scale factor

   Returns:
      feats    - a numpy array of shape (N,K), containing K-dimensional
                 feature descriptors at each of the N input locations
                 (using the default scheme suggested above, K = 72)
"""
def extract_features(image, xs, ys, scale = 1.0): 
   # check that image is grayscale
   assert image.ndim == 2, 'image should be grayscale'
   ##########################################################################
   
   #pre-processing 
   # want width = 3 
   width = 3 
   pad = int((6 * width * scale) // 2) # want 3 x 3 grid of width 3. So 9 x 9 grid. 
   new_image = pad_border(image, pad, pad)
   Ix, Iy = sobel_gradients(image)
   theta = pad_border(np.arctan2(Ix, Iy), pad, pad)
   n = len(xs)

   # finding features in a window 
   feats = []
   for i in range(n): 
    x = xs[i] + pad 
    y = ys[i] + pad 
    img_window = new_image[x - pad: x + pad + 1, y - pad: y + pad + 1] 
    theta_window = theta[x - pad: x + pad + 1, y - pad: y + pad + 1]
    X, Y = np.shape(img_window)
    box_x = np.arange(width, X + 1, width)
    box_y = np.arange(width, Y + 1, width)
    vec_list = []

    # doing box calculations 
    for x_val in box_x: 
      for y_val in box_y:
        vec = box_calc(theta_window, x_val, y_val, width)
        vec_list.append(vec)
    feature_vec = np.ravel(np.array(vec_list))
    feats.append(feature_vec)

   feats = np.array(feats)
   ##########################################################################
   return feats 

def box_calc(theta, x_val, y_val, width): 
  theta_vec = np.array([0] * 8) 
  for x in range(x_val - width, x_val): 
    for y in range(y_val - width, y_val): 
      theta_ix = theta[x, y]
      pi = math.pi 
      ix = int(np.floor(theta_ix / (-pi/4) + 4))
      if ix == 8: 
        ix = 7 #edge case 
      theta_vec[ix] += 1       
  return theta_vec 

"""
   FEATURE MATCHING (7 Points Implementation + 3 Points Write-up)

   Given two sets of feature descriptors, extracted from two different images,
   compute the best matching feature in the second set for each feature in the
   first set.

   Matching need not be (and generally will not be) one-to-one or symmetric.
   Calling this function with the order of the feature sets swapped may
   result in different returned correspondences.

   For each match, also return a real-valued score indicating the quality of
   the match.  This score could be based on a distance ratio test, in order
   to quantify distinctiveness of the closest match in relation to the second
   closest match.  It could optionally also incorporate scores of the interest
   points at which the matched features were extracted.  You are free to
   design your own criterion. Note that you are required to implement the naive
   linear NN search. For 'lsh' and 'kdtree' search mode, you could do either to
   get full credits.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices. You are required to report the efficiency comparison
   between different modes by measure the runtime (check the benchmarking related
   codes in hw2_example.py).

   Arguments:
      feats0   - a numpy array of shape (N0, K), containing N0 K-dimensional
                 feature descriptors (generated via extract_features())
      feats1   - a numpy array of shape (N1, K), containing N1 K-dimensional
                 feature descriptors (generated via extract_features())
      scores0  - a numpy array of shape (N0,) containing the scores for the
                 interest point locations at which feats0 was extracted 
                 (generated via find_interest_point())
      scores1  - a numpy array of shape (N1,) containing the scores for the
                 interest point locations at which feats1 was extracted
                 (generated via find_interest_point())
      mode     - 'naive': performs a brute force NN search

               - 'lsh': Implementing the local senstive hashing (LSH) approach
                  for fast feature matching. In LSH, the high dimensional
                  feature vectors are randomly projected into low dimension
                  space which are further binarized as boolean hashcodes. As we
                  group feature vectors by hashcodes, similar vectors may end up
                  with same 'bucket' with high propabiltiy. So that we can
                  accelerate our nearest neighbour matching through hierarchy
                  searching: first search hashcode and then find best
                  matches within the bucket.
                  Advice for impl.:
                  (1) Construct a LSH class with method like
                  compute_hash_code   (handy subroutine to project feature
                                      vector and binarize)
                  generate_hash_table (constructing hash table for all input
                                      features)
                  search_hash_table   (handy subroutine to search hash table)
                  search_feat_nn      (search nearest neighbour for input
                                       feature vector)
                  (2) It's recommended to use dictionary to maintain hashcode
                  and the associated feature vectors.
                  (3) When there is no matching for queried hashcode, find the
                  nearest hashcode as matching. When there are multiple vectors
                  with same hashcode, find the cloest one based on original
                  feature similarity.
                  (4) To improve the robustness, you can construct multiple hash tables
                  with different random project matrices and find the closest one
                  among all matched queries.
                  (5) It's recommended to fix the random seed by random.seed(0)
                  or np.random.seed(0) to make the matching behave consistenly
                  across each running.

               - 'kdtree': construct a kd-tree which will be searched in a more
                  efficient way. https://en.wikipedia.org/wiki/K-d_tree
                  Advice for impl.:
                  (1) The most important concept is to construct a KDNode. kdtree
                  is represented by its root KDNode and every node represents its
                  subtree.
                  (2) Construct a KDNode class with Variables like data (to
                  store feature points), left (reference to left node), right
                  (reference of right node) index (reference of index at original
                  point sets)and Methods like search_knn.
                  In search_knn function, you may specify a distance function,
                  input two points and returning a distance value. Distance
                  values can be any comparable type.
                  (3) You may need a user-level create function which recursively
                  creates a tree from a set of feature points. You may need specify
                  a axis on which the root-node should split to left sub-tree and
                  right sub-tree.


   Returns:
      matches  - a numpy array of shape (N0,) containing, for each feature
                 in feats0, the index of the best matching feature in feats1
      scores   - a numpy array of shape (N0,) containing a real-valued score
                 for each match
"""
def match_features(feats0, feats1, scores0, scores1, mode='naive'):
   ##########################################################################
  N0, K0 = np.shape(feats0)
  N1, K1 = np.shape(feats1)

  if mode == 'naive':
    matches, scores = brute_force_search(feats0, feats1)
    #scores = np.array([scores0[i] * scores1[matches[i]] for i in range(N0)])

  else: 
    matches, scores = kdtree_NN(feats0, feats1, 5)
     

  ###########################################################################
  return matches, scores


class kdnode():
  def __init__(self, features=None, indices=None, left=None, right=None, parent=None): 
    self.left = left 
    self.right = right 
    self.feat_indices = indices

def build_kdtree(feats, feat_indices, split_indices, medians, depth=5):

  N, K = np.shape(feats)
  kdtree = kdnode(feats, feat_indices)
  left_indices = []
  right_indices = []

  # Base case: 
  if depth == 0: 
    return None

  # Recursive step 
  for i in feat_indices: 
    if feats[i][split_indices[-depth]] < medians[split_indices[-depth]]: 
      left_indices.append(i)
    else: 
      right_indices.append(i)
  if left_indices == []:
    kdtree.left = None
  else: 
    kdtree.left = build_kdtree(feats, left_indices, split_indices, medians, depth - 1)
  if right_indices == []:
    kdtree.right = None
  else: 
    kdtree.right = build_kdtree(feats, right_indices, split_indices, medians, depth - 1)
  return kdtree 

def kdtree_NN(feats0, feats1, depth=5): 

  # pre-processing
  N0, K0 = np.shape(feats0)
  N1, K1 = np.shape(feats1)
  split1 = random.sample(range(0, K1), depth) 
  feat_indices1 = [k for k in range(N1)]
  medians = np.median(feats1, axis=0)
  kd1 = build_kdtree(feats1, feat_indices1, split1, medians, depth)
  scores = np.zeros(N0)
  matches = np.zeros(N0, dtype=int)

  for i in range(N0):
    kd1_copy = kd1 #shallow copy 
    feat_indices = kd1.feat_indices 
    for split in split1:  
      if not kd1_copy: 
        break 
      feat_indices = kd1_copy.feat_indices
      if feats0[i][split] < medians[split]: 
        kd1_copy = kd1_copy.left 
      else: 
        kd1_copy = kd1_copy.right

    matches[i], scores[i] = find_NN(feats0[i], feats1[feat_indices])

  return matches, scores 

def brute_force_search(feats0, feats1):
  X1, Y1 = np.shape(feats0)
  X2, Y2 = np.shape(feats1)
  matches = np.array([0] * X1)
  scores = np.array([0.0] * X1)
  for i in range(X1): 
    matches[i], scores[i] = find_NN(feats0[i], feats1)
  return matches, scores


def find_NN(feat0, feats1): 
  X, Y = np.shape(feats1)
  min_j = 0 
  sec_min_j = 0 
  min_dist = math.inf  
  sec_min_dist = math.inf 
  for j in range(X):
    dist = np.linalg.norm(feat0 - feats1[j])
    if dist < min_dist: 
      sec_min_dist = min_dist
      sec_min_j = min_j 
      min_dist = dist 
      min_j = j 
    elif dist < sec_min_dist: 
      sec_min_dist = dist
      sec_min_j = j
  if sec_min_dist == math.inf or sec_min_dist == 0: #case when there is only one nearest neighbor
    score = 0.8
  else: 
    score = min_dist / sec_min_dist 
  return min_j, score 

"""
   HOUGH TRANSFORM (7 Points Implementation + 3 Points Write-up)

   Assuming two images of the same scene are related primarily by
   translational motion, use a predicted feature correspondence to
   estimate the overall translation vector t = [tx ty].

   Your implementation should use a Hough transform that tallies votes for
   translation parameters.  Each pair of matched features votes with some
   weight dependant on the confidence of the match; you may want to use your
   estimated scores to determine the weight.

   In order to accumulate votes, you will need to decide how to discretize the
   translation parameter space into bins.

   In addition to your implementation, include a brief write-up (in hw2.pdf)
   of your design choices.

   Arguments:
      xs0     - numpy array of shape (N0,) containing x-coordinates of the
                interest points for features in the first image
      ys0     - numpy array of shape (N0,) containing y-coordinates of the
                interest points for features in the first image
      xs1     - numpy array of shape (N1,) containing x-coordinates of the
                interest points for features in the second image
      ys1     - numpy array of shape (N1,) containing y-coordinates of the
                interest points for features in the second image
      matches - a numpy array of shape (N0,) containing, for each feature in
                the first image, the index of the best match in the second
      scores  - a numpy array of shape (N0,) containing a real-valued score
                for each pair of matched features

   Returns:
      tx      - predicted translation in x-direction between images
      ty      - predicted translation in y-direction between images
      votes   - a matrix storing vote tallies; this output is provided for
                your own convenience and you are free to design its format
"""
def hough_votes(xs0, ys0, xs1, ys1, matches, scores):
  ###########################################################################
  offset_x = []
  offset_y = []

  for ix_0 in range(len(matches)): 
    ix_1 = matches[ix_0]
    offset_x.append(xs0[ix_0] - xs1[ix_1])
    offset_y.append(ys0[ix_0] - ys1[ix_1])

  min_x = min(offset_x)
  min_y = min(offset_y)

  dim_x = max(offset_x) - min(offset_x)
  dim_y = max(offset_y) - min(offset_y)
  votes = np.zeros((dim_x + 1, dim_y + 1))


  for ix in range(len(offset_x)): 
    x = offset_x[ix] - min_x
    y = offset_y[ix] - min_y
    votes[x, y] += scores[ix]

  val = np.unravel_index(np.argmax(votes), votes.shape)
  ty = val[1] + min_y
  tx = val[0] + min_x 

   ##########################################################################
  return tx, ty, votes
