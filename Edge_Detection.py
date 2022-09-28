import numpy as np
import cv2
from scipy import ndimage
from numpy import linalg
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (8,5)
def read_image(image_path):
  image = cv2.imread(image_path)
  return image

def gaussian_2D_filter(size , sigma):
  center = size // 2
  x, y = np.mgrid[0 - center : size - center, 0 - center : size - center]
  
  filter = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma))) 
  return filter

def imgfilter(image, filter):
  img = cv2.filter2D(image,-1,filter)
  return img
def orientedFilterMagnitude(im,sigma):
  
  filter_size = int(sigma*4+1)
  filter = gaussian_2D_filter(filter_size, sigma)
  smooth_img = (imgfilter(im, filter)).astype(np.float64)
  #blurring gaussian derivative results in sobel
  #sobel operator in horizontal,vertical and diagonal axis
  K0 = np.array([[1,0,-1],[2,0,-2],[1,0,-1]]) #0 deg
  K90 = np.array([[1,2,1],[0,0,0],[-1,-2,-1]]) #90 deg
  K45 = np.array([[0,-1,-2],[1,0,-1],[2,1,0]]) #45 deg
  K135 = np.array([[-2,-1,0],[-1,0,1],[0,1,2]]) #135 deg
  
  plt.subplot(1,4,1)
  plt.imshow(K0, extent=[0,3,0,3], aspect='auto')
  plt.title('filter along 0 deg')
  plt.subplot(1,4,2)
  plt.imshow(K45, extent=[0,3,0,3], aspect='auto')
  plt.title('filter along 45 deg')
  plt.subplot(1,4,3)
  plt.imshow(K0, extent=[0,3,0,3], aspect='auto')
  plt.title('filter along 90 deg')
  plt.subplot(1,4,4)
  plt.imshow(K0, extent=[0,3,0,3], aspect='auto')
  plt.title('filter along 135 deg')
  plt.show()
   
  #red
  F0_r = ndimage.convolve(smooth_img[:,:,0], K0)
  F90_r = ndimage.convolve(smooth_img[:,:,0], K90)
  F45_r = ndimage.convolve(smooth_img[:,:,0], K45)
  F135_r = ndimage.convolve(smooth_img[:,:,0], K135)
  magnitude_r = np.sqrt(F0_r**2 + F90_r**2+F45_r**2+F135_r**2) 

  #green
  F0_g = ndimage.convolve(smooth_img[:,:,0], K0)
  F90_g = ndimage.convolve(smooth_img[:,:,0], K90)
  F45_g = ndimage.convolve(smooth_img[:,:,0], K45)
  F135_g = ndimage.convolve(smooth_img[:,:,0], K135)
  magnitude_g = np.sqrt(F0_g**2 + F90_g**2+F45_g**2+F135_g**2) 
  
  #blue
  F0_b = ndimage.convolve(smooth_img[:,:,0], K0)
  F90_b = ndimage.convolve(smooth_img[:,:,0], K90)
  F45_b = ndimage.convolve(smooth_img[:,:,0], K45)
  F135_b = ndimage.convolve(smooth_img[:,:,0], K135)
  magnitude_b = np.sqrt(F0_b**2 + F90_b**2+F45_b**2+F135_b**2) 

  largest_f0 = np.maximum(F0_r,F0_g,F0_b)
  largest_f90 = np.maximum(F90_r,F90_g,F90_b)
  largest_f45 = np.maximum(F45_r,F45_g,F45_b)
  largest_f135 = np.maximum(F135_r,F135_g,F135_b)
  #direction
   
  overall = np.sqrt(magnitude_b**2+magnitude_g**2+magnitude_r**2)
  print("Gradient magnitude with 4 filters")
  cv2.imshow(overall)
  orient = cv2.phase(largest_f0+np.cos(np.pi/4)*largest_f45,largest_f90+np.cos(np.pi/4)*largest_f135, angleInDegrees=True)
  print("Gradient orientation with 4 filters")
  cv2.imshow(orient)
  cv2.waitKey(0)
  return overall


def edgeOrientedFilters(im):
  '''
  im: input image

  output: a soft boundary map of the image
  '''
  ## YOUR CODE HERE
  im = im.astype(np.uint8)
  edge = cv2.Canny(im,500,600)
  print("After Non-Max Suppression")
  cv2.imshow(edge)
  cv2.waitKey(0)
  return edge
img_path = "/aero.jpg" ## add the path here
im = cv2.imread(img_path)
overall = orientedFilterMagnitude(im,3)
edge = edgeOrientedFilters(overall)
