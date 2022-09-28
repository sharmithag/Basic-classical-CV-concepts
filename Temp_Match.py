import cv2
import numpy

waldo1 = cv2.imread('/waldo.jpg')
map1 = cv2.imread('/pic1.jpeg')


def sum_of_diff(map,template):
    map_grey = (cv2.cvtColor(map,cv2.COLOR_BGR2GRAY)).astype(np.float64)
    template_grey = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY).astype(np.float64)
    win_size = template_grey.shape
    
    ssd = np.ones((map_grey.shape[0]-win_size[0],map_grey.shape[1]-win_size[1]),np.float64)*255
          
    for i in range(map_grey.shape[0]-win_size[0]):
      for j in range(map_grey.shape[1]-win_size[1]):
        temp = np.sum((map_grey[i:win_size[0]+i,j:win_size[1]+j] - template_grey)**2)
        ssd[i][j] = temp
    ssd = cv2.normalize(ssd,None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX)
    
    return ssd
    
(waldoHeight, waldoWidth) = waldo1.shape[:2]
result = sum_of_diff(map1,waldo1)

topLeft = np.where(result == np.amin(result))

(_, _, minLoc, maxLoc) = cv2.minMaxLoc(result)

topLeft=minLoc
# grab the bounding box of waldo and extract him from the puzzle image

botRight = (topLeft[0] + waldoWidth, topLeft[1] + waldoHeight)
roi = map1[topLeft[1] : botRight[1], topLeft[0] : botRight[0]]

# construct a darkened transparent 'layer' to darken everything
# in the map except for Waldo
mask = np.zeros(map1.shape, dtype = "uint8")
map1 = cv2.addWeighted(map1, 0.25, mask, 0.75, 0)
map1[topLeft[1] : botRight[1], topLeft[0] : botRight[0]] = roi

# display the images
result_rgb = cv2.cvtColor(map1, cv2.COLOR_RGB2BGR)
plt.figure(figsize = (15, 15))
plt.imshow(result_rgb)
