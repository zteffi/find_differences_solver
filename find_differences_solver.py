import cv2
import numpy as np
from numpy import unravel_index
import glob
import math
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import random

MIN_PICTURE_SIZE = 50 # minimal size of a picture
images = glob.glob('./dataset/*.jpg')
translations = []
index = 0
for name in images:
    img = cv2.imread(name)    
    surf = cv2.xfeatures2d.SURF_create(3000)
    kp2, des2 = surf.detectAndCompute(img, None )
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des2,des2,k=2)
    img2 = img
    file_name= str(index).zfill(2) + '_' + name[14:19]
    file = open('results/'+ file_name + '.csv', 'w')
    file.write('distance-orig;angle-deg\n')
    angles = []
    distances = []
    start_points = []
    end_points = []
    # find feasible pairs 
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.95 * n.distance:
            # we'll assure self matching on image by swapping matches
            matches[i][0],matches[i][1] = matches[i][1], matches[i][0]
            pts = np.array([kp2[matches[i][j].trainIdx].pt for j in range(2)], np.int32)
            dist = cv2.norm(pts[0], pts[1])
            if (dist < MIN_PICTURE_SIZE):
                continue            
            dist_norm =  math.sqrt( dist * dist / (img.shape[0] * img.shape[1]))
            angle = abs(np.arcsin((pts[0][1] - pts[1][1])/dist))
          
            start_points.append(pts[0])
            end_points.append(pts[1])
            angles.append(angle)
            distances.append(dist_norm)
   
    translations.append(np.array(zip(distances, angles)))

    # find the most dense area of [translation distance, trabslation angle] space
    hist = plt.hist2d(distances, angles, bins=32)
    plt.show()
    (x,y) = unravel_index(hist[0].argmax(), hist[0].shape)
    dist_low_bound = hist[1][x]
    dist_high_bound = hist[1][x+1]
    angle_low_bound = hist[2][y]
    angle_high_bound = hist[2][y+1]

    print("angle:" + str((angle_low_bound + angle_high_bound)/2))
    print("distance:" + str((dist_low_bound+dist_high_bound)/2))
    for i,p in enumerate(start_points):
        if (angle_low_bound <= angles[i] <= angle_high_bound) and (dist_low_bound <= distances[i] <= dist_high_bound):
                cv2.line(img2, (start_points[i][0], start_points[i][1]), (end_points[i][0],end_points[i][1]) ,(255,0,255),1) 

    cv2.imwrite('results/' + file_name + '_result.png', img2)
    print('===' + str(index) + '========================================')
    index += 1