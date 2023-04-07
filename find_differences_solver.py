import cv2
import numpy as np
from numpy import unravel_index
import glob
import math
from matplotlib import pyplot as plt


MIN_PICTURE_SIZE = 50 # minimal size of a picture
images = glob.glob('./dataset/*.jpg')
translations = []
index = 0
for name in images:
    img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create(500)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img, None)
    FLANN_INDEX_KDTREE = 4
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 15)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann =  cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des1, k = 2)

    good = []
    # find feasible pairs
    for m,n in matches:
        if m.distance < 0.93 * n.distance:
            good.append((m,n))
    if len(good) < 10:
         continue
    src_pts = np.float32([ kp1[m.queryIdx].pt for (m,_) in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp1[m.trainIdx].pt for (_,m) in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    w,h = img.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    img2 = cv2.imread(name)

    cv2.imwrite('results/' + str(index) + '_src.png', img2)
    img2 = cv2.warpPerspective(img2, np.linalg.inv(M), img2.shape[:-1][::-1])
    cv2.imwrite('results/' + str(index) + '_result.png', img2)
    print('===' + str(index) + '========================================')
    index += 1