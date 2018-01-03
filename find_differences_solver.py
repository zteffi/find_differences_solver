import cv2
import numpy as np
import glob
import math
from matplotlib import pyplot as plt

images = glob.glob('./dataset/*.jpg')
translations = []
index = 0
for name in images:
    translations.append([])
    img = cv2.imread(name)    
    surf = cv2.xfeatures2d.SURF_create(3000)
    kp2, des2 = surf.detectAndCompute(img, None )
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des2,des2,k=2)
    matchesMask = [[0,0] for i in range(len(matches))]
    img2 = img
    
    file_name= str(index).zfill(2) + '_' + name[14:19]
    file = open('results/'+ file_name + '.csv', 'w')
    file.write('distance-orig;angle-deg\n')
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.95 * n.distance:
            matchesMask[i]=[1,0]
            # we'll assure self matching on image by swapping matches
            matches[i][0],matches[i][1] = matches[i][1], matches[i][0]
            pts = np.array([kp2[matches[i][j].trainIdx].pt for j in range(2)], np.int32)
            dist = cv2.norm(pts[0], pts[1])
            dist_norm =  math.sqrt( dist * dist / (img.shape[0] * img.shape[1]))
            angle = abs(np.arcsin((pts[0][1] - pts[1][1])/dist))
            print(str(dist_norm) + ' ' +str(angle))
            file.write(str(dist) +';'+ str(angle*180/math.pi) + '\n')
            row = (dist_norm, angle)
            translations[index].append(row)
            cv2.line(img2, (pts[0][0], pts[0][1]), (pts[1][0],pts[1][1]) ,(255,0,255),1)
    draw_params = dict(
                   matchesMask = matchesMask,
                   flags = 0)
    #img2 = cv2.drawMatchesKnn(img,kp2,img,kp2,matches,None,**draw_params)
    
    #img2 = cv2.drawKeypoints(img, kp2,None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('results/' + file_name + '_result_surf.png', img2)
    print('===' + str(index) + '========================================')
    index += 1