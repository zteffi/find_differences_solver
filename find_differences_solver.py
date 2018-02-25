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
    img = cv2.imread(name)    
    surf = cv2.xfeatures2d.SURF_create(500, upright = True)
    kp2, des2 = surf.detectAndCompute(img, None )
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des2,des2,k=2)
    img2 = img.copy()
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
            if dist < MIN_PICTURE_SIZE:
                continue            
            dist_norm =  math.sqrt( dist * dist / (img.shape[0] * img.shape[1]))
            angle = (np.arcsin((pts[0][1] - pts[1][1])/dist))
            if angle > 0:
                start_points.append(pts[0])
                end_points.append(pts[1])
                angles.append(angle)
                distances.append(dist)
   
    translations.append(np.array(zip(distances, angles)))

    # find the most dense area of [translation distance, trabslation angle] space
    hist = plt.hist2d(distances, angles, bins=64)
   
    (x,y) = unravel_index(hist[0].argmax(), hist[0].shape)
    dist_low_bound = hist[1][x]
    dist_high_bound = hist[1][x+1]
    angle_low_bound = hist[2][y]
    angle_high_bound = hist[2][y+1]
    
    filtered_start_points = []
    filtered_end_points= []
    translation_angle = (angle_low_bound + angle_high_bound)/2
    translation_dist = (dist_low_bound+dist_high_bound)/2
    print("angle:" + str(translation_angle))
    print("distance:" + str(translation_dist))
    for i,p in enumerate(start_points):
        if (angle_low_bound <= angles[i] <= angle_high_bound) and (dist_low_bound <= distances[i] <= dist_high_bound):
                #cv2.line(img2, (start_points[i][0], start_points[i][1]), (end_points[i][0],end_points[i][1]), (255,0,255),1) 
                filtered_start_points.append(start_points[i])
                filtered_end_points.append(end_points[i])
    x1,y1,w1,h1 = cv2.boundingRect(np.array(filtered_start_points))
    cv2.rectangle(img2, (x1,y1), (x1+w1, y1+h1), (0,100,255),3)

    '''
    x2 = x1
    if translation_angle > 0:
        x2 = x2 - (translation_dist * np.cos(translation_angle))
    else :
        x2 = x2 + (translation_dist * np.cos(translation_angle))

    y2 = y1   - (translation_dist * np.sin(translation_angle))
    
    cv2.rectangle(img2, (int(x2),int(y2)), (int(x2 + w),int(y2 + h)), (255,255,0),3)
    '''
    x2,y2,w2,h2 = cv2.boundingRect(np.array(filtered_end_points))

    roi1 = img[y1:(y1+h1), x1:(x1+w1)]
    roi2 = img[y2:(y2+h2), x2:(x2+w2)]
    # resize the rois to half the average width/height
    resize_width = int((w1+w2)/4)
    resize_height  = int((h1+h2)/4)
    resized_roi1 = cv2.resize(roi1, (resize_width, resize_height), interpolation = cv2.INTER_AREA )
    resized_roi2 = cv2.resize(roi2, (resize_width, resize_height), interpolation = cv2.INTER_AREA )
    
    diffmask = cv2.absdiff(resized_roi1, resized_roi2)
    diffmasksum = diffmask[:,:,0] + diffmask[:,:,1] + diffmask[:,:,2]
    diffmaskdraw = np.zeros([resize_height, resize_width, 3],dtype=np.uint8)
    diffmaskdraw[:,:,0] = diffmasksum
    diffmaskdraw[:,:,2] = diffmasksum

    diffmaskdraw1 = cv2.resize(diffmaskdraw, (w1, h1), interpolation = cv2.INTER_CUBIC)
    diffmaskdraw[:,:,2] = np.zeros([resize_height, resize_width],dtype=np.uint8)
    diffmaskdraw[:,:,1] = diffmasksum

    diffmaskdraw2 = cv2.resize(diffmaskdraw, (w2, h2), interpolation = cv2.INTER_CUBIC)

    img2[y1:(y1+h1), x1:(x1+w1)] =  cv2.addWeighted(diffmaskdraw1, .5, roi1, 1, 0)
    img2[y2:(y2+h2), x2:(x2+w2)] =  cv2.addWeighted(diffmaskdraw2, .5, roi2, 1, 0)

    cv2.rectangle (img2, (x2,y2), (x2+w2, y2+h2), (100,255,0),3)
    result = np.hstack((img, img2))
    cv2.imwrite('results/' + file_name + '_result.png', result)
    print('===' + str(index) + '========================================')
    index += 1