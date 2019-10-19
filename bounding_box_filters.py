#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 14:04:28 2019

@author: veli
"""

import cv2
import numpy as np

def make_bounding_box(src, frame2, out, by_filter, most_smallest_area, max_diffrence):
    # Threshold it so it becomes binary
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    #src = cv2.imread(src, 0)
    ret, thresh = cv2.threshold(src,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # You need to choose 4 or 8 for connectivity type
    connectivity = 8  
    # Perform the operation
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)

    # The fourth cell is the centroid matrix
    centroids = output[3]
    # The area of each centroid matrix
    areas=output[2][:,4]
    
    x_top = by_filter
    x_bottom = 0
    y_top = by_filter
    y_bottom = 0
    
    centroids_arr = []
    areas_arr = []
    
    for centroid, area in zip(centroids[1:], areas[1:]):
        if area > most_smallest_area:
            centroids_arr.append(centroid)
            areas_arr.append(area)
    
    
    for y_ in range(0, 720, by_filter):
        y_bottom += by_filter
        y_top += by_filter
        
        x_bottom = 0
        x_top = by_filter
        for x_ in range(0, 1280, by_filter):                
            x_bottom += by_filter
            x_top += by_filter
            
            conds_centroids = []
            for centroid in centroids_arr:
                if centroid[0] >= x_bottom and centroid[1] >= y_bottom and centroid[0] < x_top and centroid[1] < y_top:
                    conds_centroids.append(centroid)
            
            if len(conds_centroids) >= 2:
                similar = similarity_centroids(conds_centroids, max_diffrence)
                if(len(similar)>0):
                    centroids_arr = update_centroids(similar, centroids_arr)
                
    for centroid in centroids_arr:
        if not np.isnan(centroid[0]):
            cv2.rectangle(frame2, (int(centroid[0]) - 20, int(centroid[1]) - 20), (int(centroid[0]) + 20, int(centroid[1]) + 20), (255, 0, 0), 2)
    out.write(frame2)
    
    
def update_centroids(similar, centroids_arr):
    for centroid in similar:
        ind = -1
        for centroid_arr in centroids_arr:
            ind += 1
            if centroid[0] == centroid_arr[0] and centroid[1] == centroid_arr[1]:
                del centroids_arr[ind]
                
    x , y = 0, 0
    for centroid in similar:
        x += centroid[0]
        y += centroid[1]
    x /= len(similar)
    y /= len(similar)
    centroids_arr.append([x, y])
    return centroids_arr


def calculate_diff_centroids(centroid1, centroid2):
    x1 = centroid1[0] 
    x2 = centroid2[0]
    y1 = centroid1[1]
    y2 = centroid2[1]    
    diff_sqr = (x1 - x2)**2 + (y1 - y2)**2
    return diff_sqr


def similarity_centroids(conds_centroids, max_diffrence):
    diff_lst = []
    for i in range(len(conds_centroids)-1):
        for j in range(i, len(conds_centroids)):
            diff_sqr = calculate_diff_centroids(conds_centroids[i], conds_centroids[j])            
        diff_lst.append([conds_centroids[i], conds_centroids[j], diff_sqr])

    similar = []
    for diff in diff_lst:
        if diff[2] < max_diffrence:
            similar.append(np.array(diff[0]))
            similar.append(np.array(diff[1]))
    return np.asarray(similar)


cap = cv2.VideoCapture("video_cv2.mog2_no_shadows_name")
cap2 = cv2.VideoCapture("original_video_name")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'M','J','P','G') 
count = 0
out = cv2.VideoWriter('output.avi', fourcc, 15.0, (frame_width, frame_height))

while(cap.isOpened() and cap2.isOpened()):
    ret, frame = cap.read()
    ret2, frame2 = cap2.read()
    
    #filter_10by10 = [[0, 10], [0, 10]] # ((x1, x2), (y1, y2)) 
    if ret==True:
        count+= 1
        print(count)
        make_bounding_box(frame, frame2, out, by_filter=10, most_smallest_area=8)
    else:
        break
    
cap.release()
out.release()
cap2.release()
cv2.destroyAllWindows()
