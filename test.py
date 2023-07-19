import numpy as np
import math
import tensorflow as tf

# def getCentroid(arr):
#     length = arr.shape[0]
#     sum_x = np.sum(arr[:, 0])
#     sum_y = np.sum(arr[:, 1])
#     return sum_x/length, sum_y/length

# def getSSE(arr, centroid):
#     sse = 0
#     for i in arr: sse += math.dist(i, centroid)**2
#     return sse

# A, B, C, D = [1,1], [3,3], [5,1], [9,2]
# pointList = np.array([A, B, C])
# centroid = getCentroid(pointList)
# sse = getSSE(pointList, centroid)
# print(centroid)
# print(sse)