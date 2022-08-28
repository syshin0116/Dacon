# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 21:44:53 2019

@author: Jeon
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from scipy.linalg import toeplitz
# vector




# matrix
# ‘C’ means to read / write the elements using C-like index order, 
# with the last axis index changing fastest, back to the first axis index changing slowest. 
# 
# ‘F’ means to read / write the elements using Fortran-like index order, 
# with the first index changing fastest, and the last index changing slowest. 
# 
# Note that the ‘C’ and ‘F’ options take no account of the memory layout of the underlying array, 
# and only refer to the order of indexing. 
# 
# ‘A’ means to read / write the elements in Fortran-like index order if a is Fortran contiguous in memory, 
# C-like order otherwise.

a = np.array(range(10))
a.reshape((5,2), order = 'C') # default
a.reshape((5,2), order = 'F') # like R
a.reshape((2,5), order = "A")

# discussion: memory add
# C: [0,0] -> [0,1] -> [1,0] -> [1,1] ... -> [4,0] -> [4,1]
# F: [0,0] -> [1,0] -> [2,0] -> [3,0] ... -> [4,0] -> [4,1]
    
# array
# 메모리를 할당하는 방식을 이해해야 함.
a = np.array(range(24))
a.reshape((2,3,4), order = 'C')
a.reshape((2,3,4), order = 'F')

# C:[0,0,0] -> [0,0,1] -> [0,0,2] -> 
#   [0,0,3] -> [0,1,0] -> [0,1,1] -> 
# .....

# F: [0,0,0] -> [1,0,0] -> [0,1,0] -> 
#    [1,1,0] -> [0,2,0] -> [1,2,0] -> 
#    [0,0,1] -> [1,0,1] -> [0,1,1] -> 
# .....
# 즉, fotran 방식은 array 의 3-dimension이 (d_i, d_j, d_k)일때
# [i,j,k]의 index는 d_i + j*(d_i) + k*(d_i*d_j) 에 저장된다.

# C방식은 
# [i,j,k]의 index는 i*(d_j*d_k) + j*(d_k) + d_k 에 저장된다.

## 사진의 변환 (C방식 reshape의 사용)
(x,_), (_,_) = tf.keras.datasets.mnist.load_data()
x.shape # default C
n = x.shape[0]
pix_dim1 = x.shape[1]
pix_dim2 = x.shape[2]

# the ith image data
i = 1
x[[i]]
x[[i]].shape

# 하나의 dimension 의 크기가 1인 경우 해당 dimension을 제거하기 위한 방법으로
# np.squeeze  를 사용함
xs = np.squeeze(x[[i]], axis = 0)
xs.shape
plt.imshow(xs, cmap='gray')


xvec = x.reshape((n, pix_dim1*pix_dim2))
xvec[[i]]
xvec[[i]].shape
xv = np.squeeze(xvec[[i]], axis = 0)
plt.plot(xv)



# product braodcasting
# row-wise product
# 행의 크기가 클 때, 행마다 같은 상수를 곱해야 하는 경우:
# 상수 곱에 해당하는 대각 행렬의 크기가 매우 커지는 문제가 발생
# 행렬로 반복문을 사용하면 속도가 저하.
# 다음과 같이 해결한다.
 
x = np.array(range(15)).reshape((5,3))
z = np.array(range(5)).reshape((5,1))
x*z

# 열별로 같은 상수를 곱해야 하는 경우 다음과 같이 해결함
x = np.array(range(15)).reshape((5,3))
z = np.array(range(3)).reshape((1,3))
x*z


# Transpose
A = np.array(range(0,6)).reshape((2,3))
B = np.array(range(8,20)).reshape((3,4))
A
B
(A@B).T
(B.T)@(A.T)

# Transpose of block matrix
# A11 = 2 by 3 , A12 = 2 by 1
# A21 = 1 by 3   A22   1 by 1
# A is clearly 3 y 4 matrix
A = np.array(range(12)).reshape((3,4))
A11 = np.array([0,1,2,4,5,6]).reshape(2,3)
A12 = np.array([3,7]).reshape(2,1)
A21 = np.array([8,9,10]).reshape(1,3)
A22 = np.array([11]).reshape(1,1)
A
A.T

# Trace
A = np.array(range(16)).reshape((4,4))
B = np.array(range(16,32)).reshape((4,4))

np.trace(A + B)
np.trace(A) +  np.trace(B)

k = 2
np.trace(k*A)
k*np.trace(A)

A = np.array(range(24)).reshape((4,6))
B = np.array(range(24,48)).reshape((6,4))
np.trace(A@B)
np.trace(B@A)

np.trace(A.T@A)
np.trace(A@A.T)
(A**2).sum()

# inverse matrix
A = np.array(range(4)).reshape((2,2))
B = np.linalg.inv(A) 
A@B
B@A

A = np.array(range(4)).reshape((2,2))
B = np.array(range(4,8)).reshape((2,2))
np.linalg.inv(A@B)
np.linalg.inv(B)@np.linalg.inv(A)
np.linalg.inv(A.T)
(np.linalg.inv(A)).T

# 
n = 10 
p = 4
A = np.random.normal(0,1,n*p).reshape((n,p))
x = np.random.normal(0,1,p)
A@x

# more precise expression
x = x.reshape((p,1))
A@x



#  spaned space (page 29)
np.random.seed(1)
W = np.random.gamma(1,1,(10,3)).round(2)
W
a = np.random.gamma(1,1,3).round(2)
a
W@a
x[0]*W[:,0] + x[1]*W[:,1] + x[2]*W[:,2]



#  orthogonal projection (page 48)
np.random.seed(1)
Y = np.random.gamma(1,1,(5,2)).round(2)
proj_Y = Y@np.linalg.inv(Y.T@Y)@Y.T
proj_Y.shape

proj_Y
proj_Y@proj_Y
x = np.random.normal(0,1,5).round(2)
proj_Y@x
x - proj_Y@x
(proj_Y@x*(x - proj_Y@x)).sum()

# eigendecomposition (page 51)
np.random.seed(1)
tmpA = np.random.gamma(1,1,(3,3)).round(2)
A = tmpA + tmpA.T

eig_fit = np.linalg.eig(A)
type(eig_fit)
eig_fit[0] # eigen value
eig_fit[1] # eigen vector

d = eig_fit[0]
E = eig_fit[1]

eig_fit[1]@np.diag(eig_fit[0])@eig_fit[1].T
A

(E[:,0]*E[:,1]).sum()
(E[:,0]*E[:,0]).sum()
E1 = E[:,[0]]@E[:,[0]].T
# note: eig_fit[1][:,0] 대신 eig_fit[1][:,[0]] 를 사용하면, dimension collapsing  을 방지해준다. 
E2 = E[:,[1]]@E[:,[1]].T
E3 = E[:,[2]]@E[:,[2]].T

d[0]*E1 + d[1]*E2 + d[2]*E3
A



a = {"키1":10, "키2":5, "키이3": "a"}

a_keys = a.keys()
type(a_keys)

