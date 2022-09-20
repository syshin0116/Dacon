"""
@author: cau_ybh
"""

import logging

import datetime as pydatetime
import time

import os

GPU_set = str(input("Set GPU : "))

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=GPU_set

import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.losses import binary_crossentropy
import random
import sys
import h5py
from itertools import compress

from PIL import Image, ImageFont, ImageDraw
from matplotlib import pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

import datetime
import math
import cv2
import json
import skimage.transform
###


###

num_classes = 7

h_, w_ = 512, 512
input_shape = (h_, w_)

while True:
    PATH = os.path.abspath(str(input("Set Dir path : ")))
    if os.path.isdir(PATH) == True:
        break

log_dir = os.path.join(PATH,'Segmentation') + os.sep
print('Weight will save here : ' + log_dir)
###

while True:
    weight_PATH = os.path.abspath(log_dir + str(input("Set weight : ")))
    if os.path.isfile(weight_PATH) == True:
        break

print(weight_PATH)

logger = logging.getLogger()

logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

file_handler = logging.FileHandler(log_dir + '/test.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)



logger.info("start")
logger.info(datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d %H:%M:%S'))


json_list = []
for (path, dir, files) in os.walk(os.path.join(PATH,'JSON')):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        if ext == '.json':
            json_list.append(os.path.abspath("%s/%s" % (path, filename)))

file_root_list = []
file_name_list = []

for (path, dir, files) in os.walk(PATH):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        if ext == '.jpg':
            #print("%s/%s" % (path, filename))
            
            file_name_list.append(filename)
            file_root_list.append(path + os.sep + filename)


an_list_ = []
im_list_ = []
for n_, json_path in enumerate(json_list):

    with open(json_path, 'r', encoding='utf-8') as f:
        jsonText = f.read()

    jsonDict = json.loads(jsonText)
    
    an_list = jsonDict['annotations']
    im_list = jsonDict['images']
    
    an_list_.append(an_list)
    im_list_.append(im_list)

INDEX = np.load(os.path.join(PATH,'index_seg.npy'))
#im_path_list = np.load(os.path.join(PATH,'im_path_list_seg.npy'))
    
train_id = np.load(os.path.join(PATH,'train_id_seg.npy'))
valid_id = np.load(os.path.join(PATH,'valid_id_seg.npy'))
test_id = np.load(os.path.join(PATH,'test_id_seg.npy'))

logger.info("test data has been loaded")
logger.info("length of test_id : " + str(len(test_id)))

test_image_root = INDEX[test_id,2]

with open(os.path.join(PATH,'Segmentation','test_data_path.txt'), 'w') as f:
    for i in range(len(test_image_root)):
        f.write(test_image_root[i] + '\n')
        
logger.info("list of test data paths has been saved at Segmentation folder.")

###

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

def hourglass_module(bottom, cnv_dim, hgid, dims):
    lfs = left_features(bottom, hgid, dims)

    rf1 = right_features(lfs, hgid, dims)
    rf1 = convolution(rf1, 3, cnv_dim, name='cnvs.%d' % hgid)

    return rf1


def convolution(_x, k, out_dim, name, stride=1):
    padding = (k - 1) // 2
    _x = tf.keras.layers.ZeroPadding2D(padding=padding, name=name + '.pad')(_x)
    _x = tf.keras.layers.Conv2D(out_dim, k, strides=stride, use_bias=False, name=name + '.conv')(_x)
    _x = tf.keras.layers.BatchNormalization(epsilon=1e-5, name=name + '.bn')(_x)
    _x = mish(_x)
    return _x


def residual(_x, out_dim, name, stride=1):
    shortcut = _x
    num_channels = tf.keras.backend.int_shape(shortcut)[-1]
    _x = tf.keras.layers.ZeroPadding2D(padding=1, name=name + '.pad1')(_x)
    _x = tf.keras.layers.Conv2D(out_dim, 3, strides=stride, use_bias=False, name=name + '.conv1')(_x)
    _x = tf.keras.layers.BatchNormalization(epsilon=1e-5, name=name + '.bn1')(_x)
    _x = mish(_x)

    _x = tf.keras.layers.Conv2D(out_dim, 3, padding='same', use_bias=False, name=name + '.conv2')(_x)
    _x = tf.keras.layers.BatchNormalization(epsilon=1e-5, name=name + '.bn2')(_x)

    if num_channels != out_dim or stride != 1:
        shortcut = tf.keras.layers.Conv2D(out_dim, 1, strides=stride, use_bias=False, name=name + '.shortcut.0')(
                    shortcut)
        shortcut = tf.keras.layers.BatchNormalization(epsilon=1e-5, name=name + '.shortcut.1')(shortcut)

    _x = tf.keras.layers.Add(name=name + '.add')([_x, shortcut])
    _x = mish(_x)
    return _x


def pre(_x, num_channels):
    _x = convolution(_x, 3, 128, name='pre.0', stride=2)
    _x = residual(_x, num_channels, name='pre.1', stride=2)
    return _x


def left_features(bottom, hgid, dims):
    features = [bottom]
    for kk, nh in enumerate(dims):
        pow_str = ''
        for _ in range(kk):
            pow_str += '.center'
        _x = residual(features[-1], nh, name='kps.%d%s.down.0' % (hgid, pow_str), stride=2)
        _x = tf.keras.layers.Dropout(0.1)(_x)
        _x = residual(_x, nh, name='kps.%d%s.down.1' % (hgid, pow_str))
        features.append(_x)
    return features

def connect_left_right(left, right, num_channels, num_channels_next, name):
    left = residual(left, num_channels_next, name=name + 'skip.0')
    left = tf.keras.layers.Dropout(0.1)(left)
    left = residual(left, num_channels_next, name=name + 'skip.1')
    left = tf.keras.layers.Dropout(0.1)(left)

    out = residual(right, num_channels, name=name + 'out.0')
    out = tf.keras.layers.Dropout(0.1)(out)
    out = residual(out, num_channels_next, name=name + 'out.1')
    out = tf.keras.layers.Dropout(0.1)(out)
    out = tf.keras.layers.UpSampling2D(name=name + 'out.upsampleNN')(out)
    out = tf.keras.layers.Add(name=name + 'out.add')([left, out])
    return out


def bottleneck_layer(_x, num_channels, hgid):
    pow_str = 'center.' * 5
    _x = residual(_x, num_channels, name='kps.%d.%s0' % (hgid, pow_str))
    _x = tf.keras.layers.Dropout(0.1)(_x)
    _x = residual(_x, num_channels, name='kps.%d.%s1' % (hgid, pow_str))
    _x = tf.keras.layers.Dropout(0.1)(_x)
    _x = residual(_x, num_channels, name='kps.%d.%s2' % (hgid, pow_str))
    _x = tf.keras.layers.Dropout(0.1)(_x)
    _x = residual(_x, num_channels, name='kps.%d.%s3' % (hgid, pow_str))
    _x = tf.keras.layers.Dropout(0.1)(_x)
    _x = residual(_x, num_channels, name='kps.%d.%s4' % (hgid, pow_str))
    _x = tf.keras.layers.Dropout(0.1)(_x)
    return _x


def right_features(leftfeatures, hgid, dims):
    rf = bottleneck_layer(leftfeatures[-1], dims[-1], hgid)
    for kk in reversed(range(len(dims))):
        pow_str = ''
        for _ in range(kk):
            pow_str += 'center.'
        rf = connect_left_right(leftfeatures[kk], rf, dims[kk], dims[max(kk - 1, 0)], name='kps.%d.%s' % (hgid, pow_str))
    return rf

def create_heads(heads, rf1, hgid):
    _heads = []
    for head in sorted(heads):
        num_channels = heads[head]
        _x = tf.keras.layers.Conv2D(256, 3, use_bias=True, padding='same', name=head + '.%d.0.conv' % hgid)(rf1)
        _x = mish(_x)
        _x = tf.keras.layers.Conv2D(num_channels, 1, use_bias=True, name=head + '.%d.1' % hgid)(_x)
        _heads.append(_x)
    return _heads

def centernet(image_input, input_size=512, num_classes=7):

    num_stacks = 2
    cnv_dim=96
    dims=[128, 128, 192, 256]
    
    output_size = input_size // 4

    #image_input = tf.keras.layers.Input([512,512,3])

    inter = pre(image_input, cnv_dim)
    prev_inter = None
    for i in range(num_stacks):
        prev_inter = inter
        inter = hourglass_module(inter, cnv_dim, i, dims)
        if i < num_stacks - 1:
            inter_ = tf.keras.layers.Conv2D(cnv_dim, 1, use_bias=False, name='inter_.%d.0' % i)(prev_inter)
            inter_ = tf.keras.layers.BatchNormalization(epsilon=1e-5, name='inter_.%d.1' % i)(inter_)

            cnv_ = tf.keras.layers.Conv2D(cnv_dim, 1, use_bias=False, name='cnv_.%d.0' % i)(inter)
            cnv_ = tf.keras.layers.BatchNormalization(epsilon=1e-5, name='cnv_.%d.1' % i)(cnv_)

            inter = tf.keras.layers.Add(name='inters.%d.inters.add' % i)([inter_, cnv_])
            inter = mish(inter)
            #inter = Activation('relu', name='inters.%d.inters.relu' % i)(inter)
            inter = residual(inter, cnv_dim, 'inters.%d' % i)

    feature_maps = inter

    L2 = tf.keras.regularizers.l2(l=5e-4)
    
    b4 = tf.keras.layers.AveragePooling2D(pool_size=(32,32))(feature_maps)
    b4 = tf.keras.layers.Conv2D(64,(1,1),padding='same',use_bias=False)(b4)
    b4 = tf.keras.layers.BatchNormalization(epsilon=1e-5)(b4)
    b4 = mish(b4)
    b4 = tf.keras.layers.UpSampling2D((32,32),interpolation='bilinear')(b4)

    b0 = tf.keras.layers.Conv2D(64,(1,1),padding='same',use_bias=False)(feature_maps)     
    b0 = tf.keras.layers.BatchNormalization(epsilon=1e-5)(b0)
    b0 = mish(b0)
    
    
    b1 = tf.keras.layers.Conv2D(64,(3,3),strides=(1,1),dilation_rate=(6,6),padding='same',
                        use_bias=False)(feature_maps)
    b1 = tf.keras.layers.BatchNormalization(epsilon=1e-5)(b1)
    b1 = mish(b1)

    b2 = tf.keras.layers.Conv2D(64,(3,3),strides=(1,1),dilation_rate=(12,12),padding='same',
                        use_bias=False)(feature_maps)
    b2 = tf.keras.layers.BatchNormalization(epsilon=1e-5)(b2)
    b2 = mish(b2)
    
    b3 = tf.keras.layers.Conv2D(64,(3,3),strides=(1,1),dilation_rate=(18,18),padding='same',
                        use_bias=False)(feature_maps)
    b3 = tf.keras.layers.BatchNormalization(epsilon=1e-5)(b3)
    b3 = mish(b3)

    
    y0 = tf.concat([b4,b0,b1,b2,b3], axis=-1)
    
    
    y0 = tf.keras.layers.Conv2D(256,(3,3),strides=(1,1),padding='same', kernel_initializer='he_normal', kernel_regularizer=L2)(y0)
    y0 = tf.keras.layers.BatchNormalization()(y0)
    y0 = mish(y0)
    
    y0 = tf.keras.layers.Conv2D(128,(3,3),strides=(1,1),padding='same', kernel_initializer='he_normal', kernel_regularizer=L2)(y0)
    y0 = tf.keras.layers.BatchNormalization()(y0)
    y0 = mish(y0)  
    
    
    y0 = tf.keras.layers.UpSampling2D(size=(4, 4),interpolation='bilinear')(y0)

    
    y0 = tf.keras.layers.Conv2D(num_classes+1, 1, padding='same', kernel_initializer='he_normal', kernel_regularizer=L2)(y0)
    y0 = tf.keras.layers.Activation('softmax')(y0)    

    model = tf.keras.Model(inputs=image_input, outputs=y0)
   
    output_model = tf.keras.Model(inputs=image_input, outputs=y0)

    return model, output_model
###

def gaussian_radius_2(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)

def gaussian2D_2(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian_Ls(heatmap, center, radius, k=1, Ls=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D_2((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right] * Ls
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def create_model(h_,w_, num_classes):
    """create the training model"""
    K.clear_session()  # get a new session
    image_input = tf.keras.layers.Input(shape=(h_, w_, 3))

    model_body, output_model = centernet(image_input, input_size=h_,num_classes=num_classes)

    print(model_body.output)

    return model_body, output_model

train_model, prediction_model = create_model(h_,w_, num_classes)
logger.info("model has been created.")
prediction_model.load_weights(weight_PATH)
logger.info("weight has been loaded")

MIOU_list = []
IOU1 = []
IOU2 = []
IOU3 = []

non_ = np.zeros(7)
for z in range(len(test_id)):
    print(z)
    id_ = INDEX[test_id[z]]
    
    im_path = id_[2]

    img = cv2.imread(im_path)
    o_h,o_w,o_c = np.shape(img)

    img = cv2.resize(img, (512, 512),interpolation=cv2.INTER_NEAREST)
    img = np.array(img) / 255

    pred = prediction_model.predict_on_batch(np.expand_dims(img,axis=0))[0]
    pred_ = np.argmax(pred,axis=2)
    pred = np.zeros((512,512,3))

    pred[:,:,0][np.where(pred_ == 0)] = 1
    pred[:,:,1][np.where(pred_ == 1)] = 1
    pred[:,:,1][np.where(pred_ == 2)] = 1
    pred[:,:,2][np.where(pred_ == 3)] = 1
    pred[:,:,2][np.where(pred_ == 4)] = 1

    seg = an_list_[int(id_[0])][int(id_[1])]['segmentation']
    index_ = an_list_[int(id_[0])][int(id_[1])]['category_id'] 
    num_seg = len(seg)
    seg_ind = index_[len(index_)-num_seg:]

    seg_list = []
    x_ = []
    y_ = []
    for k in range(len(seg)):
        seg_xy = []
        for m in range(len(seg[k])):
            if m % 2 ==0:
                x = seg[k][m]
                x_.append(x)
            else :
                y = seg[k][m]
                y_.append(y)
                seg_xy.append([x,y])
        seg_list.append(seg_xy)

    seg_map = np.zeros((o_h,o_w), np.uint8)
    for s in range(len(seg_list)):
        pt = np.array(seg_list[s],np.int32)
        if len(pt) != 0:
            if seg_ind[s] == 1:
                seg_map = cv2.fillPoly(seg_map,[pt],(1))
            elif seg_ind[s] == 2 or seg_ind[s] == 3:
                seg_map = cv2.fillPoly(seg_map,[pt],(2))
            elif seg_ind[s] == 4 or seg_ind[s] == 5:
                seg_map = cv2.fillPoly(seg_map,[pt],(3))

    BG = skimage.transform.resize(seg_map,
                                    (512,512),
                                    mode='edge',
                                    anti_aliasing=False,
                                    anti_aliasing_sigma=None,
                                    preserve_range=True,
                                    order=0)

    truth = np.zeros((512,512,3))
    truth[:,:,0][np.where(BG == 1)] = 1
    truth[:,:,1][np.where(BG == 2)] = 1
    truth[:,:,2][np.where(BG == 3)] = 1


    IOU_ =  0
    count = 0
    for i in 0,1,2:
        union = np.sum(pred[:,:,i]) + np.sum(truth[:,:,i])
        inter = np.sum(pred[:,:,i] * truth[:,:,i])

        if np.sum(pred[:,:,i]) == 0:
            non_[i] += 1
        else:
            IOU = inter / (union - inter)
            IOU_ = IOU_ + IOU

            logger.info("Image Num : " + str(z) + ", " + "class_id : " + str(i+1) + ", " + "IOU : " + str(IOU))
            count +=1
            if i == 0:
                IOU1.append(IOU)
            elif i == 1:
                IOU2.append(IOU)
            elif i == 2:
                IOU3.append(IOU)

    if count == 0:
        pass
    else:
        logger.info("mIOU per image : " + str(IOU_/count))
        MIOU_list.append(IOU_/count)

IOU1_m = np.mean(IOU1)
IOU2_m = np.mean(IOU2)
IOU3_m = np.mean(IOU3)
mIOU_mean_per_image = np.mean(MIOU_list)
mIOU_mean_class = (IOU1_m+IOU2_m+IOU3_m)/3

logger.info("Class 1(Car) appearance frequency : " + str(round(len(IOU1)/len(test_id) * 100,2)) + "%")
logger.info("Class 2(Bus) appearance frequency : " + str(round(len(IOU2)/len(test_id) * 100,2)) + "%")
logger.info("Class 3(Truck) appearance frequency : " + str(round(len(IOU3)/len(test_id) * 100,2)) + "%")

logger.info("IOU 1 : " + str(IOU1_m))
logger.info("IOU 2 : " + str(IOU2_m))
logger.info("IOU 3 : " + str(IOU3_m))
logger.info("mean_MIOU per image : " + str(mIOU_mean_per_image))
logger.info("MIOU : " + str(mIOU_mean_class))


print(IOU1_m, IOU2_m, IOU3_m, mIOU_mean_per_image, mIOU_mean_class)

logger.info("end")
