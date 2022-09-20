"""
@author: cau_ybh
"""

import logging

import datetime
import time

import os

GPU_set = str(input("Set GPU : "))

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=GPU_set

import numpy as np
import tensorflow as tf
import keras.backend as K
import random
import sys
import h5py
from itertools import compress
import pickle

from PIL import Image, ImageFont, ImageDraw
from matplotlib import pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

import math
import cv2
import json

###

#num_classes = int(input("Set number of classes : "))
num_classes = 7

Ls_1 = 1 - 0.2 + 0.2/num_classes
Ls_0 = 0.2/num_classes

h_, w_ = 512, 512
input_shape = (h_, w_)

while True:
    PATH = os.path.abspath(str(input("Set Dir path : ")))
    #PATH = '/raid/YBH/NIA_DATA_NEW/'#!!!NIA_Det/'
    if os.path.isdir(PATH) == True:
        break

log_dir = os.path.join(PATH,'Detect') + os.sep
print('Weights were saved here : ' + log_dir)


while True:
    weight_PATH = os.path.abspath(log_dir + str(input("Set weight : ")))
    if os.path.isfile(weight_PATH) == True:
        break
        

print(weight_PATH)

###
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


INDEX = np.load(os.path.join(PATH,'index_bbox.npy'))
#im_path_list = np.load(os.path.join(PATH,'im_path_list_bbox.npy'))



train_id = np.load(PATH + os.sep + 'train_id_bbox.npy')
valid_id = np.load(PATH + os.sep + 'valid_id_bbox.npy')
test_id = np.load(PATH + os.sep + 'test_id_bbox.npy')


logger.info("test data has been loaded")
logger.info("length of test_id : " + str(len(test_id)))

test_image_root = INDEX[test_id,2]#im_path_list[test_id]

with open(os.path.join(PATH,'Detect','test_data_path.txt'), 'w') as f:
    for i in range(len(test_image_root)):
        f.write(test_image_root[i] + '\n')
        
logger.info("list of test data paths has been saved at Detect folder.")


print(len(test_id))
###    


def focal_loss(hm_pred, hm_true, Ls_1):
    Ls_1 = tf.convert_to_tensor(Ls_1, tf.float32)
    pos_mask = tf.cast(tf.equal(hm_true, Ls_1), tf.float32)#0.8285714285714286
    neg_mask = tf.cast(tf.less(hm_true, Ls_1), tf.float32)
    neg_weights = tf.pow(1 - hm_true, 4)

    pos_loss = -tf.log(tf.clip_by_value(hm_pred, 1e-4, 1. - 1e-4)) * tf.pow(1 - hm_pred, 2) * pos_mask
    neg_loss = -tf.log(tf.clip_by_value(1 - hm_pred, 1e-4, 1. - 1e-4)) * tf.pow(hm_pred, 2) * neg_weights * neg_mask

    num_pos = tf.reduce_sum(pos_mask)
    pos_loss = tf.reduce_sum(pos_loss)
    neg_loss = tf.reduce_sum(neg_loss)

    cls_loss = tf.cond(tf.greater(num_pos, 0), lambda: (pos_loss + neg_loss) / num_pos, lambda: neg_loss)
    return cls_loss

def reg_l1_loss(y_pred, y_true, indices, mask):
    b = tf.shape(y_pred)[0]
    k = tf.shape(indices)[1]
    c = tf.shape(y_pred)[-1]
    y_pred = tf.reshape(y_pred, (b, -1, c))
    indices = tf.cast(indices, tf.int32)
    y_pred = tf.gather(y_pred, indices, batch_dims=1)
    mask = tf.tile(tf.expand_dims(mask, axis=-1), (1, 1, 2))
    total_loss = tf.reduce_sum(tf.abs(y_true * mask - y_pred * mask))
    reg_loss = total_loss / (tf.reduce_sum(mask) + 1e-4)
    return reg_loss

def centernet_loss(args):
    hm_pred, wh_pred, reg_pred, hm_true, wh_true, reg_true, reg_mask, indices, Ls_1 = args
    hm_loss = focal_loss(hm_pred, hm_true,Ls_1)
    wh_loss = 0.1 * reg_l1_loss(wh_pred, wh_true, indices, reg_mask)
    reg_loss = reg_l1_loss(reg_pred, reg_true, indices, reg_mask)
    total_loss = hm_loss + wh_loss + reg_loss
    return total_loss

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

def centernet(image_input, input_size=512, max_objects=140, num_classes = 7, score_threshold=0.1, nms=False, Ls_1=1):

    num_stacks = 2
    cnv_dim=96
    dims=[128, 128, 192, 256]
    
    output_size = input_size // 4

    hm_input = tf.keras.layers.Input(shape=(output_size,output_size,num_classes))
    wh_input = tf.keras.layers.Input(shape=(max_objects, 2))
    reg_input = tf.keras.layers.Input(shape=(max_objects, 2))
    reg_mask_input = tf.keras.layers.Input(shape=(max_objects,))
    index_input = tf.keras.layers.Input(shape=(max_objects,))

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
            inter = residual(inter, cnv_dim, 'inters.%d' % i)

    feature_maps = inter

    L2 = tf.keras.regularizers.l2(l=5e-4)
    
    y0 = tf.keras.layers.Conv2D(output_size, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=L2)(feature_maps)
    y0 = tf.keras.layers.BatchNormalization()(y0)
    y0 = tf.nn.leaky_relu(y0, alpha=0.1)
    
    y1 = tf.keras.layers.Conv2D(num_classes, 1, kernel_initializer='he_normal', kernel_regularizer=L2, activation='sigmoid')(y0)

    y2 = tf.keras.layers.Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=L2)(y0)
   
    y3 = tf.keras.layers.Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=L2)(y0)    
    
    cnt_loss = tf.keras.layers.Lambda(centernet_loss, output_shape=(1,), name='centernet_loss')
    #([y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input])
    loss_ = cnt_loss([y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input, Ls_1])

    model = tf.keras.Model(inputs=[image_input, hm_input, wh_input, reg_input, reg_mask_input, index_input], outputs=[loss_])

    detections = tf.keras.layers.Lambda(lambda x: decode(*x,
                                         max_objects=max_objects,
                                         score_threshold=score_threshold,
                                         nms=nms,
                                         num_classes=num_classes))([y1, y2, y3])
    prediction_model = tf.keras.Model(inputs=image_input, outputs=detections)
    debug_model = tf.keras.Model(inputs=image_input, outputs=[y1, y2, y3])
    return model, prediction_model, debug_model


def evaluate_batch_item(batch_item_detections, num_classes, max_objects_per_class=140, max_objects=140,
                        iou_threshold=0.35, score_threshold=0.15):
    batch_item_detections = tf.boolean_mask(batch_item_detections,
                                            tf.greater(batch_item_detections[:, 4], score_threshold))
    detections_per_class = []
    for cls_id in range(num_classes):
        class_detections = tf.boolean_mask(batch_item_detections, tf.equal(batch_item_detections[:, 5], cls_id))
        nms_keep_indices = tf.image.non_max_suppression(class_detections[:, :4],
                                                        class_detections[:, 4],
                                                        max_objects_per_class,
                                                        iou_threshold=iou_threshold)
        class_detections = K.gather(class_detections, nms_keep_indices)
        detections_per_class.append(class_detections)

    batch_item_detections = K.concatenate(detections_per_class, axis=0)

    def filter():
        nonlocal batch_item_detections
        _, indices = tf.nn.top_k(batch_item_detections[:, 4], k=max_objects)
        batch_item_detections_ = tf.gather(batch_item_detections, indices)
        return batch_item_detections_

    def pad():
        nonlocal batch_item_detections
        batch_item_num_detections = tf.shape(batch_item_detections)[0]
        batch_item_num_pad = tf.maximum(max_objects - batch_item_num_detections, 0)
        batch_item_detections_ = tf.pad(tensor=batch_item_detections,
                                        paddings=[
                                            [0, batch_item_num_pad],
                                            [0, 0]],
                                        mode='CONSTANT',
                                        constant_values=0.0)
        return batch_item_detections_

    batch_item_detections = tf.cond(tf.shape(batch_item_detections)[0] >= 100,
                                    filter,
                                    pad)
    return batch_item_detections

def nms(heat, kernel=3):
    hmax = tf.nn.max_pool2d(heat, (kernel, kernel), strides=1, padding='SAME')
    heat = tf.where(tf.equal(hmax, heat), heat, tf.zeros_like(heat))
    return heat

def topk(hm, max_objects=140):
    hm = nms(hm)
    # (b, h * w * c)
    b, h, w, c = tf.shape(hm)[0], tf.shape(hm)[1], tf.shape(hm)[2], tf.shape(hm)[3]
    # hm2 = tf.transpose(hm, (0, 3, 1, 2))
    # hm2 = tf.reshape(hm2, (b, c, -1))
    hm = tf.reshape(hm, (b, -1))
    # (b, k), (b, k)
    scores, indices = tf.nn.top_k(hm, k=max_objects)
    # scores2, indices2 = tf.nn.top_k(hm2, k=max_objects)
    # scores2 = tf.reshape(scores2, (b, -1))
    # topk = tf.nn.top_k(scores2, k=max_objects)
    class_ids = indices % c
    xs = indices // c % w
    ys = indices // c // w
    indices = ys * w + xs
    return scores, indices, class_ids, xs, ys

def decode(hm, wh, reg, max_objects=140, nms=True, num_classes=7, score_threshold=0.05):

    scores, indices, class_ids, xs, ys = topk(hm, max_objects=max_objects)
    b = tf.shape(hm)[0]
    # (b, h * w, 2)
    reg = tf.reshape(reg, (b, -1, tf.shape(reg)[-1]))
    # (b, h * w, 2)
    wh = tf.reshape(wh, (b, -1, tf.shape(wh)[-1]))
    # (b, k, 2)
    topk_reg = tf.gather(reg, indices, batch_dims=1)
    # (b, k, 2)
    topk_wh = tf.cast(tf.gather(wh, indices, batch_dims=1), tf.float32)
    topk_cx = tf.cast(tf.expand_dims(xs, axis=-1), tf.float32) + topk_reg[..., 0:1]
    topk_cy = tf.cast(tf.expand_dims(ys, axis=-1), tf.float32) + topk_reg[..., 1:2]
    scores = tf.expand_dims(scores, axis=-1)
    class_ids = tf.cast(tf.expand_dims(class_ids, axis=-1), tf.float32)
    topk_x1 = topk_cx - topk_wh[..., 0:1] / 2
    topk_x2 = topk_cx + topk_wh[..., 0:1] / 2
    topk_y1 = topk_cy - topk_wh[..., 1:2] / 2
    topk_y2 = topk_cy + topk_wh[..., 1:2] / 2
    # (b, k, 6)
    detections = tf.concat([topk_x1, topk_y1, topk_x2, topk_y2, scores, class_ids], axis=-1)
    if nms:
        detections = tf.map_fn(lambda x: evaluate_batch_item(x[0],
                                                             num_classes=num_classes,
                                                             score_threshold=score_threshold),
                               elems=[detections],
                               dtype=tf.float32)
    return detections


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

def data_generator(length, batch_size, input_shape, h_, num_classes):
    '''data generator for fit_generator'''
    n = len(length)
    i = 0
    while True:
        image_data = []
        hms_data = []
        whs_data = []
        regs_data = []
        regmk_data = []
        indi_data = []
        for b_ in range(batch_size):
            if i==0:
                np.random.shuffle(length)

            id_ = INDEX[length[i]]
            bbox_ = an_list[id_]['bbox']

            index_ = an_list[id_]['category_id']       

            im_path = im_path_list[length[i]]
            img = cv2.imread(im_path)

            o_h,o_w,o_c = np.shape(img)


            i_w = random.randint(h_, o_w)
            i_h = random.randint(h_, o_h)
            img = cv2.resize(img, (i_w, i_h),interpolation=cv2.INTER_NEAREST)


            rate_w, rate_h = o_w / i_w, o_h / i_h


            crop_w = random.randint(0,i_w-h_)
            crop_h = random.randint(0,i_h-h_)    

            img = cv2.resize(img, (i_w, i_h),interpolation=cv2.INTER_NEAREST)
            img = img[crop_h:crop_h+h_,crop_w:crop_w+h_] / 255    


            bbox_n = []
            for b in range(len(bbox_)):
                x_min, y_min = bbox_[b][0]/rate_w - crop_w,bbox_[b][1]/rate_h - crop_h
                x_max, y_max = bbox_[b][2]/rate_w - crop_w,bbox_[b][3]/rate_h - crop_h

                x_min = np.maximum(0,x_min)
                y_min = np.maximum(0,y_min)
                x_max = np.minimum(h_,x_max)
                y_max = np.minimum(h_,y_max)

                bbox_n.append([x_min,y_min,x_max,y_max,index_[b]])

                if x_max < 0 or y_max < 0 or y_min > h_ or x_min > h_:
                    del bbox_n[-1]

            bbox_n = np.array(bbox_n)

            batch_hms = np.zeros((h_//4,h_//4,num_classes), dtype=np.float32)
            batch_whs = np.zeros((140, 2), dtype=np.float32)
            batch_regs = np.zeros((140, 2), dtype=np.float32)
            batch_reg_masks = np.zeros((140), dtype=np.float32)
            batch_indices = np.zeros((140), dtype=np.float32)


            for b in range(len(bbox_n)):

                bo = np.copy(bbox_n[b])

                bo_w = bo[2] - bo[0]
                bo_h = bo[3] - bo[1]

                bo_cx = (bo[0] + bo[2]) /2
                bo_cy = (bo[1] + bo[3]) /2

                cls_id = bo[4]

                x_c = bo_cx /4
                y_c = bo_cy /4
                w = bo_w /4
                h = bo_h /4     

                radius = gaussian_radius_2((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array([x_c,y_c], dtype=np.float32)
                ct_int = ct.astype(np.int32)

                for cls_ in range(num_classes):
                    if cls_id == cls_+1:
                        batch_hms[:,:,cls_] = draw_gaussian_Ls(batch_hms[:,:,cls_], ct_int, radius, 1, Ls_1)
                    else:
                        batch_hms[:,:,cls_] = draw_gaussian_Ls(batch_hms[:,:,cls_], ct_int, radius, 1, Ls_0)


                batch_whs[b] = 1. * w, 1. * h
                batch_indices[b] = ct_int[1] * 128 + ct_int[0]
                batch_regs[b] = np.reshape(ct - ct_int, (2,))
                batch_reg_masks[b] = 1                     
                
                
            image_data.append(img)
            hms_data.append(batch_hms)
            whs_data.append(batch_whs)
            regs_data.append(batch_regs)
            regmk_data.append(batch_reg_masks)
            indi_data.append(batch_indices)
            i = (i+1) % n
            
        image_data = np.array(image_data)
        hms_data = np.array(hms_data)
        whs_data = np.array(whs_data)
        regs_data = np.array(regs_data)
        regmk_data = np.array(regmk_data)
        indi_data = np.array(indi_data)
        
        yield [image_data, hms_data, whs_data, regs_data, regmk_data, indi_data], np.zeros(batch_size)

def data_generator_wrapper(length, batch_size, input_shape, h_, num_classes):
    n = len(length)
    if n==0 or batch_size<=0: return None
    return data_generator(length, batch_size, input_shape, h_, num_classes)


def create_model(h_,w_, num_classes):
    """create the training model"""
    K.clear_session()  # get a new session
    image_input = tf.keras.layers.Input(shape=(h_, w_, 3))

    model_body, prediction_model, debug_model = centernet(image_input, input_size=h_, max_objects=140, num_classes=num_classes,
                score_threshold=0.15, nms=False, Ls_1=Ls_1)

    print(model_body.output)

    return model_body, prediction_model, debug_model

def ground_truth_cal(class_num=7):
    
    gt_num = np.zeros(class_num)

    for z in range(len(test_id)):

        id_ = INDEX[test_id[z]]
        bbox_ = np.array(an_list_[int(id_[0])][int(id_[1])]['bbox'])

        index_ = np.array(an_list_[int(id_[0])][int(id_[1])]['category_id'][:len(bbox_)])

        for n in range(class_num):

            b_num = len(bbox_[index_==n+1])

            gt_num[n] += b_num

    return gt_num

def AP_cal(target):
    pr0 = target[:,6]
    re0 = target[:,7]
    check_point = 0
    re_list = []

    memory = pr0[0]
    re_start = 0
    
    for i in range(len(pr0)):
    
        if pr0[i] == 0:
            re_start = 0
            memory = pr0[i]

        if pr0[i] == 1:
            re_end = re0[i]
            memory = pr0[i]

            if i == len(pr0)-1:
                re_ = re_end - re_start
                re_start = re_end
                check_point = 1
                re_list.append((i,memory, re_, re_start))
                
        if pr0[i] < memory:

            if check_point == 0:
                re_ = re_end - re_start
                re_start = re_end
                check_point = 1
                re_list.append((i,memory, re_, re_start))

            memory = pr0[i]

        if pr0[i] >= memory:
            re_end = re0[i]
            memory = pr0[i]
            check_point = 0

                
        if i == len(pr0) - 1:
                
            re_ = re_end - re_start
            re_start = re_end
            check_point = 1
            re_list.append((i,memory, re_, re_start))                
            
    re_list_ = np.asarray(re_list)

    AP = []

    for i, re_ in enumerate(re_list_):
        if i == 0:
            area = re_[1] * re_[2]
            AP.append((re_[0],area))
            start_point = re_[3]
        else:
            if re_[1] == np.amax(re_list_[i:,1]):
                area = re_[1] * (re_[3]-start_point)
                AP.append((re_[0],area))
                start_point = re_[3]


    AP_ = np.asarray(AP)
    return np.sum(AP_[:,1])



#log_dir = '/raid/YBH/NIA_DATA_NEW/!!!NIA_Det/'

train_model, prediction_model, debug_model = create_model(h_,w_, num_classes)
logger.info("model has been created.")
prediction_model.load_weights(weight_PATH)
logger.info("weight has been loaded")


GT = ground_truth_cal() #car, sbus, lbus, truck, trailer, bic, people
GT_ = np.array((GT[0],GT[1]+GT[2],GT[3]+GT[4]))
print("Number of Ground Truth Bbox in test data :",GT)
print("Class integration for test,", "[ Car :", int(GT_[0]), " Bus :", int(GT_[1]), " Truck :", int(GT_[2]), "]")

logger.info("Number of GT Bbox in test data, Class 1(Car) : " + str(int(GT_[0])))
logger.info("Number of GT Bbox in test data, Class 2(Bus) : " + str(int(GT_[1])))
logger.info("Number of GT Bbox in test data, Class 3(Truck) : " + str(int(GT_[2])))

TorF_list = []

print("IOU Matching")
logger.info("IOU Matching Start")

for z in range(len(test_id)):
    if z%1000 == 0:
        print(str(int(z/len(test_id)*100))+"%")
                
    ##
    id_ = INDEX[test_id[z]]
    ##
    
    ####
    im_path = id_[2]
    img = cv2.imread(im_path)
    img_ = cv2.resize(img, (512, 512),interpolation=cv2.INTER_NEAREST)
    img_ = img_/ 255
    o_h, o_w, o_c = np.shape(img)
    rate_w, rate_h = o_w / 512, o_h / 512

    detections = prediction_model.predict_on_batch(np.expand_dims(img_,axis=0))[0]

    scores = detections[:,4]
    indices = np.where(scores > 0)[0]
    detections_ = detections[indices]
    detections_[:,0:4] = detections_[:,0:4] * 4

    out_boxes = detections_[:, 0:4]
    out_scores = detections_[:,4]
    out_classes = detections_[:,5] + 1
    
    out_classes = np.where(out_classes == 3, 2, out_classes)
    out_classes = np.where(out_classes == 4, 3, out_classes)
    out_classes = np.where(out_classes == 5, 3, out_classes)
    out_classes = np.where(out_classes == 6, 4, out_classes)
    out_classes = np.where(out_classes == 7, 4, out_classes)
    

    out_boxes[:,0] = out_boxes[:,0] * rate_w
    out_boxes[:,1] = out_boxes[:,1] * rate_h
    out_boxes[:,2] = out_boxes[:,2] * rate_w
    out_boxes[:,3] = out_boxes[:,3] * rate_h
    out_boxes = np.where(out_boxes < 0 , 0, out_boxes)
    ####
    
    ####
    truth_boxes = np.array(an_list_[int(id_[0])][int(id_[1])]['bbox'])
    truth_classes = np.array(an_list_[int(id_[0])][int(id_[1])]['category_id'][:len(truth_boxes)])  
    truth_boxes = np.where(truth_boxes < 0, 0, truth_boxes)
    
    truth_classes = np.where(truth_classes == 3, 2, truth_classes)
    truth_classes = np.where(truth_classes == 4, 3, truth_classes)
    truth_classes = np.where(truth_classes == 5, 3, truth_classes)
    truth_classes = np.where(truth_classes == 6, 4, truth_classes)
    truth_classes = np.where(truth_classes == 7, 4, truth_classes)
    
    ####
   
    
    if len(out_boxes) == 0:
        TorF = np.zeros((1,8))
    else:
        TorF = np.zeros((len(out_boxes),8))
        TorF[:,0] = out_classes
        TorF[:,1] = out_scores

        
    matched_list = []
    for m in range(len(out_boxes)):
        for p in range(len(truth_boxes)):
            if p in matched_list:
                #print(z,m,p)
                continue
            else:
                xx1 = np.maximum(out_boxes[m,0], truth_boxes[p,0])
                yy1 = np.maximum(out_boxes[m,1], truth_boxes[p,1])
                xx2 = np.minimum(out_boxes[m,2], truth_boxes[p,2])
                yy2 = np.minimum(out_boxes[m,3], truth_boxes[p,3])
                w = np.maximum(0., xx2 - xx1)
                h = np.maximum(0., yy2 - yy1)
                wh = w * h
                iou = wh / ((out_boxes[m,2]-out_boxes[m,0])*(out_boxes[m,3]-out_boxes[m,1])
                            + (truth_boxes[p,2]-truth_boxes[p,0])*(truth_boxes[p,3]-truth_boxes[p,1]) - wh)
                    
                if iou >= 0.5:
                    if out_classes[m] == truth_classes[p]:
                        TorF[m,2] = 1
                        matched_list.append(p)
                    else:
                        continue
    TorF_list.append(TorF)
    #logger.info("Image Num : " + str(z) + ", " + "True or False array : " + str(TorF))

logger.info("IOU matching finish")


TorF_list_ = np.array(TorF_list)
TorF_list_ = np.reshape(TorF_list_,(np.shape(TorF_list_)[0]*np.shape(TorF_list_)[1],8))
TorF_list_i = np.argsort(TorF_list_[:,1])[::-1]
TorF_list_ = TorF_list_[TorF_list_i]

TorF_L = np.copy(TorF_list_)

logger.info("True or Flase lists has been sorted")


logger.info("calculate Precision Recall Matrix per class")
TorF_T = []
for n in range(3):
    TorF_n = TorF_L[TorF_L[:,0] == n+1]
    
    for i in range(len(TorF_n)):
        if TorF_n[i,2] == 1:
            if i == 0:
                TorF_n[i,3] = 1
            else:
                TorF_n[i,3] = TorF_n[i-1,3] + 1
                TorF_n[i,4] = TorF_n[i-1,4]
                
        elif TorF_n[i,2] == 0:
            if i == 0:
                TorF_n[i,4] = 0
            else:
                TorF_n[i,3] = TorF_n[i-1,3]
                TorF_n[i,4] = TorF_n[i-1,4] + 1
    
    for i in range(len(TorF_n)):
        TorF_n[i,5] = TorF_n[i,3] + TorF_n[i,4]
        TorF_n[i,7] = TorF_n[i,3] / GT_[n]
    for i in range(len(TorF_n)):
        TorF_n[i,6] = TorF_n[i,3] / TorF_n[i,5]
    TorF_T.append(TorF_n)

TorF_1 = TorF_T[0]
logger.info("PR Matrix Class 1")
logger.info("Shape of PR Matrix Class 1 : " + str(np.shape(TorF_1)))
logger.info(TorF_1)

np.save(log_dir + os.sep + 'PR_Matrix_1.npy', TorF_1) 
#with open(log_dir + os.sep + 'PR_Matrix_1.pkl', 'wb') as f:
#    pickle.dump(TorF_1, f, pickle.HIGHEST_PROTOCOL)

logger.info("PR Matrix Class 1 has been saved")


TorF_2 = TorF_T[1]
logger.info("PR Matrix Class 2")
logger.info("Shape of PR Matrix Class 2 : " + str(np.shape(TorF_2)))
logger.info(TorF_2)

np.save(log_dir + os.sep + 'PR_Matrix_2.npy', TorF_2) 
#with open(log_dir + os.sep + 'PR_Matrix_2.pkl', 'wb') as f:
#    pickle.dump(TorF_2, f, pickle.HIGHEST_PROTOCOL)

logger.info("PR Matrix Class 2 has been saved")

TorF_3 = TorF_T[2]
logger.info("PR Matrix Class 3")
logger.info("Shape of PR Matrix Class 3 : " + str(np.shape(TorF_3)))
logger.info(TorF_3)

np.save(log_dir + os.sep + 'PR_Matrix_3.npy', TorF_3) 
#with open(log_dir + os.sep + 'PR_Matrix_3.pkl', 'wb') as f:
#    pickle.dump(TorF_3, f, pickle.HIGHEST_PROTOCOL)

logger.info("PR Matrix Class 3 has been saved")

print("Complete")
logger.info("Precision Recall Matrix has been calculated")


print("AP calculate")
logger.info("Start AP calculate")

AP_1 = AP_cal(TorF_1)
AP_2 = AP_cal(TorF_2)
AP_3 = AP_cal(TorF_3)

logger.info("AP per class has been calculated")

print("Complete")
print("AP - Car :",AP_1)
logger.info("AP 1 : " + str(AP_1))

print("AP - Bus :",AP_2)
logger.info("AP 2 : " + str(AP_2))

print("AP - Truck :",AP_3)
logger.info("AP 3 : " + str(AP_3))

print("mAP :",np.mean((AP_1,AP_2,AP_3)))
logger.info("mAP : " + str(np.mean((AP_1,AP_2,AP_3))))
