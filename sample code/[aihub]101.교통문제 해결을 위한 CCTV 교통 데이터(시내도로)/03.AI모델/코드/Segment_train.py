"""
@author: cau_ybh
"""

import os
import logging
import time
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

num_classes = 7#int(input("Set number of classes : "))
#num_classes = num_classes + 1

h_, w_ = 512, 512
input_shape = (h_, w_)

batch_size = int(input("Set number of train batch : "))
#batch_size = 8
epoch = int(input("Set epoch : "))
#epoch = 100

while True:
    PATH = os.path.abspath(str(input("Set Dir path : ")))
    if os.path.isdir(PATH) == True:
        break

log_dir = os.path.join(PATH,'Segmentation') + os.sep
print('Weight will save here : ' + log_dir)
###


logger = logging.getLogger()

logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

file_handler = logging.FileHandler(os.path.join(log_dir,"train.log"))
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
   
try:
    train_id = np.load(os.path.join(PATH,'train_id_seg.npy'))
    valid_id = np.load(os.path.join(PATH,'valid_id_seg.npy'))
    test_id = np.load(os.path.join(PATH,'test_id_seg.npy'))
except:
    id__ = np.arange(len(INDEX))
    np.random.shuffle(id__)

    train_id = id__[:int(len(id__)*0.7)]
    valid_id = id__[int(len(id__)*0.7):int(len(id__)*0.85)]
    test_id = id__[int(len(id__)*0.85):]
    np.save(os.path.join(PATH,'train_id_seg.npy'),train_id)
    np.save(os.path.join(PATH,'valid_id_seg.npy'),valid_id)
    np.save(os.path.join(PATH,'test_id_seg.npy'),test_id)


print(len(train_id), len(valid_id), len(test_id))
logger.info("length of train_id : " + str(len(train_id)))
logger.info("length of valid_id : " + str(len(valid_id)))
logger.info("length of test_id : " + str(len(test_id)))
###    

def Miou(y_true, y_pred):
        
    truth = tf.reduce_sum(tf.reduce_sum(y_true,axis=1),axis=1)
    truth = tf.where(truth > 0, truth * 0 + 1, truth * 0)
    num_ = tf.reduce_sum(truth)
        
    inter = y_true * y_pred
    inter = tf.reduce_sum(inter, axis=1)
    inter = tf.reduce_sum(inter, axis=1)
    uni = y_true + y_pred
    uni = tf.reduce_sum(uni, axis=1)
    uni = tf.reduce_sum(uni, axis=1)
    iou = tf.math.divide_no_nan(inter,(uni - inter))
    iou = tf.reduce_sum(iou) / num_
    return iou

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

    image_input = tf.keras.layers.Input([512,512,3])

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
                                                                                                                                   #y0 = tf.keras.layers.Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=L2)(y0)
    model = tf.keras.Model(inputs=image_input, outputs=y0)
   
    output_model = tf.keras.Model(inputs=image_input, outputs=y0)

    return model, output_model
###


def data_generator(length, batch_size, input_shape, h_, num_classes):
    '''data generator for fit_generator'''
    n = len(length)
    i = 0
    while True:
        image_data = []
        index_data = []
        for b_ in range(batch_size):
            if i==0:
                np.random.shuffle(length)

            id_ = INDEX[length[i]]
            seg = an_list_[int(id_[0])][int(id_[1])]['segmentation']

            index_ = an_list_[int(id_[0])][int(id_[1])]['category_id']       
            num_seg = len(seg)
            seg_ind = index_[len(index_)-num_seg:]

            im_path = id_[2]
            img = cv2.imread(im_path)

            o_h,o_w,o_c = np.shape(img)


            i_w = random.randint(h_, o_w)
            i_h = random.randint(h_, o_h)
            img = cv2.resize(img, (i_w, i_h),interpolation=cv2.INTER_NEAREST)

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

            rate_w, rate_h = o_w / i_w, o_h / i_h

            x_min, x_max = np.round(np.min(x_)/rate_w), np.round(np.max(x_)/rate_w)
            y_min, y_max = np.round(np.min(y_)/rate_h), np.round(np.max(y_)/rate_h)
            
            x_min = np.maximum(x_min,0)
            y_min = np.maximum(y_min,0)

            if x_min > i_w-512:
                crop_w = random.randint(0,i_w-512)
            else:
                crop_w = random.randint(x_min,i_w-512)

            if y_min > i_h-512:
                crop_h = random.randint(0,i_h-512)
            else:
                crop_h = random.randint(y_min,i_h-512)

            img = img[crop_h:crop_h+512,crop_w:crop_w+512] / 255


            seg_map = np.zeros((o_h,o_w), np.uint8)
            for s in range(len(seg_list)):
                pt = np.array(seg_list[s],np.int32)
                if len(pt) != 0:
                    for cls_ in range(num_classes):
                        if seg_ind[s] == cls_ + 1:
                            seg_map = cv2.fillPoly(seg_map,[pt],(cls_+1))                        
                       
            seg_map_ = skimage.transform.resize(seg_map,
                                            (i_h,i_w),
                                            mode='edge',
                                            anti_aliasing=False,
                                            anti_aliasing_sigma=None,
                                            preserve_range=True,
                                            order=0)

            seg_map_ = seg_map_[crop_h:crop_h+512,crop_w:crop_w+512]

            img2 = np.zeros((512,512,num_classes+1))

            for cls_ in range(num_classes):
                img2[:,:,cls_][np.where(seg_map_ == cls_+1)] = 1
            img2[:,:,-1][np.where(seg_map_ == 0)] = 1

            
            image_data.append(img)
            index_data.append(img2)
            i = (i+1) % n
        image_data = np.array(image_data)
        index_data = np.array(index_data)
        
        yield image_data, index_data

def data_generator_wrapper(length, batch_size, input_shape, h_, num_classes):
    n = len(length)
    if n==0 or batch_size<=0: return None
    return data_generator(length, batch_size, input_shape, h_, num_classes)


def create_model(h_,w_, num_classes):
    """create the training model"""
    K.clear_session()  # get a new session
    image_input = tf.keras.layers.Input(shape=(h_, w_, 3))

    model_body, output_model = centernet(image_input, input_size=h_,num_classes=num_classes)

    print(model_body.output)

    return model_body, output_model


#log_dir = '/raid/YBH/NIA_DATA_NEW/!!!NIA_Det/'

log_dir_ = log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

Adam = tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='Adam')

train_model, output_model = create_model(h_,w_, num_classes)
logging = tf.keras.callbacks.TensorBoard(log_dir=log_dir_)

checkpoint = tf.keras.callbacks.ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                             monitor='val_Miou', mode='max',save_weights_only=True, save_best_only=True, period=1)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_Miou', factor=0.1, patience=3, verbose=1, min_lr=1e-5)


train_model.compile(optimizer=Adam, loss=tf.keras.losses.BinaryCrossentropy(), metrics=[Miou])
print('compile success')
logger.info("model compile success")

logger.info("start training")


train_model.fit_generator(data_generator_wrapper(train_id, batch_size, input_shape, h_, num_classes),
    steps_per_epoch=max(1, len(train_id)//batch_size),
    validation_data=data_generator_wrapper(valid_id, batch_size, input_shape, h_, num_classes),
    validation_steps=max(1, len(valid_id)//batch_size),
    epochs=epoch,
    callbacks=[logging, checkpoint, reduce_lr])

train_model.save_weights(log_dir + 'trained_weights.h5')

logger.info("model training completed")
logger.info("end")





