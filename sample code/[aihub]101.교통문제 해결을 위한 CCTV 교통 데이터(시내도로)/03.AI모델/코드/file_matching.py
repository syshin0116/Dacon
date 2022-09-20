"""
@author: cau_ybh
"""

import json
import numpy as np
import os

ab_path = os.path.abspath(str(input("Set Dir path : ")))


json_list = []
for (path, dir, files) in os.walk(os.path.join(ab_path,'JSON')):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        if ext == '.json':
            json_list.append(os.path.abspath("%s/%s" % (path, filename)))

file_name_list = []

for (path, dir, files) in os.walk(ab_path):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        if ext == '.jpg':
            #print("%s/%s" % (path, filename))
            
            file_name_list.append(path.split('/')[-1] + os.sep + filename)
            file_name_list.append(path + os.sep + filename)

Index_fine = []
Index_del = []
#im_path_list = []

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
    Index_ = []
    for i in range(len(an_list)):
            bbox = an_list[i]['bbox']
            if bbox != []:
                    Index_.append(i)

    for i in Index_:
        #print(i)
        try:
            id__ = file_name_list.index(im_list[i]['file_name'])
            
            file_path_ = file_name_list[id__+1]#ab_path + os.sep + i_dir + os.sep + i_img
            os.path.isfile(file_path_)
            #print(im_list[i]['file_name'],file_path_)
            Index_fine.append((n_,i,file_path_))
            print(n_,i)
        except:
            Index_del.append((n_,i))
            print(n_,i,'Fail')

np.save(os.path.join(ab_path,"index_bbox.npy"),Index_fine)


Index_fine = []
Index_del = []
#im_path_list = []

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
    Index_ = []
    for i in range(len(an_list)):
            seg = an_list[i]['segmentation']
            if seg != []:
                    Index_.append(i)

    for i in Index_:
        #print(i)
        try:
            id__ = file_name_list.index(im_list[i]['file_name'])
            
            file_path_ = file_name_list[id__+1]#ab_path + os.sep + i_dir + os.sep + i_img
            os.path.isfile(file_path_)
            #print(im_list[i]['file_name'],file_path_)
            Index_fine.append((n_,i,file_path_))
            print(n_,i)
        except:
            Index_del.append((n_,i))
            print(n_,i,'Fail')

np.save(os.path.join(ab_path,"index_seg.npy"),Index_fine)
