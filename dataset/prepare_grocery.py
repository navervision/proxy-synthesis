'''
proxy-synthesis
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
import os
import shutil

base_path = './grocery'
trainPrefix = os.path.join(base_path, 'groceryDB/train/')
testPrefix = os.path.join(base_path, 'groceryDB/test/')
for lines in open(os.path.join(base_path, 'train.txt')):
    lines = lines.strip().split(',')
    classInd = int(lines[1])
    classInd = classInd + 1 #1 indexing it
    print(lines[0])
    try:
        fname = lines[0].split('/')[4]
    except IndexError:
        fname = lines[0].split('/')[3]
    file_path = os.path.join(base_path, lines[0])
    if classInd <= 41:
        ddr = trainPrefix + str(classInd)
        if not os.path.exists(ddr):
            os.makedirs(ddr)
        newfname = fname.split('.')[0] + 'train' + '.jpg'
        shutil.move(file_path, ddr + '/' + newfname)
    else:
        ddr = testPrefix + str(classInd)
        if not os.path.exists(ddr):
            os.makedirs(ddr)
        newfname = fname.split('.')[0] + 'train' + '.jpg'
        shutil.move(file_path, ddr + '/' + newfname)

for lines in open(os.path.join(base_path, 'test.txt')):
    lines = lines.strip().split(',')
    classInd = int(lines[1])
    classInd = classInd + 1 #1 indexing it
    print(lines[0])
    try:
        fname = lines[0].split('/')[4]
    except IndexError:
        fname = lines[0].split('/')[3]
    file_path = os.path.join(base_path, lines[0])
    if classInd <= 41:
        ddr = trainPrefix + str(classInd)
        if not os.path.exists(ddr):
            os.makedirs(ddr)
        newfname = fname.split('.')[0] + 'test' + '.jpg'
        shutil.move(file_path, ddr + '/' + newfname)
    else:
        ddr = testPrefix + str(classInd)
        if not os.path.exists(ddr):
            os.makedirs(ddr)
        newfname = fname.split('.')[0] + 'test' + '.jpg'
        shutil.move(file_path, ddr + '/' + newfname)


for lines in open(os.path.join(base_path, 'val.txt')):
    lines = lines.strip().split(',')
    classInd = int(lines[1])
    classInd = classInd + 1 #1 indexing it
    print(lines[0])
    try:
        fname = lines[0].split('/')[4]
    except IndexError:
        fname = lines[0].split('/')[3]
    file_path = os.path.join(base_path, lines[0])
    if classInd <= 41:
        ddr = trainPrefix + str(classInd)
        if not os.path.exists(ddr):
            os.makedirs(ddr)
        newfname = fname.split('.')[0] + 'val' + '.jpg'
        shutil.move(file_path, ddr + '/' + newfname)
    else:
        ddr = testPrefix + str(classInd)
        if not os.path.exists(ddr):
            os.makedirs(ddr)
        newfname = fname.split('.')[0] + 'val' + '.jpg'
        shutil.move(file_path, ddr + '/' + newfname)

#the dataset was divided into 81 fine grained classes (what we need) and 42 course grained classes (don't need)
#zero indexed, so made it one indexed
#train, test and val where separated in the dataset and had similar naming. so combined it and changed names.
#data can be downloaded from https://github.com/marcusklasson/GroceryStoreDataset

