'''
proxy-synthesis
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
import os
import shutil

base_path = './cars'
trainPrefix = os.path.join(base_path, 'carDB/train/')
testPrefix = os.path.join(base_path, 'carDB/test/')
for lines in open(os.path.join(base_path, 'cars_annos.txt')):
    lines = lines.strip().split(',')
    classInd = int(lines[1])
    fname = lines[0].split('/')[1]
    file_path = os.path.join(base_path, lines[0])
    if classInd <= 98:
        ddr = trainPrefix + str(classInd)
        if not os.path.exists(ddr):
            os.makedirs(ddr)
        shutil.move(file_path, ddr + '/' + fname)
    else:
        ddr = testPrefix + lines[1]
        if not os.path.exists(ddr):
            os.makedirs(ddr)
        shutil.move(file_path, ddr + '/' + fname)

#download the tar of all images combined from https://ai.stanford.edu/~jkrause/cars/car_dataset.html
#download the txt file from the same link
#all of the above inside cars folder
# run this file from outside the cars folder