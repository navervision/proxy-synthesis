import os
import shutil

base_path = './data/veg_fru'
trainPrefix = os.path.join(base_path, 'vegfruDB/train/')
testPrefix = os.path.join(base_path, 'vegfruDB/test/')
valPrefix = os.path.join(base_path, 'vegfruDB/val/')

for lines in open(os.path.join(base_path, 'vegfru_list/vegfru_train.txt')):
    lines = lines.strip().split(' ')
    classInd = int(lines[1])
    fname = lines[0].split('/')[2]
    file_path = os.path.join(base_path, lines[0])
    ddr = trainPrefix + str(classInd)
    if not os.path.exists(ddr):
        os.makedirs(ddr)
    shutil.move(file_path, ddr + '/' + fname)


for lines in open(os.path.join(base_path, 'vegfru_list/vegfru_test.txt')):
    lines = lines.strip().split(' ')
    classInd = int(lines[1])
    fname = lines[0].split('/')[2]
    file_path = os.path.join(base_path, lines[0])
    ddr = testPrefix + lines[1]
    if not os.path.exists(ddr):
        os.makedirs(ddr)
    shutil.move(file_path, ddr + '/' + fname)

for lines in open(os.path.join(base_path, 'vegfru_list/vegfru_val.txt')):
    lines = lines.strip().split(' ')
    classInd = int(lines[1])
    fname = lines[0].split('/')[2]
    file_path = os.path.join(base_path, lines[0])
    ddr = valPrefix + lines[1]
    if not os.path.exists(ddr):
        os.makedirs(ddr)
    shutil.move(file_path, ddr + '/' + fname)