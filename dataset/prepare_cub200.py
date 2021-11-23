import os
import shutil

base_path = './cub200'
trainPrefix = os.path.join(base_path, 'cub200DB/train/')
testPrefix = os.path.join(base_path, 'cub200DB/test/')
for lines in open(os.path.join(base_path, 'lists/files.txt')):
    line = lines.strip().split('.')
    classInd = int(line[0])
    fname = lines.split('/')[1].split('\n')[0] #the name of the file we want
    print(fname)
    file_path = os.path.join(base_path + '/images', lines.split('\n')[0])
    print(file_path)
    if classInd <= 100:
        ddr = trainPrefix + str(classInd)
        if not os.path.exists(ddr):
            os.makedirs(ddr)
        shutil.move(file_path, ddr + '/' + fname)
    else:
        ddr = testPrefix + str(classInd)
        if not os.path.exists(ddr):
            os.makedirs(ddr)
        shutil.move(file_path, ddr + '/' + fname)


#Download images and lists from http://www.vision.caltech.edu/visipedia/CUB-200.html
#place it into cub200 folder
#run prepare_cub200 from outside the cub200 folder