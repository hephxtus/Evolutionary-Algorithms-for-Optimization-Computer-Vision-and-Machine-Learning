import os
import numpy as np
import sys
from PIL import Image
from sklearn.model_selection import train_test_split
import glob
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split

def load_Img(folder, class_label):
    imgs = []
    labels = []
    images = os.listdir(folder)
    if '.directory' in images:
        images.remove('.directory')
    if '.DS_Store' in images:
        images.remove('.DS_Store')
    if '._.DS_Store' in images:
        images.remove('._.DS_Store')
    # for j in images:
    for j in images:
    #     print(j)
    #     print(images[j])
        img = Image.open(folder+'/'+j).convert('L')
        # print(img)
        # print(img.shape)
        img =img.resize((128, 128), Image.BILINEAR)
        # img.show()
        # img = np.asarray(img)
        # img.save(img[0])
        imgs.append(np.asarray(img, dtype='float32'))
        labels.append(class_label)
    return np.asarray(imgs), np.asarray(labels)

def read_images(dataset_name, path=None):
    if path is not None:
        folders_path = path+dataset_name
    else: folders_path = dataset_name
    files = os.listdir(folders_path)
    print(files)
    x_train=[]
    y_train = []
    prop = []
    prop.append('Number of Classes '+ str(len(files)))
    for i in range(0,len(files)):
        prop.append(files[i]+' ' + str(i))
        data, labels = load_Img(folders_path+'/'+files[i], i)
        print(data.shape, labels.shape, files[i])
        x_train.append(data)
        y_train.append(labels)
    print(x_train[0].shape)
    return np.asarray(x_train), np.asarray(y_train), prop

if __name__ == "__main__":
    # data_sets=['grimace', 'faces95']
    data_sets=['rafD']
    data_sets = ['f1', 'f2']
    data_sets = ['jaffe']
    for i in data_sets:
        #x1, y1,  prop = read_images(dataset_name=i, path='/local/scratch/projects/datasets/mdatasets_o/')
        x1, y1, prop = read_images(dataset_name=i, path='/local/scratch/projects/datasets/mdatasets_o/')
        x1 = np.concatenate((x1), axis=0)
        y1 = np.concatenate((y1), axis=0)
        # x1, x2, y1, y2 = train_test_split(x1, y1, test_size=0.5, random_state=12,
        #                                                     stratify=y1)
        print(x1.shape)
        np.save(i+'_data', x1)
        np.save(i+'_label', y1)
        # np.save(i + '_test_data', x2)
        # np.save(i + '_test_label', y2)
        prop.append('Instances '+ str(x1.shape[0]))
        prop.append('Image Size ' + str(x1.shape))
        # prop.append('Image Test Size ' + str(x2.shape))
        file = open(i+'_properties.txt','w')
        file.writelines(["%s\n" % item for item in prop])
        file.close()
