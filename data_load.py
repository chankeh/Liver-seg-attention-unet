from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio
# import nrrd
import cv2
import os


class NrrdReader3D(Dataset):
    def __init__(self, data_path, label_path=None, test=False, transform=None):
        self.data_path = data_path
        self.files = os.listdir(data_path)
        self.files.sort()
        self.transform = transform
        self.test = test
        if not self.test:
            self.label_path = label_path

    def __len__(self):
        return len(self.files)
    def read_img(file_path):
        # 读取path文件夹下所有文件的名字
        img_tmp = []
        imagelist = os.listdir(file_path)
        print(imagelist)
        # 输出文件列表
        # print(imagelist)

        for imgname in imagelist:
            if (imgname.endswith(".png")):
                image = cv2.imread(file_path)
                img_tmp.append(image)
        print(len(img_tmp))
        return np.array(img_tmp)

    def __getitem__(self, index):
        file_name = self.files[index]
        # data, _ = nrrd.read(self.data_path + file_name)# TODO
        datas = self.read_img(self.data_path) # read data img from dir
        print(datas.shape)
        # datas = datas.astype(np.float32)
        # datas = datas[np.newaxis, ...]
        if not self.test:
            # label, _ = nrrd.read(self.label_path + file_name) # TODO
            labels = self.read_img(self.label_path) # read label img from dir
            sample = {}
            sample.setdefault('data',[])
            sample.setdefault('label',[])
            for data,label in zip(datas,labels):
                sample['data'].append(data)
                sample['label'].append(label)
                # sample['data']['label'] = {data,labels}

            # sample = {'data': data, 'label': label}
        else:
            sample = {'data': datas}
        return sample
