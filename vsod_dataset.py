import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np


def randomCrop(image, label, flow):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    label = Image.fromarray(label)
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), np.array(label.crop(random_region)), flow.crop(random_region)


class Dataset(data.Dataset):
    def __init__(self, datasets, mode='train', transform=None, return_size=True):
        self.return_size = return_size
        if type(datasets) != list:
            datasets = [datasets]
        self.datas_id = []
        self.mode = mode
        for (i, dataset) in enumerate(datasets):

            if mode == 'train':
                data_dir = './vsod_dataset/train/{}'.format(dataset)
                imgset_path = data_dir + '/train.txt'

            else:
                data_dir = './vsod_dataset/test/{}'.format(dataset)
                imgset_path = data_dir + '/test.txt'

            imgset_file = open(imgset_path)

            for line in imgset_file:
                data = {}
                img_path = line.strip("\n").split(" ")[0]
                gt_path = line.strip("\n").split(" ")[1]
                data['img_path'] = data_dir + img_path
                data['gt_path'] = data_dir + gt_path
                if dataset == 'DUTS-TR':
                    data['split'] = dataset
                    data['img_path'] = data_dir +'/'+ img_path
                    data['gt_path'] = data_dir +'/'+ gt_path
                else:
                    data['flow_path'] = data_dir + line.strip("\n").split(" ")[2]
                    data['split'] = img_path.split('/')[-3]
                data['dataset'] = dataset
                self.datas_id.append(data)
        self.transform = transform

    def __getitem__(self, item):

        assert os.path.exists(self.datas_id[item]['img_path']), (
            '{} does not exist'.format(self.datas_id[item]['img_path']))
        assert os.path.exists(self.datas_id[item]['gt_path']), (
            '{} does not exist'.format(self.datas_id[item]['gt_path']))
        if self.datas_id[item]['dataset'] == 'DUTS-TR':
            pass
        else:
            assert os.path.exists(self.datas_id[item]['flow_path']), (
                '{} does not exist'.format(self.datas_id[item]['flow_path']))

        image = Image.open(self.datas_id[item]['img_path']).convert('RGB')
        label = Image.open(self.datas_id[item]['gt_path']).convert('L')
        label = np.array(label)

        if self.datas_id[item]['dataset'] == 'DUTS-TR':
            flow = np.zeros((image.size[1], image.size[0], 3))
            flow = Image.fromarray(np.uint8(flow))
            


        else:
            flow = Image.open(self.datas_id[item]['flow_path']).convert('RGB')
            

        if label.max() > 0:
            label = label / 255

        w, h = image.size
        size = (h, w)

        sample = {'image': image, 'label': label, 'flow': flow}
        if self.mode == 'train':
            sample['image'], sample['label'], sample['flow'] = randomCrop(sample['image'],
                                                                                           sample['label'],
                                                                                           sample['flow'])
        else:
            pass

        if self.transform:
            sample = self.transform(sample)
        if self.return_size:
            sample['size'] = torch.tensor(size)
        if self.datas_id[item]['dataset'] == 'DUTS-TR':
            sample['flow'] = torch.zeros((3, 448, 448))
        name = self.datas_id[item]['gt_path'].split('/')[-1]
        sample['dataset'] = self.datas_id[item]['dataset']
        sample['split'] = self.datas_id[item]['split']
        sample['name'] = name

        return sample

    def __len__(self):
        return len(self.datas_id)
