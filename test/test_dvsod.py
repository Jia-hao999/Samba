import argparse
import os
from dvsod_dataset import Dataset
import torch
from torchvision import transforms
import transform_dvsod
from torch.utils import data
from models.SalMamba_tri import Model
import numpy as np
import cv2

parser = argparse.ArgumentParser()
print(torch.cuda.is_available())
parser.add_argument('--cuda', type=bool, default=True)  # 是否使用cuda
#CUDA_VISIBLE_DEVICES=1 python test_tri.py
# test
parser.add_argument('--test_batch_size', type=int, default=1)
parser.add_argument('--num_thread', type=int, default=8)
parser.add_argument('--input_size', type=int, default=448)
parser.add_argument('--spatial_ckpt', type=str, default=None)
parser.add_argument('--flow_ckpt', type=str, default=None)
parser.add_argument('--depth_ckpt', type=str, default=None)
parser.add_argument('--model_path', type=str, default='./checkpoints/Samba/RDVS/Samba_RDVS.pth')
parser.add_argument('--test_dataset', type=list, default=['RDVS'])
parser.add_argument('--testsavefold', type=str, default='./Samba')

# Misc
parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
config = parser.parse_args()

composed_transforms_te = transforms.Compose([
    transform_dvsod.FixedResize(size=(config.input_size, config.input_size)),
    transform_dvsod.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    transform_dvsod.ToTensor()])

dataset = Dataset(datasets=config.test_dataset, transform=composed_transforms_te, mode='test')
test_loader = data.DataLoader(dataset, batch_size=config.test_batch_size, num_workers=config.num_thread,
                              drop_last=True, shuffle=False)

print('mode: {}'.format(config.mode))
print('------------------------------------------')
model = Model()

if config.cuda:
    model = model.cuda()
assert (config.model_path != ''), ('Test mode, please import pretrained model path!')
assert (os.path.exists(config.model_path)), ('please import correct pretrained model path!')
print('load model……all checkpoints')

model.load_pretrain_model(config.model_path)
model.eval()

if not os.path.exists(config.testsavefold):
    os.makedirs(config.testsavefold)

for i, data_batch in enumerate(test_loader):
    print("progress {}/{}\n".format(i + 1, len(test_loader)))
    image, flow, depth, name, split, size = data_batch['image'], data_batch['flow'], data_batch['depth'], \
                                            data_batch['name'], data_batch['split'], data_batch['size']
    dataset = data_batch['dataset']

    if config.cuda:
        image, flow, depth = image.cuda(), flow.cuda(), depth.cuda()
    with torch.no_grad():

        out, saliency = model(image, flow, depth)

        for i in range(config.test_batch_size):
            presavefold = os.path.join(config.testsavefold, dataset[i], split[i])

            if not os.path.exists(presavefold):
                os.makedirs(presavefold)
            pre1 = torch.nn.Sigmoid()(out[0][i])
            pre1 = (pre1 - torch.min(pre1)) / (torch.max(pre1) - torch.min(pre1))
            pre1 = np.squeeze(pre1.cpu().data.numpy()) * 255
            pre1 = cv2.resize(pre1, (int(size[0][1]), int(size[0][0])))
            cv2.imwrite(presavefold + '/' + name[i], pre1)

           
