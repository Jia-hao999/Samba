import argparse
from vsod_dataset import Dataset
from torchvision import transforms
import transform_vsod
from torch.utils import data
import torch
from collections import OrderedDict
from models.SalMamba_dual import Model
from datetime import datetime
import os
from thop import profile
import numpy as np
import IOU
import datetime
import torch.distributed as dist
import random
from Smeasure import S_object,S_region
from utils.init_func import group_weight
import torch.nn as nn
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
p = OrderedDict()
p['lr'] = 1e-4  # Learning rate
p['wd'] = 0.01  # Weight decay
p['momentum'] = 0.90  # Momentum
showEvery = 1000

CE = torch.nn.BCEWithLogitsLoss(reduction='mean')
IOU = IOU.IOU(size_average=True)

def structure_loss(pred, mask):
    bce = CE(pred, mask)
    iou = IOU(torch.nn.Sigmoid()(pred), mask)
    return bce+iou


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

parser = argparse.ArgumentParser()
print(torch.cuda.is_available())

parser.add_argument('--cuda', type=bool, default=True)  # 是否使用cuda

# train
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--epoch_save', type=int, default=5)
parser.add_argument('--save_fold', type=str, default='./checkpoints')  # 训练过程中输出的保存路径
parser.add_argument('--input_size', type=int, default=448)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_thread', type=int, default=8)
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--test_dataset1', type=list, default=['DAVIS'])
parser.add_argument('--test_dataset2', type=list, default=['SegTrack-V2'])
# Misc
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
config = parser.parse_args()

config.save_fold = config.save_fold + '/' + 'Samba/vsod_datasets'
if not os.path.exists("%s" % (config.save_fold)):
    os.mkdir("%s" % (config.save_fold))

def test1(test_loader,model,epoch,save_path):
    global best_sm1,best_epoch1
    model.eval()
    s_sum = 0
    with torch.no_grad():
        for i, data_batch in enumerate(test_loader):
            image,label, flow, name, split, size = data_batch['image'],data_batch['label'], data_batch['flow'], \
                                            data_batch['name'], data_batch['split'], data_batch['size']

            image, flow, label = image.cuda(),flow.cuda(), label.cuda()
            out, saliency = model(image,flow)
            pre1 = torch.nn.Sigmoid()(out[0])
            pre1 = (pre1 - torch.min(pre1)) / (torch.max(pre1) - torch.min(pre1))
            gt = label[0]
            gt[gt >= 0.5] = 1
            gt[gt < 0.5] = 0
            alpha = 0.5
            y = gt.mean()
            if y == 0:
                x = pre1.mean()
                Q = 1.0 - x
            elif y == 1:
                x = pre1.mean()
                Q = x
            else:
                Q = alpha * S_object(pre1, gt) + (1 - alpha) * S_region(pre1, gt)
                if Q.item() < 0:
                    Q = torch.FloatTensor([0.0])
            s_sum += Q.item()
        sm = s_sum / len(test_loader.dataset)

        if epoch==0:
            best_sm1=sm
            best_epoch1=epoch+1
            torch.save(model.state_dict(), '%s/epoch_%d_best.pth' % (save_path,epoch+1))
        else:
            if sm>=best_sm1:
                best_sm1=sm
                best_epoch1=epoch+1
                torch.save(model.state_dict(), '%s/epoch_%d_best.pth' % (save_path,epoch+1))
        print('Epoch: {} sm: {} ####  best_sm: {} bestEpoch: {}'.format(epoch+1,sm,best_sm1,best_epoch1))
        return sm

def test2(test_loader,model,epoch,save_path):
    global best_sm2,best_epoch2
    model.eval()
    s_sum = 0
    with torch.no_grad():
        for i, data_batch in enumerate(test_loader):
            image,label, flow, name, split, size = data_batch['image'],data_batch['label'], data_batch['flow'], \
                                            data_batch['name'], data_batch['split'], data_batch['size']

            image, flow, label = image.cuda(),flow.cuda(), label.cuda()
            out, saliency = model(image,flow)
            pre1 = torch.nn.Sigmoid()(out[0])
            pre1 = (pre1 - torch.min(pre1)) / (torch.max(pre1) - torch.min(pre1))
            gt = label[0]
            gt[gt >= 0.5] = 1
            gt[gt < 0.5] = 0
            alpha = 0.5
            y = gt.mean()
            if y == 0:
                x = pre1.mean()
                Q = 1.0 - x
            elif y == 1:
                x = pre1.mean()
                Q = x
            else:
                Q = alpha * S_object(pre1, gt) + (1 - alpha) * S_region(pre1, gt)
                if Q.item() < 0:
                    Q = torch.FloatTensor([0.0])
            s_sum += Q.item()
        sm = s_sum / len(test_loader.dataset)

        if epoch==0:
            best_sm2=sm
            best_epoch2=epoch+1
            # torch.save(model.state_dict(), '%s/epoch_%d_best.pth' % (save_path,epoch+1))
        else:
            if sm>=best_sm2:
                best_sm2=sm
                best_epoch2=epoch+1
                # torch.save(model.state_dict(), '%s/epoch_%d_best.pth' % (save_path,epoch+1))
        print('Epoch: {} sm: {} ####  best_sm: {} bestEpoch: {}'.format(epoch+1,sm,best_sm2,best_epoch2))
        return sm


if __name__ == '__main__':
    set_seed(1024)

    composed_transforms_ts = transforms.Compose([
        transform_vsod.RandomFlip(),
        transform_vsod.RandomRotate(),
        transform_vsod.colorEnhance(),
        transform_vsod.randomPeper(),
        transform_vsod.FixedResize(size=(config.input_size, config.input_size)),
        transform_vsod.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transform_vsod.ToTensor()])
    dataset_train = Dataset(datasets=['DAVIS','DAVSOD','FBMS'],transform=composed_transforms_ts, mode='train')

    dataloader = data.DataLoader(dataset_train, batch_size=config.batch_size, num_workers=config.num_thread,
                                 drop_last=True,
                                 shuffle=True)
    print("Training Set, DataSet Size:{}, DataLoader Size:{}".format(len(dataset_train), len(dataloader)))

    composed_transforms_te = transforms.Compose([
    transform_vsod.FixedResize(size=(config.input_size, config.input_size)),
    transform_vsod.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    transform_vsod.ToTensor()])

    dataset_test1 = Dataset(datasets=config.test_dataset1, transform=composed_transforms_te, mode='test')
    test_loader1 = data.DataLoader(dataset_test1, batch_size=1, num_workers=config.num_thread,drop_last=True, shuffle=False)
    print("Testing Set1, DataSet Size:{}, DataLoader Size:{}".format(len(dataset_test1), len(test_loader1)))

    dataset_test2 = Dataset(datasets=config.test_dataset2, transform=composed_transforms_te, mode='test')
    test_loader2 = data.DataLoader(dataset_test2, batch_size=1, num_workers=config.num_thread,drop_last=True, shuffle=False)
    print("Testing Set2, DataSet Size:{}, DataLoader Size:{}".format(len(dataset_test2), len(test_loader2)))

    model = Model()

    if config.cuda:
        model = model.cuda()
    
    # 创建一个随机输入
    input1, input2 = torch.randn(1, 3, 448, 448), torch.randn(1, 3, 448, 448)
    input1, input2 = input1.cuda(), input2.cuda()
    # 计算 FLOPs 和参数量
    flops, params = profile(model, inputs=(input1, input2))
    print(f"FLOPs: {flops}, Params: {params}")

    params_list = []
    params_list = group_weight(params_list, model, nn.BatchNorm2d, p['lr'])

    optimizer = torch.optim.AdamW(params_list, lr=p['lr'], betas=(0.9, 0.999),weight_decay=p['wd'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = 10,
            eta_min = 1e-5,
            last_epoch = -1
        )

    optimizer.zero_grad()
    # 定义梯度累积的步数
    accumulation_steps = 4

    # 一个epoch中训练iter_num个batch
    iter_num = len(dataloader)
    loss_wirte = []
    sm_wirte1 = []
    sm_wirte2 = []
    for epoch in range(config.epoch):
        loss_all = 0
        aveGrad = 0
        model.zero_grad()
        optimizer.zero_grad()
        model.train()

        for i, data_batch in enumerate(dataloader):
            image, label, flow = data_batch['image'], data_batch['label'], data_batch[
                'flow']
            if image.size()[2:] != label.size()[2:]:
                print("Skip this batch")
                continue
            if config.cuda:
                image, label, flow = image.cuda(), label.cuda(), flow.cuda()
            out, saliency = model(image, flow)
            loss = structure_loss(out, label) + structure_loss(saliency, label)
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()  # 更新参数
                optimizer.zero_grad()  # 清除累积的梯度

            loss_all += loss.data
            if i % showEvery == 0:
                print(
                    '%s || epoch: [%2d/%2d], iter: [%5d/%5d]  Loss || sum : %10.4f' % (
                        datetime.datetime.now(), epoch+1, config.epoch, i+1, iter_num, loss_all / (i + 1)))
                print('Learning rate: ' + str(optimizer.param_groups[0]['lr']))
            
            if i == iter_num - 1:
                print(
                    '%s || epoch: [%2d/%2d], iter: [%5d/%5d]  Loss || sum : %10.4f' % (
                        datetime.datetime.now(), epoch+1, config.epoch, i+1, iter_num,loss_all / (i + 1)))
                print('Learning rate: ' + str(optimizer.param_groups[0]['lr']))
        epoch_loss = loss_all / len(dataloader)
        loss_wirte.append(epoch_loss)
        scheduler.step()

        if epoch >= 0:
            sm1 = test1(test_loader1,model,epoch,config.save_fold)
            sm2 = test2(test_loader2,model,epoch,config.save_fold)
        sm_wirte1.append(sm1)
        sm_wirte2.append(sm1)
        with open("./train_loss_vsod.txt", 'w') as train_los:
            train_los.write(str(loss_wirte))
        with open("./Smeasure_vsod.txt", 'w') as train_sm:
            train_sm.write(str(sm_wirte1))
        with open("./Smeasure_vsod2.txt", 'w') as train_sm2:
            train_sm2.write(str(sm_wirte2))

        if epoch >= 10:
            torch.save(model.state_dict(),
                       '%s/epoch_%d_bone.pth' % (config.save_fold, epoch + 1))
    


