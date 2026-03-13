import numpy as np
import os
import torch
from torchvision import transforms
from PIL import Image
from torch.utils import data
import torch.nn.functional as F
from network import resnet38_SEAM

def adaptive_min_pooling_loss(x):
    n,c,h,w = x.size()
    k = h*w//4
    x = torch.max(x, dim=1)[0]
    y = torch.topk(x.view(n,-1), k=k, dim=-1, largest=False)[0]
    y = F.relu(y, inplace=False)
    loss = torch.sum(y)/(k*n)
    return loss


def max_norm(p, version='torch', e=1e-5):
    if version == 'torch':
        if p.dim() == 3:
            C, H, W = p.size()
            p = F.relu(p)
            max_v = torch.max(p.view(C,-1),dim=-1)[0].view(C,1,1)
            min_v = torch.min(p.view(C,-1),dim=-1)[0].view(C,1,1)
            p = F.relu(p-min_v-e)/(max_v-min_v+e)
        elif p.dim() == 4:
            N, C, H, W = p.size()
            p = F.relu(p)
            max_v = torch.max(p.view(N,C,-1),dim=-1)[0].view(N,C,1,1)
            min_v = torch.min(p.view(N,C,-1),dim=-1)[0].view(N,C,1,1)
            p = F.relu(p-min_v-e)/(max_v-min_v+e)
    elif version == 'numpy' or version == 'np':
        if p.ndim == 3:
            C, H, W = p.shape
            p[p<0] = 0
            max_v = np.max(p,(1,2),keepdims=True)
            min_v = np.min(p,(1,2),keepdims=True)
            p[p<min_v+e] = 0
            p = (p-min_v-e)/(max_v+e)
        elif p.ndim == 4:
            N, C, H, W = p.shape
            p[p<0] = 0
            max_v = np.max(p,(2,3),keepdims=True)
            min_v = np.min(p,(2,3),keepdims=True)
            p[p<min_v+e] = 0
            p = (p-min_v-e)/(max_v+e)
    return p

def max_onehot(x):
    n,c,h,w = x.size()
    x_max = torch.max(x[:,1:,:,:], dim=1, keepdim=True)[0]
    x[:,1:,:,:][x[:,1:,:,:] != x_max] = 0
    return x

class Mydataset(data.Dataset):
    def __init__(self, imagenamelist,dataroot,labels, transform):
        self.imgs_namelist = imagenamelist
        self.dataroot = dataroot
        self.labels = labels
        self.transforms = transform

    def __getitem__(self, index):

        img_name = self.imgs_namelist[index]
        img_path = os.path.join(self.dataroot,img_name)

        img_name = img_name.split('.')[0]
        label = self.labels[img_name]

        PIL_image = Image.open(img_path)
        data = self.transforms(PIL_image)

        return data, label

    def __len__(self):
        return len(self.imgs_namelist)

classes_list = ['background','candy','egg tart','french fries','chocolate','biscuit',
                'popcorn','pudding','ice cream','cheese butter','cake','wine','milkshake',
                'coffee','juice','milk','tea','almond','red beans','cashew','dried cranberries',
                'soy','walnut','peanut','egg','apple','date','apricot','avocado','banana','strawberry'
                'cherry','blueberry','raspberry','mango','olives','peach','lemon','pear','fig','pineapple',
                'grape','kiwi','melon','orange','watermelon','steak','pork','chicken duck','sausage','fried meat',
                'lamb','sauce','crab','fish','shellfish','shrimp','soup','bread','corn','hamburg','pizza',
                'hanamaki baozi','wonton dumplings','pasta','noodles','rice','pie','tofu','eggplant','potato',
                'garlic','cauliflower','tomato','kelp','seaweed','spring onion','rape','ginger','okra','lettuce',
                'pumpkin','cucumber','white radish','carrot','asparagus','bamboo shoots','broccoli',
                'celery stick','cilantro mint','snow peas','cabbage','bean sprouts','onion','pepper','green beans',
                'French beans','king oyster mushroom','shiitake','enoki mushroom','oyster mushroom','white button mushroom',
                'salad','other ingredients']

image_transforms = transforms.Compose([transforms.Resize(size=(448, 448)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.3),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3,3),sigma = (0.1,2.0))], p=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])
        ])

training_dataroot = r'D:\data\FoodSeg103\Images\img_dir\train'
training_imagenamelist= open(r'data\train.txt').read().splitlines()
training_imagelabel = np.load(r'data\trainglabel.npy',allow_pickle=True).item()

weights_folder = 'weights_path'

train_dataset = Mydataset(training_imagenamelist,training_dataroot,training_imagelabel,image_transforms)
train_dl = data.DataLoader(train_dataset,batch_size = 8,shuffle = True)


Mymodel = resnet38_SEAM.MyNet(104)
weights_dict = torch.load(weights_folder)

Mymodel.load_state_dict(weights_dict,strict=False)
Mymodel.cuda()
Mymodel.train()
optim = torch.optim.SGD(filter(lambda p: p.requires_grad, Mymodel.parameters()), lr=0.005)

scale_factor = 0.3

for epoch in range(30):
    print('start training   ', epoch)
    running_loss = 0
    Mymodel.train()

    for image1, label in train_dl:

        image1 = image1.to('cuda')
        N, C, H, W = image1.size()

        bg_score = torch.ones((N, 1))
        label = torch.cat((bg_score, label), dim=1)
        label = label.cuda(non_blocking=True).unsqueeze(2).unsqueeze(3)

        image2 = F.interpolate(image1, scale_factor=scale_factor, mode='bilinear', align_corners=True)

        cam1, cam_rv1 = Mymodel(image1)
        label1 = F.adaptive_avg_pool2d(cam1, (1, 1))
        loss_rvmin1 = adaptive_min_pooling_loss((cam_rv1 * label)[:, 1:, :, :])

        cam1 = F.interpolate(max_norm(cam1), scale_factor=scale_factor, mode='bilinear',align_corners=True) * label
        cam_rv1 = F.interpolate(max_norm(cam_rv1), scale_factor=scale_factor, mode='bilinear',align_corners=True) * label

        cam2, cam_rv2 = Mymodel(image2)
        label2 = F.adaptive_avg_pool2d(cam2, (1, 1))
        loss_rvmin2 = adaptive_min_pooling_loss((cam_rv2 * label)[:, 1:, :, :])
        cam2 = max_norm(cam2) * label
        cam_rv2 = max_norm(cam_rv2) * label

        loss_cls1 = F.multilabel_soft_margin_loss(label1[:, 1:, :, :], label[:, 1:, :, :])
        loss_cls2 = F.multilabel_soft_margin_loss(label2[:, 1:, :, :], label[:, 1:, :, :])

        ns, cs, hs, ws = cam2.size()
        loss_er = torch.mean(torch.abs(cam1[:, 1:, :, :] - cam2[:, 1:, :, :]))

        cam1[:, 0, :, :] = 1 - torch.max(cam1[:, 1:, :, :], dim=1)[0]
        cam2[:, 0, :, :] = 1 - torch.max(cam2[:, 1:, :, :], dim=1)[0]

        tensor_ecr1 = torch.abs(max_onehot(cam2.detach()) - cam_rv1)
        tensor_ecr2 = torch.abs(max_onehot(cam1.detach()) - cam_rv2)
        loss_ecr1 = torch.mean(torch.topk(tensor_ecr1.view(ns, -1), k=(int)(104 * hs * ws * 0.2), dim=-1)[0])
        loss_ecr2 = torch.mean(torch.topk(tensor_ecr2.view(ns, -1), k=(int)(104 * hs * ws * 0.2), dim=-1)[0])
        loss_ecr = loss_ecr1 + loss_ecr2

        loss_cls = (loss_cls1 + loss_cls2) / 2 + (loss_rvmin1 + loss_rvmin2) / 2
        loss = loss_cls + loss_er + loss_ecr

        optim.zero_grad()
        loss.backward()
        optim.step()

    with torch.no_grad():
        running_loss += loss.item()
    print('epochloss',running_loss / len(train_dl.dataset))

torch.save(Mymodel.state_dict(), os.path.join(weights_folder, 'checkpoint.pth'))
