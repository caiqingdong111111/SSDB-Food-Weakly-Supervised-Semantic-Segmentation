import numpy as np
import os
import torch
from torchvision import transforms
from PIL import Image
from torch.utils import data
from network import resnet38_cls


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

        return img_name,data, label

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

data_root = r'D:\data\FoodSeg103\Images\img_dir\train'
training_dataroot = r'D:\data\FoodSeg103\Images\img_dir\train'
training_imagenamelist= open(r'data\train.txt').read().splitlines()
training_imagelabel = np.load(r'data\trainglabel.npy',allow_pickle=True).item()

weights_folder = r'weights_path'

train_dataset = Mydataset(training_imagenamelist,training_dataroot,training_imagelabel,image_transforms)
train_dl = data.DataLoader(train_dataset, batch_size=16, shuffle=True)

Mymodel = resnet38_cls.Net(101)
weights_dict = torch.load(weights_folder)
Mymodel.load_state_dict(weights_dict,strict=False)
Mymodel.cuda()
Mymodel.train()
optim = torch.optim.SGD(filter(lambda p: p.requires_grad, Mymodel.parameters()), lr=0.001)


infertransform = transforms.Compose([transforms.Resize(size=(448, 448)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

loss_fn = torch.nn.CrossEntropyLoss()
bestf1 = 0
for epoch in range(30):
    running_loss =0
    Mymodel.train()
    for x, y in train_dl:
        x, y = x.cuda(), y.cuda()
        y_pred = Mymodel(x)
        loss = loss_fn(y_pred,y)
        optim.zero_grad()
        loss.backward()
        optim.step()

    with torch.no_grad():
        running_loss += loss.item()
    print('epochloss',running_loss / len(train_dl.dataset))

torch.save(Mymodel.state_dict(), os.path.join(weights_folder, 'last.pth'))
