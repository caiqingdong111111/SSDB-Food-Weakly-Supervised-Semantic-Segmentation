import numpy as np
import os
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from torch.autograd import Variable
from imageio import imsave

from network import resnet38_cls
from network import resnet38_SEAM

classes_list = ['background','candy','egg tart','french fries','chocolate','biscuit',
                'popcorn','pudding','ice cream','cheese butter','cake','wine','milkshake',
                'coffee','juice','milk','tea','almond','red beans','cashew','dried cranberries',
                'soy','walnut','peanut','egg','apple','date','apricot','avocado','banana','strawberry',
                'cherry','blueberry','raspberry','mango','olives','peach','lemon','pear','fig','pineapple',
                'grape','kiwi','melon','orange','watermelon','steak','pork','chicken duck','sausage','fried meat',
                'lamb','sauce','crab','fish','shellfish','shrimp','soup','bread','corn','hamburg','pizza',
                'hanamaki baozi','wonton dumplings','pasta','noodles','rice','pie','tofu','eggplant','potato',
                'garlic','cauliflower','tomato','kelp','seaweed','spring onion','rape','ginger','okra','lettuce',
                'pumpkin','cucumber','white radish','carrot','asparagus','bamboo shoots','broccoli',
                'celery stick','cilantro mint','snow peas','cabbage','bean sprouts','onion','pepper','green beans',
                'French beans','king oyster mushroom','shiitake','enoki mushroom','oyster mushroom','white button mushroom',
                'salad','other ingredients']


test_dataroot = r'D:\data\FoodSeg103\Images\img_dir\test'
test_groundroot = r'D:\data\FoodSeg103\Images\ann_dir\test'
test_imagenamelist = open(r'data\test.txt').read().splitlines()
test_imagelabel = np.load(r'data\testlabel.npy',allow_pickle=True).item()

weights_folder = r'train_weights'

Mymodel_1 = resnet38_cls.Net(103) # classification branch
weights_dict_1 = torch.load(os.path.join(weights_folder,'last.pth'))

Mymodel_2 = resnet38_SEAM.MyNet(104) # segmantation branch
weights_dict_2 = torch.load(os.path.join(weights_folder,'checkpoint.pth'))


Mymodel_1.load_state_dict(weights_dict_1,strict=False)
Mymodel_1.cuda()
Mymodel_1.eval()

Mymodel_2.load_state_dict(weights_dict_2,strict=False)
Mymodel_2.cuda()
Mymodel_2.eval()

infertransform = transforms.Compose([transforms.Resize(size=(448, 448)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],  # mean
                                                          [0.229, 0.224, 0.225])])


testimage = os.listdir(test_dataroot)

with torch.no_grad():

    outputsavepath = r'ouput'

    for testimage in testimage:
        imagepath = os.path.join(test_dataroot, testimage + '.jpg')
        img_pil = Image.open(imagepath)
        orig_img_size = img_pil.size
        img_pil = infertransform(img_pil)
        img_pil = Variable(img_pil.unsqueeze(0)).cuda()

        pred_y = Mymodel_1.forward_class(img_pil)
        _, cam_rv = Mymodel_2(img_pil,clabranch=True,seambranch=True)

        pred_y = torch.sigmoid(pred_y)
        pred_y[pred_y >= 0.5] = 1
        pred_y[pred_y < 0.5] = 0

        thecam_rv = F.interpolate(cam_rv, (orig_img_size[1], orig_img_size[0]), mode='bilinear',align_corners=False)[0]

        thecam_rv_0 = thecam_rv[1:, :, :] * pred_y.reshape(103, 1, 1)
        thecam_rv_0 = thecam_rv_0.detach().cpu().numpy()
        thecam_rv_0[thecam_rv_0 < 0] = 0
        thecam_rv_0_max = np.max(thecam_rv_0, (1, 2), keepdims=True)
        thecam_rv_0_min = np.min(thecam_rv_0, (1, 2), keepdims=True)
        thecam_rv_0[thecam_rv_0 < thecam_rv_0_min + 1e-5] = 0
        thecam_rv_0 = (thecam_rv_0 - thecam_rv_0_min - 1e-5) / (thecam_rv_0_max - thecam_rv_0_min + 1e-5)
        bg_score = [np.ones_like(thecam_rv_0[0]) * 0.1]
        pre_rv_0 = np.argmax(np.concatenate((bg_score, thecam_rv_0)), 0)

        imsave(os.path.join(outputsavepath, testimage + '.png'), pre_rv_0.astype(np.uint8))

