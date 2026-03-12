import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
np.set_printoptions(threshold=np.inf)

import network.resnet38d


class MyNet(network.resnet38d.Net):
    def __init__(self,num_classes):
        super(MyNet, self).__init__()

        self.dropout7 = torch.nn.Dropout2d(0.5)

        self.fc8 = nn.Conv2d(4096, num_classes, 1, bias=False) # 这里使用的是1*1的卷积层代替全连接层

        self.f8_3 = torch.nn.Conv2d(512, 64, 1, bias=False)
        self.f8_4 = torch.nn.Conv2d(1024, 128, 1, bias=False)
        self.f9 = torch.nn.Conv2d(192+3, 192, 1, bias=False)
        
        torch.nn.init.xavier_uniform_(self.fc8.weight)
        torch.nn.init.kaiming_normal_(self.f8_3.weight)
        torch.nn.init.kaiming_normal_(self.f8_4.weight)
        torch.nn.init.xavier_uniform_(self.f9.weight, gain=4)
        self.from_scratch_layers = [self.f8_3, self.f8_4, self.f9, self.fc8]
        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]

    def forward(self, x):
        N, C, H, W = x.size() # 2,3,448,448
        d = super().forward_as_dict(x)
        cam = self.fc8(self.dropout7(d['conv6'])) # 这里的CAM是 [N,103,56,56]大小
        n,c,h,w = cam.size()
        with torch.no_grad():
            cam_d = F.relu(cam.detach())
            cam_d_max = torch.max(cam_d.view(n,c,-1), dim=-1)[0].view(n,c,1,1)+1e-5 # 这里得到 [2,103,1] 就是各个通道的最大值但不知道什么类别
            cam_d_norm = F.relu(cam_d-1e-5)/cam_d_max #这是将数据归一化，因为relu后最小是0 所以cam_d_max = cam_d_max-0 [2,103，56，56]
            cam_d_norm[:,0,:,:] = 1-torch.max(cam_d_norm[:,1:,:,:], dim=1)[0]
            # 这里利用20类得到背景类, 在通道上做最大值，利用最大值与1的差值作为背景的数值
            cam_max = torch.max(cam_d_norm[:,1:,:,:], dim=1, keepdim=True)[0]
            cam_d_norm[:,1:,:,:][cam_d_norm[:,1:,:,:] < cam_max] = 0
            #这个操作得到了每个点上的类别并在对应channel上其他的位置都是0

        f8_3 = F.relu(self.f8_3(d['conv4'].detach()), inplace=True)  # 这样操作好像是按他们论文上说的将网络分成两部分训练, f8_3是会更新参数的
        f8_4 = F.relu(self.f8_4(d['conv5'].detach()), inplace=True)  # PCM的回传不会影响到前面的网络 conv4，conv5
        x_s = F.interpolate(x,(h,w),mode='bilinear',align_corners=True) #将原始数据缩小
        f = torch.cat([x_s, f8_3, f8_4], dim=1)   #这里将原图变小到与CAM一样大然后和f8_3与f8_4层连接在一起 # 从自己的实验来看缩小图片，CAM的范围会变大
        n,c,h,w = f.size()

        cam_rv = F.interpolate(self.PCM(cam_d_norm, f), (H,W), mode='bilinear', align_corners=True) #这个程序是他们放大了CAM
        cam = F.interpolate(cam, (H,W), mode='bilinear', align_corners=True)
        # CAM 与 CAM_RV 都是插值后与原图一样大小的图片
        return cam, cam_rv # 这两个变量的形状是

    def PCM(self, cam, f): #这里的cam 是 cam_d_norm， f 是 f8_3， f8_4 和 x_s 和拼接组合

        n,c,h,w = f.size() #这里 h, w 的大小是f8_3的大小
        cam = F.interpolate(cam, (h,w), mode='bilinear', align_corners=True).view(n,-1,h*w) # 这里的形状是 [2，21，56*56]
        f = self.f9(f) # 这里就是论文上说的利用1*1的卷积实现了 θ, φ, g 的功能,但是按照论文上说的这里只保留了θ
        f = f.view(n,-1,h*w) # 【N, 195，56*56】
        """这里对应的是方程（6）"""
        f = f/(torch.norm(f,dim=1,keepdim=True)+1e-5) # torch.norm()就是求范数，默认求2范数， dim=1相当于求每个channel的范数,得到的是[N，1，56*56]
        aff = F.relu(torch.matmul(f.transpose(1,2), f),inplace=True)
        # torch.matmul() 就是矩阵乘法 这里的transpose就是矩阵转换，这里的，1 2 代表哪两个维度换
        # 得到的数据是aff = [N，56*56,56*56]
        aff = aff/(torch.sum(aff,dim=1,keepdim=True)+1e-5)  # C(xi) = torch.sum(aff,dim=1,keepdim=True) [N，1，56*56]
        cam_rv = torch.matmul(cam, aff).view(n,-1,h,w)
        # 这里的解释是，归一化后的CAM与aff度相乘，相当于用aff对CAM加权
        # 又因为是一张图片所以目标的aff是很高的，从而更加完整了CAM
        
        return cam_rv

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        print('======================================================')
        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups

