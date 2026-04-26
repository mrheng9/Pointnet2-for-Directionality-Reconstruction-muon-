import torch
import torch.nn as nn
import math  
import torch.nn.functional as F
from models.pointnet_regression_utils import PointNetSetAbstraction,PointNetSetAbstractionMsg


class get_model(nn.Module):
    def __init__(self,num_class,normal_channel=True):
        super(get_model, self).__init__()
        in_channel = 5 if normal_channel else 3
        self.normal_channel = normal_channel
        #ssg
        # self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        # self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        # self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        
        #msg
        # self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        # self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320+3,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        # self.sa3 = PointNetSetAbstraction(None, None, None, 640+3, [256, 512, 1024], True)

        #第一层加msg512
        # self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        # self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=320+3, mlp=[128, 128, 256], group_all=False)
        # self.sa3 = PointNetSetAbstraction(None, None, None, 256+3, [256, 512, 1024], True)
         
        #第一层加msg1024
        # self.sa1 = PointNetSetAbstractionMsg(1024, [0.1, 0.2, 0.4], [16, 32, 64], in_channel, [[32, 64, 128], [64, 96, 192], [64, 96, 128]])
        # self.sa2 = PointNetSetAbstraction(npoint=256, radius=0.4, nsample=64, in_channel=448+3, mlp=[128, 192, 256], group_all=False)
        # self.sa3 = PointNetSetAbstraction(None, None, None, 256+3, [256, 512, 1024], group_all=True)

        #第一层加msg1024(加深)
        # self.sa1 = PointNetSetAbstractionMsg(npoint=1024,radius_list=[0.05, 0.1, 0.2, 0.4],nsample_list=[16, 32, 64, 128],in_channel=in_channel,mlp_list=[[32, 64, 96, 128],[64, 96, 128, 192],[64, 96, 128, 192],[64, 96, 128, 256]])

        # self.sa2 = PointNetSetAbstractionMsg(      
        #     npoint=256,
        #     radius_list=[0.2, 0.4, 0.6],
        #     nsample_list=[32, 64, 128],
        #     in_channel=(128+192+192+256)+3,         
        #     mlp_list=[
        #         [64, 96, 128, 192],
        #         [96, 128, 192, 256],
        #         [96, 128, 192, 256]
        #     ]
        # )

        # self.sa3 = PointNetSetAbstraction(
        #     npoint=64,                              
        #     radius=0.8,
        #     nsample=128,
        #     in_channel=(192+256+256)+3,
        #     mlp=[256, 384, 512],
        #     group_all=False
        # )

        # self.sa4 = PointNetSetAbstraction(          
        #     npoint=None,
        #     radius=None,
        #     nsample=None,
        #     in_channel=512+3,
        #     mlp=[512, 768, 1024],
        #     group_all=True
        # )

        #第二层加msg512
        # self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        # self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 128+3,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        # self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=640 + 3, mlp=[256, 512, 1024], group_all=True)

        #第二层加msg1024
        # self.sa1 = PointNetSetAbstraction(npoint=1024, radius=0.1, nsample=48, in_channel=in_channel, mlp=[64, 128, 256], group_all=False)
        # self.sa2 = PointNetSetAbstractionMsg(256, [0.2, 0.4, 0.6], [32, 64, 128], 256+3, [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        # self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=640 + 3, mlp=[512, 512, 1024], group_all=True)

        #ssg更改参数1024
        # self.sa1 = PointNetSetAbstraction(npoint=1024, radius=0.3, nsample=64, in_channel=in_channel, mlp=[64, 128, 256], group_all=False)
        # self.sa2 = PointNetSetAbstraction(npoint=256, radius=0.5, nsample=128, in_channel=256 + 3, mlp=[256, 256, 512], group_all=False)
        # self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[512, 1024,2048], group_all=True)    
        
        #msg更改参数1024
        self.sa1 = PointNetSetAbstractionMsg(npoint=1024,  radius_list=[0.1, 0.2, 0.3],nsample_list=[32, 48, 64],  in_channel=in_channel, mlp_list=[[32, 64, 128], [64, 96, 192], [64, 128, 256]])
        self.sa2 = PointNetSetAbstractionMsg(npoint=256, radius_list=[0.2, 0.4, 0.5], nsample_list=[64, 96, 128], in_channel=(128+192+256)+3, mlp_list=[[128, 128, 256], [128, 192, 256], [128, 256, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None,radius=None,nsample=None,in_channel=(256+256+256)+3,mlp=[512, 1024, 2048],group_all=True)
        

       
        # self.fc1 = nn.Linear(1024, 512)
        # self.bn1 = nn.BatchNorm1d(512)
        # self.drop1 = nn.Dropout(0.4)
        # self.fc2 = nn.Linear(512, 256)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.drop2 = nn.Dropout(0.4)
        # self.fc3 = nn.Linear(256, num_class)
        
        self.fc1 = nn.Linear(2048, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(512, num_class)

        #浅加深
        # self.fc1 = nn.Linear(1024, 768)
        # self.bn1 = nn.BatchNorm1d(768)
        # self.drop1 = nn.Dropout(0.4)
        # self.fc2 = nn.Linear(768, 512)
        # self.bn2 = nn.BatchNorm1d(512)
        # self.drop2 = nn.Dropout(0.4)
        # self.fc3 = nn.Linear(512, 256)
        # self.bn3 = nn.BatchNorm1d(256)
        # self.drop3 = nn.Dropout(0.3)
        # self.fc4 = nn.Linear(256, num_class)

        #加深
        # self.fc1 = nn.Linear(2048, 1024)
        # self.bn1 = nn.BatchNorm1d(1024)
        # self.drop1 = nn.Dropout(0.4)
        # self.fc2 = nn.Linear(1024, 768)
        # self.bn2 = nn.BatchNorm1d(768)
        # self.drop2 = nn.Dropout(0.4)
        # self.fc3 = nn.Linear(768, 512)
        # self.bn3 = nn.BatchNorm1d(512)
        # self.drop3 = nn.Dropout(0.4)
        # self.fc4 = nn.Linear(512, 256)
        # self.bn4 = nn.BatchNorm1d(256)
        # self.drop4 = nn.Dropout(0.3)
        # self.fc5 = nn.Linear(256, num_class)

    def forward(self, xyz):
        xyz = xyz.permute(0, 2, 1)  
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 2048)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        
        #浅加深
        # x = l3_points.view(B, 1024)
        # x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        # x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        # x = self.drop3(F.relu(self.bn3(self.fc3(x))))
        # x = self.fc4(x)
        
        #加深
        # x = l3_points.view(B, 2048)
        # x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        # x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        # x = self.drop3(F.relu(self.bn3(self.fc3(x))))
        # x = self.drop4(F.relu(self.bn4(self.fc4(x))))
        # x = self.fc5(x)

        return x, l3_points

    # def forward(self, xyz):
    #     xyz = xyz.permute(0, 2, 1)  
    #     B, _, _ = xyz.shape
    #     if self.normal_channel:
    #         norm = xyz[:, 3:, :]
    #         xyz = xyz[:, :3, :]
    #     else:
    #         norm = None
            
    #     l1_xyz, l1_points = self.sa1(xyz, norm)
    #     l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
    #     l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
    #     l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # 添加第四层处理
        
    #     # 使用 l4_points (来自SA4) 而不是 l3_points
    #     x = l4_points.view(B, 1024)
    #     x = self.drop1(F.relu(self.bn1(self.fc1(x))))
    #     x = self.drop2(F.relu(self.bn2(self.fc2(x))))
    #     x = self.fc3(x)
        
    #     return x, l4_points  # 返回 l4_points 作为特征

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        # loss = F.mse_loss(pred, target)
        loss =torch.sqrt((target[:,0]-pred[:,0])**2 + (target[:,1]-pred[:,1])**2+ (target[:,2]-pred[:,2])**2)
        return loss.mean()

