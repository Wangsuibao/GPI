'''
CNN style model
'''

import torch
import torch.nn as nn


class CNN(nn.Module):
	'''
		input： 是地震1D波形
		output： 反射系数

		模型结构： 2*[ conv_1D(1*80) + ReLU ]
	'''
	def __init__(self):
		super(CNN, self).__init__()

		self.noOfNeurons = 60   # channel,特征维
		self.dilation = 1
		self.kernel_size = 80   # 卷积核
		self.stride = 1
		self.padding = int(((self.dilation*(self.kernel_size-1)-1)/self.stride-1)/2)  # same卷积

		self.layer1 = nn.Sequential(
			nn.Conv1d(1, self.noOfNeurons, kernel_size=self.kernel_size, stride=1, padding=self.padding+1),
			nn.ReLU())

		self.layer2 = nn.Sequential(
			nn.Conv1d(self.noOfNeurons, 1, kernel_size=self.kernel_size, stride=1, padding=self.padding+2),
			nn.ReLU())

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)

		return out


class VishalNet_par(nn.Module):
	'''
		模型结构： 
					Conv_1d(1*81) + ReLU + Conv_1d(1*301) ---
															 |--cat -- conv1d --out
					Conv_1d(1*9) +  ReLU + Conv_1d(1*9) -----
	'''
	def __init__(self):
		super(VishalNet_par, self).__init__()
		self.cnn1 = nn.Conv1d(1, 60, 9, 1, 4)    # 卷积核为81, padding=(k-1)/2
		self.cnn2 = nn.Conv1d(60, 1, 9, 1, 4)    # 卷积核为301
		self.ReLU1 = nn.ReLU()
		self.ReLU2 = nn.ReLU()

		self.cnn3 = nn.Conv1d(1, 60, 81, 1, 40)    # 卷积核为81, padding=(k-1)/2
		self.cnn4 = nn.Conv1d(60, 1, 301, 1, 150)    # 卷积核为301		
		self.ReLU3 = nn.ReLU()
		self.ReLU4 = nn.ReLU()

		self.cnn_out1 = nn.Conv1d(2, 1, 15, 1, 7)

	
	def forward(self, input):
		out1 = self.ReLU1(self.cnn1(input))
		out1 = self.ReLU2(self.cnn2(out1))

		out2 = self.ReLU3(self.cnn3(input))
		out2 = self.ReLU4(self.cnn4(out2))

		# 两个支路的特征加和效果没有cat效果好。。。
		out = torch.cat([out1, out2], dim=1)
		# out = out1 + out2

		out = self.cnn_out1(out)

		return out

#################  1-D Physics-based Model by Vishal Das et al. ###################################################
class VishalNet(nn.Module):
	'''
		模型结构： Conv_1d(1*81) + ReLU + Conv_1d(1*301)
	'''
	def __init__(self, input_dim=1):
		super(VishalNet, self).__init__()
		self.cnn1 = nn.Conv1d(input_dim, 60, 81, 1, 40)    # 卷积核为81, padding=(k-1)/2
		self.cnn2 = nn.Conv1d(60, 1, 301, 1, 150)  # 卷积核为301
	
	def forward(self, input):
		out1 = nn.functional.relu(self.cnn1(input))
		out2 = self.cnn2(out1)
		return out2

class CNN_R(nn.Module):
	'''
		结构： 2*[conv_1D(1*80) + ReLU] + conv_1D(1*1)
	'''
	def __init__(self):
		super(CNN_R, self).__init__()

		self.noOfNeurons = 60   # channel,特征维
		self.dilation = 1
		self.kernel_size = 80   # 卷积核
		self.stride = 1
		self.padding = int(((self.dilation*(self.kernel_size-1)-1)/self.stride-1)/2)  # same卷积

		self.layer1 = nn.Sequential(
			nn.Conv1d(1, self.noOfNeurons, kernel_size=self.kernel_size, stride=1, padding=self.padding+1),
			nn.ReLU())

		self.layer2 = nn.Sequential(
			nn.Conv1d(self.noOfNeurons, 3, kernel_size=self.kernel_size, stride=1, padding=self.padding+2),
			nn.ReLU())

		# self.layer3 = nn.Conv1d(3, 1, kernel_size=3, padding=1)
		self.layer3 = nn.Conv1d(3, 1, kernel_size=1, padding=0)  # 拟合的形式

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)

		return out