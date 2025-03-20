'''
标准Unet的结构是：
	基本构件特征：
		1、下采样：nn.MaxPool2d(2)。  上采样：ConvTranspose2d(256,128,2,stride=2)
		2、卷积层：nn.Conv2d(in_ch, out_ch, 3, padding=1)。   激活函数：Relu
		3、skip 后是 concat。   归一化层：nn.BatchNorm2d(out_ch)
		4、最后输出接1*1的卷积层
		5、确保输入图像宽高可以被16整除，4次下采样，inchannel-64-128-256-512-1024
'''

from torch import nn
import torch
from torchsummary import summary

class conv_step(nn.Module):
	'''
		等维度卷积，归一化，激活
	'''
	def __init__(self, input_dim, num_features, kernel_size=3):
		super(conv_step, self).__init__()
		
		padding = int((kernel_size - 1) / 2)
		self.conv_1 = nn.Conv1d(input_dim, num_features, kernel_size, padding=padding)
		self.bn = nn.BatchNorm1d(num_features)
		self.IN = nn.InstanceNorm1d(num_features)  # 在图像分割领域效果好
		self.conv_2 = nn.Conv1d(num_features,num_features, kernel_size, padding=padding)
		self.relu = nn.ReLU()
		
	def forward(self,x):
		x = self.conv_1(x)
		# x = self.bn(x)
		# x = self.IN(x)
		x = self.relu(x)
		x = self.conv_2(x)
		# x = self.bn(x)
		# x = self.IN(x)
		x = self.relu(x)
		return x

class up_conv(nn.Module):
	'''
		上卷积需要完成的功能：尺度增大2倍，特征缩减到1/2
			上采样使用插值,nn.unsample(2)
	'''
	def __init__(self, input_dim):
		super(up_conv, self).__init__()
		self.upsample = nn.Upsample(scale_factor=2)
		self.conv = nn.Conv1d(input_dim, input_dim//2, 3, padding=1)
		
	def forward(self, x):
		x = self.upsample(x)
		x = self.conv(x)
		return x

class up_conv_(nn.Module):
	'''
		使用反卷积实现：尺度增加2倍，特征缩减1/2
		nn.ConvTranspose1d(input_dim, input_dim//2, 2, stride=2)
	'''
	def __init__(self, input_dim):
		super(up_conv, self).__init__()
		self.input_dim = input_dim
		self.upconv = nn.ConvTranspose1d(input_dim, input_dim//2, 2, stride=2)

	def forward(self, x):
		x = self.upconv(x)
		return x

	

class Unet_1D(nn.Module):
	def __init__(self, input_dim=1, num_features=64, kernel_size=3, out_dim=1):
		'''
			其中num_features是特征基数，特征维度的变换是：
				in_channel -->num_feature-->2*num_feature-->4*num_feature  ---> [ 8*num_feature ]
				                                                            U底

		'''
		super(Unet_1D, self).__init__()
		
		self.down_layer_1 = conv_step(input_dim, num_features, kernel_size)
		self.down_layer_2 = conv_step(num_features, num_features*2, kernel_size)
		self.down_layer_3 = conv_step(num_features*2, num_features*4, kernel_size)

		self.maxpool_1 = nn.MaxPool1d(2, 2, padding=0)
		self.maxpool_2 = nn.MaxPool1d(2, 2, padding=0)
		self.maxpool_3 = nn.MaxPool1d(2, 2, padding=0)

		self.U_layer = conv_step(num_features*4, num_features*8, kernel_size)
		
		self.up_conv_1 = up_conv(num_features*8)
		self.up_conv_2 = up_conv(num_features*4)
		self.up_conv_3 = up_conv(num_features*2)
		
		self.up_layer_1 = conv_step(num_features*8, num_features*4, kernel_size)
		self.up_layer_2 = conv_step(num_features*4, num_features*2, kernel_size)
		self.up_layer_3 = conv_step(num_features*2, num_features, kernel_size)
		
		self.final = nn.Conv1d(num_features, out_dim, 1)  # 输入channel, 输出channel, 卷积核=1

		self.softmax = nn.LogSoftmax(dim=1)   # 最后一层如果是拟合，则不需要激活函数，如果是分类，则使用激活函数
		
	def forward(self,x):
		
		
		""" 
			结构：Double_conv*2 + pool
		"""
		out_1 = self.down_layer_1(x)    # (B, 1, H) --> (B, num_feature, H)
		x = self.maxpool_1(out_1)       # (B, num_feature, H) --> (B, num_feature, H/2)
		
		out_2 = self.down_layer_2(x)
		x = self.maxpool_2(out_2)
		
		out_3 = self.down_layer_3(x)
		x = self.maxpool_3(out_3)
		
		# U底，独立的卷积层
		end = self.U_layer(x)
		

		
		"""
			up + double_conv*2
		"""
		x = self.up_conv_1(end)             # (B, 8*num_feature, H//8) --> (B, 4*num_feature, H//4)  上采样+特征缩减
		x = torch.cat([out_3,x],dim = 1)    # (B, 4*num_feature, H//4) + (B, 4*num_feature, H//4) = (B, 8*num_feature, H//4)
		x = self.up_layer_1(x)              # (B, 8*num_feature, H//4) --> (B, 4*num_feature, H//4)  特征缩减
		
		x = self.up_conv_2(x)
		x = torch.cat([out_2,x],dim = 1)
		x = self.up_layer_2(x)
		
		x = self.up_conv_3(x)
		x = torch.cat([out_1,x],dim = 1)
		x = self.up_layer_3(x)
		
		x = self.final(x)

		# xx = self.softmax(x)
		
		return x

if __name__ == "__main__":
	unet = Unet_1D(3, 16, 3).to('cpu')  # 数据的格式是（B,C,H,W）, modal中输入的参数是（in_channel, out_channel）
	print(unet)
	summary(unet, input_size=(3, 128), device='cpu')

	img =  torch.randn(1,3,128)
	out_img = unet(img)
	print(out_img.shape)