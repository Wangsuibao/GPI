'''
	主要实现不同的正演传播。
		1、用于模拟数据
		2、用于物理指导	
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class forward_model_0(nn.Module):
	'''
		来自论文3的model1D
		等长度卷积，输入和输出channel=1
	'''
	def __init__(self, resolution_ratio=1, nonlinearity="tanh"):
		super(forward_model_0, self).__init__()
		self.resolution_ratio = resolution_ratio
		self.activation =  nn.ReLU() if nonlinearity=="relu" else nn.Tanh()

		# 1、地震特征获取模块
		self.cnn = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=4, kernel_size=9, padding=4),
								 self.activation,
								 nn.Conv1d(in_channels=4, out_channels=4,kernel_size=7, padding=3),
								 self.activation,
								 nn.Conv1d(in_channels=4, out_channels=1,kernel_size=3, padding=1))

		# 2、褶积模块
		self.wavelet = nn.Conv1d(in_channels=1, out_channels=1,
							 stride=self.resolution_ratio, kernel_size=51, padding=25)

	def forward(self, x):
		x = self.cnn(x)
		x = self.wavelet(x)
		return x

class forward_model_2(nn.Module):
	'''
		不同频率--不同波长的褶积-->加和
		等长度卷积，输入和输出channel=1
	'''
	def __init__(self, resolution_ratio=1, nonlinearity="tanh"):
		super(forward_model_2, self).__init__()
		self.resolution_ratio = resolution_ratio
		self.activation =  nn.ReLU() if nonlinearity=="relu" else nn.Tanh()

		# 1、地震特征获取模块
		self.cnnd = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=4, kernel_size=101, padding=50),
								 self.activation,)
		self.cnnz = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=4,kernel_size=21, padding=10),
								 self.activation,)
		self.cnng = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=4,kernel_size=11, padding=5),
								 self.activation,)

		# 2、褶积模块
		self.wavelet = nn.Conv1d(in_channels=4, out_channels=1,
							 stride=self.resolution_ratio, kernel_size=1, padding=0)

	def forward(self, x):
		xd = self.cnnd(x)
		xz = self.cnnz(x)
		xg = self.cnng(x)
		x = xd + xz + xg
		x = self.wavelet(x)
		return x


class forward_model_1(nn.Module):
	'''
	1-D Physics-based Model, 使用雷克子波的褶积正演模型
	'''
	def __init__(self, resolution_ratio=1, frequency=40, length=0.128, dt=0.001):
		super(forward_model_1, self).__init__()
		self.wavelet = self.generate_ricker_wavelet(frequency, length, dt)
		# Register the wavelet as a buffer to handle CUDA correctly
		self.register_buffer('wavelet_buffer', self.wavelet.unsqueeze(0).unsqueeze(0))

	def generate_ricker_wavelet(self, frequency, length, dt):
		'''
		生成雷克子波
		frequency: 子波的主频 (Hz)
		length: 子波的时间长度 (秒), 一般子波长度可以根据主频设置，如50HZ对应波长20ms
		dt: 采样间隔 (秒), 即默认地震数据的分辨率也是1ms.
		'''
		t = np.arange(-length / 2, (length - dt) / 2, dt)
		y = (1.0 - 2.0 * (np.pi ** 2) * (frequency ** 2) * (t ** 2)) * np.exp(-(np.pi ** 2) * (frequency ** 2) * (t ** 2))
		return torch.tensor(y).float()

	def forward(self, x):
		'''
		输入 x 是波阻抗, 形状为 (B, channel=1, H)
		输出 (B, 1, H)
		'''
		# 1、求解反射系数
		x_d = x[..., 1:] - x[..., :-1]
		x_a = (x[..., 1:] + x[..., :-1]) / 2
		rc = x_d / x_a  # 反射系数
		rc = F.pad(rc, (1, 0), mode='replicate')

		# 2、反射系数与子波褶积
		synth = F.conv1d(rc, self.wavelet_buffer, padding='same')

		return synth





