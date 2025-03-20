'''
	来自论文15：
	M. Alfarraj and G. AlRegib, (2019), 
	”Semisupervised sequence modeling for elastic impedance inversion,” Interpretation
	
'''

import torch
from torch.nn.functional import conv1d
from torch import nn, optim

class inverse_model(nn.Module):
	'''
		input_dim
	'''
	def __init__(self, input_dim=1, resolution_ratio=6, nonlinearity="relu"):
		super(inverse_model, self).__init__()
		self.in_channels = input_dim
		self.resolution_ratio = resolution_ratio # vertical scale mismtach between seismic and EI
		self.activation =  nn.ReLU() if nonlinearity=="relu" else nn.Tanh()

		# 1、并联卷积
		'''
			1、卷积部分是，local pattern获取，基本块是：conv1D+GroupNorm + 激活
			2、GroupNorm: 
		'''
		self.cnn1 = nn.Sequential(nn.Conv1d(in_channels=self.in_channels, out_channels=8,
										   kernel_size=5, padding=2, dilation=1),
								  nn.GroupNorm(num_groups=self.in_channels, num_channels=8))

		self.cnn2 = nn.Sequential(nn.Conv1d(in_channels=self.in_channels, out_channels=8,
										   kernel_size=5, padding=6, dilation=3),
								  nn.GroupNorm(num_groups=self.in_channels, num_channels=8))

		self.cnn3 = nn.Sequential(nn.Conv1d(in_channels=self.in_channels, out_channels=8,
										   kernel_size=5, padding=12, dilation=6),
								  nn.GroupNorm(num_groups=self.in_channels, num_channels=8))

		# 2、串联卷积
		self.cnn = nn.Sequential(self.activation,
								 nn.Conv1d(in_channels=24, out_channels=16, kernel_size=3, padding=1),
								 nn.GroupNorm(num_groups=self.in_channels, num_channels=16),
								 self.activation,

								 nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
								 nn.GroupNorm(num_groups=self.in_channels, num_channels=16),
								 self.activation,

								 nn.Conv1d(in_channels=16, out_channels=16, kernel_size=1),
								 nn.GroupNorm(num_groups=self.in_channels, num_channels=16),
								 self.activation)

		# 3、串联GRU, 获取低频趋势，可地震中有低频趋势吗？答案是没有。 低频趋势来自井，并且基本固定
		'''
			1、数据维度输入：（B, seq, hidden）, 注意和卷积的数据维度（B, channel, seq）不同。
			2、数据维度输出： 模型的输出，output(B, seq, D*hidden), 另一个是模型的隐藏状态h_n(一般不用)
			3、input_size: 表示特征维度channel, 
			4、hidden_size:表示隐藏层的特征维度，
			5、num_layer: 表示隐层的深度
			*** 序列方向是循环输入的 ***
		'''
		self.gru = nn.GRU(input_size=self.in_channels,
						  hidden_size=8,
						  num_layers=3,
						  batch_first=True,
						  bidirectional=True)


		# 4、串联反卷积，因为地震波形和波阻抗的采样精度不一致，所以需要上采样，
		self.up = nn.Sequential(nn.ConvTranspose1d(in_channels=16, out_channels=8,
												   stride=1, kernel_size=3, padding=1),
								nn.GroupNorm(num_groups=self.in_channels, num_channels=8),
								self.activation,

								nn.ConvTranspose1d(in_channels=8, out_channels=8,
												   stride=1, kernel_size=3, padding=1),
								nn.GroupNorm(num_groups=self.in_channels, num_channels=8),
								self.activation)

		# 5、GRU
		self.gru_out = nn.GRU(input_size=8,
							  hidden_size=8,
							  num_layers=1,
							  batch_first=True,
							  bidirectional=True)

		# 6、线性拟合层
		self.out = nn.Linear(in_features=16, out_features=self.in_channels)

		# 初始化模型
		for m in self.modules():
			if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
				nn.init.xavier_uniform_(m.weight.data)
				m.bias.data.zero_()
			elif isinstance(m, nn.GroupNorm):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.bias.data.zero_()


		self.optimizer = optim.Adam(self.parameters(), 0.005, weight_decay=1e-4)

	def forward(self, x):
		# （B, channel, seq）
		cnn_out1 = self.cnn1(x)
		cnn_out2 = self.cnn2(x)
		cnn_out3 = self.cnn3(x)
		cnn_out = self.cnn(torch.cat((cnn_out1, cnn_out2, cnn_out3),dim=1))

		tmp_x = x.transpose(-1, -2)           # （B, seq, hidden）
		rnn_out, _ = self.gru(tmp_x)          # （B, seq, D*hidden）, 其中hidden=8
		rnn_out = rnn_out.transpose(-1, -2)

		x = rnn_out + cnn_out
		x = self.up(x)               # ============如果使用上采样，则输入输出序列大小不同==========

		tmp_x = x.transpose(-1, -2)  # （B, seq, hidden）
		x, _ = self.gru_out(tmp_x)   # （B, seq, D*hidden）

		# 全连接：输入的数据格式，（B, seq, channel）, 注意不同于1*1的一维卷积形式，卷积数据输入形式是（B, channel, seq）
		x = self.out(x)
		x = x.transpose(-1,-2)

		return x
