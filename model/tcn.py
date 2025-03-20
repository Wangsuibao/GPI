'''
** 严格TCN的形式：1、整个过程等长； 2、防止未来信息泄露； 3、深度有效信息传播； 历史长时间依赖有效
'''
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
	'''
		截取一个向量右边一段，被截掉的长为chomp_size; 保证输入输出等长
		chomp_size = padding = (kernel_size-1)*dialation_size
	'''
	def __init__(self, chomp_size):
		super(Chomp1d, self).__init__()
		self.chomp_size = chomp_size

	def forward(self, x):
		'''
			输入：x 的数据格式是 （B, c, T） 最后一维是时间方向数据
			输出： （B,c, T-chomp_size）
			1、删除一个向量后chomp_size个块
			2、深拷贝后返回
		'''
		return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
	'''
		实现 2*(DC+WN+R+Drop) + Resblock
	'''
	def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
		super(TemporalBlock, self).__init__()
		self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
										   stride=stride, padding=padding, dilation=dilation))
		self.chomp1 = Chomp1d(padding)
		self.relu1 = nn.ReLU()
		self.dropout1 = nn.Dropout(dropout)

		self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
										   stride=stride, padding=padding, dilation=dilation))
		self.chomp2 = Chomp1d(padding)
		self.relu2 = nn.ReLU()
		self.dropout2 = nn.Dropout(dropout)

		self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
								 self.conv2, self.chomp2, self.relu2, self.dropout2)
		self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
		self.relu = nn.ReLU()
		self.init_weights()

	def init_weights(self):
		self.conv1.weight.data.normal_(0, 0.01)
		self.conv2.weight.data.normal_(0, 0.01)
		if self.downsample is not None:
			self.downsample.weight.data.normal_(0, 0.01)

	def forward(self, x):
		out = self.net(x)
		res = x if self.downsample is None else self.downsample(x)
		return self.relu(out + res)


class TemporalConvNet(nn.Module):
	'''
		多个 2*(DC+WN+R+Drop) + Resblock 的链接
		d在指数增加
	'''
	def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
		super(TemporalConvNet, self).__init__()
		layers = []
		num_levels = len(num_channels)
		for i in range(num_levels):
			dilation_size = 2 ** i
			in_channels = num_inputs if i == 0 else num_channels[i-1]
			out_channels = num_channels[i]
			layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
									 padding=(kernel_size-1) * dilation_size, dropout=dropout)]

		self.network = nn.Sequential(*layers)

	def forward(self, x):
		return self.network(x)


# ---------------------------------------------------具体化------------------------------
class TCN_IV_1D_C(nn.Module):
	'''
		输入的数据维度是（B, C, T）
		模型的最后一层使用1*1卷积层，完成channel上的合并，channel：5-->1。
		可以认为是channel上的拟合
	'''
	def __init__(self, input_dim=1):
		super(TCN_IV_1D_C, self).__init__()

		# 论文4中模型配置： num_channels=[3,6,6,6,6,6,5], kernel_size=5, dropout=0.2，
		# 论文4超参数设置： lr=0.001, epoch=3000, trace=19
		self.tcn_local = TemporalConvNet(num_inputs=input_dim, num_channels=[3,6,6,6,6,5], kernel_size=5, dropout=0.2)
		self.regression = nn.Conv1d(in_channels=5, out_channels=1, kernel_size=1)

	def forward(self, input):
		out = self.tcn_local(input)
		out = self.regression(out)
		return out

if __name__ == '__main__':
	model = TCN_IV_1D_C()
	print(model)