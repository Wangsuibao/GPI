'''
	地震相分类
	1、（B, 1, H）--->（B, n_classs, H）
'''
import torch
import torch.nn as nn
import torch.optim as optim


class Facies_model_class(nn.Module):
	'''
		波阻抗--->地震相 ,分类模型
		只适合有地震相的数据，输入波阻抗，输出地震相。在train_multitask.py中使用。

		波阻抗-->地震相，因为是分割输出，所以模型的输出是（B, n_classs, H）
		等长度卷积，输入channel=1
	'''
	def __init__(self, facies_n=4, nonlinearity="tanh"):
		super(Facies_model_class, self).__init__()
		self.activation =  nn.ReLU()

		# 1、特征提取的形式
		self.cnn = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
								 self.activation,
								 nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
								 self.activation,
								 nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
								 )

		# 拟合
		self.fit = nn.Conv1d(in_channels=128, out_channels=facies_n, kernel_size=1, padding=0)


	def forward(self, x):
		x = self.cnn(x)
		x = self.fit(x)
		return x


if __name__ == '__main__':
	model = Facies_model_class(num_classes=5)
	