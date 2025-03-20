'''
实现全连接神经网络（Fully Connected Neural Network）FCNN。也称多层感知机(Multilayer Perceptron)
'''

import torch
import torch.nn as nn

class FCNN(nn.Module):
	'''
	'''
	def __init__(self):
		super(FCNN, self).__init__()
		self.hidden1=nn.Sequential(
				nn.Linear(in_features=1, out_features=30, bias=True),
				nn.ReLU())
		self.hidden2=nn.Sequential(
				nn.Linear(in_features=30, out_features=10, bias=True),
				nn.ReLU())
		self.hidden3=nn.Sequential(
				nn.Linear(in_features=10, out_features=1, bias=True),
				nn.Sigmoid())

	def forward(self,x):
		fc1=self.hidden1(x)
		fc2=self.hidden2(fc1)
		output=self.hidden3(fc2)
		return output


if __name__ == '__main__':
	FCNN = FCNN()
	x = torch.randn(size=(10, 1))
	y = FCNN(x)
	print(y)