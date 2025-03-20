'''
	所有模型可调用的公共子模块
'''

import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class VGGExtractor(nn.Module):
	''' 
		VGG extractor for ASR described in https://arxiv.org/pdf/1706.02737.pdf
		因为地震反演的输入是低频，输出需要是高频，所以这里不能下采样
		2D特征获取，注意原始数据是seq的1D特征。2D特征可以通过小波变换新增一个频率维度。形成（batch, channel, seq, F）
	'''

	def __init__(self, input_dim):
		super(VGGExtractor, self).__init__()
		self.init_dim = 64
		self.hide_dim = 128
		in_channel, freq_dim, out_dim = self.check_dim(input_dim)
		self.in_channel = in_channel
		self.freq_dim = freq_dim
		self.out_dim = out_dim

		self.extractor = nn.Sequential(
			nn.Conv2d(in_channel, self.init_dim, 3, padding=1),
			nn.ReLU(),
			nn.Conv2d(self.init_dim, self.init_dim, 3, padding=1),
			nn.ReLU(),
			nn.Conv2d(self.init_dim, self.hide_dim, 3, padding=1),
			nn.ReLU(),
			nn.Conv2d(self.hide_dim, self.hide_dim, 3, padding=1),
			nn.ReLU(),
		)

	def check_dim(self, input_dim):
		# Check input dimension, delta feature should be stack over channel.
		if input_dim % 13 == 0:
			# MFCC feature
			return int(input_dim/13), 13, (13//4)*self.hide_dim
		elif input_dim % 40 == 0:
			# Fbank feature
			return int(input_dim/40), 40, (40//4)*self.hide_dim
		else:
			raise ValueError(
			'Acoustic feature dimension for VGG should be 13/26/39(MFCC) or 40/80/120(Fbank) but got '+input_dim)

	def forward(self, feature):
		'''
			in： （B, channel, seq）
			out:  (B, channel, seq)
		'''
		feature = self.extractor(feature)

		return feature


class CNNExtractor(nn.Module):
	'''
		 A simple 2-layer CNN extractor for acoustic feature
	'''

	def __init__(self, input_dim, out_dim):
		super(CNNExtractor, self).__init__()

		self.out_dim = out_dim
		self.extractor = nn.Sequential(
			nn.Conv1d(input_dim, out_dim, 3, padding=1),
			nn.Conv1d(out_dim, out_dim, 3, padding=1),
		)

	def forward(self, feature):

		'''
			in: (batch, channel, seq)
			out: (batch, channel, seq)
		'''
		feature = self.extractor(feature)

		return feature


class RNNLayer(nn.Module):
	''' 
		RNN ： 
			output的处理过程 LayerNorm_ + dropout_ + time-downsampling + tanh(Linear)_  ： (seq, batch, 2*channel) 
			hidden的处理过程 torch.cat((hidden[-2,:,:], hidden[-1,:,:]) + tanh(Linear)  :  (Batch, dec_hid_dim)
	'''

	def __init__(self, input_dim, dim,  bidirectional, dropout, layer_norm, proj):
		super(RNNLayer, self).__init__()

		rnn_out_dim = 2*dim if  bidirectional else dim
		self.out_dim = rnn_out_dim      # 计算RNN输出的特征维度
		self.layer_norm = layer_norm    # RNN后是否加normal层
		self.dropout = dropout          # RNN后是否加dropout层        
		self.proj = proj                # RNN后是否有一层全连接层

		# 1、RNN层
		'''
			GRU(in_dim, hid_dim, bi=True, layers=1, batch_first=True)
				in: (seq, batch, channel)
				out: (seq, batch, 2*channel)
		'''
		self.layer = nn.GRU(input_dim, dim, bidirectional= bidirectional, num_layers=1, batch_first=False)

		# 2、归一化层和dropout层
		if self.layer_norm:
			self.ln = nn.LayerNorm(rnn_out_dim)              # in: (seq, batch, 2*channel)
		if self.dropout > 0:
			self.dp = nn.Dropout(p=dropout)                  # in: (seq, batch, 2*channel)

		# 3、线性映射层
		if self.proj:
			self.pj = nn.Linear(rnn_out_dim, rnn_out_dim)    # in: (seq, batch, 2*channel)

	def forward(self, input_x):
		'''
			input_x : (seq, batch, channel)
			output  : (seq, batch, 2*channel) ; hidden: (Batch, dec_hid_dim)
		'''

		# 1、RNN层
		output, hidden = self.layer(input_x)

		# 2、归一化层和dropout层
		if self.layer_norm:
			output = self.ln(output)
		if self.dropout > 0:
			output = self.dp(output)
		# 3、 线性映射层
		if self.proj:
			output = torch.tanh(self.pj(output))

		# (batch, seq, 2*channel)
		return output, hidden




