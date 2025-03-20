'''
需要额外使用的损失函数
	1、判别损失
'''

import torch
import torch.nn.functional as F

def cosine_distance_loss(x, y):
	'''
	优点：1、无尺度性，余弦只关注向量之间的方向而非大小，因此适合于特征值范围差异大的情况。2、适合文本和高维稀疏数据。

	示例：
		x = torch.randn((10, 1, 128))  # 10个样本，每个样本128维，特征维度是1
		y = torch.randn((10, 1, 128))
		loss = cosine_distance_loss(x, y)
		print(loss)
	'''
	cos_sim = F.cosine_similarity(x, y, dim=2)
	loss = 1 - cos_sim
	return loss.mean()




class CycleGANLoss(nn.Module):
	# 定义对抗性损失、循环一致性损失和身份损失
	def __init__(self, lambda_cycle=10.0, lambda_identity=5.0):
		super(CycleGANLoss, self).__init__()
		self.lambda_cycle = lambda_cycle
		self.lambda_identity = lambda_identity
		self.criterionGAN = nn.MSELoss()  # 对抗性损失（对抗损失应该使用那个？）
		self.criterionCycle = nn.L1Loss()  # 循环一致性损失
		self.criterionIdentity = nn.L1Loss()  # 身份损失

	def generator_loss(self, fake_output):
		return self.criterionGAN(fake_output, torch.ones_like(fake_output))

	def discriminator_loss(self, real_output, fake_output):
		real_loss = self.criterionGAN(real_output, torch.ones_like(real_output))
		fake_loss = self.criterionGAN(fake_output, torch.zeros_like(fake_output))
		return (real_loss + fake_loss) * 0.5

	def cycle_consistency_loss(self, real_A, rec_A, real_B, rec_B):
		loss_A = self.criterionCycle(real_A, rec_A)
		loss_B = self.criterionCycle(real_B, rec_B)
		return self.lambda_cycle * (loss_A + loss_B)

	def identity_loss(self, real_A, same_A, real_B, same_B):
		loss_A = self.criterionIdentity(real_A, same_A)
		loss_B = self.criterionIdentity(real_B, same_B)
		return self.lambda_identity * (loss_A + loss_B)



if __name__ == '__main__':
	# 示例用法
	cycle_gan_loss = CycleGANLoss(lambda_cycle=10.0, lambda_identity=5.0)

	# 假设存在生成器G_A和G_B，判别器D_A和D_B，以及优化器
	optimizer_G = optim.Adam(list(G_A.parameters()) + list(G_B.parameters()), lr=0.0002, betas=(0.5, 0.999))
	optimizer_D_A = optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
	optimizer_D_B = optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

	# 输入数据
	real_A = torch.randn(B, 1, T)  # 来自域A的真实样本
	real_B = torch.randn(B, 1, T)  # 来自域B的真实样本

	# 生成假样本
	fake_B = G_A(real_A)
	fake_A = G_B(real_B)

	# 重新生成样本
	rec_A = G_B(fake_B)
	rec_B = G_A(fake_A)

	# 身份映射
	same_A = G_A(real_A)
	same_B = G_B(real_B)

	# 计算生成器损失
	loss_G_A = cycle_gan_loss.generator_loss(D_B(fake_B))
	loss_G_B = cycle_gan_loss.generator_loss(D_A(fake_A))
	loss_cycle = cycle_gan_loss.cycle_consistency_loss(real_A, rec_A, real_B, rec_B)
	loss_identity = cycle_gan_loss.identity_loss(real_A, same_A, real_B, same_B)
	loss_G = loss_G_A + loss_G_B + loss_cycle + loss_identity

	# 更新生成器
	optimizer_G.zero_grad()
	loss_G.backward()
	optimizer_G.step()

	# 计算判别器损失
	real_output_A = D_A(real_A)
	fake_output_A = D_A(fake_A.detach())
	loss_D_A = cycle_gan_loss.discriminator_loss(real_output_A, fake_output_A)

	real_output_B = D_B(real_B)
	fake_output_B = D_B(fake_B.detach())
	loss_D_B = cycle_gan_loss.discriminator_loss(real_output_B, fake_output_B)

	# 更新判别器A
	optimizer_D_A.zero_grad()
	loss_D_A.backward()
	optimizer_D_A.step()

	# 更新判别器B
	optimizer_D_B.zero_grad()
	loss_D_B.backward()
	optimizer_D_B.step()

	print(f'Generator loss: {loss_G.item()}, Discriminator A loss: {loss_D_A.item()}, Discriminator B loss: {loss_D_B.item()}')