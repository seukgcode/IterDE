from cmath import inf, nan
from multiprocessing import reduction
from .Strategy import Strategy
import torch.nn.functional as F
import torch

class NegativeSampling_kd(Strategy):

	def __init__(self, model = None, t_model = None, loss = None, batch_size = 256, regul_rate = 0.0, l3_regul_rate = 0.0, l = 1):
		super(NegativeSampling_kd, self).__init__()
		self.model = model
		self.loss = loss
		self.batch_size = batch_size
		self.regul_rate = regul_rate
		self.l3_regul_rate = l3_regul_rate
		self.t_model = t_model
		self.l = l #软标签的占比


	def _get_positive_score(self, score):
		positive_score = score[:self.batch_size]
		positive_score = positive_score.view(-1, self.batch_size).permute(1, 0)
		return positive_score

	def _get_negative_score(self, score):
		negative_score = score[self.batch_size:]
		negative_score = negative_score.view(-1, self.batch_size).permute(1, 0)
		return negative_score

	def forward(self, data):
		# 计算硬标签损失
		score = self.model(data)
		p_score = self._get_positive_score(score)
		n_score = self._get_negative_score(score)
		loss_res = self.loss(p_score, n_score)
		if self.regul_rate != 0:
			loss_res += self.regul_rate * self.model.regularization(data)
		if self.l3_regul_rate != 0:
			loss_res += self.l3_regul_rate * self.model.l3_regularization()
		# 计算软标签损失
		if self.t_model != None:
			# for k,v in self.t_model.named_parameters():
			# 	v.requires_grad=False    #固定参数
			# 分数蒸馏
			for k,v in self.t_model.named_parameters():
					v.requires_grad = False
			self.t_model.cuda(2)
			t_score = self.t_model(data)
			t_loss_res = F.kl_div(score.softmax(dim=-1).log(), t_score.softmax(dim=-1), reduction='none').mean()
			if t_loss_res == nan or t_loss_res == inf:
				t_loss_res = 0
			# t_loss_res = t_loss_res.mean()
			# k = F.kl_div(x.softmax(dim=-1).log(), y.softmax(dim=-1), reduction='sum')
			# print("t_loss_res1:")
			# print(t_loss_res)
			# 结构蒸馏1

			# batch_h = data['batch_h']
			# batch_t = data['batch_t']
			# s_h = self.model.ent_embeddings(batch_h)
			# s_t = self.model.ent_embeddings(batch_t)
			# t_h = self.t_model.ent_embeddings(batch_h)
			# t_t = self.t_model.ent_embeddings(batch_t)
			# s_h_norma = F.normalize(s_h)
			# s_t_norma = F.normalize(s_t)
			# t_h_norma = F.normalize(t_h)
			# t_t_norma = F.normalize(t_t)
			# t_loss_res += F.smooth_l1_loss((s_h_norma * s_t_norma).sum(dim=1), (t_h_norma * t_t_norma).sum(dim=1))

			# print("t_loss_res2:")
			# print(t_loss_res)
			# 结构蒸馏2

			# s_h_norm = torch.norm(s_h)
			# s_t_norm = torch.norm(s_t)
			# t_h_norm = torch.norm(t_h)
			# t_t_norm = torch.norm(t_t)
			# t_loss_res += F.smooth_l1_loss(torch.div(s_h_norm, s_t_norm), torch.div(t_h_norm, t_t_norm))

			# print("t_loss_res3:")
			# print(t_loss_res)
			# 总的软标签损失
			# print("hard_loss:")
			# print(loss_res)
			# print("soft_loss:")
			# print(t_loss_res)
			loss_res = self.l * t_loss_res + loss_res
		# print("t_loss_resall:")
		# print(loss_res)
		# print("```````````````````````````````````````````")
		return loss_res