from cmath import inf, nan
from multiprocessing import reduction

from .Strategy import Strategy
import torch.nn.functional as F
import torch

class kd_huber_new(Strategy):

	def __init__(self, model = None, t_model = None, loss = None, batch_size = 256, regul_rate = 0.0, l3_regul_rate = 0.0, l = 1):
		super(kd_huber_new, self).__init__()
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

	def norm(self, x):
		if x < 1.0:
			while x < 1.0:
				x *= 10.0
		elif x >= 10.0:
			while x >= 10.0:
				x /= 10.0
		return x

	def forward(self, data, half_epoch):
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
			for k,v in self.t_model.named_parameters():
				v.requires_grad = False
			self.t_model.cuda()
			t_score = self.t_model(data)
			# t_loss_res = F.kl_div(score.softmax(dim=-1).log(), t_score.softmax(dim=-1), reduction='none').mean()
			t_loss_res = F.smooth_l1_loss(score, t_score)
			if t_loss_res == nan or t_loss_res == inf:
				t_loss_res = 0
			if half_epoch == False:
				loss_cmp = t_loss_res * loss_res
				loss_cmp = self.norm(loss_cmp)
			else:
				loss_cmp = 1
			loss_res = self.l / loss_cmp * t_loss_res + loss_res
		return loss_res