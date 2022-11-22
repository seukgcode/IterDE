from cmath import inf, nan
from multiprocessing import reduction
from .Strategy import Strategy
import torch.nn.functional as F
import torch

class kd_mulde(Strategy):

	def __init__(self, model = None, t_models = None, loss = None, batch_size = 256, regul_rate = 0.0, l3_regul_rate = 0.0, l = 1):
		super(kd_mulde, self).__init__()
		self.model = model
		self.loss = loss
		self.batch_size = batch_size
		self.regul_rate = regul_rate
		self.l3_regul_rate = l3_regul_rate
		self.t_models = t_models
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
		if self.t_models != None:
			# for k,v in self.t_model.named_parameters():
			# 	v.requires_grad=False    #固定参数
			# 分数蒸馏
			t_model1 = self.t_models[0]
			t_model2 = self.t_models[1]
			t_model3 = self.t_models[2]
			t_model4 = self.t_models[3]
			for k,v in t_model1.named_parameters():
				v.requires_grad = False
			for k,v in t_model2.named_parameters():
				v.requires_grad = False
			for k,v in t_model3.named_parameters():
				v.requires_grad = False
			for k,v in t_model4.named_parameters():
				v.requires_grad = False
			t_model1.cuda()
			t_model2.cuda()
			t_model3.cuda()
			t_model4.cuda()
			# self.t_models.cuda()
			t_score1 = t_model1(data)
			t_score2 = t_model2(data)
			t_score3 = t_model3(data)
			t_score4 = t_model4(data)
			t_score = (t_score1 + t_score2 + t_score3 + t_score4) / 4
			# t_loss_res = F.kl_div(score.softmax(dim=-1).log(), t_score.softmax(dim=-1), reduction='none').mean()
			t_loss_res = F.smooth_l1_loss(score, t_score)
			if t_loss_res == nan or t_loss_res == inf:
				t_loss_res = 0
			loss_res = self.l * t_loss_res + loss_res
		# print("t_loss_resall:")
		# print(loss_res)
		# print("```````````````````````````````````````````")
		return loss_res