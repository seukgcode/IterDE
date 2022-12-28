from cmath import inf, nan
from multiprocessing import reduction
from .Strategy import Strategy
import torch.nn.functional as F
import torch

class kd_structure(Strategy):

	def __init__(self, model = None, t_model = None, loss = None, batch_size = 256, regul_rate = 0.0, l3_regul_rate = 0.0, l = 1, kind = 0):
		super(kd_structure, self).__init__()
		self.model = model
		self.loss = loss
		self.batch_size = batch_size
		self.regul_rate = regul_rate
		self.l3_regul_rate = l3_regul_rate
		self.t_model = t_model
		self.l = l #软标签的占比
		self.kind = kind


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
			# 分数蒸馏
			# for k,v in self.t_model.named_parameters():
			# 	v.requires_grad = False
			self.t_model.cuda()
			t_score = self.t_model(data)
			t_loss_res = F.smooth_l1_loss(score, t_score)
			if t_loss_res == nan or t_loss_res == inf:
				t_loss_res = 0
			if self.kind == 1: #蒸馏com
				# 结构蒸馏1
				batch_h = data['batch_h']
				batch_t = data['batch_t']
				t_h_re = self.t_model.ent_re_embeddings(batch_h)
				t_h_im = self.t_model.ent_im_embeddings(batch_h)
				t_t_re = self.t_model.ent_re_embeddings(batch_t)
				t_t_im = self.t_model.ent_im_embeddings(batch_t)
				s_h_re = self.model.ent_re_embeddings(batch_h)
				s_h_im = self.model.ent_im_embeddings(batch_h)
				s_t_re = self.model.ent_re_embeddings(batch_t)
				s_t_im = self.model.ent_im_embeddings(batch_t)
				s_h_re_norma = F.normalize(s_h_re)
				s_t_re_norma = F.normalize(s_t_re)
				s_h_im_norma = F.normalize(s_h_im)
				s_t_im_norma = F.normalize(s_t_im)
				t_h_re_norma = F.normalize(t_h_re)
				t_t_re_norma = F.normalize(t_t_re)
				t_h_im_norma = F.normalize(t_h_im)
				t_t_im_norma = F.normalize(t_t_im)
				t_loss_res += F.smooth_l1_loss((s_h_re_norma * s_t_re_norma).sum(dim=1), (t_h_re_norma * t_t_re_norma).sum(dim=1))
				t_loss_res += F.smooth_l1_loss((s_h_im_norma * s_t_im_norma).sum(dim=1), (t_h_im_norma * t_t_im_norma).sum(dim=1))

				# 结构蒸馏2
				t_h_re_norm = torch.norm(t_h_re)
				t_t_re_norm = torch.norm(t_t_re)
				s_h_re_norm = torch.norm(s_h_re)
				s_t_re_norm = torch.norm(s_t_re)
				t_h_im_norm = torch.norm(t_h_im)
				t_t_im_norm = torch.norm(t_t_im)
				s_h_im_norm = torch.norm(s_h_im)
				s_t_im_norm = torch.norm(s_t_im)
				t_loss_res += F.smooth_l1_loss(torch.div(s_h_re_norm, s_t_re_norm), torch.div(t_h_re_norm, t_t_re_norm))
				t_loss_res += F.smooth_l1_loss(torch.div(s_h_im_norm, s_t_im_norm), torch.div(t_h_im_norm, t_t_im_norm))
			else: # 蒸馏transe, simple
			# 结构蒸馏1
				batch_h = data['batch_h']
				batch_t = data['batch_t']
				s_h = self.model.ent_embeddings(batch_h)
				s_t = self.model.ent_embeddings(batch_t)
				t_h = self.t_model.ent_embeddings(batch_h)
				t_t = self.t_model.ent_embeddings(batch_t)
				s_h_norma = F.normalize(s_h)
				s_t_norma = F.normalize(s_t)
				t_h_norma = F.normalize(t_h)
				t_t_norma = F.normalize(t_t)
				# print(s_h_norma.shape)
				# print(s_t_norma.shape)
				# print(t_h_norma.shape)
				# print(t_t_norma.shape)
				t_loss_res += F.smooth_l1_loss((s_h_norma * s_t_norma).sum(dim=1), (t_h_norma * t_t_norma).sum(dim=1))

				# 结构蒸馏2
				s_h_norm = torch.norm(s_h)
				s_t_norm = torch.norm(s_t)
				t_h_norm = torch.norm(t_h)
				t_t_norm = torch.norm(t_t)
				t_loss_res += F.smooth_l1_loss(torch.div(s_h_norm, s_t_norm), torch.div(t_h_norm, t_t_norm))
			loss_res = self.l * t_loss_res + loss_res
		return loss_res