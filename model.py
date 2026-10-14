import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from flow_matching.path import AffineProbPath
from flow_matching.scheduler import (
	CondOTScheduler,
	CosineScheduler,
	LinearVPScheduler,
	PolynomialConvexScheduler,
	VPScheduler,
)
from params import args
from utils.utils import *

init = nn.init.xavier_uniform_

class Model(nn.Module):
	def __init__(self, image_embedding, text_embedding, audio_embedding=None):
		super(Model, self).__init__()

		self.uEmbeds = nn.Parameter(init(torch.empty(args.user, args.latdim)))
		self.iEmbeds = nn.Parameter(init(torch.empty(args.item, args.latdim)))
		self.gcnLayers = nn.Sequential(*[GCNLayer() for _ in range(args.gnn_layer)])
		self.edgeDropper = SpAdjDropEdge(args.keepRate)

		if args.trans == 1:
			self.image_trans = nn.Linear(args.image_feat_dim, args.latdim)
			self.text_trans = nn.Linear(args.text_feat_dim, args.latdim)
		elif args.trans == 0:
			self.image_trans = nn.Parameter(init(torch.empty(size=(args.image_feat_dim, args.latdim))))
			self.text_trans = nn.Parameter(init(torch.empty(size=(args.text_feat_dim, args.latdim))))
		else:
			self.image_trans = nn.Parameter(init(torch.empty(size=(args.image_feat_dim, args.latdim))))
			self.text_trans = nn.Linear(args.text_feat_dim, args.latdim)

		if audio_embedding is not None:
			if args.trans == 1:
				self.audio_trans = nn.Linear(args.audio_feat_dim, args.latdim)
			else:
				self.audio_trans = nn.Parameter(init(torch.empty(size=(args.audio_feat_dim, args.latdim))))

		self.image_embedding = image_embedding
		self.text_embedding = text_embedding
		self.audio_embedding = audio_embedding

		if audio_embedding is not None:
			self.modal_weight = nn.Parameter(torch.Tensor([0.3333, 0.3333, 0.3333]))
		else:
			self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))

		self.softmax = nn.Softmax(dim=0)
		self.leakyrelu = nn.LeakyReLU(0.2)

		self.rop2 = ROProjection2(args.clear_rank, args.clear_lam, args.clear_text) if args.clear_rank > 0 else None
		self.rop3 = ROProjection3(args.clear_rank, args.clear_lam) if args.clear_rank > 0 else None

	def getItemEmbeds(self):
		return self.iEmbeds

	def getUserEmbeds(self):
		return self.uEmbeds
	

	def _modal_feats(self, kind=None):
		if args.trans == 0:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			text_feats = self.leakyrelu(torch.mm(self.text_embedding, self.text_trans))
		elif args.trans == 1:
			image_feats = self.image_trans(self.image_embedding)
			text_feats = self.text_trans(self.text_embedding)
		else:
			image_feats = self.leakyrelu(torch.mm(self.image_embedding, self.image_trans))
			text_feats = self.text_trans(self.text_embedding)

		if self.audio_embedding is not None:
			if args.trans != 1:
				audio_feats = self.leakyrelu(torch.mm(self.audio_embedding, self.audio_trans))
			else:
				audio_feats = self.audio_trans(self.audio_embedding)
		else:
			audio_feats = None


		if self.rop2 is not None and args.data != 'tiktok':
			image_feats, text_feats = self.rop2(image_feats, text_feats)
		if self.rop3 is not None and args.data == 'tiktok' :	
			image_feats, text_feats, audio_feats = self.rop3(image_feats, text_feats, audio_feats)
		
		return image_feats, text_feats, audio_feats

	def getImageFeats(self):
		image_feats, _, _ = self._modal_feats()
		return image_feats

	def getTextFeats(self):
		_, text_feats, _ = self._modal_feats()
		return text_feats
	
	def getAudioFeats(self):
		_, _, audio_feats = self._modal_feats()
		return audio_feats

	def forward_MM(self, adj, image_adj, text_adj, audio_adj=None):


		image_feats, text_feats, audio_feats = self._modal_feats(kind='GA')
		weight = self.softmax(self.modal_weight)


		embedsImageAdj = torch.concat([self.uEmbeds, self.iEmbeds])
		embedsImageAdj = torch.spmm(image_adj, embedsImageAdj)


		embedsImage = torch.concat([self.uEmbeds, F.normalize(image_feats)])
		embedsImage = torch.spmm(adj, embedsImage)


		embedsImage_ = torch.concat([embedsImage[:args.user], self.iEmbeds])
		embedsImage_ = torch.spmm(adj, embedsImage_)


		embedsImage += embedsImage_
		embedsImage += args.ris_adj_lambda * embedsImageAdj

		embedsTextAdj = torch.concat([self.uEmbeds, self.iEmbeds])
		embedsTextAdj = torch.spmm(text_adj, embedsTextAdj)

		embedsText = torch.concat([self.uEmbeds, F.normalize(text_feats)])
		embedsText = torch.spmm(adj, embedsText)

		embedsText_ = torch.concat([embedsText[:args.user], self.iEmbeds])
		embedsText_ = torch.spmm(adj, embedsText_)

		embedsText += embedsText_
		embedsText += args.ris_adj_lambda * embedsTextAdj

		if audio_adj is not None:
			embedsAudioAdj = torch.concat([self.uEmbeds, self.iEmbeds])
			embedsAudioAdj = torch.spmm(audio_adj, embedsAudioAdj)

			embedsAudio = torch.concat([self.uEmbeds, F.normalize(audio_feats)])
			embedsAudio = torch.spmm(adj, embedsAudio)

			embedsAudio_ = torch.concat([embedsAudio[:args.user], self.iEmbeds])
			embedsAudio_ = torch.spmm(adj, embedsAudio_)

			embedsAudio += embedsAudio_
			embedsAudio += args.ris_adj_lambda * embedsAudioAdj

		if audio_adj is None:
			embedsModal = weight[0] * embedsImage + weight[1] * embedsText
		else:
			embedsModal = weight[0] * embedsImage + weight[1] * embedsText + weight[2] * embedsAudio
		

		embeds = embedsModal
		embedsLst = [embeds]
		for gcn in self.gcnLayers:
			embeds = gcn(adj, embedsLst[-1])
			embedsLst.append(embeds)

		embeds = sum(embedsLst)
		embeds = embeds + args.ris_lambda * F.normalize(embedsModal)
		return embeds[:args.user], embeds[args.user:]

	def forward_cl_MM(self, adj, image_adj, text_adj, audio_adj=None):


		image_feats, text_feats, audio_feats = self._modal_feats(kind='CL')

		embedsImage = torch.concat([self.uEmbeds, F.normalize(image_feats)])
		embedsImage = torch.spmm(image_adj, embedsImage)

		embedsText = torch.concat([self.uEmbeds, F.normalize(text_feats)])
		embedsText = torch.spmm(text_adj, embedsText)

		if audio_adj is not None:
			embedsAudio = torch.concat([self.uEmbeds, F.normalize(audio_feats)])
			embedsAudio = torch.spmm(audio_adj, embedsAudio)

		embeds1 = embedsImage
		embedsLst1 = [embeds1]
		for gcn in self.gcnLayers:
			embeds1 = gcn(adj, embedsLst1[-1])
			embedsLst1.append(embeds1)
		embeds1 = sum(embedsLst1)

		embeds2 = embedsText
		embedsLst2 = [embeds2]
		for gcn in self.gcnLayers:
			embeds2 = gcn(adj, embedsLst2[-1])
			embedsLst2.append(embeds2)
		embeds2 = sum(embedsLst2)

		if audio_adj is not None:
			embeds3 = embedsAudio
			embedsLst3 = [embeds3]
			for gcn in self.gcnLayers:
				embeds3 = gcn(adj, embedsLst3[-1])
				embedsLst3.append(embeds3)
			embeds3 = sum(embedsLst3)

		if audio_adj is None:
			return embeds1[:args.user], embeds1[args.user:], embeds2[:args.user], embeds2[args.user:]
		return embeds1[:args.user], embeds1[args.user:], embeds2[:args.user], embeds2[args.user:], embeds3[:args.user], embeds3[args.user:]

	def reg_loss(self):
		ret = 0
		ret += self.uEmbeds.norm(2).square()
		ret += self.iEmbeds.norm(2).square()
		return ret

class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()

	def forward(self, adj, embeds):
		return torch.spmm(adj, embeds)

class SpAdjDropEdge(nn.Module):
	def __init__(self, keepRate):
		super(SpAdjDropEdge, self).__init__()
		self.keepRate = keepRate

	def forward(self, adj):
		vals = adj._values()
		idxs = adj._indices()
		edgeNum = vals.size()
		mask = ((torch.rand(edgeNum, device=vals.device) + self.keepRate).floor()).type(torch.bool)
		newVals = vals[mask] / self.keepRate
		newIdxs = idxs[:, mask]
		return torch.sparse_coo_tensor(newIdxs, newVals, adj.shape)


class ROProjection2(nn.Module):
	def __init__(self, rank=4, lam=1.0, project_text=True):
		super(ROProjection2, self).__init__()
		self.rank = rank
		self.lam = lam
		self.project_text = project_text

	def forward(self, x, y):
		x_c = F.normalize(x - x.mean(0, keepdim=True), dim=-1)
		y_c = F.normalize(y - y.mean(0, keepdim=True), dim=-1)
		c = x_c.t() @ y_c
		u, _, vh = torch.linalg.svd(c, full_matrices=False)
		ur = u[:, :self.rank]
		vr = vh[:self.rank, :].t()
		x_perp = x - self.lam * (x @ ur) @ ur.t()
		if self.project_text:
			y_perp = y - self.lam * (y @ vr) @ vr.t()
		else:
			y_perp = y
		return x_perp, y_perp


class ROProjection3(nn.Module):
    def __init__(self, rank=4, lam=1.0, project_modalities=(True, True, True),
                 detach_basis=True, eps=1e-6):
        super(ROProjection3, self).__init__()
        self.rank = rank
        self.lam = lam
        self.project_modalities = project_modalities
        self.detach_basis = detach_basis
        self.eps = eps
	

    def _pair_svd_dirs(self, x, y):


        x_c = x - x.mean(0, keepdim=True)
        y_c = y - y.mean(0, keepdim=True)

        c = x_c.t() @ y_c / x_c.size(0)

        u, _, vh = torch.linalg.svd(c, full_matrices=False)

        rank = min(self.rank, u.size(1), vh.size(0))

        ux = u[:, :rank]
        vy = vh[:rank, :].t()

        return ux, vy
	

    def _orth_basis(self, b):


        if b.numel() == 0:
            return b

        u, s, _ = torch.linalg.svd(b, full_matrices=False)

        if s.numel() == 0:
            return b[:, :0]

        tol = self.eps * max(b.shape) * s[0].detach()
        r = int((s > tol).sum().item())

        if r == 0:
            return b[:, :0]

        return u[:, :r]
	

    def _project_null(self, x, q, enabled=True):


        if not enabled or q.numel() == 0:
            return x

        return x - self.lam * (x @ q) @ q.t()
	

    def _build_bases(self, x1, x2, x3):


        u12, w12 = self._pair_svd_dirs(x1, x2)


        u13, w13 = self._pair_svd_dirs(x1, x3)


        u23, w23 = self._pair_svd_dirs(x2, x3)


        b1 = torch.cat([u12, u13], dim=1)


        b2 = torch.cat([w12, u23], dim=1)


        b3 = torch.cat([w13, w23], dim=1)


        q1 = self._orth_basis(b1)
        q2 = self._orth_basis(b2)
        q3 = self._orth_basis(b3)

        return q1, q2, q3

    def forward(self, x1, x2, x3):


        if self.detach_basis:
            with torch.no_grad():
                q1, q2, q3 = self._build_bases(x1, x2, x3)
        else:
            q1, q2, q3 = self._build_bases(x1, x2, x3)

        x1_perp = self._project_null(x1, q1, self.project_modalities[0])
        x2_perp = self._project_null(x2, q2, self.project_modalities[1])
        x3_perp = self._project_null(x3, q3, self.project_modalities[2])

        return x1_perp, x2_perp, x3_perp

class _TimeCondResBlock(nn.Module):


	def __init__(self, dim, time_dim, dropout=0.0):
		super().__init__()
		self.norm = nn.LayerNorm(dim)
		self.linear1 = nn.Linear(dim, dim)
		self.linear2 = nn.Linear(dim, dim)
		self.act = nn.SiLU()
		self.drop = nn.Dropout(dropout)
		self.time_mlp = nn.Linear(time_dim, dim * 2)
		nn.init.zeros_(self.linear2.weight)
		nn.init.zeros_(self.linear2.bias)

	def forward(self, x, t_emb):
		h = self.norm(x)
		scale, shift = self.time_mlp(t_emb).chunk(2, dim=-1)
		h = h * (1 + scale) + shift
		h = self.act(self.linear1(h))
		h = self.drop(h)
		h = self.linear2(h)
		return x + h

class VelocityField(nn.Module):
	def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5):
		super().__init__()
		self.input_dim = in_dims[0]
		self.output_dim = out_dims[-1]
		self.time_emb_dim = emb_size
		self.norm = norm

		hidden = in_dims[1] if len(in_dims) > 1 else max(256, self.input_dim // 4)
		bottleneck = max(hidden // 2, 128)

		self.time_mlp = nn.Sequential(
			nn.Linear(emb_size, emb_size * 2),
			nn.SiLU(),
			nn.Linear(emb_size * 2, emb_size),
		)

		self.proj_in = nn.Linear(self.input_dim, hidden)
		self.enc_block = _TimeCondResBlock(hidden, emb_size, dropout)
		self.down_proj = nn.Linear(hidden, bottleneck)

		self.mid_block1 = _TimeCondResBlock(bottleneck, emb_size, dropout)
		self.mid_block2 = _TimeCondResBlock(bottleneck, emb_size, dropout)

		self.up_proj = nn.Linear(bottleneck, hidden)
		self.skip_linear = nn.Linear(hidden * 2, hidden)
		self.dec_block = _TimeCondResBlock(hidden, emb_size, dropout)

		self.proj_out = nn.Linear(hidden, self.output_dim)
		self.drop = nn.Dropout(dropout)

		if self.input_dim == self.output_dim:
			self.shortcut = None
		else:
			self.shortcut = nn.Linear(self.input_dim, self.output_dim, bias=False)

		self._init_weights()

	def _init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear) and m is not self.proj_out:
				nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					nn.init.zeros_(m.bias)
		nn.init.zeros_(self.proj_out.weight)
		nn.init.zeros_(self.proj_out.bias)

	def _time_embed(self, timesteps, device):
		half = self.time_emb_dim // 2
		freqs = torch.exp(
			-math.log(10000) * torch.arange(half, dtype=torch.float32, device=device) / half
		)
		angles = timesteps[:, None].float() * freqs[None]
		emb = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
		if self.time_emb_dim % 2:
			emb = F.pad(emb, (0, 1))
		return self.time_mlp(emb)

	def forward(self, x, timesteps, mess_dropout=True):
		t_emb = self._time_embed(timesteps, x.device)

		if self.norm:
			x = F.normalize(x, dim=-1)
		x_in = x
		if mess_dropout:
			x = self.drop(x)

		h = self.proj_in(x)
		h = self.enc_block(h, t_emb)
		h_skip = h

		h = self.down_proj(h)
		h = self.mid_block1(h, t_emb)
		h = self.mid_block2(h, t_emb)

		h = self.up_proj(h)
		h = self.skip_linear(torch.cat([h, h_skip], dim=-1))
		h = self.dec_block(h, t_emb)

		out = self.proj_out(h)
		if self.shortcut is not None:
			out = out + self.shortcut(x_in)
		else:
			out = out + x_in
		return out

class FlowMatching(nn.Module):
	def __init__(self, noise_scale=0.1, scheduler_name='cosine', ode_method='midpoint'):
		super(FlowMatching, self).__init__()
		self.noise_scale = noise_scale
		self.ode_method = ode_method.lower()
		self.prob_path = AffineProbPath(scheduler=_build_scheduler(scheduler_name))

	def _velocity_to_target(self, velocity, x_t, t):
		s = self.prob_path.scheduler(t)
		denom = s.d_alpha_t * s.sigma_t - s.d_sigma_t * s.alpha_t
		a_t = (-s.d_sigma_t / denom).view(-1, 1)
		b_t = (s.sigma_t / denom).view(-1, 1)
		return a_t * x_t + b_t * velocity

	def _ode_step(self, model, x, t_val, dt):
		batch_size = x.shape[0]
		t0 = torch.full((batch_size,), t_val, device=x.device)
		if self.ode_method == 'rk4':
			t1 = torch.full((batch_size,), t_val + 0.5 * dt, device=x.device)
			t2 = torch.full((batch_size,), t_val + dt, device=x.device)
			v1 = model(x, t0 * 1000, mess_dropout=False)
			v2 = model(x + 0.5 * dt * v1, t1 * 1000, mess_dropout=False)
			v3 = model(x + 0.5 * dt * v2, t1 * 1000, mess_dropout=False)
			v4 = model(x + dt * v3, t2 * 1000, mess_dropout=False)
			return x + (dt / 6.0) * (v1 + 2.0 * v2 + 2.0 * v3 + v4)
		if self.ode_method == 'midpoint':
			t1 = torch.full((batch_size,), t_val + 0.5 * dt, device=x.device)
			v1 = model(x, t0 * 1000, mess_dropout=False)
			v2 = model(x + 0.5 * dt * v1, t1 * 1000, mess_dropout=False)
			return x + dt * v2
		v = model(x, t0 * 1000, mess_dropout=False)
		return x + dt * v

	def p_sample(self, model, x_start, steps):
		if steps <= 0:
			return x_start
		x = x_start.mean(dim=0, keepdim=True) + torch.randn_like(x_start) * self.noise_scale
		dt = 1.0 / steps
		for i in range(steps):
			x = self._ode_step(model, x, i * dt, dt)
			x = x + torch.randn_like(x) * (self.noise_scale * math.sqrt(dt))
		return x

	def training_losses(self, model, x_start, itmEmbeds, batch_index, model_feats):
		batch_size = x_start.size(0)
		if args.data == 'baby':
			t = torch.distributions.Beta(2.0, 2.0).sample((batch_size,)).to(x_start.device)
		else:
			ts = torch.randint(0, max(1, args.steps), (batch_size,), device=x_start.device)
			t = (ts.float() + torch.rand(batch_size, device=x_start.device)) / float(max(1, args.steps))
		t = t.clamp(1e-4, 1.0 - 1e-4)
		batch_mean = x_start.mean(dim=0, keepdim=True)
		x_0 = batch_mean + torch.randn_like(x_start) * self.noise_scale
		x_1 = x_start

		path_sample = self.prob_path.sample(x_0=x_0, x_1=x_1, t=t)
		pred_v = model(path_sample.x_t, t * 1000)
		diff2 = (pred_v - path_sample.dx_t) ** 2
		flow_loss = self.mean_flat(diff2)
		x1_pred = self._velocity_to_target(pred_v, path_sample.x_t, t)
		

		usr_model_embeds = torch.mm(x1_pred, model_feats)
		usr_id_embeds = torch.mm(x_1, itmEmbeds)
		gc_loss = self.mean_flat((usr_model_embeds - usr_id_embeds) ** 2)
		return flow_loss, gc_loss

	def mean_flat(self, tensor):
		return tensor.mean(dim=list(range(1, len(tensor.shape))))

def _build_scheduler(name):
	name = name.lower()
	if name == 'cosine':
		return CosineScheduler()
	if name == 'vp':
		return VPScheduler()
	if name == 'linear_vp':
		return LinearVPScheduler()
	if name == 'poly2':
		return PolynomialConvexScheduler(n=2)
	if name == 'poly3':
		return PolynomialConvexScheduler(n=3)
	return CondOTScheduler()
