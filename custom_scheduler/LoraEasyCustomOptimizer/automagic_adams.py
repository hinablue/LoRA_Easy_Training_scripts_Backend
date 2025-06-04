import torch
import math
from typing import List
import torch.nn.functional as F
from torch.nn.functional import normalize

class Automagic_AdamS(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-6,
        min_lr=1e-7,
        max_lr=1e-3,
        lr_bump=3e-6,
        eps=1e-8,
        clip_threshold=1.0,
        betas=(0.5, 0.98, 0.99),
        alpha_decay=0.9995,
        eta=2,
        d_coef=2,
        weight_decay=4e-5,
        warmup_steps=500,
        full_finetune=False,
    ):
        self.lr = lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_bump = lr_bump
        self.full_finetune = full_finetune
        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            betas=betas,
            alpha_decay=alpha_decay,
            eta=eta,
            d_coef=d_coef,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            full_finetune=full_finetune,
        )
        super().__init__(params, defaults)
        self.weight_decay = weight_decay
        self._step = 1
        self.warmup_steps = warmup_steps

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5 + 1e-10)

    def _get_group_lr(self, group):
        group_lrs = []
        for p in group["params"]:
            state = self.state[p]
            if 'avg_lr' in state:
                group_lrs.append(state['avg_lr'])
        return float(torch.mean(torch.tensor(group_lrs))) if group_lrs else self.lr

    # Implementation from: https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability/blob/main/orthograd.py
    def orthograd_(self, p, grad):
        if p.norm(2) <= 1e-30:
            return grad

        G_shape = grad.shape
        w = p.view(-1)
        g = grad.view(-1)
        g_norm = g.norm(2)

        proj = torch.dot(w, g) / torch.dot(w, w).add(1e-30)
        g_orth = g.sub_(w, alpha=proj)
        g_orth_scaled = g_orth.mul_(g_norm / g_orth.norm(2).add(1e-30))

        return g_orth_scaled.view(G_shape)

    def _ratio(self, new_p, p, pre):
        curr_norm, prev_norm = torch.norm(new_p - pre), torch.norm(p - pre)
        ratio = (curr_norm - prev_norm) / (curr_norm + 1e-8)
        return torch.nn.functional.hardtanh(ratio, 0.0, 1.0)

    def _init_state(self, p, group=None):
        device = p.device
        shape = p.shape
        state = self.state[p]
        state.setdefault("lr_max", 1e-6)
        state.setdefault("decay_step", 0)
        state.setdefault("need_ortho", False)
        state.setdefault("step", 0)
        state.setdefault('lr_mask', torch.ones(shape, device=device, dtype=torch.float16) * self.lr)
        state.setdefault('avg_lr', float(self.lr))
        state.setdefault('last_polarity', torch.zeros(shape, dtype=torch.bool, device=device))
        state.setdefault("exp_avg", torch.zeros_like(p))
        if group['full_finetune'] == False:
            state.setdefault("pre", None)
            # ==== ALLoRA ====
            #ALLoRA: Adaptive Learning Rate Mitigates LoRA Fatal Flaws
            #https://arxiv.org/abs/2410.09692
            if len(p.shape) == 2:
                row_norm = p.norm(dim=1, keepdim=True)
                state["row_scaling"] = 1.0 / torch.sqrt(row_norm + 1.0 / (group['eta']**2))
        else:
            state.setdefault("pre", p.clone())

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        smoothing = 0.9 
        for group in self.param_groups:
            grads_this_group = []
            for p in group["params"]:
                if p.grad is not None:
                    grads_this_group.append(p.grad.view(-1))
            if len(grads_this_group) == 0:
                continue
            all_group_grads = torch.cat(grads_this_group)
            abs_all_group_grads = torch.abs(all_group_grads)
            sum_abs_all_group_grads = torch.sum(abs_all_group_grads) + 1e-12

            for p in group["params"]:
                if p.grad is None or not p.requires_grad:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    self._init_state(p, group)
                if 'step' not in state:
                    state['step'] = 0
                state["step"] += 1
                self._step = state["step"] + 1
                
                # === grad 初始化 ===
                grad = p.grad

                # ==== AGR自適應梯度正則 ====
                #Adaptive Gradient Regularization: A Faster and Generalizable Optimization Technique for Deep Neural Networks
                #https://arxiv.org/pdf/2407.16944
                abs_grad = torch.abs(grad)
                agr = abs_grad / sum_abs_all_group_grads
                grad = grad * (1 - agr)
                
                beta1, beta2, beta3 = group["betas"]
                eps = group["eps"]
                alpha = (1 - beta1) / (1 - beta3)
                # ===   ===
                exp_avg = state['exp_avg']
                interval = int(math.ceil(0.5 / (1 - beta3)))
                if interval > 0 and state["step"] % interval == 0:
                    cos_sim = F.cosine_similarity(exp_avg.view(-1), p.view(-1), dim=0)
                    if cos_sim < -0.9:
                        exp_avg.copy_(self.orthograd_(p, exp_avg))
                        state["need_ortho"] = True
                    elif state["need_ortho"]:
                        cos_sim = F.cosine_similarity(grad.view(-1), p.view(-1), dim=0)
                        if cos_sim > -0.9:
                            state["need_ortho"] = False
                if state["need_ortho"] and state["step"] > group["warmup_steps"]:
                    grad = self.orthograd_(p, grad)

                exp_avg.mul_(beta3).add_(grad)
                alpha_grad = alpha * grad
                alpha_grad_p2 = alpha_grad ** 2
                final_exp_avg =  beta1 * exp_avg + alpha * grad
                final_exp_avg_p2 =final_exp_avg ** 2
                exp_avg_sq = final_exp_avg_p2.mul_(beta2).add_(alpha_grad_p2, alpha=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(eps)
                update = final_exp_avg / denom

                #=== Cautious ===
                #Cautious Optimizers: Improving Training with One Line of Code
                #https://arxiv.org/abs/2411.16085
                #https://github.com/kyleliang919/C-Optim
                mask = (update * grad > 0).to(grad.dtype)
                mask_ratio = mask.mean()
                mask.div_(mask_ratio.clamp_(min=1e-3))
                update = update * mask

                # ==== Automagic lrmask ====
                # https://github.com/ostris/ai-toolkit/blob/main/toolkit/optimizers/automagic.py

                if state["step"] < group["warmup_steps"]:
                    last_polarity = state['last_polarity']
                    current_polarity = (grad > 0)
                    sign_agree = torch.where(last_polarity == current_polarity, 1.0, -1.0)
                    state['last_polarity'] = current_polarity
                    lr_mask = state['lr_mask']
                    condition = -torch.sum(p.grad * p)
                    if state["step"] < group["warmup_steps"] / 2:
                        lr_bump_pos = self.lr_bump * group['d_coef'] if condition > 0.0 else self.lr_bump
                        lr_bump_neg = self.lr_bump * group['d_coef'] if condition < 0.0 else self.lr_bump
                    else:
                        lr_bump_pos, lr_bump_neg = self.lr_bump, self.lr_bump
                    new_lr = torch.where(
                        sign_agree > 0,
                        lr_mask + lr_bump_pos,
                        lr_mask - lr_bump_neg
                    )
                    if group["lr"] >= state["lr_max"]:
                        state["lr_max"] = group["lr"]
                    new_lr = torch.clamp(new_lr, min=self.min_lr, max=self.max_lr)
                    state['lr_mask'] = new_lr
                    state['avg_lr'] = torch.mean(new_lr).item()
                else:
                    if 'last_polarity' in state:
                        del state['last_polarity']
                    new_lr = state['lr_mask']
                    if group["lr"] >= state["lr_max"]:
                        state["decay_step"] = 0
                        state["lr_max"] = group["lr"]
                    elif group["lr"] < state["lr_max"]:
                        new_lr = new_lr * max(group["lr"] / state["lr_max"],0.1)

                if "row_scaling" in state:
                    new_lr = new_lr * state["row_scaling"]

                update.mul_(new_lr)

                # === SPD 選擇性投影decay ===
                #Rethinking Weight Decay for Robust Fine-Tuning of Foundation Models
                #https://arxiv.org/abs/2411.01713
                #https://github.com/GT-RIPL/Selective-Projection-Decay/tree/main
                #Mirror, Mirror of the Flow: How Does Regularization Shape Implicit Bias?
                #https://arxiv.org/abs/2504.12883
                do_spd = False
                if state["step"] < group["warmup_steps"]:
                    pre = torch.zeros_like(p)
                    if condition < 0.0:
                        do_spd = True
                        new_p = p - update
                        ratio = self._ratio(new_p, p, pre)
                        new_p = new_p - group["weight_decay"] * ratio * (new_p - pre)
                        p.copy_(new_p)
                if not do_spd:
                    p.add_(-update)

        return loss

    def state_dict(self):
        state = super().state_dict()
        state['magic_version'] = 1
        return state

    def load_state_dict(self, state_dict):
        if 'magic_version' not in state_dict or state_dict['magic_version'] != 1:
            print('[WARNING] 您載入了非預期state dict，某些動態mask參數可能未正確同步！')
        super().load_state_dict(state_dict)
