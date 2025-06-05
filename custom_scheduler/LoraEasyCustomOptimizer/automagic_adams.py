import torch
import math
from typing import List
import torch.nn.functional as F
from torch.nn.functional import normalize

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None
    print("[Automagic_AdamS] 找不到 bitsandbytes，將以 FP16 儲存狀態。")

class Automagic_AdamS(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-6,
        min_lr: float = 1e-7,
        max_lr: float = 1e-3,
        lr_bump: float = 3e-6,
        eps: float = 1e-8,
        clip_threshold: float = 1.0,
        betas: tuple = (0.5, 0.98, 0.99),
        alpha_decay: float = 0.9995,
        eta: float = 2,
        d_coef: float = 2,
        weight_decay: float = 1.0,
        weight_decay2: float = 4e-5,
        warmup_steps: int = 500,
        full_finetune: bool = False,
        use_8bit: bool = False,
    ):
        self.lr = lr
        self.use_8bit = bool(use_8bit and bnb is not None)
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
            weight_decay2=weight_decay2,
            warmup_steps=warmup_steps,
            full_finetune=full_finetune,
        )
        super().__init__(params, defaults)
        self.weight_decay = weight_decay
        self._step = 1
        self.warmup_steps = warmup_steps

    def _q(self, t: torch.Tensor):
        """Quantize tensor to (int8, scale) 兩元組。"""
        if not self.use_8bit:
            return t
        q, s = bnb.functional.quantize_8bit(t)
        return (q, s)

    def _dq(self, q_or_t):
        """還原成 FP16/FP32 張量。"""
        if not self.use_8bit:
            return q_or_t
        q, s = q_or_t
        return bnb.functional.dequantize_8bit(q, s)

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
    def orthograd_(self, p, grad, state):
        w = p.view(-1)
        w_norm = w.norm(2)
        if w_norm < 1e-30:
            return grad
        G_shape = grad.shape
        g = grad.view(-1)
        g_norm = g.norm(2)
        dot_wd = torch.dot(w, g)
        if G_shape[0] * G_shape[1] > 50 ** 2:
            ema_decay = 0.9
            cos_val = dot_wd / (w_norm * g_norm)
            if "cos_sim" not in state or state["cos_sim"] == 0:
                state["cos_sim"] = cos_val.item()
            else:
                state["cos_sim"] = (ema_decay * state["cos_sim"] + (1 - ema_decay) * cos_val.item())

        if state["cos_sim"] < - 0.8 or G_shape[0] * G_shape[1] <= 50 ** 2:
            dot_ww = torch.dot(w, w)
            proj = dot_wd / (dot_ww + 1e-30)
            g_orth = g - w * proj
            g_orth_scaled = g_orth * (g_norm / (g_orth.norm(2) + 1e-30))
            return g_orth_scaled.view(G_shape)
        else:
            return grad

    def _ratio(self, delta_new, delta_p):
        curr_norm, prev_norm = torch.norm(delta_new), torch.norm(delta_p)
        ratio = (curr_norm - prev_norm) / (curr_norm + 1e-8)
        return torch.nn.functional.hardtanh(ratio, 0.0, 1.0)

    def soft_collision_update(self, weight: torch.Tensor,
                             grad: torch.Tensor,
                             coll_coef: float = 0.1) -> torch.Tensor:

        w_norm = F.normalize(weight, dim=1)           # (N, D)
        cos_w = w_norm @ w_norm.t()                   # (N, N)
        cos_w.fill_diagonal_(0.0)
        g_norm = F.normalize(grad, dim=1)
        cos_g = g_norm @ g_norm.t()
        cos_g.fill_diagonal_(0.0)
        coeff = cos_w * cos_g
        delta_g = - coeff @ grad
        new_grad = grad + coll_coef * delta_g
        return new_grad

    def _init_state(self, p, group=None):
        device, shape = p.device, p.shape
        state = self.state[p]
        state.setdefault("lr_max", 1e-6)
        state.setdefault("step", 0)
        state.setdefault("decay_step", 0)
        state.setdefault("cos_sim", 0)
        # lr_mask
        lr_init = torch.ones(shape, device=device, dtype=torch.float16) * self.lr
        state.setdefault("lr_mask", self._q(lr_init))
        state.setdefault("avg_lr", float(self.lr))
        state.setdefault('last_polarity', torch.zeros(shape, dtype=torch.bool, device=device))
        exp_init = torch.zeros_like(p)
        state.setdefault("exp_avg", self._q(exp_init))
        if group['full_finetune'] == False:
            state.setdefault("pre", None)
            # ==== ALLoRA ====
            #ALLoRA: Adaptive Learning Rate Mitigates LoRA Fatal Flaws
            #https://arxiv.org/abs/2410.09692
            if len(p.shape) == 2:
                row_norm = p.norm(dim=1, keepdim=True)
                state["row_scaling"] = 1.0 / torch.sqrt(row_norm + 1.0 / (group['eta']**2))
        else:
            pre_init = p.clone()
            state.setdefault("pre", self._q(pre_init))

    def power_iteration(self, W, num_iters=3):
        device = W.device
        v = torch.randn(W.shape[1], 1, device=device)
        v = v / v.norm()
        for _ in range(num_iters):
            v = W.t() @ (W @ v)
            v = v / v.norm()
        sigma = (W @ v).norm()
        return sigma.item()

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
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

            if self._step < self.warmup_steps / 2 and self.weight_decay > 0:
                mean_norm = abs_all_group_grads.mean()
                std_norm = abs_all_group_grads.std(unbiased=False) + 1e-12

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
                exp_avg = state['exp_avg']

                # === 正交梯度 ===
                #Grokking at the Edge of Numerical Stability

                #https://arxiv.org/abs/2501.04697
                #https://github.com/LoganBooker/prodigy-plus-schedule-free/tree/dev
                interval = int(math.ceil(0.5 / (1 - beta3)))
                if p.ndim == 2 and grad.ndim == 2:
                    if state["cos_sim"] < -0.8 or p.data.shape[0] * p.data.shape[1] <= 50 ** 2:
                        grad.copy_(self.orthograd_(p, grad, state))
                    elif interval > 0 and state["step"] % interval == 0:
                        exp_avg.copy_(self.orthograd_(p, exp_avg, state))

                # ==== Simplified-AdEMAMix ====
                #Connections between Schedule-Free Optimizers, AdEMAMix, and Accelerated SGD Variants
                #https://arxiv.org/abs/2502.02431
                #https://github.com/DepenM/Simplified-AdEMAMix
                exp_avg.mul_(beta3).add_(grad)
                alpha_grad = alpha * grad
                alpha_grad_p2 = alpha_grad ** 2
                final_exp_avg =  beta1 * exp_avg + alpha * grad
                final_exp_avg_p2 =final_exp_avg ** 2

                # ==== AdamS ====
                #AdamS: Momentum Itself Can Be A Normalizer for LLM Pretraining and Post-training
                #https://arxiv.org/abs/2502.02431
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

                if state["step"] < group["warmup_steps"]:
                    delta_p = p - state["pre"] if state["pre"] else p
                    pre = state["pre"] if state["pre"] else torch.zeros_like(p)
                    condition = -torch.sum(p.grad * delta_p)
                else:
                    if 'pre' in state:
                        del state["pre"]

                # ==== Automagic lrmask ====
                # https://github.com/ostris/ai-toolkit/blob/main/toolkit/optimizers/automagic.py
                lr_decay = 1
                if state["step"] < group["warmup_steps"]:
                    last_polarity = state['last_polarity']
                    current_polarity = (grad > 0)
                    sign_agree = torch.where(last_polarity == current_polarity, 1.0, -1.0)
                    state['last_polarity'] = current_polarity
                    lr_mask = state['lr_mask']
                    #Prodigy: An Expeditiously Adaptive Parameter-Free Learner
                    #https://arxiv.org/pdf/2306.06101
                    #https://github.com/konstmish/prodigy
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
                        #Neural Thermodynamic Laws for Large Language Model Training
                        #https://arxiv.org/abs/2505.10559
                        state["decay_step"] += 1
                        decay_progress = min(state["decay_step"], 3000) / 3000
                        allowed_min_ratio = 1.0 - decay_progress
                        lr_decay = max(max(group["lr"] / state["lr_max"], allowed_min_ratio), 0.1)

                allora =  state["row_scaling"] if "row_scaling" in state else 1

                # ==== VRAdam ====
                #A Physics-Inspired Optimizer: Velocity Regularized Adam
                #https://arxiv.org/abs/2505.13196
                vr = 1 / (1+ min(3 * (final_exp_avg_p2).sum(),10))

                lr_tweak = lr_decay * allora * vr
                new_lr = new_lr * lr_tweak
                update.mul_(new_lr)

                #Mirror, Mirror of the Flow: How Does Regularization Shape Implicit Bias?
                #https://arxiv.org/abs/2504.12883
                do_spd = False
                if state["step"] < group["warmup_steps"]:
                    if p.ndim == 2 and p.data.shape[0] * p.data.shape[1] <= 50 ** 2:
                        if state["step"] < group["warmup_steps"] / 2:
                            #Adaptive Weight Decay for Deep Neural Networks
                            #https://arxiv.org/abs/1907.08931
                            param_abs_grad = torch.abs(p.grad).mean()
                            norm_grad = (param_abs_grad - mean_norm) / std_norm
                            ada_alpha = 4
                            theta = 2 / (1 + torch.exp(-ada_alpha * norm_grad))
                            p.data.mul_(1 - new_lr * group["weight_decay2"] * theta)
                    else:
                        # === SPD 選擇性投影 weight decay ===
                        #Rethinking Weight Decay for Robust Fine-Tuning of Foundation Models
                        #https://arxiv.org/abs/2411.01713
                        #https://github.com/GT-RIPL/Selective-Projection-Decay/tree/main
                        if condition < 0.0:
                            do_spd = True
                            new_p = p - update
                            delta_new = new_p - pre
                            ratio = self._ratio(delta_new, delta_p)
                            new_p = new_p - group["weight_decay"] * ratio * delta_new
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
