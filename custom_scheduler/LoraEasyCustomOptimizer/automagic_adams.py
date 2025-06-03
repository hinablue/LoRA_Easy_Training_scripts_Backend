import torch
import math
from typing import List, Dict, Optional
import torch.nn.functional as F
from torch.nn.functional import normalize
import gc
import weakref

class TensorCache:
    """張量快取池，用於重用相同形狀的張量以減少記憶體分配"""
    def __init__(self, max_size: int = 100):
        self.cache: Dict[tuple, List[torch.Tensor]] = {}
        self.max_size = max_size
        self.access_count = 0

    def get_tensor(self, shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """獲取指定形狀的張量，優先從快取中取得"""
        key = (shape, dtype, device)

        if key in self.cache and self.cache[key]:
            tensor = self.cache[key].pop()
            tensor.zero_()  # 清零重用
            return tensor

        # 快取中沒有，創建新張量
        return torch.zeros(shape, dtype=dtype, device=device)

    def return_tensor(self, tensor: torch.Tensor):
        """將張量歸還到快取池"""
        if not tensor.is_cuda:  # 只快取 CPU 張量以避免 VRAM 碎片
            return

        key = (tuple(tensor.shape), tensor.dtype, tensor.device)

        if key not in self.cache:
            self.cache[key] = []

        if len(self.cache[key]) < self.max_size:
            self.cache[key].append(tensor.detach())

    def clear(self):
        """清空快取"""
        self.cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        cache_size=50,  # 新增：快取大小
        memory_efficient=True,  # 新增：是否啟用記憶體效率模式
    ):
        self.lr = lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_bump = lr_bump
        self.full_finetune = full_finetune
        self.memory_efficient = memory_efficient

        # 初始化張量快取池
        self.tensor_cache = TensorCache(cache_size) if memory_efficient else None

        # 快取常用的計算結果
        self._computation_cache = {}
        self._cache_step = 0

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
            memory_efficient=memory_efficient,
        )
        super().__init__(params, defaults)
        self.weight_decay = weight_decay
        self._step = 1
        self.warmup_steps = warmup_steps

    @staticmethod
    def _rms(tensor):
        """計算 RMS，使用 in-place 操作優化"""
        return tensor.norm(2) / (tensor.numel() ** 0.5 + 1e-10)

    def _get_group_lr(self, group):
        """獲取群組平均學習率，加入快取機制"""
        cache_key = f"group_lr_{id(group)}"
        if cache_key in self._computation_cache and self._cache_step == self._step:
            return self._computation_cache[cache_key]

        group_lrs = []
        for p in group["params"]:
            state = self.state[p]
            if 'avg_lr' in state:
                group_lrs.append(state['avg_lr'])

        result = float(torch.mean(torch.tensor(group_lrs))) if group_lrs else self.lr
        self._computation_cache[cache_key] = result
        return result

    def orthograd_(self, p, grad):
        """正交梯度計算，優化記憶體使用"""
        if p.norm(2) <= 1e-30:
            return grad

        # 使用 view 避免複製
        w = p.view(-1)
        g = grad.view(-1)
        g_norm = g.norm(2)

        # in-place 計算投影
        proj = torch.dot(w, g) / torch.dot(w, w).add_(1e-30)

        # 重用輸入張量進行 in-place 操作
        g_orth = g.clone()  # 只在必要時複製
        g_orth.sub_(w, alpha=proj)
        g_orth_norm = g_orth.norm(2).add_(1e-30)
        g_orth.mul_(g_norm / g_orth_norm)

        return g_orth.view(grad.shape)

    def _ratio(self, new_p, p, pre):
        """比率計算，優化記憶體分配"""
        # 使用 in-place 操作
        curr_norm = torch.norm(new_p - pre)
        prev_norm = torch.norm(p - pre)
        ratio = (curr_norm - prev_norm) / (curr_norm + 1e-8)
        return torch.nn.functional.hardtanh(ratio, 0.0, 1.0)

    def _init_state(self, p, group=None):
        """延遲初始化狀態，只初始化必要的部分"""
        device = p.device
        shape = p.shape
        state = self.state[p]

        # 基本狀態
        state.setdefault("lr_max", 1e-6)
        state.setdefault("decay_step", 0)
        state.setdefault("need_ortho", False)
        state.setdefault("step", 0)
        state.setdefault('avg_lr', float(self.lr))

        # 延遲初始化大型張量
        if 'exp_avg' not in state:
            if self.memory_efficient and self.tensor_cache:
                state["exp_avg"] = self.tensor_cache.get_tensor(shape, p.dtype, device)
            else:
                state["exp_avg"] = torch.zeros_like(p)

        # 只在 warmup 期間初始化學習率遮罩
        if state["step"] < group.get('warmup_steps', 500):
            if 'lr_mask' not in state:
                if group.get('memory_efficient', True):
                    # 使用 float16 減少記憶體
                    state['lr_mask'] = torch.full(shape, self.lr, device=device, dtype=torch.float16)
                else:
                    state['lr_mask'] = torch.ones(shape, device=device, dtype=torch.float32) * self.lr

            if 'last_polarity' not in state:
                state['last_polarity'] = torch.zeros(shape, dtype=torch.bool, device=device)

        # ALLoRA 初始化（僅在需要時）
        if group.get('full_finetune', True) == False and 'row_scaling' not in state:
            if len(p.shape) == 2:
                row_norm = p.norm(dim=1, keepdim=True)
                state["row_scaling"] = 1.0 / torch.sqrt(row_norm + 1.0 / (group.get('eta', 2)**2))

        # pre 狀態處理
        if 'pre' not in state:
            if group.get('full_finetune', True):
                state["pre"] = p.clone() if self.memory_efficient else p.detach().clone()
            else:
                state["pre"] = None

    def _cleanup_cache(self):
        """定期清理快取以釋放記憶體"""
        if self._step % 100 == 0:  # 每100步清理一次
            self._computation_cache.clear()
            if self.tensor_cache:
                # 只保留最近使用的快取
                for key in list(self.tensor_cache.cache.keys()):
                    if len(self.tensor_cache.cache[key]) > 10:
                        self.tensor_cache.cache[key] = self.tensor_cache.cache[key][:10]

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        smoothing = 0.9

        # 更新快取步數
        self._cache_step = self._step

        for group in self.param_groups:
            # 預計算群組級別的梯度統計（一次性計算）
            grads_this_group = []
            for p in group["params"]:
                if p.grad is not None:
                    grads_this_group.append(p.grad.view(-1))

            if len(grads_this_group) == 0:
                continue

            # 使用 in-place 操作
            all_group_grads = torch.cat(grads_this_group)
            sum_abs_all_group_grads = torch.sum(torch.abs(all_group_grads)).add_(1e-12)

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

                # 獲取梯度並進行 AGR 正則化
                grad = p.grad
                if self.memory_efficient:
                    # in-place AGR 計算
                    abs_grad = torch.abs(grad)
                    agr = abs_grad / sum_abs_all_group_grads
                    grad = grad * (1 - agr)  # 可以 in-place 如果不需要保留原始梯度
                else:
                    abs_grad = torch.abs(grad)
                    agr = abs_grad / sum_abs_all_group_grads
                    grad = grad * (1 - agr)

                beta1, beta2, beta3 = group["betas"]
                eps = group["eps"]
                alpha = (1 - beta1) / (1 - beta3)

                exp_avg = state['exp_avg']

                # 正交梯度檢查（減少頻率以節省計算）
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

                # 使用 in-place 操作更新動量
                exp_avg.mul_(beta3).add_(grad)

                # 計算更新方向（優化記憶體分配）
                alpha_grad = alpha * grad
                final_exp_avg = beta1 * exp_avg + alpha_grad

                if self.memory_efficient:
                    # 重用張量進行計算
                    alpha_grad_p2 = alpha_grad.pow_(2)  # in-place
                    final_exp_avg_p2 = final_exp_avg.pow(2)
                    exp_avg_sq = final_exp_avg_p2.mul_(beta2).add_(alpha_grad_p2, alpha=1.0 - beta2)
                else:
                    alpha_grad_p2 = alpha_grad ** 2
                    final_exp_avg_p2 = final_exp_avg ** 2
                    exp_avg_sq = final_exp_avg_p2.mul_(beta2).add_(alpha_grad_p2, alpha=1.0 - beta2)

                denom = exp_avg_sq.sqrt_().add_(eps)  # in-place sqrt
                update = final_exp_avg / denom

                # Cautious 優化器遮罩
                mask = (update * grad > 0).to(grad.dtype)
                mask_ratio = mask.mean()
                mask.div_(mask_ratio.clamp_(min=1e-3))
                update.mul_(mask)  # in-place

                # 學習率遮罩處理
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

                    # in-place 更新學習率遮罩
                    lr_mask.add_(torch.where(sign_agree > 0, lr_bump_pos, -lr_bump_neg))

                    if group["lr"] >= state["lr_max"]:
                        state["lr_max"] = group["lr"]

                    lr_mask.clamp_(min=self.min_lr, max=self.max_lr)
                    state['avg_lr'] = torch.mean(lr_mask.float()).item()
                    new_lr = lr_mask.float()
                else:
                    # 清理不需要的狀態以節省記憶體
                    if 'last_polarity' in state:
                        del state['last_polarity']
                        if 'lr_mask' in state:
                            # 轉換為標量以節省記憶體
                            state['lr_scalar'] = state['avg_lr']
                            del state['lr_mask']

                    new_lr = state.get('lr_scalar', state.get('avg_lr', self.lr))
                    if group["lr"] >= state["lr_max"]:
                        state["decay_step"] = 0
                        state["lr_max"] = group["lr"]
                    elif group["lr"] < state["lr_max"]:
                        new_lr = new_lr * max(group["lr"] / state["lr_max"], 0.1)

                # ALLoRA 縮放
                if "row_scaling" in state:
                    if isinstance(new_lr, torch.Tensor):
                        new_lr = new_lr * state["row_scaling"]
                    else:
                        new_lr = new_lr * state["row_scaling"]

                update.mul_(new_lr)  # in-place

                # SPD 選擇性投影衰減
                do_spd = False
                if state["step"] < group["warmup_steps"]:
                    condition = -torch.sum(p.grad * p)
                    if condition < 0.0:
                        do_spd = True
                        new_p = p - update
                        pre = state["pre"] if state["pre"] is not None else torch.zeros_like(p)
                        ratio = self._ratio(new_p, p, pre)
                        new_p.sub_(pre, alpha=group["weight_decay"] * ratio).add_(pre)
                        p.copy_(new_p)

                if not do_spd:
                    p.sub_(update)  # in-place

        # 定期清理快取
        self._cleanup_cache()

        return loss

    def state_dict(self):
        """保存狀態字典，包含快取資訊"""
        state = super().state_dict()
        state['magic_version'] = 2  # 更新版本號
        state['memory_efficient'] = self.memory_efficient
        return state

    def load_state_dict(self, state_dict):
        """載入狀態字典，處理版本相容性"""
        version = state_dict.get('magic_version', 1)
        if version < 2:
            print('[WARNING] 載入舊版本的狀態字典，某些記憶體優化功能可能無法使用')

        # 恢復記憶體效率設定
        self.memory_efficient = state_dict.get('memory_efficient', True)

        super().load_state_dict(state_dict)

    def cleanup(self):
        """手動清理記憶體"""
        if self.tensor_cache:
            self.tensor_cache.clear()
        self._computation_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self):
        """析構時清理資源"""
        try:
            self.cleanup()
        except:
            pass  # 忽略析構時的錯誤
