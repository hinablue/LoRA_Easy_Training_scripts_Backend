import torch
from typing import List, Dict, Any, Optional, Tuple, Deque
from dataclasses import dataclass
from collections import deque
import math
import random

@dataclass
class ImprovedOptimizerConfig:
    """改進版優化器配置，專門針對 LoRA 訓練優化."""
    lr: float = 1e-6
    min_lr: float = 1e-7
    max_lr: float = 1e-3
    lr_bump: float = 3e-6
    eps: Tuple[float, float, float] = (1e-30, 1e-16, 1e-8)
    clip_threshold: float = 1.0
    betas: Tuple[float, float, float] = (0.8, 0.99, 0.999)
    eta: float = 2.0
    beta1_decay: float = 0.9995
    weight_decay: float = 5e-4
    d_coef: float = 2.0
    warmup_steps: int = 500
    edge_threshold: float = 0.6
    came: bool = True
    full_finetune: bool = False
    verbose: bool = False
    # 新增：邊緣和背景過擬合控制參數
    edge_suppression: bool = True
    edge_penalty: float = 0.1
    background_regularization: bool = True
    spatial_awareness: bool = True
    frequency_penalty: float = 0.05
    detail_preservation: float = 0.8
    # 新增：LoRA 特定優化參數
    lora_rank_penalty: bool = True
    rank_penalty_strength: float = 0.01
    low_rank_emphasis: float = 1.2
    # 新增：記憶體優化參數
    enable_cache: bool = True
    max_cache_size: int = 100
    use_approximate_svd: bool = True
    # 新增：雙動量系統參數
    enable_dual_momentum: bool = True
    long_term_beta: float = 0.99  # beta3 for long-term momentum
    alpha_mix_ratio: float = None  # auto-compute if None: (1-beta1)/(1-beta3)

class TensorCache:
    """張量緩存管理器，用於重用計算結果和緩衝區."""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache: Dict[str, torch.Tensor] = {}
        self.buffer_pool: Dict[tuple, List[torch.Tensor]] = {}
        self.access_count: Dict[str, int] = {}

    def get_buffer(self, shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """獲取指定形狀的緩衝區張量."""
        key = (shape, dtype, device)
        if key in self.buffer_pool and self.buffer_pool[key]:
            return self.buffer_pool[key].pop()
        return torch.zeros(shape, dtype=dtype, device=device)

    def return_buffer(self, tensor: torch.Tensor) -> None:
        """歸還緩衝區張量."""
        key = (tuple(tensor.shape), tensor.dtype, tensor.device)
        if key not in self.buffer_pool:
            self.buffer_pool[key] = []
        if len(self.buffer_pool[key]) < 5:  # 限制每種類型的緩衝區數量
            tensor.zero_()  # 清零重用
            self.buffer_pool[key].append(tensor)

    def get_cached(self, key: str) -> Optional[torch.Tensor]:
        """獲取快取的計算結果."""
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None

    def set_cached(self, key: str, value: torch.Tensor) -> None:
        """設定快取的計算結果."""
        if len(self.cache) >= self.max_size:
            # 移除最少使用的項目
            least_used = min(self.access_count, key=self.access_count.get)
            del self.cache[least_used]
            del self.access_count[least_used]
        self.cache[key] = value.detach()
        self.access_count[key] = 1

class ImprovedBaseOptimizer(torch.optim.Optimizer):
    """改進版基礎優化器，包含邊緣和背景過擬合控制."""

    def __init__(self, params, config: ImprovedOptimizerConfig):
        self.config = config
        eta_value = float(config.eta) if isinstance(config.eta, (int, float)) else 2.0

        defaults = dict(
            lr=config.lr,
            eps=config.eps,
            clip_threshold=config.clip_threshold,
            betas=config.betas,
            eta=eta_value,
            beta1_decay=config.beta1_decay,
            weight_decay=config.weight_decay,
            warmup_steps=config.warmup_steps,
            edge_threshold=config.edge_threshold,
            came=config.came,
            full_finetune=config.full_finetune,
            d_coef=config.d_coef,
            edge_suppression=config.edge_suppression,
            edge_penalty=config.edge_penalty,
            background_regularization=config.background_regularization,
            spatial_awareness=config.spatial_awareness,
            frequency_penalty=config.frequency_penalty,
            detail_preservation=config.detail_preservation,
            lora_rank_penalty=config.lora_rank_penalty,
            rank_penalty_strength=config.rank_penalty_strength,
            low_rank_emphasis=config.low_rank_emphasis,
        )
        super().__init__(params, defaults)
        self.base_lrs: List[float] = [config.lr for group in self.param_groups]

        # 初始化記憶體優化組件
        self.tensor_cache = TensorCache(config.max_cache_size) if config.enable_cache else None
        self._precomputed_constants = self._precompute_constants()

        # 新增：為每個參數群組設定雙動量參數
        for group in self.param_groups:
            group.setdefault('enable_dual_momentum', config.enable_dual_momentum)
            group.setdefault('long_term_beta', config.long_term_beta)
            group.setdefault('alpha_mix_ratio', config.alpha_mix_ratio)

    def _precompute_constants(self) -> Dict[str, float]:
        """預計算常用的常數值."""
        return {
            'sqrt_2_pi': math.sqrt(2.0 * math.pi),
            'ln_2': math.log(2.0),
            'inv_sqrt_2': 1.0 / math.sqrt(2.0),
            'beta_decay_factor': self.config.beta1_decay,
            'rank_decay': math.exp(-0.1),  # 用於 LoRA 排名懲罰
        }

    @staticmethod
    def _rms(tensor: torch.Tensor) -> torch.Tensor:
        """計算張量的均方根值."""
        return tensor.norm(2) / (tensor.numel() ** 0.5 + 1e-10)

    @staticmethod
    def _ratio(new_p: torch.Tensor, p: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        """計算選擇性投影衰減的比率."""
        with torch.no_grad():  # 純計算部分，不需要梯度
            curr_norm, prev_norm = torch.norm(new_p - pre), torch.norm(p - pre)
            ratio = (curr_norm - prev_norm) / (curr_norm + 1e-8)
            return torch.nn.functional.hardtanh(ratio, 0.0, 1.0)

    def _compute_edge_penalty_optimized(self, grad: torch.Tensor, threshold: float = 0.6,
                                      cache_key: Optional[str] = None) -> torch.Tensor:
        """
        優化版邊緣懲罰計算，使用緩存和簡化算法.
        """
        if len(grad.shape) < 2:
            return torch.zeros_like(grad)

        # 檢查緩存
        if cache_key and self.tensor_cache:
            cached = self.tensor_cache.get_cached(f"edge_{cache_key}")
            if cached is not None and cached.shape == grad.shape:
                return cached

        with torch.no_grad():
            # 獲取緩衝區而不是創建新張量
            if self.tensor_cache:
                laplacian = self.tensor_cache.get_buffer(grad.shape, grad.dtype, grad.device)
            else:
                laplacian = torch.zeros_like(grad)

            # 簡化的邊緣檢測：只計算最重要的方向
            if len(grad.shape) == 2 and grad.shape[0] > 2 and grad.shape[1] > 2:
                # 使用原地操作
                laplacian[1:-1, :] = grad[2:, :] - 2 * grad[1:-1, :] + grad[:-2, :]
                laplacian[:, 1:-1] += grad[:, 2:] - 2 * grad[:, 1:-1] + grad[:, :-2]

            # 計算邊緣強度（簡化版本）
            edge_strength = torch.abs(laplacian)
            edge_mask = (edge_strength > threshold).float()
            result = edge_mask * edge_strength

            # 緩存結果
            if cache_key and self.tensor_cache:
                self.tensor_cache.set_cached(f"edge_{cache_key}", result)

            return result

    def _compute_frequency_penalty_simplified(self, grad: torch.Tensor) -> torch.Tensor:
        """
        簡化版頻率懲罰計算，使用近似方法.
        """
        if len(grad.shape) < 2:
            return torch.zeros_like(grad)

        with torch.no_grad():
            # 使用簡化的高頻檢測：計算相鄰元素差異
            if len(grad.shape) == 2:
                h, w = grad.shape
                if h > 1 and w > 1:
                    # 計算水平和垂直差異
                    h_diff = torch.abs(grad[:, 1:] - grad[:, :-1])
                    v_diff = torch.abs(grad[1:, :] - grad[:-1, :])

                    # 創建結果張量
                    if self.tensor_cache:
                        result = self.tensor_cache.get_buffer(grad.shape, grad.dtype, grad.device)
                    else:
                        result = torch.zeros_like(grad)

                    # 組合差異信息
                    result[:, 1:] += h_diff
                    result[1:, :] += v_diff

                    return result

            return torch.zeros_like(grad)

    def _lora_rank_regularization_fast(self, param: torch.Tensor, rank_strength: float = 0.01,
                                     use_approx: bool = True) -> torch.Tensor:
        """
        快速 LoRA 低秩正則化，使用近似 SVD.
        """
        if len(param.shape) != 2:
            return torch.zeros_like(param)

        with torch.no_grad():
            if use_approx and self.config.use_approximate_svd:
                # 使用近似方法：只考慮最大的幾個奇異值
                # 計算 A^T A 的特徵值（避免完整 SVD）
                if param.shape[0] <= param.shape[1]:
                    cov = torch.mm(param, param.t())
                else:
                    cov = torch.mm(param.t(), param)

                # 只取前幾個特徵值
                eigenvals, _ = torch.linalg.eigh(cov)
                large_eigenvals = eigenvals[eigenvals.argsort(descending=True)[10:]]
                rank_penalty_scalar = torch.sum(large_eigenvals) * rank_strength

                # 創建梯度近似
                return param * rank_penalty_scalar
            else:
                # 完整 SVD（如果需要）
                U, S, Vh = torch.linalg.svd(param, full_matrices=False)
                rank_penalty = torch.sum(S[S.argsort(descending=True)[10:]])
                penalty_grad = U @ torch.diag(S * rank_strength) @ Vh
                return penalty_grad

    def _init_state(self, p: torch.Tensor, group: Optional[Dict[str, Any]] = None) -> None:
        """初始化優化器狀態."""
        device = p.device
        shape = p.shape
        state = self.state[p]

        # 基本狀態初始化
        state.setdefault("lr_max", 1e-6)
        state.setdefault("step", 0)

        # 學習率遮罩初始化
        state.setdefault('lr_mask', torch.ones(shape, device=device, dtype=torch.float32) * self.config.lr)
        state.setdefault('avg_lr', float(self.config.lr))
        state.setdefault('last_polarity', torch.zeros(shape, dtype=torch.bool, device=device))

        # 動量和變異數初始化
        state.setdefault("exp_avg", torch.zeros_like(p))
        state.setdefault("s", torch.zeros_like(p))

        if group and group.get('came', True):
            state.setdefault("exp_avg_sq", torch.zeros_like(p))

        state.setdefault("exp_avg_res", torch.zeros_like(p))

        # 新增：邊緣和背景過擬合控制狀態
        if group and group.get('edge_suppression', True):
            state.setdefault("edge_history", torch.zeros_like(p))
            state.setdefault("edge_momentum", torch.zeros_like(p))

        if group and group.get('spatial_awareness', True):
            state.setdefault("spatial_variance", torch.ones_like(p))
            state.setdefault("detail_tracker", torch.zeros_like(p))

        # ALLoRA 和 LoRA 特定初始化
        if group and not group.get('full_finetune', True):
            if len(p.shape) == 2:
                row_norm = p.norm(dim=1, keepdim=True)
                state["row_scaling"] = 1.0 / torch.sqrt(row_norm + 1.0 / (group.get('eta', 2.0)**2))

                # LoRA 低秩結構追蹤
                if group.get('lora_rank_penalty', True):
                    state["rank_tracker"] = torch.zeros(min(p.shape), device=device)

        # 新增：雙動量系統狀態
        if group and group.get('enable_dual_momentum', True):
            state.setdefault("long_term_momentum", torch.zeros_like(p))  # 長期動量
            state.setdefault("mixed_momentum_sq", torch.zeros_like(p))   # 混合動量的方差估計

    @staticmethod
    def _orthograd(p: torch.Tensor, grad: torch.Tensor,
                  temp_buffer: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        === 正交梯度（記憶體優化版）===
        使用提供的緩衝區減少記憶體分配
        """
        if p.norm(2) <= 1e-30:
            return grad

        with torch.no_grad():
            G_shape = grad.shape
            w = p.view(-1)
            g = grad.view(-1)
            g_norm = g.norm(2)

            proj = torch.dot(w, g) / torch.dot(w, w).add(1e-30)

            # 使用原地操作
            if temp_buffer is not None and temp_buffer.shape == g.shape:
                g_orth = temp_buffer
                g_orth.copy_(g)
                g_orth.sub_(w, alpha=proj)
            else:
                g_orth = g.sub(w, alpha=proj)

            g_orth_scaled = g_orth.mul_(g_norm / g_orth.norm(2).add(1e-30))

            return g_orth_scaled.view(G_shape)

    @staticmethod
    def _compute_cosine_similarity_efficient(exp_avg: torch.Tensor, scaled_grad: torch.Tensor,
                                            temp_buffer: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        高效計算餘弦相似度，避免兩次正規化操作.

        使用公式: cos(θ) = (a · b) / (||a|| * ||b||)
        相比兩次 normalize 操作，這種方法：
        1. 減少記憶體分配（不創建正規化張量）
        2. 減少計算量（只需一次內積和兩次範數計算）
        3. 提升數值穩定性
        """
        with torch.no_grad():
            # 計算內積 (向量化操作)
            dot_product = torch.sum(exp_avg * scaled_grad, dim=0, keepdim=True)

            # 計算兩個向量的範數
            exp_avg_norm = torch.norm(exp_avg, p=2, dim=0, keepdim=True)
            scaled_grad_norm = torch.norm(scaled_grad, p=2, dim=0, keepdim=True)

            # 計算餘弦相似度，加入數值穩定性保護
            denominator = exp_avg_norm * scaled_grad_norm
            denominator = torch.clamp(denominator, min=1e-12)  # 避免除零

            cosine_sim = dot_product / denominator

            # 將相似度廣播到原始張量的形狀
            if cosine_sim.shape != exp_avg.shape:
                # 如果需要廣播，使用高效的方式
                if temp_buffer is not None and temp_buffer.shape == exp_avg.shape:
                    temp_buffer.fill_(1.0)
                    temp_buffer.mul_(cosine_sim)
                    return temp_buffer
                else:
                    return cosine_sim.expand_as(exp_avg)

            return cosine_sim

    @staticmethod
    def _compute_correlation_fast(exp_avg: torch.Tensor, scaled_grad: torch.Tensor,
                                temp_buffer: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        快速計算向量相關性，用於動量更新.

        這個版本直接計算相關性而不是逐元素相乘正規化向量：
        - 使用餘弦相似度作為相關性度量
        - 避免創建中間正規化張量
        - 利用向量化操作提升效率
        """
        # 如果張量形狀相同且較小，使用簡化計算
        if exp_avg.shape == scaled_grad.shape and exp_avg.numel() < 10000:
            return ImprovedBaseOptimizer._compute_cosine_similarity_efficient(
                exp_avg, scaled_grad, temp_buffer)

        # 對於大張量或不同形狀，使用分塊計算
        with torch.no_grad():
            if len(exp_avg.shape) > 1:
                # 展平張量進行計算，然後重塑
                exp_avg_flat = exp_avg.view(-1)
                scaled_grad_flat = scaled_grad.view(-1)

                # 計算全局餘弦相似度
                dot_prod = torch.dot(exp_avg_flat, scaled_grad_flat)
                norm_exp = torch.norm(exp_avg_flat)
                norm_scaled = torch.norm(scaled_grad_flat)

                # 避免除零並計算相似度
                denominator = norm_exp * norm_scaled
                if denominator < 1e-12:
                    cosine_sim = torch.tensor(0.0, device=exp_avg.device, dtype=exp_avg.dtype)
                else:
                    cosine_sim = dot_prod / denominator

                # 創建相關性張量
                if temp_buffer is not None and temp_buffer.shape == exp_avg.shape:
                    temp_buffer.fill_(cosine_sim.item())
                    return temp_buffer
                else:
                    return torch.full_like(exp_avg, cosine_sim.item())
            else:
                # 1D 張量的簡單情況
                return ImprovedBaseOptimizer._compute_cosine_similarity_efficient(
                    exp_avg, scaled_grad, temp_buffer)

    def _compute_dual_momentum_update(
        self,
        grad: torch.Tensor,
        state: Dict[str, Any],
        group: Dict[str, Any],
        temp_buffers: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        計算雙動量系統的更新值，結合 AdamS 的雙動量理念與 CAME 的信心度機制.

        Args:
            grad: 當前梯度
            state: 優化器狀態
            group: 參數群組設定
            temp_buffers: 臨時緩衝區字典

        Returns:
            改進的動量更新值
        """
        if not group.get('enable_dual_momentum', True):
            return grad  # 如果未啟用，返回原始梯度

        beta1, beta2, beta3 = group["betas"]
        long_term_beta = group.get('long_term_beta', 0.99)

        # 計算 alpha 混合比例 (類似 AdamS)
        alpha_mix = group.get('alpha_mix_ratio')
        if alpha_mix is None:
            alpha_mix = (1 - beta1) / (1 - long_term_beta)

        # 獲取狀態變數
        exp_avg = state["exp_avg"]           # 短期動量 (CAME 原有)
        long_term_momentum = state["long_term_momentum"]  # 長期動量 (新增)
        mixed_momentum_sq = state["mixed_momentum_sq"]    # 混合動量方差 (新增)

        # === 步驟 1：更新長期動量 (類似 AdamS 的 exp_avg.mul_(beta3).add_(grad)) ===
        long_term_momentum.mul_(long_term_beta).add_(grad)

        # === 步驟 2：計算混合動量 (類似 AdamS 的 final_exp_avg) ===
        # 使用緩衝區計算混合動量
        mixed_momentum_key = f"mixed_momentum_{grad.shape}"
        if self.tensor_cache and mixed_momentum_key not in temp_buffers:
            temp_buffers[mixed_momentum_key] = self.tensor_cache.get_buffer(
                grad.shape, grad.dtype, grad.device)

        if mixed_momentum_key in temp_buffers:
            mixed_momentum = temp_buffers[mixed_momentum_key]
            # mixed_momentum = beta1 * long_term_momentum + alpha_mix * grad
            mixed_momentum.copy_(long_term_momentum)
            mixed_momentum.mul_(beta1).add_(grad, alpha=alpha_mix)
        else:
            mixed_momentum = beta1 * long_term_momentum + alpha_mix * grad

        # === 步驟 3：更新混合動量的方差估計 (改進版) ===
        # 計算縮放梯度項 (類似 AdamS 的 alpha_grad)
        scaled_grad_key = f"scaled_grad_dual_{grad.shape}"
        if self.tensor_cache and scaled_grad_key not in temp_buffers:
            temp_buffers[scaled_grad_key] = self.tensor_cache.get_buffer(
                grad.shape, grad.dtype, grad.device)

        if scaled_grad_key in temp_buffers:
            scaled_grad_term = temp_buffers[scaled_grad_key]
            scaled_grad_term.copy_(grad)
            scaled_grad_term.mul_(alpha_mix).pow_(2)
        else:
            scaled_grad_term = (alpha_mix * grad).pow(2)

        # 更新混合動量方差: mixed_momentum_sq = beta2 * mixed_momentum_sq + (1-beta2) * [mixed_momentum² + scaled_grad_term]
        mixed_momentum_var_key = f"mixed_momentum_var_{grad.shape}"
        if self.tensor_cache and mixed_momentum_var_key not in temp_buffers:
            temp_buffers[mixed_momentum_var_key] = self.tensor_cache.get_buffer(
                grad.shape, grad.dtype, grad.device)

        if mixed_momentum_var_key in temp_buffers:
            momentum_var_term = temp_buffers[mixed_momentum_var_key]
            momentum_var_term.copy_(mixed_momentum)
            momentum_var_term.pow_(2).add_(scaled_grad_term)
        else:
            momentum_var_term = mixed_momentum.pow(2) + scaled_grad_term

        mixed_momentum_sq.mul_(beta2).add_(momentum_var_term, alpha=1.0 - beta2)

        # === 步驟 4：計算最終更新 (結合 CAME 的思想) ===
        eps1 = group["eps"][0] if len(group["eps"]) > 0 else 1e-30

        # 使用改進的方差估計進行正規化
        if mixed_momentum_var_key in temp_buffers:
            normalized_update = temp_buffers[mixed_momentum_var_key]
            normalized_update.copy_(mixed_momentum)
            normalized_update.div_(mixed_momentum_sq.sqrt().add_(eps1))
        else:
            normalized_update = mixed_momentum / (mixed_momentum_sq.sqrt() + eps1)

        # === 步驟 5：與 CAME 的信心度機制結合 ===
        # 保持與原有 exp_avg 的相容性，更新為混合動量
        exp_avg.copy_(mixed_momentum)

        return normalized_update

class Automagic_CameAMP_Improved(ImprovedBaseOptimizer):
    """改進版 Automagic_CameAMP 優化器，專門優化 LoRA 訓練並減少邊緣、背景過擬合."""

    def __init__(self, params, **kwargs):
        config = ImprovedOptimizerConfig(**kwargs)
        super().__init__(params, config)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """執行單一優化步驟（記憶體優化版）."""
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            # 計算群組梯度統計
            grads_this_group = []
            for p in group["params"]:
                if p.grad is None or not p.requires_grad:
                    continue
                grads_this_group.append(p.grad.view(-1))

            if not grads_this_group:
                continue

            all_group_grads = torch.cat(grads_this_group)
            sum_abs_all_group_grads = torch.sum(torch.abs(all_group_grads))

            # 預計算常用值
            warmup_half = group.get("warmup_steps", 500) // 2
            has_warmup_params = any(self.state.get(p, {}).get("step", 0) < warmup_half
                                  for p in group["params"] if p.grad is not None)

            if has_warmup_params and group["weight_decay"] > 0:
                abs_all_group_grads = torch.abs(all_group_grads)
                mean_norm = abs_all_group_grads.mean()
                std_norm = abs_all_group_grads.std(unbiased=False)

            # 為這個群組創建緩衝區
            temp_buffers = {}

            for p in group["params"]:
                if p.grad is None or not p.requires_grad:
                    continue

                grad = p.grad
                state = self.state[p]

                # 狀態初始化
                if len(state) == 0:
                    self._init_state(p, group)

                if 'step' not in state:
                    state['step'] = 0
                state["step"] += 1

                # 清理 warmup 後不需要的狀態
                if state["step"] == group.get("warmup_steps", 500):
                    for key in ['s', 'last_polarity']:
                        if key in state:
                            del state[key]
                    if 'pre' in state and state.get("pre") is not None:
                        del state['pre']

                # === 改進 1：增強的 AGR 自適應梯度正則化（使用原地操作）===
                with torch.no_grad():
                    abs_grad = torch.abs(grad)
                    alpha = abs_grad / (sum_abs_all_group_grads + 1e-10)

                    # 新增：邊緣感知的梯度正則化（使用緩存）
                    if group.get('edge_suppression', True):
                        cache_key = f"p_{id(p)}_{state['step']}"
                        edge_penalty = self._compute_edge_penalty_optimized(
                            grad, group.get('edge_threshold', 0.6), cache_key)
                        edge_factor = 1.0 + group.get('edge_penalty', 0.1) * edge_penalty
                        alpha.mul_(edge_factor)  # 原地操作

                    # 原地更新梯度
                    grad.mul_(1 - alpha)

                    # === 改進 2：頻率感知的梯度調整（簡化版）===
                    if group.get('spatial_awareness', True) and len(grad.shape) >= 2:
                        freq_penalty = self._compute_frequency_penalty_simplified(grad)
                        freq_factor = group.get('frequency_penalty', 0.05)
                        grad.sub_(freq_penalty, alpha=freq_factor)  # 原地操作

                    # === 改進 3：LoRA 低秩正則化（快速版）===
                    if group.get('lora_rank_penalty', True) and len(p.shape) == 2:
                        rank_penalty = self._lora_rank_regularization_fast(
                            p, group.get('rank_penalty_strength', 0.01))
                        grad.add_(rank_penalty)  # 原地操作

                # 原始 CAME 核心處理
                beta1, beta2, beta3 = 0.9, 0.999, 0.9999
                eps1, eps2 = group["eps"][:2]

                # CAME 自適應記憶體高效優化
                if group.get('came', True):
                    # 獲取或創建緩衝區
                    buffer_key = f"update_p_{grad.shape}"
                    if self.tensor_cache and buffer_key not in temp_buffers:
                        temp_buffers[buffer_key] = self.tensor_cache.get_buffer(
                            grad.shape, grad.dtype, grad.device)

                    update_p = temp_buffers.get(buffer_key, grad.pow(2))
                    if buffer_key in temp_buffers:
                        update_p.copy_(grad.pow(2))
                    update_p.add_(eps1)  # 原地操作

                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).add_(update_p, alpha=1 - beta2)

                    # 獲取 scaled_grad 緩衝區
                    scaled_grad_key = f"scaled_grad_{grad.shape}"
                    if self.tensor_cache and scaled_grad_key not in temp_buffers:
                        temp_buffers[scaled_grad_key] = self.tensor_cache.get_buffer(
                            grad.shape, grad.dtype, grad.device)

                    if scaled_grad_key in temp_buffers:
                        scaled_grad = temp_buffers[scaled_grad_key]
                        scaled_grad.copy_(grad)
                        scaled_grad.mul_(exp_avg_sq.rsqrt())
                    else:
                        scaled_grad = grad * exp_avg_sq.rsqrt()

                    # 原地梯度裁剪
                    rms_val = self._rms(scaled_grad)
                    clip_factor = group["clip_threshold"] / rms_val.clamp_(min=1.0)
                    scaled_grad.mul_(clip_factor)
                else:
                    scaled_grad = grad

                # === 改進 4A：雙動量系統處理（新增）===
                if group.get('enable_dual_momentum', True):
                    # 應用雙動量系統，獲得改進的動量更新
                    dual_momentum_update = self._compute_dual_momentum_update(
                        scaled_grad, state, group, temp_buffers)

                    # 將雙動量結果作為後續處理的輸入
                    processed_grad = dual_momentum_update
                else:
                    processed_grad = scaled_grad

                # === 改進 4：增強的動量處理（原地操作）===
                if state["step"] < warmup_half:
                    # Torque-Aware Momentum with LoRA adaptation
                    decay_rate = self._precomputed_constants['beta_decay_factor']
                    if 's' in state:
                        s, exp_avg = state['s'], state['exp_avg']

                        # 使用高效餘弦相似度計算替代兩次正規化
                        corr_buffer_key = f"corr_{exp_avg.shape}"
                        corr_buffer = temp_buffers.get(corr_buffer_key)
                        if self.tensor_cache and corr_buffer is None:
                            corr_buffer = self.tensor_cache.get_buffer(
                                exp_avg.shape, exp_avg.dtype, exp_avg.device)
                            temp_buffers[corr_buffer_key] = corr_buffer

                        # 計算相關性（餘弦相似度）
                        corr = self._compute_correlation_fast(exp_avg, processed_grad, corr_buffer)

                        s.mul_(decay_rate).add_(corr, alpha=1.0 - decay_rate)

                        # LoRA 特定調整：強調低秩方向
                        if group.get('lora_rank_penalty', True) and len(p.shape) == 2:
                            low_rank_factor = group.get('low_rank_emphasis', 1.2)
                            s.mul_(low_rank_factor)

                        # 使用緩衝區計算 d
                        d_key = f"d_{processed_grad.shape}"
                        if self.tensor_cache and d_key not in temp_buffers:
                            temp_buffers[d_key] = self.tensor_cache.get_buffer(
                                processed_grad.shape, processed_grad.dtype, processed_grad.device)

                        if d_key in temp_buffers:
                            d = temp_buffers[d_key]
                            d.copy_(s)
                            d.add_(1.0).div_(2.0).add_(eps1).mul_(processed_grad)
                        else:
                            d = ((1.0 + s) / 2.0).add_(eps1).mul_(processed_grad)

                        exp_avg.mul_(beta1).add_(d)
                else:
                    # Cautious Optimizers: Improving Training with One Line of Code
                    beta1, beta2, beta3 = group["betas"]
                    beta1_t = max(beta1 * (self._precomputed_constants['beta_decay_factor'] ** state["step"]), 0.4)
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(beta1_t).add_(processed_grad, alpha=1 - beta1_t)

                # CAME 核心：信心引導的記憶體高效優化
                exp_avg_res = state["exp_avg_res"]

                # 使用緩衝區計算 res
                res_key = f"res_{processed_grad.shape}"
                if self.tensor_cache and res_key not in temp_buffers:
                    temp_buffers[res_key] = self.tensor_cache.get_buffer(
                        processed_grad.shape, processed_grad.dtype, processed_grad.device)

                if res_key in temp_buffers:
                    res = temp_buffers[res_key]
                    res.copy_(processed_grad)
                    res.sub_(exp_avg).pow_(2).add_(eps2)
                else:
                    res = (processed_grad - exp_avg).pow(2) + eps2

                exp_avg_res.mul_(beta3).add_(res, alpha=1.0 - beta3)

                # 獲取 update_p 緩衝區
                update_p_key = f"update_p_final_{exp_avg.shape}"
                if self.tensor_cache and update_p_key not in temp_buffers:
                    temp_buffers[update_p_key] = self.tensor_cache.get_buffer(
                        exp_avg.shape, exp_avg.dtype, exp_avg.device)

                if update_p_key in temp_buffers:
                    update_p = temp_buffers[update_p_key]
                    update_p.copy_(exp_avg)
                    update_p.mul_(exp_avg_res.rsqrt())
                else:
                    update_p = exp_avg * exp_avg_res.rsqrt()

                # === 改進 5：增強的 Automagic 學習率遮罩（原地操作）===
                if state["step"] < group.get("warmup_steps", 500):
                    if 'last_polarity' in state:
                        last_polarity = state['last_polarity']
                        current_polarity = (processed_grad > 0)
                        sign_agree = torch.where(last_polarity == current_polarity, 1.0, -1.0)
                        last_polarity.copy_(current_polarity)  # 原地更新

                        lr_mask = state['lr_mask']

                        # 新增：空間感知的學習率調整
                        if group.get('spatial_awareness', True):
                            spatial_var = state.get('spatial_variance', torch.ones_like(lr_mask))
                            detail_factor = group.get('detail_preservation', 0.8)
                            spatial_factor = (spatial_var * detail_factor).clamp(0.5, 1.5)
                            lr_bump = self.config.lr_bump * spatial_factor
                        else:
                            lr_bump = self.config.lr_bump

                        condition = -torch.sum(p.grad * p)
                        if state["step"] < group.get("warmup_steps", 500) / 2:
                            lr_bump_pos = lr_bump * group['d_coef'] if condition > 0.0 else lr_bump
                            lr_bump_neg = lr_bump * group['d_coef'] if condition < 0.0 else lr_bump
                        else:
                            lr_bump_pos, lr_bump_neg = lr_bump, lr_bump

                        # 原地更新學習率遮罩
                        lr_delta = torch.where(
                            sign_agree > 0,
                            lr_bump_pos,
                            -lr_bump_neg
                        )
                        lr_mask.add_(lr_delta)

                        if group["lr"] > state["lr_max"]:
                            lr_mask.add_(group["lr"] - state["lr_max"])
                            state["lr_max"] = group["lr"]

                        lr_mask.clamp_(min=self.config.min_lr, max=self.config.max_lr)
                        state['avg_lr'] = torch.mean(lr_mask).item()
                else:
                    lr_mask = state['lr_mask']
                    if group["lr"] > state["lr_max"]:
                        state["lr_max"] = group["lr"]
                    if group["lr"] < state["lr_max"]:
                        lr_mask.mul_(group["lr"] / state["lr_max"])

                # 正交梯度處理（使用緩衝區）
                if state["step"] < warmup_half:
                    ortho_buffer_key = f"ortho_{update_p.shape}"
                    ortho_buffer = temp_buffers.get(ortho_buffer_key)
                    if self.tensor_cache and ortho_buffer is None:
                        ortho_buffer = self.tensor_cache.get_buffer(
                            update_p.view(-1).shape, update_p.dtype, update_p.device)
                        temp_buffers[ortho_buffer_key] = ortho_buffer

                    update_p = self._orthograd(p, update_p, ortho_buffer)

                # === 改進 6：智能梯度方向控制（原地操作）===
                if state["step"] < warmup_half:
                    # Grams with edge awareness（原地操作）
                    update_p.abs_().mul_(processed_grad.sign())

                    # 新增：邊緣抑制調整
                    if group.get('edge_suppression', True):
                        edge_history = state.get('edge_history', torch.zeros_like(update_p))
                        current_edge = self._compute_edge_penalty_optimized(update_p)
                        edge_history.mul_(0.9).add_(current_edge, alpha=0.1)
                        edge_suppression_factor = 1.0 - group.get('edge_penalty', 0.1) * edge_history
                        edge_suppression_factor.clamp_(0.1, 1.0)
                        update_p.mul_(edge_suppression_factor)
                        state['edge_history'] = edge_history
                else:
                    # Cautious optimization with spatial awareness
                    mask_key = f"mask_{update_p.shape}"
                    if self.tensor_cache and mask_key not in temp_buffers:
                        temp_buffers[mask_key] = self.tensor_cache.get_buffer(
                            update_p.shape, update_p.dtype, update_p.device)

                    if mask_key in temp_buffers:
                        mask = temp_buffers[mask_key]
                        mask.copy_(update_p * processed_grad > 0)
                        mask = mask.to(processed_grad.dtype)
                        mask.div_(mask.mean().clamp_(min=1e-3))
                    else:
                        mask = (update_p * processed_grad > 0).to(processed_grad.dtype)
                        mask.div_(mask.mean().clamp_(min=1e-3))

                    # 新增：背景正則化
                    if group.get('background_regularization', True):
                        grad_variance = torch.var(processed_grad) if processed_grad.numel() > 1 else torch.tensor(0.0)
                        if grad_variance < 1e-6:
                            mask.mul_(0.5)

                    update_p.mul_(mask)

                # 應用學習率遮罩（原地操作）
                update_p.mul_(lr_mask)

                # === 改進 7：增強的選擇性投影衰減===
                do_spd = False
                if state["step"] < group.get("warmup_steps", 500):
                    if "pre" not in state:
                        state["pre"] = p.clone()

                    pre = state["pre"]
                    condition = -torch.sum(p.grad * p)

                    if condition < 0.0:
                        do_spd = True
                        # 使用緩衝區計算 new_p
                        new_p_key = f"new_p_{p.shape}"
                        if self.tensor_cache and new_p_key not in temp_buffers:
                            temp_buffers[new_p_key] = self.tensor_cache.get_buffer(
                                p.shape, p.dtype, p.device)

                        if new_p_key in temp_buffers:
                            new_p = temp_buffers[new_p_key]
                            new_p.copy_(p)
                            new_p.sub_(update_p)
                        else:
                            new_p = p - update_p

                        ratio = self._ratio(new_p, p, pre)

                        # 新增：LoRA 感知的權重衰減
                        if group.get('lora_rank_penalty', True) and len(p.shape) == 2:
                            # 簡化版本：使用預計算的衰減因子
                            weighted_decay = group["weight_decay"] * (1.0 + self._precomputed_constants['rank_decay'])
                        else:
                            weighted_decay = group["weight_decay"]

                        # 原地更新
                        decay_term = ratio * weighted_decay
                        new_p.sub_(new_p - pre, alpha=decay_term)
                        p.copy_(new_p)

                    # 更新 pre 狀態
                    state["pre"].copy_(p)

                # 最終參數更新
                if not do_spd:
                    p.sub_(update_p)

                # === 改進 8：更新空間感知狀態（原地操作）===
                if group.get('spatial_awareness', True):
                    # 更新空間變異數追蹤
                    if len(processed_grad.shape) >= 2:
                        current_variance = torch.var(processed_grad, dim=-1, keepdim=True) if processed_grad.shape[-1] > 1 else torch.ones_like(processed_grad)
                        spatial_var = state.get('spatial_variance', torch.ones_like(current_variance))
                        spatial_var.mul_(0.9).add_(current_variance, alpha=0.1)
                        state['spatial_variance'] = spatial_var

            # 歸還緩衝區
            if self.tensor_cache:
                for buffer in temp_buffers.values():
                    self.tensor_cache.return_buffer(buffer)

        if self.config.verbose:
            avg_lrs = [torch.mean(state['lr_mask']).item() if 'lr_mask' in state else group["lr"]
                      for group in self.param_groups for state in [self.state.get(p, {}) for p in group["params"] if p.grad is not None]]
            print(f"平均學習率: {avg_lrs}")

        return loss

    def state_dict(self) -> Dict[str, Any]:
        """獲取優化器狀態字典."""
        state = super().state_dict()
        state['magic_version'] = 2  # 標記為記憶體優化版本
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """載入優化器狀態字典."""
        if 'magic_version' not in state_dict:
            print('[警告] 您載入了舊版本的狀態字典，某些新功能可能無法正常工作！')
        elif state_dict['magic_version'] != 2:
            print(f'[警告] 狀態字典版本不匹配：期望版本 2，實際版本 {state_dict["magic_version"]}')
        super().load_state_dict(state_dict)

    def get_memory_stats(self) -> Dict[str, Any]:
        """獲取記憶體使用統計."""
        if not self.tensor_cache:
            return {"cache_disabled": True}

        cache_size = len(self.tensor_cache.cache)
        buffer_pools = {key: len(buffers) for key, buffers in self.tensor_cache.buffer_pool.items()}

        return {
            "cache_size": cache_size,
            "max_cache_size": self.tensor_cache.max_size,
            "buffer_pools": buffer_pools,
            "total_buffers": sum(buffer_pools.values())
        }

    def clear_cache(self) -> None:
        """清理緩存和緩衝區池."""
        if self.tensor_cache:
            self.tensor_cache.cache.clear()
            self.tensor_cache.buffer_pool.clear()
            self.tensor_cache.access_count.clear()
            print("已清理所有緩存和緩衝區。")