# 有效传播数 R_eff 的形式推导

**版本**: V1.0
**日期**: 2026-07-24
**对应模型**: `docs/task2_model_definition_v1.md`

---

## 一、定义

### 1.1 流行病学类比

在经典 SIR 模型中，基本再生数 R₀ = β/γ，其中 β 是传播率，γ 是恢复率。R₀ > 1 意味着传染病会在人群中扩散；R₀ < 1 意味着会自然消亡。

在我们的模型中，传播过程是：

```
U ──(β·w_ji^s·𝟙[z_j=A])──> E ──(σ(α₀+α₁|o|+α₃Γ+...))──> A
```

"感染"对应 U→E→A 过程。"恢复"对应 A→D。

但与 SIR 的关键区别在于：**激活概率 P(E→A) 依赖于 agent 的观点、情绪和意见气候**。这意味着 R_eff 不是固定的系统参数，而是由当前的观点分布内生决定的。

### 1.2 下一代矩阵

采用 van den Driessche & Watmough (2002) 的下一代矩阵方法。

将系统状态分为：
- **感染仓室**：E（瞬态暴露）和 A（活跃传播）
- **未感染仓室**：U（未知）和 D（休眠）

在无传播平衡点（DFE = Disease-Free Equilibrium），A = E = 0，所有 agent 处于 U 或 D。

在 DFE 处引入一个活跃 agent j。该 agent j 能够：
1. 暴露邻居 i，概率为 β · w_ji^s
2. 暴露后 i 激活为 A，概率为 P_i = σ(α₀ + α₁|o_i| + α₂h_i + α₃Γ_i - α₄f_i - α₅c_i)

因此，agent j 对 agent i 的期望次级激活数为：

\[
\boxed{M_{ij} = \beta \cdot w_{ji}^s \cdot \sigma(\alpha_0 + \alpha_1|o_i| + \alpha_2 h_i + \alpha_3 \Gamma_i - \alpha_4 f_i - \alpha_5 c_i)}
\]

下一代矩阵 M ∈ ℝ^{N×N} 的第 (i,j) 个元素即 M_{ij}。

### 1.3 有效传播数

\[
\boxed{R_{\text{eff}} = \rho(M)}
\]

其中 ρ(·) 表示矩阵的谱半径（最大特征值的模）。

**传播阈值定理**：
- R_eff > 1 ⇒ 传播将增长（cascade grows）
- R_eff < 1 ⇒ 传播将衰减（cascade dies out）
- R_eff = 1 ⇒ 临界状态

---

## 二、修正的下一代矩阵

### 2.1 考虑活跃持续时间

原推导中 $M_{ij} = \beta \cdot w_{ji}^s \cdot P_i^{\text{activate}}$ 缺少一个关键因子：活跃 agent j 的**期望持续感染时间**。

在 SEIR 类比中，一个感染者能产生的次级感染数 = 传播率 × 接触率 × 感染持续时间。在我们的模型中：

- 传播率 = $\beta$
- 接触率 = $w_{ji}^s$（agent j 对 i 的暴露权重）
- 激活概率 = $P_i^{\text{activate}}$（i 暴露后激活的概率）
- **活跃持续时间** = $\mathbb{E}[T_j^A] = 1 / g_j(t)$，其中 $g_j(t) = P(A_j \to D_j)$ 是每步衰减概率

修正后的下一代矩阵：

\[
\boxed{K_{ij}(t) = S_i(t) \cdot \beta \cdot w_{ji}^s \cdot q_j(t) \cdot P_i^{\text{activate}}(t) \cdot \frac{1}{g_j(t)}}
\]

其中：
- $S_i(t) = \mathbf{1}[z_i(t) = \mathbf{U}]$：i 仍处于可暴露状态（susceptible）
- $q_j(t)$：j 的内容影响力
- $P_i^{\text{activate}}(t)$：i 暴露后激活的概率（含沉默螺旋修正）
- $g_j(t) = \sigma(\gamma_0 + \gamma_1 f_j + \gamma_2 s - \gamma_3 n)$：j 的每日衰减概率
- $1/g_j(t) \approx$ j 的期望活跃天数

### 2.2 有效传播数

\[
\boxed{R_{\text{eff}}(t) = \rho(K(t))}
\]

### 2.3 平均场近似

在均质混合假设下，用平均场近似：

\[
R_{\text{eff}}^{\text{MF}} \approx \beta \cdot \bar{k}_{\text{eff}} \cdot \bar{P}_{\text{activate}} \cdot \bar{S} \cdot \frac{1}{\bar{g}}
\]

其中：
- $\bar{S} = N_U / N$：可暴露比例
- $\bar{g} = \sigma(\gamma_0 + \gamma_1\bar{f} + \gamma_2\bar{s} - \gamma_3\bar{n})$：平均衰减率
- $1/\bar{g}$：期望活跃持续时间（若 $\bar{g}=0.5$，则持续约 2 天）

**注意**：此推导为初步近似。完整推导需要处理：(1) $\beta$ 与 $w_{ji}^s$ 的乘法可能导致 $\Lambda_i > 1$ 时的截断非线性；(2) $P_i^{\text{activate}}$ 本身依赖 $K$ 中的网络状态（通过 $\Gamma_i$）；(3) 离散时间下的生成函数形式。当前版本适用于参数数量级估计和数值敏感性分析，不作为严格的流行病学阈值定理。
- k̄_eff：有效平均度，考虑有向网络中出度和入度的相关性：

\[
\bar{k}_{\text{eff}} = \frac{\langle k_{\text{out}} \cdot k_{\text{in}} \rangle}{\langle k_{\text{out}} \rangle}
\]

这是对"友谊悖论"（你的粉丝比你拥有更多粉丝）的修正——高度数节点更可能被传播链选中。

- P̄_activate：平均激活概率：

\[
\bar{P}_{\text{activate}} = \frac{1}{N} \sum_{i=1}^{N} \sigma(\alpha_0 + \alpha_1|o_i| + \alpha_2 h_i + \alpha_3 \Gamma_i - \alpha_4 f_i - \alpha_5 c_i)
\]

---

## 三、核心定理

### 定理 1：观点极端性增加传播潜力

\[
\frac{\partial R_{\text{eff}}}{\partial \alpha_1} > 0
\]

**证明**：

M 的元素为 M_{ij} = β · w_ji^s · σ(z_i)，其中 z_i = α₀ + α₁|o_i| + ...。

σ'(x) = σ(x)(1-σ(x)) > 0 对所有 x 成立。

∂M_{ij}/∂α₁ = β · w_ji^s · σ'(z_i) · |o_i| ≥ 0。

由于 w_ji^s ≥ 0 且 |o_i| ≥ 0，M 的每个元素对 α₁ 是非递减的。

根据 Perron-Frobenius 定理（M 是非负矩阵），ρ(M) 对每个矩阵元素是单调非递减的。且当存在至少一个 i 满足 |o_i| > 0 和至少一个 j 满足 w_ji^s > 0 时，导数是严格的：

∂R_eff/∂α₁ > 0。

**推论 1.1**：在观点极化的群体中（|o_i| 均值高），R_eff 显著高于观点温和的群体。

**推论 1.2**：α₁ 是连接观点动力学和传播动力学的核心参数——它使得传播阈值内生地依赖于观点分布。

### 定理 2：意见气候一致性放大传播

\[
\frac{\partial R_{\text{eff}}}{\partial \alpha_3} > 0
\]

**证明**：类似定理 1。∂z_i/∂Γ_i = α₃ > 0，σ 单调递增。Γ_i 越大（个体越感到"安全"），激活概率越高。当 α₃ > 0 时，R_eff 对 α₃ 单调递增。

**推论 2.1**：在意见气候同质化的社区中（Γ_i 系统性地高），R_eff 更高——这解释了"回声室促进传播"的机制。

### 定理 3：沉默螺旋抑制少数意见的传播贡献

\[
\frac{\partial R_{\text{eff}}}{\partial \lambda_{\text{spiral}}} \leq 0
\]

**证明**：

沉默螺旋修正仅对 Γ_i < 0.5 的 agent 生效：

P_i^final = P_i^base · (1 - λ_spiral · (0.5 - Γ_i))

∂P_i^final/∂λ_spiral = -P_i^base · (0.5 - Γ_i) ≤ 0 当 Γ_i < 0.5。

由于 ρ(M) 对 P_i 单调非递减（σ 内的激活概率降低 → M 元素减小 → 谱半径不增），∂R_eff/∂λ_spiral ≤ 0。

当且仅当存在 Γ_i < 0.5 的 agent（即少数意见者）时，导数为严格负。

**推论 3.1**：沉默螺旋的传播抑制效应取决于少数意见者的比例和他们的 Γ_i 分布。在完全同质的社区中（所有 Γ_i > 0.5），λ_spiral 对 R_eff 无影响。

**推论 3.2**：社区结构可以通过提高 Γ_i（少数派也感到安全）来"免疫"沉默螺旋的传播抑制效应。

### 定理 4：情绪唤醒的传播放大效应

\[
\frac{\partial R_{\text{eff}}}{\partial \bar{h}} > 0 \quad \text{当} \quad \alpha_2 > 0
\]

平均情绪唤醒 h̄ 越高 → 平均激活概率越高 → R_eff 越高。这解释了"情绪化事件的病毒式传播"——不是因为信息本身更有价值，而是因为它触发的情绪状态降低了表达阈值。

---

## 四、数值验证

我们用数值仿真验证上述四个定理。验证脚本见 `dynamics_simulation/reff.py`。

### 验证 1：∂R_eff/∂|ō| > 0

将全部 agent 的观点设为均匀的 |o| 值，扫描 |o| ∈ [0, 0.8]，计算 R_eff。

**预期**：R_eff 随 |ō| 单调递增。

### 验证 2：∂R_eff/∂α₁ > 0

在固定状态下，用有限差分计算导数。

**预期**：导数为正。

### 验证 3：∂R_eff/∂λ_spiral ≤ 0

比较 λ_spiral = 0 和 λ_spiral = 0.85 下的 R_eff，在存在 Γ_i < 0.5 的 agent 的子群中。

**预期**：R_eff(λ_spiral=0.85) ≤ R_eff(λ_spiral=0)。

### 验证 4：R_eff 与网络结构的关系

在不同网络类型（ER/BA/WS/SBM）上计算 k̄_eff 和 R_eff。

**预期**：SBM（社区内高密度）的 R_eff > ER（均匀随机）的 R_eff，在相同 β 下。

---

## 五、与经典 SIR 的对比

| 特征 | 经典 SIR | 本模型 |
|------|---------|--------|
| 传播阈值 | R₀ = β·k̄/γ（常数） | R_eff = ρ(M({o_i, h_i, Γ_i}))（状态依赖） |
| 传播潜力决定因素 | 网络拓扑 + 固定参数 | 网络拓扑 + 观点分布 + 情绪状态 + 意见气候 |
| 超级传播者 | 高度数节点 | 高度数 + 极端观点 + 高情绪节点 |
| 干预效应 | 降低 β（行为干预） | 降低 β（信息管控）+ 降低 α₁|ō|（舆论引导）+ 提高 λ_spiral（社会压力）|

---

## 六、仿真可检验的预测

从上述定理可以导出以下可检验的预测：

1. **极化加剧传播**：当观点从温和（|ō| ≈ 0.2）变为极化（|ō| ≈ 0.7）时，R_eff 应增加 50-200%

2. **回声室效应**：在 SBM 网络（社区内 p_in >> 社区间 p_out）中，R_eff 的社区内分量 >> 社区间分量

3. **沉默螺旋的阈值行为**：λ_spiral 对 R_eff 的效应仅在 Γ_i < 0.5 的 agent 比例超过 ~20% 时才显著

4. **情绪放大**：在冲击事件后（Shock(t) 注入），R_eff 应瞬时上升然后随疲劳累积而下降

---

## 参考文献

- van den Driessche, P., & Watmough, J. (2002). Reproduction numbers and sub-threshold endemic equilibria for compartmental models of disease transmission. *Mathematical Biosciences*, 180(1-2), 29-48.
- Diekmann, O., Heesterbeek, J. A. P., & Metz, J. A. J. (1990). On the definition and the computation of the basic reproduction ratio R₀ in models for infectious diseases in heterogeneous populations. *Journal of Mathematical Biology*, 28(4), 365-382.
- Perron, O. (1907). Zur Theorie der Matrices. *Mathematische Annalen*, 64(2), 248-263.
- Friedkin, N. E., & Johnsen, E. C. (1990). Social influence and opinions. *Journal of Mathematical Sociology*, 15(3-4), 193-206.
