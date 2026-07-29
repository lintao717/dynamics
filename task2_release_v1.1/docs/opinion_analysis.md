# 公开意见偏差与极化稳定性分析

**日期**: 2026-07-24
**对应模型**: `docs/task2_model_definition_v1.md`

---

## 第一部分：公开意见偏差 B_obs

### 1.1 定义

公开意见偏差度量了"群体真正相信的"和"群体公开表达的"之间的系统性差距：

\[
B_{\text{obs}}(t) = \left| \bar{o}^{\text{private}}(t) - \bar{o}^{\text{public}}(t) \right|
\]

其中：

\[
\bar{o}^{\text{private}}(t) = \frac{1}{N} \sum_i o_i(t), \quad
\bar{o}^{\text{public}}(t) = \frac{1}{|\mathcal{A}(t)|} \sum_{i \in \mathcal{A}(t)} \hat{o}_i(t)
\]

这里 A(t) = {i : z_i(t) = A} 是活跃 agent 集合。

### 1.2 偏差的两种来源

公开表达偏差 δ_i(t) = ô_i(t) - o_i(t) 有两项：

\[
\delta_i(t) = \underbrace{\lambda_c \cdot [\bar{o}_{\text{pub,local}}(t) - o_i(t)]}_{\text{从众偏差}} + \underbrace{\lambda_h \cdot h_i(t) \cdot o_i(t)}_{\text{情绪放大}}
\]

**来源 1（从众偏差）**：个体公开表达向本地气候靠拢。当 λ_c > 0 时，少数意见者表达时"软化"自己的立场，多数意见者表达时更加自信。

**来源 2（情绪放大）**：高唤醒状态放大了个体原本的立场方向。当 o_i > 0 时表达更积极，当 o_i < 0 时表达更消极。

### 1.3 选择偏差 vs 表达偏差

B_obs 包含两种机制上不同的偏差。首先定义**有符号偏差**（signed bias）：

\[
\Delta_{\text{obs}}(t) = \bar{\hat{o}}_{\mathcal{A}}(t) - \bar{o}(t)
\]

可以严格分解为：

\[
\boxed{\Delta_{\text{obs}}(t) = \underbrace{(\bar{o}_{\mathcal{A}} - \bar{o})}_{\text{选择偏差 } \Delta_{\text{sel}}} + \underbrace{(\bar{\hat{o}}_{\mathcal{A}} - \bar{o}_{\mathcal{A}})}_{\text{表达偏差 } \Delta_{\text{expr}}}}
\]

绝对偏差 $B_{\text{obs}} = |\Delta_{\text{obs}}|$ 满足三角不等式：

\[
B_{\text{obs}} \leq |\Delta_{\text{sel}}| + |\Delta_{\text{expr}}|
\]

（注意：一般 $B_{\text{obs}} \neq B_{\text{sel}} + B_{\text{expr}}$，因为两个偏差可能部分抵消。）

**选择偏差** $\Delta_{\text{sel}}$：活跃 agent 集合的私人观点均值与全体的差异。这是沉默螺旋的直接结果——少数派选择不表达，所以活跃群体向多数派倾斜。

**表达偏差** $\Delta_{\text{expr}}$：活跃 agent 的公开表达与他们私人观点的差异。这是从众偏差 + 情绪放大的结果。

### 1.4 命题 5：选择偏差的符号（充分条件）

**命题 5**：在以下条件下，选择偏差与多数意见同向：
1. $\lambda_{\text{spiral}} > 0$，且存在 $\Gamma_i < 0.5$ 的 agent；
2. 本地意见气候与全局多数意见同向；
3. 观点极端性（$\alpha_1$）、情绪（$\alpha_2$）和疲劳（$\alpha_4$）等激活因素在正负两侧条件对称。

在这些条件下：

\[
\text{sign}(\bar{o}_{\mathcal{A}} - \bar{o}) = \text{sign}(\bar{o})
\]

**证明**：沉默螺旋惩罚降低了少数意见者（Γ_i < 0.5）的 P(E→A)。少数意见者 o_i 的符号与多数意见 ō 相反。因此：

- 若 ō > 0（多数支持），少数 o_i < 0 的 agent 被沉默 → A 集合的均值 o 高于全体均值 → 选择偏差为正
- 若 ō < 0（多数反对），对称地 → 选择偏差为负

形式上：
\[
\bar{o}_{\mathcal{A}} - \bar{o} = \frac{\sum_{i \in \mathcal{A}} o_i}{|\mathcal{A}|} - \frac{\sum_i o_i}{N}
\]

由于 P(i ∈ A) 随 Γ_i 递减，且 Γ_i = 1 - |o_i - climate|/2 与 o_i 的符号相关，A 集合的分布向多数意见偏移。

### 1.5 命题 6：双向耦合对 B_obs 的数值效应

**命题 6**（数值观察）：在当前默认参数和 N ≥ 500 的 SBM 网络配置下，双向耦合模式下的平均公开偏差低于无耦合模式（B_obs 约降低 37%）。这一效应不是普适的——其强度和方向依赖于参数配置和网络结构。

**机制分析**：

双向耦合对 B_obs 存在两个竞争效应：
1. **降低选择偏差**：观点更新使私人观点向本地气候靠拢，减少 ō_𝒜 与 ō 的差距
2. **引入表达偏差**：从众效应（λ_c）和情绪放大（λ_h）使 ô 偏离 o

净效应取决于哪个更强。在以下情况下 B_obs 可能不降反升：
- 高 λ_c 或 λ_h 参数
- 低 μ_i（观点更新慢，反馈弱）
- 频繁的新 agent 进入（U→E→A）打破稳态
- 不同社区之间存在相反方向的从众压力
- 锚定项 ζ 将观点拉离当前气候
- 随机噪声 ξ_i

**结论**：双向耦合降低 B_obs 是一个**参数依赖的数值现象**，而非普遍定理。它需要在特定条件下才能成立，但确实揭示了反馈机制的重要定性特征。

### 1.6 定理 7：无耦合下 B_obs 的分歧条件

**定理 7**：在无耦合条件下（λ_c = λ_h = 0, λ_spiral = 0），当仅存在选择偏差时：

\[
B_{\text{obs}} = |\bar{o} - \bar{o}_{\mathcal{A}}| = \left|\frac{\sum_i o_i \cdot (1 - p_i)}{N} \cdot \frac{N}{|\mathcal{A}|}\right|
\]

其中 p_i = P(i ∈ A)。选择偏差的大小取决于 p_i 与 o_i 的相关性。

当激活概率依赖于 |o_i|（即 α₁ > 0）时：
- |o_i| 大的 agent 更可能活跃 → A 集合包含更多极端 agent
- 但极端 agent 均匀分布在正负两侧 → 选择偏差可能很小
- B_obs 主要来自表达偏差，而非选择偏差

当激活概率依赖于 Γ_i（即 α₃ > 0）时：
- Γ_i 高的 agent（意见与气候一致）更可能活跃
- 如果气候偏向多数派 → 少数派被选择性地排除
- 产生系统性选择偏差

---

## 第二部分：极化稳定性

### 2.1 观点动力学系统的稳态

将私人观点更新方程写成离散动力系统：

\[
o_i(t+1) = \Pi_{[-1,1]}\left[ o_i(t) + \mu_i \cdot F_i(\mathbf{o}(t)) + \xi_i(t) \right]
\]

其中：
\[
F_i(\mathbf{o}) = \zeta_i[o_i(0) - o_i] + (1-\zeta_i)\sum_j w_{ji}^o \cdot \Phi_i(\hat{o}_j - o_i) + \eta_i I_i + \chi_i u_i
\]

### 2.2 命题 8：锚定项抑制观点塌缩的充分条件

**命题 8**：当 agent 具有异质初始锚定（ζ_i > 0）且初始观点非完全同质时，锚定项提供了向初始观点的回复力，能够抑制观点向全局单一共识塌缩。

**论证**：

在无噪声（ξ_i = 0）且无外部信息（I_i = u_i = 0）时，观点更新包含弹性回复力 -ζ_i[o_i - o_i(0)]。

每个 agent 的观点存在一个以初始值为中心的"引力盆地"。锚定强度 ζ_i 越大，盆地越深。社会影响项可以扰动观点偏离初始值，但锚定力持续将其拉回。

**注意**：这不是一个严格的 Lyapunov 稳定性证明——社会影响项在有界置信截断下是非线性的，全局收敛性需要更强的条件。此外，如果所有 agent 初始观点相同，系统可以从一开始就处于共识（平凡情况）。

**推论 8.1**：ζ-ε 相位图（见数值实验）显示，在合理的参数范围内（ζ ∈ [0.1, 0.9], ε ∈ [0.1, 0.9]），系统表现出三种定性不同的区域：共识（低 ζ + 高 ε）、极化（中 ζ + 中 ε）、和类碎片化（高 ζ + 低 ε）。

**推论 8.2**：极化（两个分离的观点簇）可以在以下候选条件下维持：
1. ζ_i 适中（不太大，允许社会影响；不太小，防止塌缩）
2. ε_i 适中（允许簇内收敛但阻止簇间融合）
3. 初始观点分布是双峰的

**数值验证**：ζ-ε 相位图实验（6×6 网格，T=100）显示，在 36 个参数组合中，33 个维持极化（σ_o > 0.15），仅 3 个达到共识（低 ζ + 高 ε 组合）。

### 2.3 命题 9：极化稳定的候选条件

**命题 9**（候选充分条件）：在无噪声且初始观点呈对称双峰分布（两个簇中心分别在 +a 和 -a，a > 0）的简化设定下，一个候选的极化维持条件为：

\[
\epsilon < 2a \cdot (1 - \zeta_{\text{eff}})
\]

其中 $\zeta_{\text{eff}}$ 是群体平均锚定强度。

**解释**：
- 如果 ε 太小：簇内也不收敛 → 碎片化（多簇）
- 如果 ε 适中且 ζ 适中：两个簇各自内部收敛，但簇间距离 > ε → 极化维持
- 如果 ε 太大：两个簇融合 → 共识
- 如果 ζ 太大：所有 agent 弹回初始值 → 社会影响消失

**推导**（简化双簇模型）：

考虑两个簇的中心 c₁ ≈ +a 和 c₂ ≈ -a。簇间距离 d = |c₁ - c₂| ≈ 2a。

在均衡处，锚定向初始值的拉力与社会影响向簇均值的拉力平衡。对于簇 k 内的 agent i：

\[
o_i^* = \frac{\zeta_i}{\zeta_i + (1-\zeta_i)s_i} \cdot o_i(0) + \frac{(1-\zeta_i)s_i}{\zeta_i + (1-\zeta_i)s_i} \cdot \bar{o}_{\text{cluster }k}
\]

其中 $s_i$ 是簇内邻居比例。簇中心向初始均值偏移但不会完全等同于初始值。

两簇维持分离的条件为：

\[
|c_1^* - c_2^*| > \epsilon_{\text{eff}}
\]

其中 $\epsilon_{\text{eff}}$ 是簇间有界置信阈值。上述不等式在 ζ 足够大或 ε 足够小时成立。

**注意**：上述推导是简化的双簇降维分析，不是严格的"当且仅当"条件。完整的极化条件需要更细致的数学处理（包括网络拓扑、度分布、权重异质性等的影响）。当前版本将其作为数值实验的候选理论指导，ζ-ε 相位图（见仿真验证）提供了经验支持。

### 2.4 仿真预测

从定理 8-9 可以导出以下可检验的预测：

1. **ζ 扫描**：ζ → 1 时观点完全冻结在初始值；ζ → 0 时观点塌缩为 HK 模型的均衡
2. **ε 扫描**：存在临界值 ε_crit，当 ε > ε_crit 时两个簇融合
3. **ζ-ε 相位图**：在 (ζ, ε) 平面上存在三个区域——共识、极化、碎片化
4. **T=500 下的 σ_o 不趋近于 0**（已验证于 longterm_polarization 实验）

---

## 第三部分：数值验证

### 3.1 B_obs 偏差分解实验

对实验 5（双向耦合对比）的数据进行事后分析，将 B_obs 分解为选择偏差和表达偏差。

关键假设：双向耦合的 B_obs < 单向耦合的 B_obs < 无耦合的 B_obs。

### 3.2 ζ-ε 相位图

扫描 ζ ∈ [0, 1] 和 ε ∈ [0, 0.8]，测量 T=200 时的 σ_o 和簇数，绘制相位图。

预期：三个区域——
- 左上（高 ζ + 低 ε）：碎片化（多簇，高 σ_o）
- 中部：极化（2-3 簇，中等 σ_o ≈ 0.4-0.6）
- 右下（低 ζ + 高 ε）：共识（1 簇，低 σ_o < 0.1）

---

## 参考文献

- Friedkin, N. E., & Johnsen, E. C. (1990). Social influence and opinions. *Journal of Mathematical Sociology*.
- Hegselmann, R., & Krause, U. (2002). Opinion dynamics and bounded confidence: models, analysis and simulation. *Journal of Artificial Societies and Social Simulation*.
- Lorenz, J. (2007). Continuous opinion dynamics under bounded confidence: A survey. *International Journal of Modern Physics C*.
- Noelle-Neumann, E. (1974). The spiral of silence: A theory of public opinion. *Journal of Communication*.
