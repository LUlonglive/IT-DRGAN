# End-to-End Graph Neural Network Framework for Precise Localization of Internal Leakage Valves in Marine Pipelines Based on Intelligent Graphs

# 2. Problem Statement and Approach Overview

海洋管道阀门内泄漏声发射信号具有高度非平稳和高频瞬态特性，其能量在多个频带间呈现复杂的耦合分布。在实际海洋环境中，这类信号易受管道材质、阀门结构和背景噪声的干扰，给泄漏定位带来巨大挑战。传统时域或频域分析方法难以全面刻画信号的复杂特性及其频带间的动态关联。此外，现有的深度学习方法在处理此类信号时也面临诸多限制：一方面，传统深度学习架构难以有效捕捉泄露事件间的空间拓扑关系；另一方面，虽然图神经网络在处理非平稳信号方面展现出潜力，但在实际应用中仍存在图结构表征能力不足和深层网络训练不稳定等问题。

针对上述挑战，本研究提出了一种基于声发射信号的海洋管道阀门内泄漏定位方法。如Fig. {workflow}所示，该方法通过声发射信号采集获取泄漏特征，并结合图神经网络技术实现精确定位。本研究的创新点在于提出了一种新的信号处理和特征提取方法，能够有效处理复杂海洋环境下的泄漏信号，为后续的精确定位提供可靠的技术支持。具体的技术细节将在第3节中详细阐述。

# 3. Methodology

## 3.1 Acoustic Emission Signal Characteristics

声发射（Acoustic Emission, AE）是材料在变形过程中自发释放弹性能的现象，其信号可被AE传感器检测。作为本研究Intelligent Graph框架的数据基础，声发射技术为阀门内部状态监测提供了关键信号源。AE信号源于多种物理现象，如塑性变形、裂纹扩展、泄漏、燃烧及断裂。其中，管道阀门内部泄漏产生的声学信号是典型的AE现象，构成本研究中Intelligent Graph构建的关键输入。

当阀门内部发生介质泄漏时，压力差导致流体在阀门内部产生湍流。这不仅扰乱了介质的正常流动，还促使弹性波与阀壁相互作用，进而产生宽频带声学信号。这些携带泄漏信息的弹性波传播至阀门表面并由AE传感器捕获，其采集的信号构成本研究Intelligent Graph分析的原始数据。通过对所采集AE信号的系统化处理，本研究提出的IT-DRGAN能够有效识别发生内泄漏的管道阀门。

## 3.2 Intrinsic Spatiotemporal Topological Graph Generation

内生时空拓扑图（Intrinsic Spatiotemporal Topological Graph, ISTG）是Intelligent Graph构建的第一阶段，旨在利用小波分解与子带能量分析，将声发射信号的时空相似性及其多尺度动态特性映射为拓扑结构，为后续Intelligent Graph的构建奠定基础。

ISTG的构建始于各频带能量序列的计算。对于给定信号 $x(t)$，采用特定小波基进行 $L$ 层小波包分解（Wavelet Packet Decomposition, WPD），将其分解为 $2^L$ 个子带信号 $S_{L,k}(t)$：

$$
S_{L,k}(t) = \text{WPD}\left(x(t), \psi_{L,k}\right), \quad k=1,2,\ldots,2^L \tag{1}
$$

其中 $\psi_{L,k}$ 代表第 $L$ 层第 $k$ 个小波包的基函数。每个子带信号 $S_{L,k}(t)$ 表征了原始信号在不同频带上的特性。为捕捉信号的动态特性，针对每个子带信号 $S_{L,k}(t)$，应用滑动窗口技术计算其局部能量序列。设滑动窗口大小为 $W_s$，重叠步长为 $O_s$，则第 $j$ 个窗口的能量 $E_{j,k}$ 计算如下：

$$
E_{j,k} = \sum_{n=(j-1)O_s+1}^{(j-1)O_s + W_s} \left| S_{L,k}(n) \right|^2, \quad j=1,2,\ldots,N_w \tag{2}
$$

其中 $N_w$ 为窗口总数。由此获得的能量序列 $\mathbf{E}_k = [E_{1,k}, E_{2,k}, \ldots, E_{N_w,k}]$ 反映了第 $k$ 个频带能量随时间的变化。


随后，进行节点间相似性的度量以构建ISTG的邻接矩阵。在ISTG中，各频带被视为图的节点，节点间的边权重由结合时间域与频域信息的组合相似性定义。时间域相似性旨在量化不同子带能量序列间的方向一致性，本研究采用余弦相似度进行计算。对于两个节点 $p$ 和 $q$ 的能量序列 $\mathbf{E}_p$ 和 $\mathbf{E}_q$，其时间域相似性 $\text{Sim}_{\text{time}}(p,q)$ 定义为：

$$
\text{Sim}_{\text{time}}(p,q) = \frac{\mathbf{E}_p \cdot \mathbf{E}_q}{\|\mathbf{E}_p\|_2 \|\mathbf{E}_q\|_2} \tag{3}
$$

其中 $\|\cdot\|_2$ 表示向量的欧几里得范数。频域相似性则基于子带信号中心频率的差异进行量化。设节点 $p$ 和 $q$ 对应的中心频率分别为 $f_p$与$f_q$，其频域相似性 $\text{Sim}_{\text{freq}}(p,q)$ 计算如下：

$$
\text{Sim}_{\text{freq}}(p,q) = \frac{1}{1 + |f_p - f_q|} \tag{4}
$$

中心频率 $f_k$ 定义为对应频带功率谱的加权平均频率：

$$
f_k = \frac{\sum_{m=1}^{M_k} f_{k,m} \cdot P(f_{k,m})}{\sum_{m=1}^{M_k} P(f_{k,m})} \tag{5}
$$

其中 $f_{k,m}$ 是频带 $k$ 内的第 $m$ 个频率分量，$P(f_{k,m})$ 是其对应的功率，$M_k$ 是该频带内的频率分量总数。最终，通过对时间域相似性与频域相似性的加权组合，定义节点间的联合相似性 $\text{Sim}_{\text{joint}}(p,q)$：

$$
\text{Sim}_{\text{joint}}(p,q) = \alpha \cdot \text{Sim}_{\text{freq}}(p,q) + (1-\alpha) \cdot \text{Sim}_{\text{time}}(p,q) \tag{6}
$$

其中 $\alpha \in [0,1]$ 是用以平衡频域和时域信息贡献的权重系数。基于此联合相似性构建ISTG的邻接矩阵 $\mathbf{A}^{\text{ISTG}}$：

$$
A_{pq}^{\text{ISTG}} = \begin{cases} 
\text{Sim}_{\text{joint}}(p,q), & \text{if } \text{Sim}_{\text{joint}}(p,q) > \theta_{\text{sim}} \\
0, & \text{otherwise}
\end{cases} \tag{7}
$$

其中 $\theta_{\text{sim}}$ 是预设的相似性阈值。内生时空拓扑图的完整构建流程如图Fig. {ISTG}所示。

Fig. {ISTG} Process of constructing Intrinsic Spatiotemporal Topological Graph.

## 3.3 Laplacian Feature Extraction

为将内生时空拓扑图（ISTG）中的结构与拓扑信息转化为Intelligent Graph的节点特征，本研究提出了一种基于图论的特征提取方法。该方法通过分析ISTG的拉普拉斯矩阵特性，生成能够有效表征原始信号特征的向量，并将其作为Intelligent Graph构建的核心输入。

图论中常用的矩阵包括邻接矩阵和拉普拉斯矩阵。在Intelligent Graph的构建中，本研究聚焦于拉普拉斯矩阵的应用。拉普拉斯矩阵的计算首先需要确定图的度矩阵。随后，通过对拉普拉斯矩阵进行特征值分解所获得的特征值向量，将作为Intelligent Graph中节点的特征表示。


在ISTG的特征提取阶段，权重矩阵 $\mathbf{W}$ 采用ISTG邻接矩阵 $\mathbf{A}^{\text{ISTG}}$，其元素 $W_{pq}$ 表示节点 $p$ 与 $q$ 之间的连接强度。图的度矩阵 $\mathbf{D}$ 是一个对角矩阵，其对角元素 $D_{pp}$ 表示节点 $p$ 的度，即与该节点连接的所有边的权重之和：

$$
D_{pp} = \sum_{q} W_{pq}, \quad D_{pq} = 0 \text{ for } p \neq q \tag{8}
$$

拉普拉斯矩阵 $\mathbf{L}$ 是一个对称半正定矩阵，能反映图的全局拓扑特性：

$$
\mathbf{L} = \mathbf{D} - \mathbf{W} \tag{9}
$$

对拉普拉斯矩阵进行特征值分解：

$$
\mathbf{L} = \mathbf{U} \mathbf{\Lambda} \mathbf{U}^T \tag{10}
$$

其中，$\mathbf{\Lambda} = \text{diag}(\lambda_0, \lambda_1, \ldots, \lambda_{N_n-1})$ 为按升序排列的特征值对角矩阵，$N_n$ 为图中节点数量，等于频带总数 $2^L$。$\mathbf{U} = [\mathbf{u}_0, \mathbf{u}_1, \ldots, \mathbf{u}_{N_n-1}]$ 是对应的特征向量矩阵。因此，每个ISTG均可由其拉普拉斯特征值向量 $\boldsymbol{\lambda} = [\lambda_0, \lambda_1, \ldots, \lambda_{N_n-1}]$ 表示，该向量将作为Intelligent Graph构建模型的一维特征输入。

## 3.4 Adaptive Intelligent Graph Learning

本节详细阐述动态显式正则化（Dynamic Explicit Regularization, DER）方法。DER专用于优化Intelligent Graph的邻接矩阵 $\mathbf{A}^{\text{IG}}$，揭示样本间的内在关联结构。借鉴迭代图学习思想[59]，DER将图结构优化视为渐进过程，通过多轮迭代完善样本间关联。在此框架中，每个样本由其对应的内生时空拓扑图（ISTG）拉普拉斯特征值向量 $\boldsymbol{\lambda}$ 表示。相比传统的预定义或静态图连接方式，DER采用数据驱动策略，自适应地优化样本间图结构，从而准确刻画其深层关系。这种动态学习机制结合特定的正则化更新策略，确保构建的图结构能优化反映数据内在特性，提升后续图学习任务性能。

该过程包括初始图构建和迭代优化两个阶段。假设有 $N$ 个样本，每个样本 $i$ 由其特征向量 $\mathbf{x}_i \in \mathbb{R}^D$ 表示（此特征向量 $\mathbf{x}_i$ 即为该样本对应ISTG的拉普拉斯特征值向量 $\boldsymbol{\lambda}_i$，维度为 $D$)。样本 $i$ 和样本 $j$ 之间的欧几里得距离 $d_{ij}$ 定义为：

$$
d_{ij} = \| \mathbf{x}_i - \mathbf{x}_j \|_2 = \sqrt{\sum_{k=1}^{D} (x_{ik} - x_{jk})^2} \tag{11}
$$

其中 $x_{ik}$ 表示特征向量 $\mathbf{x}_i$ 的第 $k$ 个分量。所有样本对间的距离共同构成了距离矩阵 $\mathbf{D}^{\text{dist}} \in \mathbb{R}^{N \times N}$。随后，利用高斯核函数将距离矩阵 $\mathbf{D}^{\text{dist}}$ 转化为初始相似性矩阵 $\mathbf{W}^{\text{sim}}$：

$$
W_{ij}^{\text{sim}} = \exp\left(-\frac{d_{ij}^2}{2\sigma_g^2}\right) \tag{12}
$$

其中 $\sigma_g$ 是高斯核的长度尺度参数，控制相似性随距离衰减的速度。为构建稀疏的初始邻接矩阵 $\mathbf{A}^{(0)}$，对每个节点 $i$，仅保留其与 $K$ 个最近邻居间的相似性值，其余则置零：

$$
A_{ij}^{(0)} = \begin{cases} 
W_{ij}^{\text{sim}}, & j \in \mathcal{N}_K(i) \\
0, & \text{otherwise}
\end{cases} \tag{13}
$$

其中 $\mathcal{N}_K(i)$ 表示节点 $i$ 的 $K$ 个最近邻居集合。由于构造的 $\mathbf{A}^{(0)}$ 可能非对称，因此进行对称化处理：

$$
\mathbf{A}^{(0)} = \frac{\mathbf{A}^{(0)} + (\mathbf{A}^{(0)})^T}{2} \tag{14}
$$

通过上述步骤，即可获得一个稀疏且对称的初始邻接矩阵 $\mathbf{A}^{(0)}$。该矩阵在保留样本局部几何结构的同时，亦保证了其稀疏性。

在获得初始邻接矩阵 $\mathbf{A}^{(0)}$ 后，通过迭代学习过程对其进行优化，以动态调整图的拓扑结构。本研究采用如下特定更新规则：

$$
A_{ij}^{(k+1)} = \beta \cdot \log\left(\sum_{p=1}^{N} A_{ip}^{(k)} + 1\right) + \gamma \cdot \tilde{A}_{ij}^{(k)} \tag{15}
$$

其中，$\tilde{A}_{ij}^{(k)}$ 表示在第 $k$ 次迭代中基于样本原始特征 $\mathbf{x}_i, \mathbf{x}_j$重新评估的节点 $i$ 与 $j$ 间的相似度。该更新规则通过对数项 $\beta \cdot \log(\cdot)$ 对节点度进行正则化调整，并结合 $\gamma \cdot \tilde{A}_{ij}^{(k)}$ 项整合当前迭代的相似性信息，从而实现图结构的动态学习。参数 $\beta$ 与 $\gamma$ 用以平衡这两个组成部分的影响。迭代过程的收敛性通过邻接矩阵Frobenius范数的变化量进行衡量：

$$
\| \Delta \mathbf{A}^{(k)} \|_F = \| \mathbf{A}^{(k+1)} - \mathbf{A}^{(k)} \|_F \tag{16}
$$

当该变化量 $\| \Delta \mathbf{A}^{(k)} \|_F$ 小于预设的收敛阈值 $\epsilon$ 时，迭代停止；否则，优化过程将持续进行，直至达到预设的最大迭代次数。Fig.{DER} 中展示了DER的原理示意图。

通过前述方法构建的图称为Intelligent Graph。该图可表示为 $G^{\text{IG}} = (V^{\text{IG}}, E^{\text{IG}}, \mathbf{A}^{\text{IG}})$，其中：
- $V^{\text{IG}}$ 代表Intelligent Graph的节点集合，每个节点的特征向量来源于内生时空拓扑图的拉普拉斯矩阵特征值向量 $\boldsymbol{\lambda}$。
- $E^{\text{IG}}$ 代表边集合，表示节点之间通过动态显式正则化方法学习到的连接关系。
- $\mathbf{A}^{\text{IG}} \in \mathbb{R}^{M \times M}$ 为最终的邻接矩阵，其中 $M$ 为Intelligent Graph中节点的数量，表示任意两个节点之间的连接强度。
  
## 3.5 Intelligent Graph Enhanced DRGAN Leakage Localization

在完成Intelligent Graph（$G^{\text{IG}}$）的构建后，本节将介绍一种专门针对海洋管道阀门内部泄漏定位任务的动态残差图聚合网络（Dynamic Residual Graph Aggregation Network, DRGAN），其架构如Fig. {DRGAN}所示。该网络基于Graph Sample and Aggregate思想设计，通过引入残差连接与动态门控机制，有效增强了对$G^{\text{IG}}$节点特征的学习能力。DRGAN以ISTG的拉普拉斯特征值向量$\boldsymbol{\lambda}$作为节点特征输入，这种设计显著提升了对图结构拓扑特征的表征能力，从而能够更准确地进行泄漏定位，并使模型训练过程更加稳定。与采用全局信息处理的传统图卷积网络（GCN）相比，GraphSAGE专注于节点的局部邻域操作，这种局部化特性使其更适合处理本研究中的Intelligent Graph结构。在DRGAN框架下，每个节点的最终表示是通过$L_g$层深度的邻居信息逐层聚合而成。

设 $\text{AGG}$ 表示聚合操作，GraphSAGE 的核心步骤如下：
1) **邻域聚合**：在第 $l$ 层，其中 $l$ 是一个索引，取值范围从 1 到 $L_g$，节点 $v$ 聚合其一阶邻居 $\mathcal{N}(v)$ 的第 $l-1$ 层表示，形成聚合向量 $\mathbf{h}_{\mathcal{N}(v)}^{l}$：
   
$$
\mathbf{h}_{\mathcal{N}(v)}^{l} = \text{AGG}\left(\{\mathbf{h}_u^{l-1} | u \in \mathcal{N}(v)\}\right) \tag{17}
$$

其中初始节点特征设为 $\mathbf{h}_v^{0}$ 等于 $\mathbf{x}_v$。常用的聚合器包括均值聚合器、LSTM聚合器和池化聚合器。
1) **更新节点表示**：将节点 $v$ 的第 $l-1$ 层表示 $\mathbf{h}_v^{l-1}$ 与聚合得到的邻域向量 $\mathbf{h}_{\mathcal{N}(v)}^{l}$ 拼接，然后通过带有非线性激活函数 $\sigma_a$ 的全连接层变换：
   
$$
\mathbf{h}_v^{l} = \sigma_a\left(\mathbf{W}^{l} \cdot \text{CONCAT}\left(\mathbf{h}_v^{l-1}, \mathbf{h}_{\mathcal{N}(v)}^{l}\right)\right) \tag{18}
$$

其中 $\mathbf{W}^{l}$ 是第 $l$ 层的可学习权重矩阵，新生成的特征 $\mathbf{h}_v^{l}$ 作为下一层的输入。

1) **输出表示**：经过 $L_g$ 层迭代后，节点 $v$ 的最终嵌入表示为 $\mathbf{z}_v = \mathbf{h}_v^{L_g}$。
   
2) **Output Representation**: After $L_g$ layers of iteration, the final embedding representation of node $v$ is $\mathbf{z}_v = \mathbf{h}_v^{L_g}$.

DRGAN的基本构建单元是残差块，每个残差块包含多层GraphSAGE卷积，用于实现深层特征提取。为稳定训练过程，在卷积层间引入了批归一化和ReLU激活函数。残差块的主路径处理流程如下：输入特征经过多层GraphSAGE卷积，每层后接批归一化和ReLU激活函数，最终输出经过批归一化、ReLU激活和Dropout的残差特征。这些残差特征通过残差连接与输入特征相结合。

DRGAN通过堆叠多个残差块构建深层结构。DRGAN引入动态门控机制以更好地融合残差块间传递的特征信息。动态门控机制通过全连接层和Sigmoid激活函数计算动态门控权重，表示为 $\mathbf{g}_k$：

$$
\mathbf{g}_k = \sigma_s\left(\mathbf{W}_g \mathbf{h}^{(k-1)} + \mathbf{b}_g\right) \tag{26}
$$

随后利用门控权重动态调整当前残差块主路径输出与前一层特征的融合比例。融合通过以下方式进行：将门控权重 $\mathbf{g}_k$ 与前一层特征 $\mathbf{h}^{(k-1)}$ 进行哈达玛积，并将此结果与当前残差块输出的结果相加，其中 $\mathbf{h}_{k}^{\text{res}}$ 代表当前残差块的主路径输出：

$$
\mathbf{h}^{(k)} = \mathbf{g}_k \odot \mathbf{h}^{(k-1)} + \left(1 - \mathbf{g}_k\right) \odot \mathbf{h}_{k}^{\text{res}}
$$

这种动态特征融合机制使网络能够根据输入数据自适应地调整信息流，有效平衡浅层和深层特征的贡献，从而更有效地从Intelligent Graph中学习判别性特征。经过残差块的堆叠处理后，DRGAN将学习到的最终节点嵌入通过一个分类层，输出针对各个样本代表不同阀门或工况的泄漏状态预测，从而实现对海洋管道中发生内部泄漏的阀门的精确定位。

