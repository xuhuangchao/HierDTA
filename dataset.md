# HierDTA 图数据说明

每个 DTA 样本包含一个分子 atom 图、一个分子 motif 图和一个 protein residue 图。三类图的节点数量均不固定。本文档仅描述输入图的数据结构和特征，不涉及任何交互模块设计。

## 1. Atom graph

Atom graph 表示分子的原子级拓扑。

### 1.1 节点

每个节点对应一个原子，节点特征维度为 37：

| 特征 | 维度 | 描述 |
|---|---:|---|
| 原子类型 one-hot | 13 | B、Br、C、Ca、Cl、F、H、I、N、Na、O、P、S |
| 原子序数 | 1 | RDKit atomic number，数值特征 |
| 显式化合价 one-hot | 7 | 0–6 |
| 氢原子数量 one-hot | 5 | 0–4 |
| 杂化方式 one-hot | 5 | SP、SP2、SP3、SP3D、SP3D2 |
| 是否芳香 | 1 | 二值特征 |
| 是否位于环中 | 1 | 二值特征 |
| CIP 手性 one-hot | 3 | R、S、unknown |
| 是否可能具有手性 | 1 | 二值特征 |

```text
atom_x shape = [N_atom, 37]
```

### 1.2 边

两个原子之间存在化学键时建立边。每条无向化学键被展开为两个方向：

```text
atom_i → atom_j
atom_j → atom_i
```

边特征维度为 13：

| 特征 | 维度 | 描述 |
|---|---:|---|
| 化学键类型 one-hot | 4 | single、double、triple、aromatic |
| 是否位于环中 | 1 | 二值特征 |
| 是否共轭 | 1 | 二值特征 |
| 立体化学 one-hot | 7 | none、any、Z、E、cis、trans、unknown |

```text
atom_edge_index shape = [2, E_atom]
atom_edge_attr  shape = [E_atom, 13]
```

输入边中没有显式自环。

## 2. Motif graph

Motif graph 表示比原子更高层次的分子子结构。Motif 包括识别出的官能团、环和未归入这些结构的剩余局部结构。

不同 motif 可以共享原子，因此一个原子可能同时属于多个 motif。未被其他 motif 覆盖的原子会形成单原子 motif，所以每个原子至少属于一个 motif，并且每个 motif 至少包含一个原子。

### 2.1 节点

每个节点对应一个 motif，节点特征维度为 50，由两部分拼接：

$$
x_m =
\left[
\sum_{i\in m}x_i^{\mathrm{atom}}
\;\Vert\;
\sum_{e\in m}x_e^{\mathrm{bond}}
\right].
$$

| 特征 | 维度 | 描述 |
|---|---:|---|
| motif 内所有原子特征之和 | 37 | 对成员原子的 37 维特征求和 |
| motif 内部所有化学键特征之和 | 13 | 仅统计两个端点都属于该 motif 的键 |

```text
motif_x shape = [N_motif, 50]
```

Motif 节点特征使用求和而不是均值，因此保留了一定的 motif 大小和组成信息。

### 2.2 Motif–motif 边

两个 motif 在以下任一情况下连接：

1. 两个 motif 共享至少一个原子；
2. 两个 motif 不共享原子，但它们的成员原子之间存在化学键。

每个无向 motif 连接被展开为两个方向。边特征维度为 37：

- 如果两个 motif 共享原子，边特征为所有共享原子的 37 维特征之和；
- 如果两个 motif 通过化学键连接，边特征为该键两个端点原子的 37 维特征之和。

```text
motif_edge_index shape = [2, E_motif]
motif_edge_attr  shape = [E_motif, 37]
```

相同 motif 对的重复连接会被去重，输入边中没有显式自环。

### 2.3 Atom–motif membership 关系

另外保存原子到所属 motif 的有向归属关系：

```text
atom_i → motif_j
```

每条 membership edge 的特征固定为标量 1：

```text
membership_edge_index shape = [2, E_membership]
membership_edge_attr  shape = [E_membership, 1]
```

重要性质：

- 一个原子可以指向多个 motif；
- 每个 motif 至少接收一条 membership edge；
- membership edge 只表达组成关系，不是普通化学键。

## 3. Protein residue graph

Protein graph 表示蛋白质三维结构中的残基级关系。

### 3.1 节点

每个节点对应一个标准氨基酸残基，节点特征维度为 41：

| 特征 | 维度 | 描述 |
|---|---:|---|
| SASA z-score | 1 | 残基溶剂可及表面积，在每个蛋白内部标准化 |
| 主链二面角 φ | 1 | 除以 180 归一化；缺失时为 0 |
| 主链二面角 ψ | 1 | 除以 180 归一化；缺失时为 0 |
| DSSP 二级结构 one-hot | 6 | H、B、E、G、T、S；其他状态为全 0 |
| AAPHY7 描述符 | 7 | 氨基酸物理化学性质描述符 |
| BLOSUM62 描述符 | 23 | 氨基酸替换特征 |
| phosphorylated flag | 1 | 蛋白磷酸化标志，未提供时为 0 |
| mutated flag | 1 | 蛋白突变标志，未提供时为 0 |

最后两个蛋白级标志会复制到该蛋白的所有残基节点。

```text
protein_x shape = [N_residue, 41]
```

### 3.2 边

Protein graph 使用有向残基边。首先将以下残基对作为候选：

- 两个残基的任意重原子距离不超过 7 Å；
- 或两个残基在同一条链上共价相邻。

候选无向残基对被展开为两个方向。只有至少存在一种相互作用的候选残基对才会保留为边。

边特征为 10 维二值向量：

| 索引 | 相互作用 |
|---:|---|
| 0 | 共价连接 |
| 1 | 疏水接触，cutoff 4 Å |
| 2 | 氢键：源残基 donor → 目标残基 acceptor，cutoff 3.5 Å |
| 3 | 氢键：源残基 acceptor → 目标残基 donor，cutoff 3.5 Å |
| 4 | 盐桥：源残基 cation → 目标残基 anion，cutoff 4 Å |
| 5 | 盐桥：源残基 anion → 目标残基 cation，cutoff 4 Å |
| 6 | cation–π：源残基 aromatic → 目标残基 cation，cutoff 5 Å |
| 7 | cation–π：源残基 cation → 目标残基 aromatic，cutoff 5 Å |
| 8 | 平行 π–π 堆积，cutoff 5 Å |
| 9 | 垂直 π–π 堆积，cutoff 5 Å |

```text
protein_edge_index shape = [2, E_protein]
protein_edge_attr  shape = [E_protein, 10]
```

一条残基边可以同时具有多种相互作用，因此 10 维边特征是 multi-hot，而不是单一类别。氢键、盐桥和 cation–π 特征具有方向性。

## 4. 独立全局特征

每个蛋白还有一个独立的 1280 维全序列 ESM-2 全局表示：

```text
esm_global shape = [1280]
```

它由 ESM-2 残基表示在完整序列上平均池化得到，不属于 protein graph 的 41 维残基节点特征。

每个分子还有一个独立的 2048 维 ECFP4 指纹：

```text
fingerprint shape = [2048]
```

它不属于 atom graph 或 motif graph 的节点特征。
