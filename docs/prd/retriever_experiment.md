# 实验验证与分析

## 实验目标

实验环节主要对RAG检索器进行评估测试，验证检索器的有效性和性能。

## 评测指标
本研究采用以下指标衡量检索器的准确性：
- Hit Rate @K (命中率): 前 K 个检索结果中是否包含正确答案chunk_id。K根据经验进行提前确认取值。它只关心一件事：在检索回来的前 K 个结果里，有没有包含那个“标准答案（Ground Truth）”，它不在乎答案排在第 1 位还是第 $K$ 位。计算公式如下
给定测试集 $Q$，其规模为 $N$，对于任一查询 $q_i \in Q$，记其标准答案块为 $g_i$，检索器生成的 Top-K 候选集合为 $R_{i,K}$。则 $\mathrm{HR@K}$ 的计算公式定义为：

$$
\mathrm{HR@K} = \frac{1}{N}\sum_{i=1}^{N}\delta\bigl(g_i, R_{i,K}\bigr)
$$

其中，$\delta$ 为命中指示函数，若 $g_i \in R_{i,K}$ 则 $\delta=1$，反之为 $0$。在本实验中，$K$ 分别取值 $\{5, 10, 20\}$ 以评估系统在不同召回深度下的性能。
- MRR (Mean Reciprocal Rank): 衡量正确答案在检索列表中的排名靠前程度。关注排名了。认为正确答案排得越靠前，系统的性能就越好。它取正确答案排名的倒数作为分数。计算公式如下

$$
\mathrm{MRR} = \frac{1}{N}\sum_{i=1}^{N}\frac{1}{\mathrm{rank}_i}
$$

- NDCG (Normalized Discounted Cumulative Gain)：NDCG 解决“一个问题对应多个正确答案”或者“答案相关度有等级之分”的情况。它包含三个核心概念：CG (Cumulative Gain): 把前 $K$ 个结果的相关度直接相加。DCG (Discounted CG): 考虑到人的阅读习惯，排在后面的结果价值要“打折”（除以排名的对数）。NDCG (Normalized DCG): 归一化。用实际得分除以“理想情况下（最好的结果全排在最前面）”的得分，得到一个 0 到 1 之间的值。

前 $K$ 个结果的 DCG 定义为：

$$
\mathrm{DCG}_K = \sum_{i=1}^{K}\frac{\mathrm{rel}_i}{\log_2(i+1)}
$$

归一化 NDCG（理想排序下的最大 DCG 记为 $\mathrm{IDCG}_K$）为：

$$
\mathrm{NDCG}_K = \frac{\mathrm{DCG}_K}{\mathrm{IDCG}_K}
$$

其中 $\mathrm{rel}_i$ 是第 $i$ 个位置的相关度得分（可以是 0/1，也可以是 0-3 级）
等级2 (完美相关)：检索到的Chunk包含了回答查询所需的精确且完整的信息。

等级1 (部分相关)：检索到的Chunk提供了部分背景或侧面信息，但不足以直接回答问题。

等级0 (不相关)：与查询无关。