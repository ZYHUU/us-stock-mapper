# 2026-08-21: candidate_count 训练/线上特征偏差修复

## 背景

`candidates_in_message`（这条消息一共有几个候选公司）是结构化特征之一，本意是给模型一个
"消息本身有多歧义"的信号。但训练脚本和线上推理各自算出来的值不是一回事：

- **训练时**：直接数 `candidate_pairs.jsonl` 里同一条消息对应的候选行数——这个数字已经被
  `random_topup`/`background` 负样本填充策略拉高了，同一条消息经常被人为拆成 3-4 个候选行。
- **线上时**：数 `mapper.identify()` 真实召回的候选个数——大多数消息只有 1 个候选。

结果是模型在训练阶段学到的"候选数量"分布和线上完全不一致（典型的 train/serve skew）。LightGBM
这种树模型对这类泄漏特征比 LogisticRegression 更敏感（分裂点更锐利），offline 指标好看，
线上却明显失真——这个 bug 是通过 smoke test（拿一条明确该命中的合成消息，走真实的
`SemanticMatcher.score()` 推理路径跑一遍，而不是只看 batch 评估脚本的数字）发现的，offline
指标再好看也不能替代这一步。

## 怎么修的

`candidates_in_message` 的训练时取值改为直接使用 `build_training_dataset.py` 在负样本填充**之前**
记录下来的 `real_candidate_count` 字段，和线上"规则 + 上游候选的真实并集大小"算法保持一致。

## 结果

在同一个数据快照上重新训练两个分类器，并在冻结测试集上重新评估（阈值全部来自验证集搜索，
测试集只套用、不调参）：

| 模型 | 阈值 | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| classic-lr-v3 | 0.6555 | 0.873 | 0.982 | 0.924 |
| lightgbm-shadow-v2 | 0.5606 | 0.937 | 0.982 | 0.959 |
| 规则基线 | — | — | — | 0.427 |

## 结论 / 后续

- LightGBM 修复后测试集 F1 依然更高（0.959 vs 0.924），但多候选消息的 p99 延迟接近/可能超出
  100ms 预算，暂不接管正式输出，只在 `ws_client.py --shadow-lightgbm` 影子运行里跟规则/LR
  做分歧对照，持续积累人工复核样本。
- classic-lr-v3 的 Precision（0.873）还没到 98% 的上线目标，先作为正式基线继续用，同时靠
  影子分歧复核积累修正样本（`src/shadow_review.py`）。

## 关于这条记录

这次训练发生在 `experiments/`/`run.json` 这套记录机制建立之前，`run.json` 里的数字是事后从
`reports/final_test_evaluation.md`（私有工作区内）和代码注释里转录的，不是训练时实时写入的——
放进来当第一条 `experiments/` 记录，正好说明为什么要建这套机制：没有它，这类信息只能靠事后
翻代码注释和报告文件拼凑，容易丢失训练当时的上下文（比如具体用了哪个数据快照，只能靠时间戳
反推）。
