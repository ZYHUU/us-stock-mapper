# 模型版本记录

这份文件记录**面向外部的正式发布版本**（语义化版本号，`vX.Y.Z`，对应 GitHub Release）和
**内部训练/实验代号**（`classic-lr-vN` / `lightgbm-shadow-vN`，代码 `src/semantic_matcher.py`
里的 `MODEL_VERSION` 常量）之间的对应关系——这两套编号服务于不同目的，不要混用：

- **`vX.Y.Z`**：外部用户下载模型时看到的版本，遵循语义化版本：
  - `MAJOR`：输入/输出接口或候选体系发生不兼容变化
  - `MINOR`：换了模型/特征/新增训练数据后重新训练
  - `PATCH`：只是阈值校准、序列化格式修复等，不改变行为接口
- **`classic-lr-vN` / `lightgbm-shadow-vN`**：内部训练迭代/影子对照代号，`shadow_predictions`
  数据表用它区分"这条打分是哪个模型跑的"，跟外部发布节奏无关——不是每次内部迭代都会升级成
  一个正式的 `vX.Y.Z` 发布。

每次训练在私有工作区会自动生成 `models/company_classifier/<variant>/meta.json`（训练时间、
数据快照名、验证集指标、当时的 git commit）。只有确认值得让别人用的训练结果，才会打包发布成
`vX.Y.Z`，对应的完整训练记录放进这个仓库的 `experiments/<run_id>/`。

## 版本对照表

| 发布版本 | 内部代号 | 训练记录 | 状态 |
| --- | --- | --- | --- |
| [`v0.1.1`](../../releases/tag/v0.1.1) | `classic-lr-v6`（未变） | [`experiments/2026-09-01-crwd-candidate-recall-fix`](experiments/2026-09-01-crwd-candidate-recall-fix/) | **当前发布版本**，候选层（规则层）修复，分类模型本身未重训 |
| [`v0.1.0`](../../releases/tag/v0.1.0) | `classic-lr-v6` | [`experiments/2026-08-24-frozen-test-and-privacy-fix`](experiments/2026-08-24-frozen-test-and-privacy-fix/) | `experimental`（见 [Model Card](model_cards/company-matcher-v0.1.0.md)），上一个发布版本 |
| （从未发布） | `classic-lr-v5` | 见上面同一份训练记录 | 修了词表整句泄漏（第一轮隐私检查），但第二轮更彻底的复查发现词表仍残留真实用户名片段，修复后直接跳到 v6，v5 本身未发布 |
| （从未发布） | `classic-lr-v4` | 见上面同一份训练记录 | 修了 train/val/test 切分泄漏，但打包发布前的隐私检查发现词表整句泄漏问题，v4 本身未发布 |
| （从未发布） | `classic-lr-v3` | [`experiments/2026-08-21-candidate-count-fix`](experiments/2026-08-21-candidate-count-fix/) | 私有工作区曾经的正式基线，已被 v6 取代 |
| （不计划发布） | `lightgbm-shadow-v2` | [`experiments/2026-08-21-candidate-count-fix`](experiments/2026-08-21-candidate-count-fix/) | 影子模型，F1 曾经更高但多候选延迟接近预算上限，只在私有工作区跑影子对照，不打包发布 |
| （不计划发布） | `lightgbm-shadow-v3` | [`experiments/2026-08-24-frozen-test-and-privacy-fix`](experiments/2026-08-24-frozen-test-and-privacy-fix/) | 相对 v2 是权衡而非净提升（Precision 下降），未转正，继续跑影子观察 |

---

## v0.1.1（候选层修复，`classic-lr-v6` 分类模型未变）

**分类模型没有升级，仍然是 `classic-lr-v6`；这不是"整体股票识别全面提升"，
是一次经过验证的、范围明确的候选层（规则层）定向修复。**

- **改了什么**：`data/companies.csv` 新增一行 `CrowdStrike`（`NASDAQ:CRWD`）。
  修复前，公司库只覆盖了 securities 主数据里自动生成的公司全称
  （`CrowdStrike Holdings, Inc.`），没有注册裸公司名 `CrowdStrike`/
  `Crowdstrike`——真实文本里更常见的是不带公司后缀的裸名写法，导致这类消息
  在候选层就漏召回，分类器根本没有机会打分。这次只加了一行别名注册，不涉及
  任何分类模型改动。
- **定向测试结果**（`CRWD` 单项，独立盲标确认后按同一文本、修复前/后两个
  候选层快照重放对比，方法论详见训练记录）：

  | 指标 | 修复前 | 修复后 | 变化 |
  | --- | ---: | ---: | ---: |
  | Precision | 0.805 | 0.837 | +3.18pt |
  | Recall | 0.786 | 0.976 | +19.0pt |

- **上线观察**（小范围发布后的真实生产数据，详见训练记录）：累计 8,007 条
  自然消息，其中独立确认 11 条真实 CrowdStrike 相关内容；非 CrowdStrike
  文本误触发 `NASDAQ:CRWD` 的数量为 0；其他公司的候选识别结果相对修复前
  基线的差异数量为 0；生产进程全程无异常。五条预先锁定的验收条件全部
  满足后转为稳定发布。
- **新增测试**：`tests/test_mapper.py` 新增 3 个 CrowdStrike 候选层回归
  用例，修复前全部失败、修复后全部通过。
- **明确的边界（如实披露，不夸大范围）**：
  - 分类器**没有**升级——`classic-lr-v6` 原样保留，本次改动完全在候选层，
    不改变分类模型的输入/输出接口或行为。
  - 这次评估过程里，同一批候选层修复清单还包含 Nebius（`NBIS`）、Marvell
    （`MRVL`）、JPMorgan（`JPM`）、Alibaba（`BABA`）四项，但**均未进入本次
    发布**：Nebius 定向复核测得真实的精确率下降（裸"Nebius"这个词在非正式
    语境里比股票代码更容易被随意提及，误触发比修复前更多）；Marvell、
    Alibaba 召回率提升方向正确但统计证据不足；JPMorgan 在测试样本里基线
    召回率已经是 100%，没有可提升的空间。这四项继续留在内部改进清单，
    本次不发布，避免夹带未经验证或已知有代价的改动。
  - 不能把这次发布理解成"股票识别能力整体提升"——范围仅限于
    CrowdStrike 这一家公司的候选层召回修复。

---

## v0.1.0（`classic-lr-v6`）

第一个正式的公开发布版本，标记为 `experimental`（Precision 未达到项目自定的 98% 长期目标，
详见 [Model Card](model_cards/company-matcher-v0.1.0.md)）。

- **模型**：`tfidf_structured__balanced`（LogisticRegression + TF-IDF word/char n-gram + 12 个结构化特征，
  输入文本先经过 `src/text_sanitize.py` 脱敏）
- **测试集指标**（冻结测试集，阈值只在验证集上选，测试集只套用不调参）：

  | 变体 | 阈值 | Precision | Recall | F1 |
  | --- | ---: | ---: | ---: | ---: |
  | classic-lr-v3（上一版本，同一测试集重新打分） | 0.6555 | 0.868 | 0.982 | 0.921 |
  | classic-lr-v5（未发布，词表仍有用户名残留） | 0.6804 | 0.938 | 0.968 | 0.953 |
  | **classic-lr-v6（本次发布）** | 0.6396 | **0.907** | 0.975 | **0.940** |

- **改了什么**：详见训练记录 [`experiments/2026-08-24-frozen-test-and-privacy-fix`](experiments/2026-08-24-frozen-test-and-privacy-fix/)。
  简要说：(1) 修了训练/验证/测试集切分逻辑里的近重复内容泄漏问题；(2) 第一轮隐私检查发现词表里
  混入了近乎完整句子的中文原文片段，修了 TF-IDF 词分词器的正则（→v5）；(3) 第二轮更彻底的隐私
  复查发现词表仍残留真实 Twitter 用户名片段，加了统一的文本脱敏（`sanitize_message_text()`，
  训练和线上推理共用同一个函数）后重训（→v6）。
- **如实披露代价**：脱敏让 Precision 从未发布的 v5（0.938）降到实际发布的 v6（0.907）——模型
  原来部分依赖对特定 handle 字符串的死记硬背，脱敏后这部分信号被移除。即便如此，v6 相对上一个
  真正发布过的版本 v3 仍是明确提升。
- **发布前检查**：数据泄漏检查、两轮隐私检查、真实 serving smoke test、延迟测试、回滚验证均已
  通过，新增 `tests/test_text_sanitize.py` 锁定脱敏行为，详见训练记录。
- **已知局限**：Precision 未达 98%，见 Model Card。

---

<!-- 下次正式发布新版本时在上面表格加一行，并在下面补一节说明改了什么、为什么。 -->
