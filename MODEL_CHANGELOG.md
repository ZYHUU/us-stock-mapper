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
| [`v0.1.0`](../../releases/tag/v0.1.0) | `classic-lr-v5` | [`experiments/2026-08-24-frozen-test-and-privacy-fix`](experiments/2026-08-24-frozen-test-and-privacy-fix/) | **当前发布版本**，`experimental`（见 [Model Card](model_cards/company-matcher-v0.1.0.md)），正式线上基线 |
| （从未发布） | `classic-lr-v4` | 见上面同一份训练记录 | 打包发布前的隐私检查发现词表泄漏问题，修复后直接跳到 v5，v4 本身未发布 |
| （从未发布） | `classic-lr-v3` | [`experiments/2026-08-21-candidate-count-fix`](experiments/2026-08-21-candidate-count-fix/) | 私有工作区曾经的正式基线，已被 v5 取代 |
| （不计划发布） | `lightgbm-shadow-v2` | [`experiments/2026-08-21-candidate-count-fix`](experiments/2026-08-21-candidate-count-fix/) | 影子模型，F1 曾经更高但多候选延迟接近预算上限，只在私有工作区跑影子对照，不打包发布 |
| （不计划发布） | `lightgbm-shadow-v3` | [`experiments/2026-08-24-frozen-test-and-privacy-fix`](experiments/2026-08-24-frozen-test-and-privacy-fix/) | 相对 v2 是权衡而非净提升（Precision 下降），未转正，继续跑影子观察 |

---

## v0.1.0（`classic-lr-v5`）

第一个正式的公开发布版本，标记为 `experimental`（Precision 未达到项目自定的 98% 长期目标，
详见 [Model Card](model_cards/company-matcher-v0.1.0.md)）。

- **模型**：`tfidf_structured__balanced`（LogisticRegression + TF-IDF word/char n-gram + 12 个结构化特征）
- **测试集指标**（冻结测试集，阈值只在验证集上选，测试集只套用不调参）：

  | 变体 | 阈值 | Precision | Recall | F1 |
  | --- | ---: | ---: | ---: | ---: |
  | classic-lr-v3（上一版本，同一测试集重新打分） | 0.6555 | 0.868 | 0.982 | 0.921 |
  | **classic-lr-v5（本次发布）** | 0.6804 | **0.938** | 0.968 | **0.953** |

- **改了什么**：详见训练记录 [`experiments/2026-08-24-frozen-test-and-privacy-fix`](experiments/2026-08-24-frozen-test-and-privacy-fix/)。
  简要说：(1) 修了训练/验证/测试集切分逻辑里的近重复内容泄漏问题；(2) 打包发布前的隐私检查
  发现词表里混入了近乎完整句子的中文原文片段，修了 TF-IDF 词分词器的正则、重新训练。
- **发布前检查**：真实 serving smoke test、延迟测试、回滚验证均已通过，详见训练记录。
- **已知局限**：Precision 未达 98%，见 Model Card。

---

<!-- 下次正式发布新版本时在上面表格加一行，并在下面补一节说明改了什么、为什么。 -->
