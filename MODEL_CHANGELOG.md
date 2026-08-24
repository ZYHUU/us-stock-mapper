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
| （尚未发布） | `classic-lr-v3` | [`experiments/2026-08-21-candidate-count-fix`](experiments/2026-08-21-candidate-count-fix/) | 私有工作区的正式基线，还没打包成 Release |
| （尚未发布） | `lightgbm-shadow-v2` | [`experiments/2026-08-21-candidate-count-fix`](experiments/2026-08-21-candidate-count-fix/) | 私有工作区的影子模型，F1 更高但多候选延迟接近预算上限，还没打包成 Release |

---

<!-- 下次正式发布新版本时在上面表格加一行，并在下面补一节说明改了什么、为什么。 -->
