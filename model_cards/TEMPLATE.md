# Model Card: company-matcher-vX.Y.Z

- **发布日期**：
- **对应 GitHub Release**：`vX.Y.Z`（附件见 Release 页面：`model.joblib` / `manifest.json` / `SHA256SUMS`）
- **对应内部实验代号**：`classic-lr-vN` 或 `lightgbm-shadow-vN`（见 `MODEL_CHANGELOG.md` 的版本对照表）
- **对应训练记录**：`experiments/<run_id>/`

## 这个模型做什么

输入一条社交媒体/新闻消息文本 + 一个候选公司（来自规则匹配召回），输出这个候选是否真的在说这家公司
（二分类：真实提及 / 通用词撞库等假阳性）。不是"从任意文本里识别股票"的通用分类器——上游已经用正则
做过一轮候选召回，这个模型只做消歧。

## 训练数据

不随代码/模型公开（原始消息含第三方社交媒体内容，涉及版权/隐私，见仓库根目录 README 的说明）。
`examples/synthetic_messages.jsonl` 提供了一批可复现的合成示例，覆盖已知的典型假阳性/真阳性模式。

## 指标（冻结测试集）

| 指标 | 数值 |
| --- | --- |
| Precision | |
| Recall | |
| F1 | |

## 已知局限

-

## 加载方式

```python
import joblib

bundle = joblib.load("model.joblib")
# 具体字段见 src/semantic_matcher.py 里 SemanticMatcher 的加载逻辑
```
