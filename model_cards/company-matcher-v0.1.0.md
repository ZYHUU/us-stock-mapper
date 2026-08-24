# Model Card: company-matcher-v0.1.0

- **发布日期**：2026-08-24
- **对应 GitHub Release**：`v0.1.0`（附件见 Release 页面：`model.joblib` / `manifest.json` / `SHA256SUMS`）
- **对应内部实验代号**：`classic-lr-v5`（见 `MODEL_CHANGELOG.md` 的版本对照表）
- **对应训练记录**：[`experiments/2026-08-24-frozen-test-and-privacy-fix/`](../experiments/2026-08-24-frozen-test-and-privacy-fix/)
- **状态**：`experimental`——Precision 未达到项目自定的 98% 长期目标（当前 0.938），发布方决定
  98% 是持续优化方向而不是这次发布的硬门槛，见下方"已知局限"。

## 这个模型做什么

输入一条社交媒体/新闻消息文本 + 一个候选公司（来自规则匹配召回），输出这个候选是否真的在说这家公司
（二分类：真实提及 / 通用词撞库等假阳性）。不是"从任意文本里识别股票"的通用分类器——上游已经用正则
做过一轮候选召回，这个模型只做消歧。

## 训练数据

不随代码/模型公开（原始消息含第三方社交媒体内容，涉及版权/隐私，见仓库根目录 README 的说明）。
`examples/synthetic_messages.jsonl` 提供了一批可复现的合成示例，覆盖已知的典型假阳性/真阳性模式。

训练集/验证集/测试集按时间切分，并做了近重复内容聚类去重（同一条转发/复制粘贴文案换个消息 id
不会同时出现在训练集和测试集里），细节见训练记录。测试集从 2026-08-24 起永久冻结，未来版本
沿用同一批消息 id 做比较，不再每次重新切分。

## 指标（冻结测试集）

| 指标 | 数值 |
| --- | --- |
| Precision | 0.938 |
| Recall | 0.968 |
| F1 | 0.953 |
| 阈值 | 0.6804 |

对照：候选生成阶段（规则+上游）单独当分类器的 baseline F1 约 0.42——这个模型相对纯规则候选
是明显的提升，但 Precision 还没到项目自定的 98% 目标线。

## 已知局限

- **Precision 未达 98% 长期目标**：0.938，意味着大约每16个模型判定为"真实提及"的候选里，
  有1个其实是通用词撞库等假阳性（`Strategy`/`NOW`/`Oracle`/`BB` 等，详见 README 里链接的
  假阳性模式记录）。98% 是持续优化方向，不是这次发布的门槛。
- **候选召回率本身有上限**：这个模型只对上游规则已经召回的候选做二次判断，规则层召回率本身
  （约 97.5%）会独立限制端到端整体正确率上限，这个模型没法把漏掉的候选找回来。
- **训练语料以英文/中文社交媒体与新闻为主**：其他语言、其他文体（比如财报原文、监管文件）
  上的表现没有专门验证过。
- **TF-IDF 词表包含训练语料里出现过的真实词/用户名/加密钱包地址片段**（这是词袋模型的固有特性，
  不是可修复的bug）：发布前做过隐私检查，排除了"整句中文被当成一个token"这种结构性泄漏
  （详见 `MODEL_CHANGELOG.md` classic-lr-v5 一节），但像 Twitter 用户名、加密钱包地址这类
  短 token 仍会出现在词表里——这些内容本身来自公开社交媒体帖子，不是私密通信。

## 加载方式

```python
import joblib

bundle = joblib.load("model.joblib")
# 具体字段见 src/semantic_matcher.py 里 SemanticMatcher 的加载逻辑
# bundle 的 key: word_vectorizer / char_vectorizer / classifier / use_structured / structured_feature_names
```

推荐阈值 **0.6804**（在验证集上选定，Precision=0.985，Recall=0.978；冻结测试集上的实际表现见上表）。
