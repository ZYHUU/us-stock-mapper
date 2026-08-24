# 美股/港股/A股 公司识别器

输入一条中文或英文消息，识别其中提到的股票公司，并输出带交易所的标准股票代码——不是让模型自由生成代码，而是从受控的公司主数据库里做匹配，所以不会凭空编出不存在的股票代码。

架构是两层：
1. **规则层**（`src/mapper.py`）：公司全称、简称、品牌、股票代码的词典匹配，不需要模型就能跑。
2. **消歧层**（`src/semantic_matcher.py`，可选）：规则层召回的候选词很多是通用词撞库（比如英文单词 `now` 撞上 ServiceNow 的股票代码 `NOW`），这一层用训练好的分类器判断"这个候选是不是真的在说这家公司"。规则层单独也能跑，消歧层是在此基础上的精度提升。

完整的数据准备、模型训练和上线步骤见 [训练与上线规划](docs/TRAINING_PLAN.md)；模型迭代记录见 [MODEL_CHANGELOG.md](MODEL_CHANGELOG.md) 和 [experiments/](experiments/)。

## 这个仓库包含什么、不包含什么

这是这个项目的**公开仓库**，只放代码、文档和合成示例：

- ✅ 全部推理/训练/评估/标注工具代码
- ✅ 公司主数据字典 `data/companies.csv`、公开市场数据快照 `data/binance_tradfi_instruments.csv`
- ✅ 合成示例消息 `examples/synthetic_messages.jsonl`（自己编的，不是真实抓取内容）
- ✅ 训练记录 `experiments/`、模型版本对照 `MODEL_CHANGELOG.md`
- ❌ **不包含训练用的原始消息语料**——这些是抓取自社交媒体的第三方内容，涉及版权和隐私，不会公开分发。想跑通训练脚本，需要你自己的消息数据源。
- ❌ **不包含训练好的模型二进制文件**——通过 GitHub Release 发布（见下方"下载训练好的模型"），不进 git 历史。
- ❌ **不包含 WS 数据源的具体连接地址**——那是我自己接的一个上游数据源，属于私有信息，见下方"接入你自己的消息源"。

也就是说，**规则层**（词典匹配）克隆下来就能跑；**消歧层**（模型）需要下载已发布的模型权重，或者接入你自己的消息源自己训练。

## 运行环境

- Python 3.10 或更高版本
- 所有命令都在本项目目录执行

```powershell
python -m pip install -r requirements.txt
copy .env.example .env
```

## 快速开始：只用规则层

不需要任何模型、任何账号，克隆下来就能跑：

```powershell
python -m src.cli
```

Windows 上如果 `python` 不可用，也可以尝试 `py -m src.cli`。

示例输入：

```text
微软与英伟达扩大 AI 合作
```

示例输出：

```json
{
  "status": "matched",
  "companies": [
    {
      "company_id": "microsoft",
      "company_name": "微软",
      "canonical_code": "NASDAQ:MSFT",
      "mention": "微软",
      "match_type": "alias",
      "confidence": 0.98
    },
    {
      "company_id": "nvidia",
      "company_name": "英伟达",
      "canonical_code": "NASDAQ:NVDA",
      "mention": "英伟达",
      "match_type": "alias",
      "confidence": 0.98
    }
  ]
}
```

想跑更完整的公司范围（含通过币安 TradFi 数据同步进来的美股/港股/A股全集，而不只是 `companies.csv` 里手工维护的那一小批），先跑一次公开数据同步（不需要任何密钥，读的是币安和美国 SEC 的公开接口）：

```powershell
python -m src.sync_binance_tradfi
```

这会在本地建一个 `data/stock_mapper.db`（只有 `securities`/`platform_instruments` 两张主数据表，不含任何消息数据），`src/mapper.py` 的 `default_mapper()` 会自动合并这个数据库和 `data/companies.csv`。

## 用合成示例测试消歧层（可选，需要模型）

`examples/synthetic_messages.jsonl` 里是一批自己编的示例，覆盖了真实数据里反复出现的假阳性/真阳性模式（通用词撞库、handle 子串误撞、品牌真实提及等，每条都写了 `note` 说明）。下载模型权重后（见下方），可以拿它验证消歧层的行为：

```python
import json
from src.mapper import default_mapper
from src.semantic_matcher import default_lr_matcher

mapper = default_mapper()
matcher = default_lr_matcher()
code_to_company = {c.canonical_code: c for c in mapper.companies}

with open("examples/synthetic_messages.jsonl", encoding="utf-8") as f:
    for line in f:
        example = json.loads(line)
        rule_matches = mapper.identify(example["text"])
        scored = matcher.score(example["text"], "twitter", rule_matches, code_to_company)
        model_codes = [s.canonical_code for s in scored if s.model_predicted]
        print(example["id"], "expected=", example["expected_codes"], "model=", model_codes)
```

## 下载训练好的模型

模型二进制文件不进 git 历史，通过 [GitHub Release](../../releases) 发布。版本号规则和当前版本状态见 [MODEL_CHANGELOG.md](MODEL_CHANGELOG.md)。下载后放进 `models/company_classifier/<variant>/model.joblib`（跟 `src/semantic_matcher.py` 里 `LR_MODEL_PATH`/`LIGHTGBM_MODEL_PATH` 指向的路径对应），并核对 Release 附件里的 `SHA256SUMS`。

`model.joblib` 是 `joblib`/`pickle` 序列化格式，反序列化时会执行任意代码——只加载官方 Release 发布的文件，不要加载来源不明的 `.joblib`。

## 运行自动测试

```powershell
python -m unittest discover -s tests -v
```

## 启动 HTTP API

```powershell
python -m uvicorn src.api:app --reload
```

浏览器访问 `http://127.0.0.1:8000/docs`，可以直接测试接口。请求格式：

```json
{
  "message": "特斯拉与英伟达宣布合作"
}
```

## 添加公司和别名

编辑 `data/companies.csv`。其中：

- `exchange` 使用 `NASDAQ` 或 `NYSE`
- `aliases` 是公司全称、简称和常见称呼，使用 `|` 分隔
- `brands` 是能明确指向公司的产品或品牌，使用 `|` 分隔
- `negative_contexts` 是用于排除歧义的词，例如苹果公司的"水果"和"果农"
- 不要随便加入过于宽泛的词，否则容易误报

例如，不能直接把"苹果"始终视为苹果公司。规则层通过 `negative_contexts` 排除一部分明显的水果语境；更复杂的歧义交给消歧层的模型判断（参见 `examples/synthetic_messages.jsonl` 里的假阳性案例）。

## 接入你自己的消息源

`src/message_parser.py` 可以处理 Twitter 类事件，从 `content_cn`/`content`/`title`/`new_full_text_cn`/`new_full_text`/`quo_text_cn`/`quo_text` 等字段抽取文本，相同内容只保留一次，空字符串/`null`/`{}` 会被忽略；头像、用户名、图片地址等字段不参与公司识别。

```python
from src.mapper import default_mapper
from src.message_parser import identify_event

result = identify_event(ws_event, default_mapper())
```

`src/ws_client.py` 是一个通用的 WebSocket 采集客户端，但**不含任何具体数据源的连接信息**——地址、鉴权信息全部通过环境变量（`.env`，已在 `.gitignore` 里）传入：

```powershell
$env:WS_URL = "wss://你自己的消息源地址"
python -m src.ws_client
```

不设置 `WS_URL` 时会用一个明显的占位地址，直接连接会失败并提示配置。消息、预测和原始事件会保存到本地 SQLite 文件 `data/stock_mapper.db`。`source_id` 或事件哈希具有唯一约束，程序重启后仍能自动去重。按 `Ctrl+C` 可以安全停止监听，连接意外中断时程序会自动重连。

如果连接需要鉴权：

```powershell
$env:WS_AUTHORIZATION = "Bearer 你的令牌"
$env:WS_COOKIE = "你的Cookie"
```

如果服务要求指定来源或在连接后发送订阅消息：

```powershell
python -m src.ws_client --origin "https://你的前端域名" --subscribe '{"type":"subscribe"}'
```

如果还希望额外保留 JSONL 备份：

```powershell
python -m src.ws_client --jsonl-backup data/ws_events_new.jsonl
```

## 人工标注

采集到自己的消息后，启动本地标注页面：

```powershell
python -m uvicorn src.annotation_api:app --port 8001
```

浏览器打开 `http://127.0.0.1:8001`。页面支持：

- 按 `1` 接受当前预测
- 按 `0` 标记为不含当前公司库中的公司
- 勾选一家或多家公司后按 `Enter` 保存
- 按 `S` 暂时跳过

页面每次只从 SQLite 读取最多 100 条未标注消息。标注以追加方式写入 `data/stock_mapper.db`，同一条消息重新标注时以最新版本为准。每条标签还会保存标注时的公司代码范围、标注来源和置信度。

## 影子模型分歧复核

如果同时跑规则和一到两个模型（`--shadow`/`--shadow-lightgbm`），可以用 `src/shadow_review.py` / `src/shadow_review_api.py` 拉出"模型和规则判断不一致"的消息做人工复核，复核结果直接写回标注表，用于后续重训。方法论和实战案例见 [docs/SHADOW_REVIEW.md](docs/SHADOW_REVIEW.md)。

## 训练自己的模型

需要你自己积累的标注数据（见上面"人工标注"）。完整流程见 [训练与上线规划](docs/TRAINING_PLAN.md)，简要步骤：

```powershell
python -m src.create_data_snapshot --name my-snapshot
python -m src.build_training_dataset --snapshot-name my-snapshot
python -m src.train_classic_model --snapshot-name my-snapshot --model-version classic-lr-v6
python -m src.train_lightgbm_model --snapshot-name my-snapshot --model-version lightgbm-shadow-v4
```

每次训练会在 `models/company_classifier/<variant>/meta.json` 里自动记录训练时间、数据快照名、验证集指标和当时的 git commit。确认某次训练值得发布后，把训练记录整理进 `experiments/<run_id>/`，模型权重打包发布成 GitHub Release，并更新 `MODEL_CHANGELOG.md`。

## 同步币安 TradFi 主数据

币安 Futures 的 TradFi 分类同时包含个股、ETF、商品和 Pre-IPO 标的，不能全部当作股票训练数据：

```powershell
python -m src.sync_binance_tradfi
```

命令从币安官方 Futures 接口读取当前合约，并使用美国 SEC 公司代码表核对美股名称、交易所和标准代码。结果写入 SQLite 的 `securities`、`platform_instruments` 两张表，同时导出 `data/binance_tradfi_instruments.csv` 供人工检查。

`mapper_candidate=1` 表示属于当前目标市场（美股、A股或港股）的上市公司，可以进入后续公司别名整理；ETF、商品、韩国股票和未确认标的默认不会进入公司识别器。

识别器启动时会自动合并 SQLite 中 `mapper_candidate=1` 的公司和 `companies.csv`：SQLite 提供完整公司范围与标准代码，CSV 中的人工别名、品牌和排除语境优先保留。因此新增或更新币安公司时不需要把整张主表复制进 CSV；只有需要补充简称或处理歧义时才编辑 CSV。

## License

[MIT](LICENSE)
