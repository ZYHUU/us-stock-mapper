# 影子模型分歧复核

WS 影子运行保持开启，不需要停止采集：

```powershell
python -m src.ws_client --shadow --shadow-lightgbm
```

另开一个终端启动分歧复核页面：

```powershell
python -m uvicorn src.shadow_review_api:app --port 8002
```

浏览器打开 `http://127.0.0.1:8002`。

队列只收录 LR 和 LightGBM 都已完成打分，并且至少一个模型与规则结果不同的消息。排序时优先展示“两套影子模型结论一致但与规则不同”的消息，其次展示两套模型互相不一致的消息；多候选消息会额外提高优先级。

页面会同时显示规则结果、两套模型的结果和候选分数。人工可以采用其中一个结果，也可以手工勾选正确公司或标记为不含目标公司。最终结果以 `shadow_review` 标注者追加写入现有 `annotations` 表；保存后消息自动从待复核队列消失，并可直接进入后续训练数据导出流程。

命令行汇总仍可使用：

```powershell
python -m src.shadow_report --limit 50
```
