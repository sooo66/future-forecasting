# build_result 输出字段说明

`build_result()` 函数生成的每条预测结果包含以下字段：

## 必填字段（始终存在）

| 字段 | 类型 | 说明 |
|------|------|------|
| `market_id` | str | 市场唯一标识 |
| `domain` | str | 问题领域 |
| `difficulty` | str | 难度等级 |
| `predicted_prob` | float | 模型预测的 YES 概率 [0, 1] |
| `label` | int | 真实标签（0=NO, 1=YES） |
| `brier_score` | float | Brier 分数 (predicted_prob - label)² |
| `trajectory` | list[dict] | Agent 执行轨迹（工具调用、步骤等） |
| `reasoning_summary` | str | 模型输出的推理摘要 |
| `latency_sec` | float | 预测耗时（秒） |
| `total_tokens` | int \| None | 总 token 数 |

## 可选字段（有值时才会出现）

| 字段 | 类型 | 说明 |
|------|------|------|
| `steps_count` | int | Agent LLM 调用总次数 |
| `error` | str | 运行错误信息（仅失败时出现） |

---

## 输出示例

```json
{
  "market_id": "Will_AAPL_close_above_180_2024_01_15",
  "method_name": "agentic_nomem",
  "domain": "equity",
  "difficulty": "medium",
  "open_time": "2024-01-01T00:00:00Z",
  "resolve_time": "2024-01-15T21:00:00Z",
  "sample_time": "2024-01-10T00:00:00Z",
  "cutoff_time": "2024-01-10T00:00:00Z",
  "predicted_prob": 0.65,
  "label": 1,
  "accuracy": 1,
  "brier_score": 0.1225,
  "log_loss": 0.4308,
  "trajectory": [
    {"step": "assistant_before_tool_1", "content": "I need to check AAPL recent price movement..."},
    {"step": "tool_call_1", "tool_name": "openbb", "arguments": {"function": " equity.price.historical", "params": {"symbol": "AAPL"}}},
    {"step": "openbb_result_1", "tool_name": "openbb", "function": "equity.price.historical", "result_count": 30, "results_preview": [...]},
    {"step": "assistant", "content": "Based on the recent price trend..."},
    {"step": "final", "raw_response": "{\"predicted_prob\": 0.65, \"reasoning_summary\": \"AAPL has shown...\"}"}
  ],
  "reasoning_summary": "AAPL has shown bullish momentum with strong volume...",
  "latency_sec": 12.34,
  "prompt_tokens": 1500,
  "completion_tokens": 280,
  "total_tokens": 1780,
  "retrieved_source_types": ["news", "blog"],
  "steps_count": 3,
  "tool_usage_counts": {"openbb": 1, "search": 2}
}
```

---

## build_failed_result（错误情况）

当方法执行出错时，返回固定的默认结果：

```json
{
  "market_id": "...",
  "method_name": "...",
  "predicted_prob": 0.5,
  "label": ...,
  "accuracy": int(label == 1),
  "brier_score": 0.25,
  "log_loss": 0.6931471805599453,
  "trajectory": [{"step": "error", "message": "<error details>"}],
  "reasoning_summary": "fallback due to runtime error",
  "latency_sec": 0.0,
  "prompt_tokens": null,
  "completion_tokens": null,
  "total_tokens": null,
  "error": "<error details>"
}
```
