# 🧬 Self-Evolving Skill Engine

自我进化技能引擎 — 让 Agent 踩过的坑变成下次不踩的规则，但安全边界永远不让 LLM 改。

## 核心思想

```
固定核心规则（只读） + 动态错误附录（可写）
```

- **固定核心规则** — 安全边界、物理约束、前置条件，LLM 碰不了
- **动态错误附录** — 每次出错记录：什么条件下触发、怎么修

## 架构

```
LLM 生成技能草案
       ↓
  符号验证器校验（纯代码，不走 LLM）
       ↓
  执行时出错 → 确定性规则归类（4 种失败类型）
       ↓
  精准修复 → 写入动态错误附录
       ↓
  命中 N 次 → 晋升提案（附录 → 核心规则，需人工审核）
```

## 4 种失败类型

| 类型 | 说明 | 修复策略 |
|---|---|---|
| 领域规则违反 | 触发符号验证拦截 | 回溯约束，生成 safe_fallback |
| 感知输出提取错误 | 多模态对齐失败 | 交互式探测消除不确定性 |
| 上下文误解/幻觉 | 实体识别错误 | 环境接地检查 |
| 不完整执行 | 超时/断电 | 断点续传 |

## 使用

### 环境变量

```bash
# MIMO / OpenAI 兼容接口
export LLM_ENDPOINT="https://token-plan-cn.xiaomimimo.com/v1"
export LLM_API_KEY="your-api-key"
export LLM_MODEL="xiaomi/mimo-v2.5-pro"

# 数据持久化目录（可选，默认 ./data）
export SKILL_DATA_DIR="./data"
```

### 运行

```bash
python self_evolving_skill.py
```

输出示例：

```
✅ 技能草案 [select_swim_gift_for_7yo_girl] 生成成功
🛡️ 核心规则校验通过
🔧 MIMO 生成修复方案: 针对尺码表OCR失败，使用备用尺寸数据库查询...
💾 修复经验已写入动态附录 (ID: aa51b5)
💾 技能已持久化: skill_a1b2c3d4.json
🚀 触发晋升提案: 附录命中 5 次 → 建议固化为核心规则，请人工审核！
```

### 持久化

所有技能和附录自动保存到 `data/` 目录：
- `data/skill_<id>.json` — 每个技能的完整状态（含动态附录）
- `data/appendix_store.json` — 全局附录向量库

重启后自动加载，数据不丢失。

## 为什么不用 LLM 判断对错？

LLM 自己生成、自己验证 = 不可靠的闭环。

这个引擎的每一步都是确定性可验证的：
- 校验用代码，不用 LLM
- 归类用规则映射，不用 LLM
- 修复由 LLM（MIMO）执行，但归因不由 LLM 判断

LLM 只在阶段4参与：根据确定的失败类型和上下文生成修复方案。归类这一步永远是代码决定。

## 配置

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_ENDPOINT` | `https://token-plan-cn.xiaomimimo.com/v1` | LLM API 地址 |
| `LLM_API_KEY` | - | API Key |
| `LLM_MODEL` | `xiaomi/mimo-v2.5-pro` | 模型名称 |
| `SKILL_DATA_DIR` | `./data` | 持久化目录 |

## License

MIT
