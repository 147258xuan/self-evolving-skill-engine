# Self-Evolving Skill Engine

一个轻量级、可持久化的「自我进化技能引擎」原型，基于「固定核心 + 动态附录」混合架构，接入 MIMO (小米大模型) 实现真实的 LLM 驱动修复。

## 核心理念

> 既不让 AI 完全放飞（纯开放式进化容易失控），也不完全固化（纯符号系统无法应对未知错误）。
> **固定核心保证可预测，动态附录实现经验复用。**

## 架构：固定核心 + 动态附录

```
┌─────────────────────────────────────────────┐
│              Skill (技能手册)                │
├─────────────────────────────────────────────┤
│  📌 Fixed Core Rules (固定核心规则)          │
│     - 不可被运行时修改                       │
│     - 符号验证器强制执行                     │
│     - 例: budget_limit, safety_color_ban     │
├─────────────────────────────────────────────┤
│  📝 Dynamic Error Appendix (动态错误附录)    │
│     - 从失败经验中自动提取                   │
│     - 通过触发条件复用                       │
│     - 达到阈值后可晋升为固定规则             │
└─────────────────────────────────────────────┘
```

## 运行流程

```
1. 生成 (Generate)     → LLM 生成技能草案
2. 校验 (Validate)     → 符号验证器检查核心规则
3. 归类 (Classify)     → 确定性规则映射失败类型
4. 修复 (Repair)       → MIMO 生成修复方案 → 写入附录
5. 晋升 (Promote)      → 命中阈值 → 固化为核心规则
```

## v2.0 新增功能

### 🔄 回滚机制 (Rollback)

- 每次修复前自动拍摄技能状态快照
- 修复后检测回归：如果错误数显著增加（>1.5x），自动回滚到上一个快照
- 冷启动保护：错误数 < 3 时不触发回归检测，避免误判

### 📦 失败记录保留 (Attempt History)

- **所有**修复尝试都记录在案，无论成功、失败还是回滚
- 每条记录包含：尝试时间、失败类型、修复方案、修复前后错误数、最终结果
- 存储在 `data/fix_history.json`，可追溯可分析

### 🛡️ 评估逻辑隔离 (Rules Guard)

- `classify_failure` 的确定性规则从外部 JSON 文件加载
- 启动时计算 SHA256 校验和，运行时每次调用前验证完整性
- 如果规则文件被篡改（包括被 LLM 意外修改），立即拒绝执行并报警
- 规则文件：`data/classification_rules.json`

## 快速开始

```bash
# 设置环境变量
export LLM_API_KEY="your-mimo-api-key"
export LLM_ENDPOINT="https://token-plan-cn.xiaomimimo.com/v1"  # 可选
export LLM_MODEL="xiaomi/mimo-v2.5-pro"                        # 可选

# 运行演示
python self_evolving_skill.py
```

## 数据文件说明

运行后会在 `data/` 目录生成：

| 文件 | 说明 |
|------|------|
| `skill_*.json` | 技能数据（含核心规则 + 动态附录） |
| `appendix_store.json` | 附录向量存储 |
| `fix_history.json` | 所有修复尝试记录 |
| `classification_rules.json` | 受保护的分类规则 |
| `snapshots_*.json` | 技能状态快照 |

## 与其他自进化系统对比

| 特性 | 本项目 | Meta HyperAgents | evolve (DGM-H) |
|------|--------|-----------------|-----------------|
| 语言 | Python | Python | TypeScript |
| 核心机制 | 固定核心 + 动态附录 | Archive + Staged Eval | Darwin Gödel Machine |
| LLM 驱动 | ✅ MIMO | ✅ 多模型 | ✅ Claude |
| 回滚机制 | ✅ | ❌ | ❌ |
| 评估隔离 | ✅ 校验和保护 | ❌ | ✅ 隐藏评估代码 |
| 失败记录 | ✅ 全量保留 | ✅ Archive | ✅ 所有变体保留 |
| 持久化 | ✅ JSON 文件 | ❌ 内存 | ✅ 文件系统 |

## 许可证

MIT
