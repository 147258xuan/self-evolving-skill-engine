import sys
import os
import json
import csv
import uuid
import hashlib
import copy
import requests
from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

# Windows 编码修复
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# ==========================================
# 0. LLM 配置 (MIMO / OpenAI 兼容)
# ==========================================

LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "https://token-plan-cn.xiaomimimo.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")  # 必须通过环境变量设置，不要硬编码
LLM_MODEL = os.getenv("LLM_MODEL", "xiaomi/mimo-v2.5-pro")

DATA_DIR = Path(os.getenv("SKILL_DATA_DIR", Path(__file__).parent / "data"))


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """调用 MIMO (OpenAI 兼容接口) 获取 LLM 回复"""
    resp = requests.post(
        f"{LLM_ENDPOINT}/chat/completions",
        headers={
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 1024,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ==========================================
# 1. 技能数据结构定义 (固定核心 + 动态附录)
# ==========================================


class FailureType(Enum):
    """四类失败精准归类枚举"""
    RULE_VIOLATION = "领域规则违反"
    PERCEPTION_ERROR = "感知输出提取错误"
    CONTEXT_HALLUCINATION = "上下文误解/幻觉"
    INCOMPLETE_EXECUTION = "不完整执行"


@dataclass
class ErrorAppendixEntry:
    """动态错误附录条目"""
    entry_id: str
    failure_type: FailureType
    trigger_condition: str
    fix_patch: str
    hit_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        d = asdict(self)
        d["failure_type"] = self.failure_type.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ErrorAppendixEntry":
        d["failure_type"] = FailureType(d["failure_type"])
        return cls(**d)


# ==========================================
# [新增] 尝试记录：保留所有修复尝试（成功+失败）
# ==========================================


class AttemptOutcome(Enum):
    SUCCESS = "成功"
    FAILED = "失败"
    REGRESSION = "回归退化"
    ROLLBACK = "已回滚"


@dataclass
class FixAttempt:
    """每次修复尝试的完整记录，无论成功失败"""
    attempt_id: str
    entry_id: str  # 关联的附录条目
    skill_id: str
    failure_type: str
    trigger_condition: str
    fix_patch: str
    outcome: AttemptOutcome
    error_before: int  # 修复前错误数
    error_after: int   # 修复后错误数
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        d = asdict(self)
        d["outcome"] = self.outcome.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FixAttempt":
        d["outcome"] = AttemptOutcome(d["outcome"])
        return cls(**d)


# ==========================================
# [新增] 版本快照：用于回滚机制
# ==========================================


@dataclass
class SkillSnapshot:
    """技能在某次修复前的完整状态快照"""
    snapshot_id: str
    skill_id: str
    skill_state: dict  # Skill.to_dict() 的深拷贝
    error_count_at_snapshot: int
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Skill:
    """可复用技能手册结构"""
    skill_id: str
    name: str
    version: str
    fixed_core_rules: Dict[str, Any]
    dynamic_error_appendix: List[ErrorAppendixEntry] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    error_count: int = 0  # [新增] 累计错误计数，用于回归检测

    def to_dict(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "version": self.version,
            "fixed_core_rules": self.fixed_core_rules,
            "dynamic_error_appendix": [e.to_dict() for e in self.dynamic_error_appendix],
            "steps": self.steps,
            "error_count": self.error_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Skill":
        return cls(
            skill_id=d["skill_id"],
            name=d["name"],
            version=d["version"],
            fixed_core_rules=d["fixed_core_rules"],
            dynamic_error_appendix=[ErrorAppendixEntry.from_dict(e) for e in d.get("dynamic_error_appendix", [])],
            steps=d.get("steps", []),
            error_count=d.get("error_count", 0),
        )


# ==========================================
# [新增] 评估逻辑隔离：校验和保护 classify_failure 规则
# ==========================================


class ClassificationRulesGuard:
    """
    保护 classify_failure 的确定性规则不被 LLM 篡改。
    规则从外部 JSON 文件加载，启动时计算校验和。
    每次使用前验证校验和，确保规则未被篡改。
    """

    def __init__(self, rules_path: Path):
        self.rules_path = rules_path
        self.rules: Dict[str, Any] = {}
        self._checksum: str = ""
        self._load_and_seal()

    def _load_and_seal(self):
        """加载规则并计算校验和，从此锁定"""
        if not self.rules_path.exists():
            # 首次运行：创建默认规则文件
            self._create_default_rules()

        raw = self.rules_path.read_bytes()
        self._checksum = hashlib.sha256(raw).hexdigest()
        self.rules = json.loads(raw)
        print(f"🔒 分类规则已加载并锁定 (SHA256: {self._checksum[:16]}...)")

    def _create_default_rules(self):
        """创建默认的分类规则文件"""
        default_rules = {
            "_comment": "classify_failure 的确定性映射规则。此文件受校验和保护，不可被运行时修改。",
            "code_mapping": {
                "SAFETY_BOUNDARY_EXCEEDED": "RULE_VIOLATION",
                "KINEMATIC_LIMIT": "RULE_VIOLATION",
                "SENSOR_NOISE": "PERCEPTION_ERROR",
                "ENTITY_NOT_FOUND": "CONTEXT_HALLUCINATION",
                "TIMEOUT": "INCOMPLETE_EXECUTION",
                "POWER_LOSS": "INCOMPLETE_EXECUTION"
            },
            "message_patterns": {
                "OCR_FAIL": "PERCEPTION_ERROR",
                "HALLUCINATION_DETECTED": "CONTEXT_HALLUCINATION"
            },
            "default_fallback": "CONTEXT_HALLUCINATION"
        }
        self.rules_path.parent.mkdir(parents=True, exist_ok=True)
        self.rules_path.write_text(
            json.dumps(default_rules, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"📝 首次运行，已创建默认规则文件: {self.rules_path}")

    def verify_integrity(self) -> bool:
        """验证规则文件未被篡改"""
        current = self.rules_path.read_bytes()
        current_hash = hashlib.sha256(current).hexdigest()
        if current_hash != self._checksum:
            print(f"🚨 警告: 分类规则文件被篡改！")
            print(f"   期望: {self._checksum[:16]}...")
            print(f"   实际: {current_hash[:16]}...")
            return False
        return True

    def classify(self, error_signal: dict) -> FailureType:
        """使用受保护的规则进行分类（每次调用前自动校验）"""
        if not self.verify_integrity():
            raise SecurityError("分类规则文件被篡改，拒绝执行分类！请恢复原始规则文件。")

        err_code = error_signal.get("code", "")
        err_msg = error_signal.get("message", "")

        # 1. 按 err_code 查表
        code_map = self.rules.get("code_mapping", {})
        if err_code in code_map:
            return FailureType[code_map[err_code]]

        # 2. 按 err_msg 模式匹配
        msg_patterns = self.rules.get("message_patterns", {})
        for pattern, type_name in msg_patterns.items():
            if pattern in err_msg:
                return FailureType[type_name]

        # 3. 降级默认
        fallback_name = self.rules.get("default_fallback", "CONTEXT_HALLUCINATION")
        print(f"⚠️ 未知错误类型 (code={err_code})，降级为 {fallback_name}")
        return FailureType[fallback_name]


class SecurityError(Exception):
    """评估规则被篡改时抛出"""
    pass


# ==========================================
# 2. 核心引擎：生成、校验、归类、修复 + 持久化 + 回滚
# ==========================================


class SelfEvolvingSkillEngine:
    # 回归检测阈值：修复后错误数超过此倍数则触发回滚
    REGRESSION_MULTIPLIER = 1.5
    # 最少需要多少次错误才能触发回归检测（避免冷启动误判）
    MIN_ERRORS_FOR_REGRESSION = 1

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.skill_registry: Dict[str, Skill] = {}
        self.appendix_vector_store: List[ErrorAppendixEntry] = []

        # [新增] 尝试历史：保留所有修复记录
        self.fix_history: List[FixAttempt] = []

        # [新增] 快照存储：skill_id -> [snapshots]
        self._snapshots: Dict[str, List[SkillSnapshot]] = {}

        # [新增] 评估逻辑隔离：加载受保护的分类规则
        self.rules_guard = ClassificationRulesGuard(self.data_dir / "classification_rules.json")

        # 启动时自动加载已有数据
        self._load_all()

    # ---------- 持久化 ----------

    def _skill_path(self, skill_id: str) -> Path:
        return self.data_dir / f"skill_{skill_id}.json"

    def _appendix_path(self) -> Path:
        return self.data_dir / "appendix_store.json"

    def _fix_history_path(self) -> Path:
        return self.data_dir / "fix_history.json"

    def _snapshots_path(self, skill_id: str) -> Path:
        return self.data_dir / f"snapshots_{skill_id}.json"

    def _save_skill(self, skill: Skill):
        path = self._skill_path(skill.skill_id)
        path.write_text(json.dumps(skill.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"💾 技能已持久化: {path.name}")

    def _save_appendix_store(self):
        path = self._appendix_path()
        data = [e.to_dict() for e in self.appendix_vector_store]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _save_fix_history(self):
        path = self._fix_history_path()
        data = [a.to_dict() for a in self.fix_history]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _save_snapshots(self, skill_id: str):
        path = self._snapshots_path(skill_id)
        snapshots = self._snapshots.get(skill_id, [])
        data = [
            {
                "snapshot_id": s.snapshot_id,
                "skill_id": s.skill_id,
                "skill_state": s.skill_state,
                "error_count_at_snapshot": s.error_count_at_snapshot,
                "created_at": s.created_at,
            }
            for s in snapshots
        ]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_all(self):
        """从 data_dir 加载所有技能、附录、尝试历史和快照"""
        for f in self.data_dir.glob("skill_*.json"):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                skill = Skill.from_dict(d)
                self.skill_registry[skill.skill_id] = skill
                print(f"📂 加载技能: {skill.name} (v{skill.version})")
            except Exception as e:
                print(f"⚠️ 加载 {f.name} 失败: {e}")

        appendix_file = self._appendix_path()
        if appendix_file.exists():
            try:
                data = json.loads(appendix_file.read_text(encoding="utf-8"))
                self.appendix_vector_store = [ErrorAppendixEntry.from_dict(d) for d in data]
                print(f"📂 加载附录: {len(self.appendix_vector_store)} 条记录")
            except Exception as e:
                print(f"⚠️ 加载附录失败: {e}")

        # [新增] 加载尝试历史
        history_file = self._fix_history_path()
        if history_file.exists():
            try:
                data = json.loads(history_file.read_text(encoding="utf-8"))
                self.fix_history = [FixAttempt.from_dict(d) for d in data]
                print(f"📂 加载修复历史: {len(self.fix_history)} 条记录")
            except Exception as e:
                print(f"⚠️ 加载修复历史失败: {e}")

        # [新增] 加载快照
        for f in self.data_dir.glob("snapshots_*.json"):
            try:
                skill_id = f.stem.replace("snapshots_", "")
                data = json.loads(f.read_text(encoding="utf-8"))
                self._snapshots[skill_id] = [
                    SkillSnapshot(
                        snapshot_id=s["snapshot_id"],
                        skill_id=s["skill_id"],
                        skill_state=s["skill_state"],
                        error_count_at_snapshot=s["error_count_at_snapshot"],
                        created_at=s["created_at"],
                    )
                    for s in data
                ]
                print(f"📂 加载快照: {skill_id} ({len(self._snapshots[skill_id])} 个)")
            except Exception as e:
                print(f"⚠️ 加载快照 {f.name} 失败: {e}")

    # ---------- [新增] 快照 & 回滚 ----------

    def _take_snapshot(self, skill: Skill) -> SkillSnapshot:
        """修复前对技能状态做深拷贝快照"""
        snap = SkillSnapshot(
            snapshot_id=str(uuid.uuid4())[:8],
            skill_id=skill.skill_id,
            skill_state=copy.deepcopy(skill.to_dict()),
            error_count_at_snapshot=skill.error_count,
        )
        if skill.skill_id not in self._snapshots:
            self._snapshots[skill.skill_id] = []
        self._snapshots[skill.skill_id].append(snap)
        self._save_snapshots(skill.skill_id)
        print(f"📸 快照已保存: {snap.snapshot_id} (错误数={snap.error_count_at_snapshot})")
        return snap

    def _rollback(self, skill: Skill, snapshot: SkillSnapshot) -> Skill:
        """回滚技能到指定快照状态"""
        restored = Skill.from_dict(snapshot.skill_state)
        self.skill_registry[skill.skill_id] = restored
        self._save_skill(restored)
        print(f"⏪ 已回滚技能 [{skill.name}] 到快照 {snapshot.snapshot_id}")
        print(f"   错误数: {skill.error_count} → {snapshot.error_count_at_snapshot}")
        return restored

    def _detect_regression(self, skill: Skill, snapshot: SkillSnapshot) -> bool:
        """检测是否发生回归：修复后错误数显著增加"""
        before = snapshot.error_count_at_snapshot
        after = skill.error_count

        # 冷启动保护：错误太少时不判断回归
        if before < self.MIN_ERRORS_FOR_REGRESSION:
            return False

        if after > before * self.REGRESSION_MULTIPLIER:
            print(f"🚨 回归检测: 错误数 {before} → {after} (>{self.REGRESSION_MULTIPLIER}x)")
            return True
        return False

    # ---------- [新增] 执行 & 错误计数 ----------

    def execute_and_count_errors(self, skill: Skill, results: List[dict]) -> int:
        """
        执行后统计错误数，自动更新 skill.error_count。

        调用方式：
          results = [
            {"name": "test_login", "passed": True},
            {"name": "test_payment", "passed": False, "error": "timeout"},
            {"name": "test_refund", "passed": False, "error": "assertion failed"},
          ]
          error_count = engine.execute_and_count_errors(skill, results)

        参数:
            skill: 目标技能
            results: 执行结果列表，每项包含:
                - name: 测试/步骤名称
                - passed: 是否通过 (True/False)
                - error: 失败原因 (可选)
                - severity: 严重级别 (可选, 默认 "error", 可选 "warning"/"critical")

        返回:
            错误总数
        """
        errors = [r for r in results if not r.get("passed", True)]
        warnings = [r for r in results if r.get("severity") == "warning" and not r.get("passed", True)]
        criticals = [r for r in results if r.get("severity") == "critical" and not r.get("passed", True)]

        # critical 算 2 分，error 算 1 分，warning 算 0.5 分
        error_count = len(criticals) * 2 + (len(errors) - len(criticals)) * 1 + len(warnings) * 0.5
        error_count = int(error_count)

        # 更新技能错误计数
        old_count = skill.error_count
        skill.error_count = error_count

        # 输出统计
        total = len(results)
        passed = total - len(errors)
        print(f"\n📊 执行结果统计 [{skill.name}]:")
        print(f"   总计: {total} | 通过: {passed} | 失败: {len(errors)}")
        if criticals:
            print(f"   🔴 严重: {len(criticals)} — {[r['name'] for r in criticals]}")
        if errors:
            normal_errors = [r for r in errors if r.get('severity', 'error') == 'error']
            if normal_errors:
                print(f"   ❌ 错误: {len(normal_errors)} — {[r['name'] for r in normal_errors]}")
        if warnings:
            print(f"   ⚠️ 警告: {len(warnings)} — {[r['name'] for r in warnings]}")
        print(f"   错误分数: {old_count} → {error_count}")

        # 持久化
        self._save_skill(skill)

        return error_count

    # ---------- 阶段1: 生成 ----------

    def generate_skill_draft(self, llm_output: dict) -> Optional[Skill]:
        """阶段1: Agent自动生成技能草案"""
        try:
            skill = Skill(
                skill_id=str(uuid.uuid4())[:8],
                name=llm_output["name"],
                version="v0.1",
                fixed_core_rules=llm_output["fixed_core_rules"],
                steps=llm_output["steps"],
            )
            self.skill_registry[skill.skill_id] = skill
            self._save_skill(skill)
            print(f"✅ 技能草案 [{skill.name}] 生成成功")
            return skill
        except KeyError as e:
            print(f"❌ 技能生成失败: LLM输出缺失关键字段 {e}")
            return None

    # ---------- 阶段2: 校验 ----------

    def validate_core_rules(self, skill: Skill) -> bool:
        """阶段2: 符号验证器校验 (独立于LLM的确定性代码)"""
        rules = skill.fixed_core_rules

        # 通用校验
        if rules.get("max_retries", 0) < 0:
            print("🚫 符号验证拦截: max_retries 不能为负数")
            return False

        if rules.get("timeout", 0) < 0:
            print("🚫 符号验证拦截: timeout 不能为负数")
            return False

        if rules.get("budget_limit", float("inf")) < 0:
            print("🚫 符号验证拦截: 预算限制不能为负数")
            return False

        # 兼容旧的购物场景校验
        if "safety_color_ban" in rules:
            banned = rules["safety_color_ban"]
            if any(c in banned for c in rules.get("target_color", [])):
                print(f"🚫 符号验证拦截: 目标颜色 {rules['target_color']} 在禁用列表 {banned} 中")
                return False

        # 代码修复场景校验
        if "forbidden_imports" in rules:
            print(f"🔒 禁止导入模块: {rules['forbidden_imports']}")

        if rules.get("require_type_hints") == True:
            print("🔒 要求: 修复代码必须包含类型提示")

        print("🛡️ 核心规则校验通过")
        return True

    # ---------- 阶段3: 归类 [已改造：使用受保护的规则] ----------

    def classify_failure(self, error_signal: dict) -> FailureType:
        """阶段3: 失败自动归类 — 使用受校验和保护的确定性规则，防止 LLM 篡改"""
        return self.rules_guard.classify(error_signal)

    # ---------- 阶段4: 修复 (接入 MIMO + 快照 + 尝试记录) ----------

    def auto_repair_and_update(self, skill: Skill, failure_type: FailureType, context: dict) -> str:
        """阶段4: 精准修复 & 动态附录更新 — 调用 MIMO 生成修复方案，带快照和回归检测"""

        # 4.0 修复前：拍快照
        snapshot = self._take_snapshot(skill)

        # 4.1 构造修复 Prompt
        system_prompt = (
            "你是一个自动化修复专家。根据以下失败类型和上下文，"
            "生成一段具体的修复补丁描述（1-3句话）。"
            "只输出修复方案，不要解释原因。"
        )
        user_prompt = (
            f"技能名称: {skill.name}\n"
            f"失败类型: {failure_type.value}\n"
            f"触发条件: {json.dumps(context, ensure_ascii=False)}\n"
            f"当前步骤: {skill.steps}\n"
            f"核心规则: {json.dumps(skill.fixed_core_rules, ensure_ascii=False)}\n\n"
            "请生成修复方案："
        )

        try:
            patch_strategy = call_llm(system_prompt, user_prompt)
            print(f"🔧 MIMO 生成修复方案: {patch_strategy}")
        except Exception as e:
            fallback = {
                FailureType.RULE_VIOLATION: "回溯符号约束，生成safe_fallback动作序列",
                FailureType.PERCEPTION_ERROR: "生成交互式探测动作以消除感知不确定性",
                FailureType.CONTEXT_HALLUCINATION: "执行环境接地检查，请求澄清或更新实体白名单",
                FailureType.INCOMPLETE_EXECUTION: "读取最近Checkpoint，生成断点续传指令",
            }
            patch_strategy = fallback[failure_type]
            print(f"⚠️ MIMO 调用失败({e})，降级为模板: {patch_strategy}")

        # 4.2 写入动态错误附录
        new_entry = ErrorAppendixEntry(
            entry_id=str(uuid.uuid4())[:6],
            failure_type=failure_type,
            trigger_condition=json.dumps(context, ensure_ascii=False),
            fix_patch=patch_strategy,
        )
        skill.dynamic_error_appendix.append(new_entry)
        self.appendix_vector_store.append(new_entry)

        # 4.3 记录这次修复尝试
        attempt = FixAttempt(
            attempt_id=str(uuid.uuid4())[:8],
            entry_id=new_entry.entry_id,
            skill_id=skill.skill_id,
            failure_type=failure_type.value,
            trigger_condition=json.dumps(context, ensure_ascii=False),
            fix_patch=patch_strategy,
            outcome=AttemptOutcome.SUCCESS,  # 先假设成功，回归检测后可能改
            error_before=snapshot.error_count_at_snapshot,
            error_after=skill.error_count,
        )

        # 4.4 回归检测
        if self._detect_regression(skill, snapshot):
            attempt.outcome = AttemptOutcome.REGRESSION
            attempt.notes = f"回归! 错误数 {snapshot.error_count_at_snapshot} → {skill.error_count}，触发回滚"
            self.fix_history.append(attempt)
            self._save_fix_history()

            # 自动回滚
            skill = self._rollback(skill, snapshot)
            attempt.outcome = AttemptOutcome.ROLLBACK
        else:
            self.fix_history.append(attempt)

        self._save_skill(skill)
        self._save_appendix_store()
        self._save_fix_history()
        print(f"💾 修复经验已写入动态附录 (ID: {new_entry.entry_id})")

        return patch_strategy

    # ---------- 阶段5: 晋升 ----------

    def check_promotion_threshold(self, skill: Skill, threshold: int = 5):
        """阶段5: 晋升机制 (动态经验 -> 固定规则的安全转化)"""
        for entry in skill.dynamic_error_appendix:
            if entry.hit_count >= threshold:
                print(f"🚀 触发晋升提案: 附录 [{entry.entry_id}] 命中 {entry.hit_count} 次")
                print(f"   ➡️ 建议将 '{entry.fix_patch}' 固化为固定核心规则，请人工审核！")

    # ---------- [新增] 诊断报告 ----------

    def get_report(self, skill_id: Optional[str] = None) -> dict:
        """生成诊断报告：包含尝试历史、回归统计、快照信息"""
        attempts = self.fix_history
        if skill_id:
            attempts = [a for a in attempts if a.skill_id == skill_id]

        total = len(attempts)
        successes = sum(1 for a in attempts if a.outcome == AttemptOutcome.SUCCESS)
        rollbacks = sum(1 for a in attempts if a.outcome == AttemptOutcome.ROLLBACK)
        regressions = sum(1 for a in attempts if a.outcome == AttemptOutcome.REGRESSION)

        report = {
            "总尝试次数": total,
            "成功": successes,
            "回滚": rollbacks,
            "回归检测": regressions,
            "成功率": f"{successes/total*100:.1f}%" if total > 0 else "N/A",
            "最近5次尝试": [
                {
                    "时间": a.created_at,
                    "技能": a.skill_id,
                    "类型": a.failure_type,
                    "结果": a.outcome.value,
                    "修复方案": a.fix_patch[:50] + "..." if len(a.fix_patch) > 50 else a.fix_patch,
                }
                for a in attempts[-5:]
            ],
        }
        return report


# ==========================================  [新增] Benchmark 系统  # ==========================================

class BenchmarkScorer:
    """评分器：比较预测结果和标准答案"""

    @staticmethod
    def exact_match(predicted: str, expected: str) -> float:
        """精确匹配（忽略大小写和首尾空格）"""
        return 1.0 if predicted.strip().lower() == expected.strip().lower() else 0.0

    @staticmethod
    def contains_match(predicted: str, expected: str) -> float:
        """包含匹配（预测结果包含标准答案）"""
        return 1.0 if expected.strip().lower() in predicted.strip().lower() else 0.0

    @staticmethod
    def numeric_tolerance(predicted: str, expected: str, tolerance: float = 0.05) -> float:
        """数值容差匹配（±5%）"""
        try:
            pred_num = float(''.join(c for c in predicted if c.isdigit() or c in '.-'))
            exp_num = float(''.join(c for c in expected if c.isdigit() or c in '.-'))
            if exp_num == 0:
                return 1.0 if pred_num == 0 else 0.0
            return 1.0 if abs(pred_num - exp_num) / abs(exp_num) <= tolerance else 0.0
        except (ValueError, ZeroDivisionError):
            return 0.0

    @staticmethod
    def llm_judge(predicted: str, expected: str, rubric: str = "") -> float:
        """LLM-as-judge 评分"""
        if not rubric:
            rubric = "判断预测答案是否正确。只输出 1（正确）或 0（错误）。"
        prompt = f"标准答案: {expected}\n预测答案: {predicted}\n评分标准: {rubric}\n\n只输出 1 或 0:"
        try:
            resp = call_llm("你是一个严格的评分员。", prompt)
            return 1.0 if "1" in resp.strip() else 0.0
        except Exception:
            return BenchmarkScorer.exact_match(predicted, expected)


class BenchmarkDataset:
    """加载测试数据集（支持 JSON 和 CSV）"""

    def __init__(self, path: Path):
        self.path = path
        self.cases: List[dict] = []
        self._load()

    def _load(self):
        if self.path.suffix == ".json":
            with open(self.path, "r", encoding="utf-8") as f:
                self.cases = json.load(f)
        elif self.path.suffix == ".csv":
            import csv
            with open(self.path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.cases = list(reader)
        else:
            raise ValueError(f"不支持的格式: {self.path.suffix}")
        print(f"📂 加载 benchmark 数据集: {len(self.cases)} 条测试用例")

    def split(self, train_ratio: float = 0.7) -> tuple:
        """拆分训练集和验证集"""
        split_idx = int(len(self.cases) * train_ratio)
        return self.cases[:split_idx], self.cases[split_idx:]


class BenchmarkRunner:
    """运行 benchmark 并跟踪进化历史"""

    def __init__(self, dataset: BenchmarkDataset, scorer_type: str = "exact"):
        self.dataset = dataset
        self.scorer_type = scorer_type
        self.history: List[dict] = []  # 每次迭代的准确率

    def run(self, predict_fn, cases: Optional[List[dict]] = None) -> dict:
        """
        运行一轮 benchmark。
        predict_fn: 函数，输入 question 字符串，输出预测答案字符串
        返回: {accuracy, total, passed, failed_cases, all_results}
        """
        if cases is None:
            cases = self.dataset.cases

        results = []
        for case in cases:
            question = case.get("question", case.get("input", ""))
            expected = case.get("answer", case.get("expected", case.get("ground_truth", "")))

            try:
                predicted = predict_fn(question)
            except Exception as e:
                results.append({"question": question, "expected": expected, "predicted": "", "score": 0.0, "error": str(e)})
                continue

            score = self._score(str(predicted), str(expected))
            results.append({"question": question, "expected": expected, "predicted": str(predicted), "score": score})

        accuracy = sum(r["score"] for r in results) / len(results) if results else 0.0
        failed = [r for r in results if r["score"] < 1.0]

        iteration = len(self.history) + 1
        self.history.append({
            "iteration": iteration,
            "accuracy": accuracy,
            "total": len(results),
            "passed": len(results) - len(failed),
            "failed": len(failed),
            "timestamp": datetime.now().isoformat(),
        })

        return {"accuracy": accuracy, "total": len(results), "passed": len(results) - len(failed), "failed_cases": failed, "all_results": results}

    def _score(self, predicted: str, expected: str) -> float:
        scorers = {
            "exact": BenchmarkScorer.exact_match,
            "contains": BenchmarkScorer.contains_match,
            "numeric": BenchmarkScorer.numeric_tolerance,
            "llm": BenchmarkScorer.llm_judge,
        }
        return scorers.get(self.scorer_type, BenchmarkScorer.exact_match)(predicted, expected)

    def get_frontier(self) -> dict:
        """获取历史最佳"""
        if not self.history:
            return {"accuracy": 0.0, "iteration": 0}
        return max(self.history, key=lambda x: x["accuracy"])

    def print_progress(self):
        """打印进度表（类似 EvoSkill）"""
        if not self.history:
            return
        print(f"\n {'Iter':<6} {'Accuracy':<10} {'Delta':<8} {'Passed':<8} {'Frontier':<10} {'Status'}")
        print(" " + "-" * 55)
        frontier_acc = self.get_frontier()["accuracy"]
        for h in self.history:
            delta = h["accuracy"] - self.history[h["iteration"] - 2]["accuracy"] if h["iteration"] > 1 else 0
            is_frontier = h["accuracy"] >= frontier_acc
            status = "★ best" if is_frontier and h["iteration"] > 1 else ("baseline" if h["iteration"] == 1 else "")
            print(f" {h['iteration']:<6} {h['accuracy']:<10.1%} {delta:<+8.1%} {h['passed']}/{h['total']:<6} {frontier_acc:<10.1%} {status}")


# ==========================================
# 3. 运行演示 — 代码 Bug 自动修复 & 技能生成 & Benchmark
# ==========================================

if __name__ == "__main__":
    engine = SelfEvolvingSkillEngine()

    print("\n" + "=" * 60)
    print("🚀 Self-Evolving Skill Engine v2.0")
    print("=" * 60)
    print("场景1: 代码 Bug 自动修复")
    print("场景2: 技能生成 & 进化")
    print("场景3: Benchmark 驱动进化")
    print("=" * 60)

    # ========================================
    # 场景1: 代码 Bug 自动修复
    # ========================================
    print("\n" + "-" * 60)
    print("📝 场景1: 代码 Bug 自动修复")
    print("-" * 60)

    # 1.1 生成「代码修复」技能
    code_fix_skill = engine.generate_skill_draft({
        "name": "python_bug_auto_fixer",
        "fixed_core_rules": {
            "max_retries": 3,
            "timeout": 30,
            "forbidden_imports": ["os.system", "subprocess.Popen"],  # 安全约束
            "require_type_hints": True,
        },
        "steps": [
            "读取报错堆栈", "定位出错文件和行号",
            "分析错误类型", "生成修复代码", "运行测试验证",
        ],
    })

    if code_fix_skill:
        engine.validate_core_rules(code_fix_skill)

        # 1.2 模拟: 运行测试，发现3个错误
        print("\n💥 模拟: Agent 修复后运行测试...")
        test_results_1 = [
            {"name": "test_type_hints", "passed": True},
            {"name": "test_import_safety", "passed": True},
            {"name": "test_basic_run", "passed": False, "error": "TypeError: cannot unpack non-iterable NoneType"},
            {"name": "test_edge_case", "passed": False, "error": "KeyError: 'result'"},
            {"name": "test_timeout", "passed": False, "error": "timeout after 30s", "severity": "warning"},
        ]
        engine.execute_and_count_errors(code_fix_skill, test_results_1)

        err = {"code": "KINEMATIC_LIMIT", "message": "TypeError: cannot unpack non-iterable NoneType object"}
        ft = engine.classify_failure(err)
        engine.auto_repair_and_update(
            skill=code_fix_skill, failure_type=ft,
            context={"file": "api_handler.py", "line": 42, "error": "TypeError"},
        )

        # 1.3 模拟: 又遇到导入安全问题，但测试结果改善
        print("\n💥 模拟: Agent 修复后再次测试...")
        test_results_2 = [
            {"name": "test_type_hints", "passed": True},
            {"name": "test_import_safety", "passed": False, "error": "forbidden import: os.system", "severity": "critical"},
            {"name": "test_basic_run", "passed": True},
            {"name": "test_edge_case", "passed": True},
            {"name": "test_timeout", "passed": True},
        ]
        engine.execute_and_count_errors(code_fix_skill, test_results_2)

        err2 = {"code": "SAFETY_BOUNDARY_EXCEEDED", "message": "forbidden import: os.system"}
        ft2 = engine.classify_failure(err2)
        engine.auto_repair_and_update(
            skill=code_fix_skill, failure_type=ft2,
            context={"file": "utils.py", "line": 15, "blocked_import": "os.system"},
        )

        # 1.4 模拟: 修复后测试全崩了！触发回滚
        print("\n💥 模拟: Agent 激进修复后测试全崩，触发回滚...")
        # 先拍快照（此时错误数=2，是上次修复后的状态）
        snapshot_before = engine._take_snapshot(code_fix_skill)
        # 再执行测试，错误数飙升到 7
        test_results_3 = [
            {"name": "test_type_hints", "passed": False, "error": "SyntaxError", "severity": "critical"},
            {"name": "test_import_safety", "passed": False, "error": "ImportError", "severity": "critical"},
            {"name": "test_basic_run", "passed": False, "error": "crash"},
            {"name": "test_edge_case", "passed": False, "error": "crash"},
            {"name": "test_timeout", "passed": False, "error": "crash"},
        ]
        engine.execute_and_count_errors(code_fix_skill, test_results_3)
        # 回归检测：错误数 2 → 7，触发回滚！
        if engine._detect_regression(code_fix_skill, snapshot_before):
            code_fix_skill = engine._rollback(code_fix_skill, snapshot_before)
            print("✅ 回滚成功！技能恢复到修复前状态")

    # ========================================
    # 场景2: 技能生成 & 进化
    # ========================================
    print("\n" + "-" * 60)
    print("📝 场景2: 技能生成 & 进化")
    print("-" * 60)

    # 2.1 从 LLM 输出生成新技能
    api_skill = engine.generate_skill_draft({
        "name": "api_data_fetcher",
        "fixed_core_rules": {
            "max_retries": 3,
            "timeout": 10,
            "rate_limit": "100/min",
        },
        "steps": [
            "构建请求参数", "发送 HTTP 请求", "校验响应状态码",
            "解析 JSON 响应", "提取目标字段", "返回结构化数据",
        ],
    })

    if api_skill:
        engine.validate_core_rules(api_skill)

        # 2.2 模拟多次执行中遇到不同错误，技能逐步进化
        errors = [
            ({"code": "SENSOR_NOISE", "message": "OCR_FAIL on response field"},
             {"api": "weather", "field": "temperature", "raw_value": "??°C"},
             [{"name": "test_parse", "passed": True}, {"name": "test_field", "passed": False, "error": "OCR_FAIL"}]),
            ({"code": "ENTITY_NOT_FOUND", "message": "HALLUCINATION_DETECTED: field 'result' not in response"},
             {"api": "stock", "response_keys": ["data", "meta", "ts"]},
             [{"name": "test_parse", "passed": True}, {"name": "test_field", "passed": False, "error": "missing field"}, {"name": "test_schema", "passed": False, "error": "schema mismatch"}]),
            ({"code": "TIMEOUT", "message": "request timeout after 10s"},
             {"api": "payment", "retry_count": 0},
             [{"name": "test_parse", "passed": False, "error": "timeout", "severity": "critical"}, {"name": "test_field", "passed": False, "error": "timeout"}, {"name": "test_schema", "passed": False, "error": "timeout"}]),
            ({"code": "SENSOR_NOISE", "message": "OCR_FAIL parsing nested JSON"},
             {"api": "order", "nested_depth": 3},
             [{"name": "test_parse", "passed": True}, {"name": "test_nested", "passed": False, "error": "OCR_FAIL"}]),
            ({"code": "SENSOR_NOISE", "message": "OCR_FAIL on decimal numbers"},
             {"api": "price", "format": "¥1,234.56"},
             [{"name": "test_parse", "passed": True}, {"name": "test_decimal", "passed": False, "error": "format error"}]),
        ]

        for err_signal, ctx, test_results in errors:
            engine.execute_and_count_errors(api_skill, test_results)
            ft = engine.classify_failure(err_signal)
            engine.auto_repair_and_update(skill=api_skill, failure_type=ft, context=ctx)

        # 2.3 晋升检查
        print("\n" + "-" * 60)
        print("📊 技能进化状态")
        print("-" * 60)
        for entry in api_skill.dynamic_error_appendix:
            entry.hit_count = 5  # 模拟命中次数
        engine.check_promotion_threshold(api_skill)

    # ========================================
    # 场景3: Benchmark 驱动进化
    # ========================================
    print("\n" + "-" * 60)
    print("📝 场景3: Benchmark 驱动进化")
    print("-" * 60)

    # 3.1 加载 benchmark 数据集
    benchmark_path = Path(__file__).parent / "benchmark_questions.json"
    if benchmark_path.exists():
        dataset = BenchmarkDataset(benchmark_path)
        train_cases, val_cases = dataset.split(train_ratio=0.7)
        print(f"   训练集: {len(train_cases)} 条 | 验证集: {len(val_cases)} 条")

        # 3.2 创建 benchmark runner
        runner = BenchmarkRunner(dataset, scorer_type="contains")

        # 3.3 模拟 Agent 回答（第一轮：基础能力，60% 准确率）
        def agent_v1(question: str) -> str:
            """v1: 只能回答简单问题"""
            simple_answers = {
                "反转列表": "list.reverse()",
                "读取 JSON": "json.load()",
                "捕获异常": "except Exception",
                "虚拟环境": "python -m venv",
                "格式化字符串": "f-string",
            }
            for key, ans in simple_answers.items():
                if key in question:
                    return ans
            return "不知道"

        print("\n🔄 迭代 1: Agent v1 (基础能力)")
        result1 = runner.run(agent_v1, train_cases)
        runner.print_progress()
        print(f"   失败案例: {[r['question'][:20] for r in result1['failed_cases']]}")

        # 3.4 模拟进化：Agent v2 学会了更多
        def agent_v2(question: str) -> str:
            """v2: 修复了排序和类型检查"""
            answers = {
                "反转列表": "list.reverse()",
                "读取 JSON": "json.load()",
                "捕获异常": "except Exception",
                "虚拟环境": "python -m venv",
                "格式化字符串": "f-string",
                "排序字典": "sorted()",
                "检查类型": "isinstance()",
            }
            for key, ans in answers.items():
                if key in question:
                    return ans
            return "不知道"

        print("\n🔄 迭代 2: Agent v2 (修复了排序和类型检查)")
        result2 = runner.run(agent_v2, train_cases)
        runner.print_progress()

        # 3.5 模拟进化：Agent v3 进一步提升
        def agent_v3(question: str) -> str:
            """v3: 几乎全对"""
            answers = {
                "反转列表": "list.reverse()",
                "读取 JSON": "json.load()",
                "捕获异常": "except Exception",
                "虚拟环境": "python -m venv",
                "格式化字符串": "f-string",
                "排序字典": "sorted()",
                "检查类型": "isinstance()",
                "删除文件": "os.remove()",
                "连接列表": "extend()",
                "当前时间": "datetime.now()",
            }
            for key, ans in answers.items():
                if key in question:
                    return ans
            return "不知道"

        print("\n🔄 迭代 3: Agent v3 (全面进化)")
        result3 = runner.run(agent_v3, train_cases)
        runner.print_progress()

        # 3.6 用验证集测试最佳版本
        print("\n📊 用验证集测试最佳 Agent...")
        val_result = runner.run(agent_v3, val_cases)
        print(f"   验证集准确率: {val_result['accuracy']:.1%}")

        # 3.7 输出进化摘要
        frontier = runner.get_frontier()
        print(f"\n🏆 最佳迭代: #{frontier['iteration']}，准确率: {frontier['accuracy']:.1%}")
    else:
        print("⚠️ benchmark_questions.json 不存在，跳过场景3")

    # ========================================
    # 汇总报告
    # ========================================
    print("\n" + "=" * 60)
    print("📊 全局诊断报告")
    print("=" * 60)
    report = engine.get_report()
    for k, v in report.items():
        if isinstance(v, list):
            print(f"  {k}:")
            for item in v:
                print(f"    - [{item['结果']}] {item['类型']}: {item['修复方案']}")
        else:
            print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("🛡️ 评估规则完整性: " + ("✅ 通过" if engine.rules_guard.verify_integrity() else "❌ 被篡改!"))
    print("=" * 60)
