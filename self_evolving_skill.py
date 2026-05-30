import sys
import os
import json
import uuid
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


@dataclass
class Skill:
    """可复用技能手册结构"""
    skill_id: str
    name: str
    version: str
    fixed_core_rules: Dict[str, Any]
    dynamic_error_appendix: List[ErrorAppendixEntry] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "version": self.version,
            "fixed_core_rules": self.fixed_core_rules,
            "dynamic_error_appendix": [e.to_dict() for e in self.dynamic_error_appendix],
            "steps": self.steps,
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
        )


# ==========================================
# 2. 核心引擎：生成、校验、归类、修复 + 持久化
# ==========================================


class SelfEvolvingSkillEngine:
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.skill_registry: Dict[str, Skill] = {}
        self.appendix_vector_store: List[ErrorAppendixEntry] = []
        # 启动时自动加载已有数据
        self._load_all()

    # ---------- 持久化 ----------

    def _skill_path(self, skill_id: str) -> Path:
        return self.data_dir / f"skill_{skill_id}.json"

    def _appendix_path(self) -> Path:
        return self.data_dir / "appendix_store.json"

    def _save_skill(self, skill: Skill):
        path = self._skill_path(skill.skill_id)
        path.write_text(json.dumps(skill.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"💾 技能已持久化: {path.name}")

    def _save_appendix_store(self):
        path = self._appendix_path()
        data = [e.to_dict() for e in self.appendix_vector_store]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_all(self):
        """从 data_dir 加载所有技能和附录"""
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

        if rules.get("budget_limit", float("inf")) < 0:
            print("🚫 符号验证拦截: 预算限制不能为负数")
            return False

        if "safety_color_ban" in rules:
            banned = rules["safety_color_ban"]
            if any(c in banned for c in rules.get("target_color", [])):
                print(f"🚫 符号验证拦截: 目标颜色 {rules['target_color']} 在禁用列表 {banned} 中")
                return False

        print("🛡️ 核心规则校验通过")
        return True

    # ---------- 阶段3: 归类 ----------

    def classify_failure(self, error_signal: dict) -> FailureType:
        """阶段3: 失败自动归类 (确定性规则映射，不让LLM判断对错)"""
        err_code = error_signal.get("code", "")
        err_msg = error_signal.get("message", "")

        if err_code in ("SAFETY_BOUNDARY_EXCEEDED", "KINEMATIC_LIMIT"):
            return FailureType.RULE_VIOLATION
        elif err_code == "SENSOR_NOISE" or "OCR_FAIL" in err_msg:
            return FailureType.PERCEPTION_ERROR
        elif err_code == "ENTITY_NOT_FOUND" or "HALLUCINATION_DETECTED" in err_msg:
            return FailureType.CONTEXT_HALLUCINATION
        elif err_code in ("TIMEOUT", "POWER_LOSS"):
            return FailureType.INCOMPLETE_EXECUTION
        else:
            print("⚠️ 未知错误类型，降级调用LLM进行深度归因...")
            return FailureType.CONTEXT_HALLUCINATION

    # ---------- 阶段4: 修复 (接入 MIMO) ----------

    def auto_repair_and_update(self, skill: Skill, failure_type: FailureType, context: dict) -> str:
        """阶段4: 精准修复 & 动态附录更新 — 调用 MIMO 生成真实修复方案"""

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
            # LLM 调用失败时降级为模板
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
        self._save_skill(skill)
        self._save_appendix_store()
        print(f"💾 修复经验已写入动态附录 (ID: {new_entry.entry_id})")

        return patch_strategy

    # ---------- 阶段5: 晋升 ----------

    def check_promotion_threshold(self, skill: Skill, threshold: int = 5):
        """阶段5: 晋升机制 (动态经验 -> 固定规则的安全转化)"""
        for entry in skill.dynamic_error_appendix:
            if entry.hit_count >= threshold:
                print(f"🚀 触发晋升提案: 附录 [{entry.entry_id}] 命中 {entry.hit_count} 次")
                print(f"   ➡️ 建议将 '{entry.fix_patch}' 固化为固定核心规则，请人工审核！")
                # 实际系统中此处应发送审批流，审核通过后修改 skill.fixed_core_rules


# ==========================================
# 3. 运行演示
# ==========================================

if __name__ == "__main__":
    engine = SelfEvolvingSkillEngine()

    # Step 1: LLM 生成技能草案
    draft_data = {
        "name": "select_swim_gift_for_7yo_girl",
        "fixed_core_rules": {
            "budget_limit": 200,
            "safety_color_ban": ["blue", "white"],
            "target_color": ["fluorescent_pink"],
        },
        "steps": ["检索山姆三件套", "校验库存", "组合View泳镜", "下单"],
    }
    skill = engine.generate_skill_draft(draft_data)

    if skill:
        # Step 2: 符号验证器校验
        is_valid = engine.validate_core_rules(skill)

        if is_valid:
            # Step 3: 模拟执行时发生【感知输出提取错误】
            mock_error_signal = {"code": "SENSOR_NOISE", "message": "Size chart OCR_FAIL"}
            failure_type = engine.classify_failure(mock_error_signal)

            # Step 4: 调用 MIMO 生成修复方案并写入附录
            engine.auto_repair_and_update(
                skill=skill,
                failure_type=failure_type,
                context={"sku": "sam_mickey_001", "step_index": 1},
            )

            # Step 5: 模拟多次命中触发晋升检查
            skill.dynamic_error_appendix[-1].hit_count = 5
            engine.check_promotion_threshold(skill)
