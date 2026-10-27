# -*- coding: utf-8 -*-
"""使用 LangChain 通过 OpenAI 兼容接口调用智谱 GLM 模型，支持多轮对话记忆。

记忆以多会话形式持久化在 sessions/ 目录（每个会话一个 JSON 文件），
启动时列出可用会话，由用户决定恢复哪个上下文或开始新对话：
- 环境变量 MEMORY_DIR 可改会话目录；MEMORY_FILE 指旧版单文件（仅用于自动导入检测）
- 对话中 /save 保存到当前会话、/load 重新选择会话

上下文长度限制可配置、可动态调整：
- 启动前：环境变量 MODEL_CONTEXT_WINDOW / MAX_CONTEXT_TOKENS / MAX_MESSAGES（或写入 .env）
- 对话中：/window <n> 调整模型窗口、/tokens <n> 调整历史预算、/msgs <n> 调整条数上限，/config 查看占用

模型上下文窗口（MODEL_CONTEXT_WINDOW）是硬上限：glm-5.3-flash 官方为 1M。
历史 token 预算（MAX_CONTEXT_TOKENS）不能超过窗口，默认等于窗口值；
调小预算可让对话只保留最近内容。注意输出 token 也占窗口，预算建议留出输出空间。

流式输出按智谱官方 LangChain 集成文档实现：模型设置 streaming=True，
配合 StreamingStdOutCallbackHandler 逐字实时打印，回复完毕后完整内容才入记忆。

条件冲突处理（工程层）：每轮用户消息记入「条件账本」并按轮次/时间戳版本化，
显式修正（“改成/我是说”等）自动作废最近一条生效条件，疑似矛盾打标并提示模型
先澄清；账本随会话持久化，/conds 查看变更历史。
"""
import json
import os
import re
import sys
import time
from datetime import datetime

import tiktoken
from dotenv import load_dotenv
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# 从项目根目录的 .env 文件加载环境变量
load_dotenv()

# 智谱 AI 提供的 OpenAI 兼容接口地址
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"

# ===== 上下文限制配置（默认值，可用环境变量覆盖，对话中还可动态调整）=====
# 模型上下文窗口大小（硬上限）。glm-5.3-flash 官方窗口为 1M，可换算：1M = 1048576。
# 换用窗口更小的模型时需同步调小此值（或对话中 /window <n> 调整）。
DEFAULT_MODEL_CONTEXT_WINDOW = int(os.getenv("MODEL_CONTEXT_WINDOW", str(1024 * 1024)))
# 发送给模型的上下文 token 预算（含系统提示词），默认等于模型窗口。
# 留空/为 0 表示跟随模型窗口；最大值会被窗口封顶。
DEFAULT_MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "0")) or DEFAULT_MODEL_CONTEXT_WINDOW
# 历史消息条数上限（不含开头的 SystemMessage）
DEFAULT_MAX_MESSAGES = int(os.getenv("MAX_MESSAGES", "40"))

# 每条消息的固定开销（按 OpenAI 约定每条约 4 token）
_TOKENS_PER_MESSAGE = 4

# GLM 的分词器与 tiktoken 不完全兼容，这里用 cl100k_base 近似估算。
# 该编码对中文偏保守（估算略高于实际值），只会提前裁剪、不会撑爆上下文。
_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(messages: list) -> int:
    """估算一次请求的总 token 数（各消息内容 + 每条固定开销）。"""
    return sum(len(_ENCODER.encode(str(m.content))) for m in messages) + len(messages) * _TOKENS_PER_MESSAGE


def build_model() -> ChatOpenAI:
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        sys.exit("请先设置环境变量 ZHIPUAI_API_KEY（在 https://open.bigmodel.cn 获取）")

    return ChatOpenAI(
        model="glm-5.3-flash",
        api_key=api_key,
        base_url=ZHIPU_BASE_URL,
        temperature=0.7,
        max_retries=2,
        timeout=60,  # 显式超时（秒）：服务无响应/网络异常时最多等 60 秒，避免长时间卡住
        # 流式输出（智谱官方 LangChain 集成文档推荐方式）
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )


def clamp_budget_to_window(config: dict) -> None:
    """把 token 预算封顶到模型窗口（预算是窗口内的软预算，窗口才是硬上限）。"""
    if config["max_tokens"] > config["window"]:
        config["max_tokens"] = config["window"]


# ===== 记忆持久化（多会话）=====
# 会话目录：环境变量 MEMORY_DIR 可覆盖，默认为脚本同目录的 sessions/
DEFAULT_MEMORY_DIR = os.getenv("MEMORY_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions"))
# 旧版单文件记忆（存在时启动会被询问是否导入）
LEGACY_MEMORY_FILE = os.getenv("MEMORY_FILE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.json"))


def _session_path(session_id: str) -> str:
    return os.path.join(DEFAULT_MEMORY_DIR, f"{session_id}.json")


def _new_session_id() -> str:
    """生成新会话 ID：时间戳 + 短随机后缀，避免同秒创建冲突。"""
    import random
    import string

    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{stamp}_{suffix}"


def list_sessions() -> list[dict]:
    """列出所有会话，按最后保存时间倒序。返回 [{id, path, mtime, count, preview}]。"""
    if not os.path.isdir(DEFAULT_MEMORY_DIR):
        return []
    result = []
    for name in os.listdir(DEFAULT_MEMORY_DIR):
        if not name.endswith(".json"):
            continue
        path = os.path.join(DEFAULT_MEMORY_DIR, name)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            messages = data["messages"]
            # 预览第一条用户消息（LangChain 序列化结构为 {type, data: {content, ...}}）
            preview = next(
                (str(m.get("data", {}).get("content", "")) for m in messages if m.get("type") == "human"),
                "（无用户消息）",
            )
            preview = preview if len(preview) <= 30 else preview[:30] + "…"
            result.append({
                "id": name[:-5],
                "path": path,
                "mtime": os.path.getmtime(path),
                "count": max(len(messages) - 1, 0),
                "preview": preview,
            })
        except (json.JSONDecodeError, KeyError, TypeError, OSError):
            continue  # 跳过损坏的会话文件
    result.sort(key=lambda s: s["mtime"], reverse=True)
    return result


def save_session(messages: list, session_id: str, ledger: "ConditionLedger | None" = None) -> str:
    """保存会话到 sessions/<id>.json（原子写入），返回文件路径。

    ledger 为条件账本（可空）；旧版文件无 conditions 字段，读取时按空账本处理。
    """
    from langchain_core.messages import messages_to_dict

    os.makedirs(DEFAULT_MEMORY_DIR, exist_ok=True)
    path = _session_path(session_id)
    data = {
        "version": 1,
        "session_id": session_id,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "messages": messages_to_dict(messages),
        "conditions": ledger.to_dict() if ledger else {"turn": 0, "entries": []},
    }
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)
    return path


def load_session(session_id: str) -> tuple[list, "ConditionLedger"]:
    """按会话 ID 加载消息列表与条件账本；损坏时抛 ValueError，由调用方处理。

    旧版文件缺 conditions 字段时返回空账本（向后兼容）。
    """
    from langchain_core.messages import messages_from_dict

    path = _session_path(session_id)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    messages = messages_from_dict(data["messages"])
    if not messages or type(messages[0]).__name__ != "SystemMessage":
        raise ValueError("缺少 SystemMessage")
    return messages, ConditionLedger.from_dict(data.get("conditions"))


def trim_history(
    messages: list,
    max_tokens: int,
    max_messages: int,
) -> int:
    """裁剪历史消息以满足 token 预算与条数限制，始终保留开头的 SystemMessage。

    条数限制从最旧处裁剪；token 超预算时也丢弃最旧的历史（保留最新）。
    返回裁剪后的总 token 数。若单条最新消息就超出预算，则只保留该消息。
    """
    # 条数上限（-1 是排除开头的 SystemMessage）
    while len(messages) - 1 > max_messages:
        del messages[1]

    total = count_tokens(messages)
    while total > max_tokens and len(messages) > 2:
        # 至少保留 SystemMessage + 最新一条消息
        del messages[1]
        total = count_tokens(messages)
    return total


def _read_int_command(prompt: str, current: int, minimum: int) -> int | None:
    """解析 `/tokens <n>`、`/msgs <n>` 这类动态调整命令，返回新值。"""
    parts = prompt.split()
    if len(parts) != 2:
        print(f"用法: {parts[0]} <数字>   （当前值: {current}）")
        return None
    try:
        value = int(parts[1])
    except ValueError:
        print(f"无效数字: {parts[1]!r}")
        return None
    if value < minimum:
        print(f"值太小，至少为 {minimum}。")
        return None
    return value


# 系统提示词：内置多轮对话的条件冲突处理规则（显式修正直接覆盖、隐式矛盾必须澄清、
# 无法澄清取最新并声明假设、硬约束优先于软偏好、绝不静默替用户取舍）
SYSTEM_PROMPT = """你是一个乐于助人的中文助手。回答时请结合上文语境。

用户会在多轮对话中不断补充、修正条件。遇到前后条件冲突时，按以下规则处理：
1. 显式修正（“改成/不对/我是说”）：直接以最新说法为准，并确认“已更新为X，原Y作废”。
2. 隐式矛盾（无法共存的时序约束）：不要擅自二选一，先向用户澄清：
   “您先后提到A和B，二者冲突，请确认以哪个为准，或说明优先级。”
3. 硬约束优先于软偏好：先区分硬约束（必须满足）与软偏好（尽量满足），
   冲突时保硬约束，软偏好的取舍需说明理由。
4. 无法澄清时：默认采用最新条件，并显式声明假设：“按您最新的要求B处理，之前提到的A不再适用。”
5. 涉及删除数据、花费金额、安全等不可逆操作的冲突，必须问清楚再执行。
核心原则：绝不静默替用户做取舍——冲突要么问清楚，要么说清楚。

对话中会附带 [条件一致性提示] 的消息：那是系统自动检测到新输入与既有条件疑似矛盾时
注入的提醒，不是用户本人发言。收到该提示时，按规则2先澄清；若判定其实可共存，正常
回答并说明判定理由即可，不必反问用户。"""


# ── 条件冲突处理（工程层）──────────────────────────────────────────────
class ConditionLedger:
    """条件账本：按轮次/时间戳版本化记录用户条件，支持作废、矛盾打标与变更历史。

    每条记录：{"turn", "timestamp", "condition", "kind", "status", "reason"}
    kind: 硬约束 / 软偏好 | status: 生效 / 作废 / 待澄清
    """

    STATUS_ACTIVE = "生效"
    STATUS_VOID = "作废"
    STATUS_CLARIFY = "待澄清"
    EXPLICIT_MARKERS = ("改成", "不对", "我是说", "改为", "换成了")

    def __init__(self, entries=None, turn=0):
        self.entries: list[dict] = list(entries or [])
        self._turn = int(turn)

    def advance_turn(self) -> int:
        self._turn += 1
        return self._turn

    def _entry(self, condition, kind, status, reason):
        return {
            "turn": self._turn,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "condition": condition.strip(),
            "kind": kind,
            "status": status,
            "reason": reason,
        }

    def record(self, condition, kind="软偏好", status=None, reason="新增条件"):
        """记录一条新条件（status 缺省为生效；显式修正覆盖时作废旧条件）。"""
        status = status or self.STATUS_ACTIVE
        if status == self.STATUS_ACTIVE and self.EXPLICIT_MARKERS and any(
            m in condition for m in self.EXPLICIT_MARKERS
        ):
            self.explicit_override(condition)
            return
        self.entries.append(self._entry(condition, kind, status, reason))

    def explicit_override(self, condition):
        """规则1：显式修正——最新条件直接生效，最近一条生效条件作废。"""
        for entry in reversed(self.entries):
            if entry["status"] == self.STATUS_ACTIVE:
                entry["status"] = self.STATUS_VOID
                entry["reason"] = "显式修正（最新条件直接覆盖）"
                break
        self.entries.append(
            self._entry(condition, "硬约束", self.STATUS_ACTIVE, "显式修正覆盖生效")
        )

    def flag_conflict(self, condition, kind="硬约束"):
        """规则2：疑似隐式矛盾——打上待澄清标记，由注入提示引导模型先澄清。"""
        self.entries.append(
            self._entry(condition, kind, self.STATUS_CLARIFY, "疑似与既有条件矛盾，需澄清")
        )

    def resolve(self, condition, kind="硬约束"):
        """规则4：澄清完成——新条件转正为生效。"""
        self.entries.append(
            self._entry(condition, kind, self.STATUS_ACTIVE, "澄清后生效")
        )

    def active_entries(self):
        return [e for e in self.entries if e["status"] == self.STATUS_ACTIVE]

    def summary(self):
        """/conds 输出：完整的条件变更历史。"""
        if not self.entries:
            return "（本会话暂无条件记录）"
        return "\n".join(
            f"[第{e['turn']}轮 {e['timestamp']}] {e['status']} | {e['kind']} | "
            f"{e['condition']}（{e['reason']}）"
            for e in self.entries
        )

    def to_dict(self):
        return {"turn": self._turn, "entries": self.entries}

    @classmethod
    def from_dict(cls, data):
        data = data or {}
        return cls(entries=data.get("entries"), turn=data.get("turn", 0))


def _extract_condition_topics(text: str) -> set:
    """从条件文本提取粗粒度主题词：按标点/空白切分后，再去掉句首的礼貌/否定前缀。"""
    cleaned = "".join(ch for ch in text if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
    words = re.split(r"[，。！？、；：,.!?;:\s（）()\[\]{}\"'～~\-_/\\]+", cleaned)
    result = set()
    for w in words:
        if len(w) < 2:
            continue
        for prefix in ("请", "麻烦", "帮我", "帮忙", "不要", "不能", "别", "禁止", "无需", "不用"):
            if w.startswith(prefix) and len(w) > len(prefix):
                w = w[len(prefix):]
                break
        if len(w) >= 2:
            result.add(w)
    return result


_NEGATION_HINTS = ("不要", "别", "禁止", "不用", "无需", "不能", "改成", "换成了", "排除")


def _looks_like_conflict(text: str, ledger: "ConditionLedger") -> bool:
    """启发式矛盾检测：新输入与既有生效条件主题重叠且极性相反（否定词不同）时疑似冲突。

    纯词面启发式，宁可漏报不误报；命中的条件会打「待澄清」标，由注入提示引导模型判断。
    """
    new_topics = _extract_condition_topics(text)
    new_neg = any(h in text for h in _NEGATION_HINTS)
    for entry in ledger.active_entries():
        old_topics = _extract_condition_topics(entry["condition"])
        if not (new_topics & old_topics):
            continue
        old_neg = any(h in entry["condition"] for h in _NEGATION_HINTS)
        if new_neg != old_neg:
            return True
    return False


def _inject_condition_hints(messages, ledger):
    """存在「待澄清」矛盾时，把一致性提示注入消息流（置于本轮用户消息之前）。

    返回新列表，不修改原列表；无待澄清项时原样返回。
    """
    pending = [e for e in ledger.entries if e["status"] == ConditionLedger.STATUS_CLARIFY]
    if not pending or len(messages) < 2:
        return messages
    active = ledger.active_entries()
    active_text = "；".join(e["condition"] for e in active[-5:]) or "无"
    hint = (
        "[条件一致性提示] 系统检测到用户新输入疑似与既有条件矛盾。"
        f"既有生效条件：{active_text}。待澄清新条件：{pending[-1]['condition']}。"
        "请按系统提示中的冲突规则处理：若确实无法共存，不要擅自取舍，先澄清；"
        "若判定可共存，正常回答并说明判定理由。"
    )
    return messages[:-1] + [HumanMessage(content=hint), messages[-1]]


def _fresh_messages() -> list:
    return [SystemMessage(content=SYSTEM_PROMPT)]


def _maybe_import_legacy() -> None:
    """旧版单文件 memory.json 存在时，询问用户是否导入为多会话格式。"""
    if not os.path.exists(LEGACY_MEMORY_FILE):
        return
    try:
        with open(LEGACY_MEMORY_FILE, encoding="utf-8") as f:
            data = json.load(f)
        from langchain_core.messages import messages_from_dict
        messages = messages_from_dict(data["messages"])
        assert type(messages[0]).__name__ == "SystemMessage"
    except Exception:
        print(f"提示: 旧记忆文件 {LEGACY_MEMORY_FILE} 无法读取，已跳过导入。")
        return

    answer = input(
        f"检测到旧版记忆文件 {LEGACY_MEMORY_FILE}（{len(messages) - 1} 条历史），是否导入为会话？[Y/n] "
    ).strip().lower()
    if answer in {"", "y", "yes"}:
        sid = _new_session_id()
        save_session(messages, sid)
        print(f"已导入为会话 {sid}。")
    if input("是否删除旧记忆文件？[y/N] ").strip().lower() in {"y", "yes"}:
        os.remove(LEGACY_MEMORY_FILE)
        print("已删除。")


def choose_session() -> tuple[str | None, list, "ConditionLedger"]:
    """启动菜单：列出历史会话，由用户决定恢复哪个或开始新对话。

    返回 (会话ID, 消息列表, 条件账本)；新对话时会话 ID 为 None（首次保存时再生成）。
    """
    _maybe_import_legacy()
    sessions = list_sessions()

    if sessions:
        print("\n可用会话（按最后保存时间排序）:")
        for i, s in enumerate(sessions, 1):
            stamp = time.strftime("%m-%d %H:%M", time.localtime(s["mtime"]))
            print(f"  {i}. [{stamp}] {s['id']}  {s['count']}条  | {s['preview']}")
        print("  n. 开始新对话")
        raw = input("选择要恢复的会话编号（直接回车=新对话）: ").strip().lower()
        if raw.isdigit() and 1 <= int(raw) <= len(sessions):
            chosen = sessions[int(raw) - 1]
            try:
                loaded, ledger = load_session(chosen["id"])
                return chosen["id"], loaded, ledger
            except (json.JSONDecodeError, ValueError, OSError) as e:
                print(f"警告: 会话 {chosen['id']} 损坏（{e}），请选择其他会话或开始新对话。")
                return None, _fresh_messages(), ConditionLedger()
        if raw == "n" or raw == "":
            return None, _fresh_messages(), ConditionLedger()
        print("无效选择，已开始新对话。")
        return None, _fresh_messages(), ConditionLedger()

    print("没有找到历史会话，开始新对话。")
    return None, _fresh_messages(), ConditionLedger()


def handle_command(
    prompt: str,
    config: dict,
    messages: list,
    ledger: "ConditionLedger | None" = None,
) -> bool:
    """处理斜杠命令。返回 True 表示 prompt 是命令且已处理。"""
    if prompt == "/clear":
        del messages[1:]
        print("已清空对话记忆。")
        return True
    if prompt == "/config":
        used = count_tokens(messages)
        print(
            f"当前配置: 模型窗口={config['window']}，token预算={config['max_tokens']}，"
            f"条数上限={config['max_messages']}\n"
            f"当前占用: {used}/{config['max_tokens']} tokens"
            f"（占模型窗口 {used}/{config['window']}），"
            f"消息 {len(messages) - 1} 条（不含系统提示词）"
        )
        return True
    if prompt.startswith("/window"):
        value = _read_int_command(prompt, config["window"], minimum=1024)
        if value is not None:
            config["window"] = value
            clamp_budget_to_window(config)
            trim_history(messages, config["max_tokens"], config["max_messages"])
            print(
                f"模型窗口已调整为 {value}，token 预算封顶为 {config['max_tokens']}，"
                f"当前占用 {count_tokens(messages)} tokens。"
            )
        return True
    if prompt.startswith("/tokens"):
        value = _read_int_command(prompt, config["max_tokens"], minimum=64)
        if value is not None:
            config["max_tokens"] = value
            clamp_budget_to_window(config)
            total = trim_history(messages, config["max_tokens"], config["max_messages"])
            print(f"已将 token 预算调整为 {value}（封顶后 {config['max_tokens']}），当前占用 {total} tokens。")
        return True
    if prompt.startswith("/msgs"):
        value = _read_int_command(prompt, config["max_messages"], minimum=2)
        if value is not None:
            config["max_messages"] = value
            trim_history(messages, config["max_tokens"], value)
            print(f"已将历史条数上限调整为 {value}。")
        return True
    if prompt == "/save":
        path = save_session(messages, config["session_id"], ledger)
        print(f"会话已保存: {path}")
        return True
    if prompt == "/load":
        new_id, loaded, new_ledger = choose_session()
        config["session_id"] = new_id
        messages[:] = loaded
        ledger.entries[:] = new_ledger.entries
        ledger._turn = new_ledger._turn
        return True
    if prompt == "/conds":
        if ledger is None:
            print("（本会话暂无条件记录）")
        else:
            print("条件账本（变更历史，最新在最下）:")
            print(ledger.summary())
        return True
    if prompt.startswith("/"):
        print("未知命令。可用命令: /clear、/config、/save、/load、/conds、/window <n>、/tokens <n>、/msgs <n>")
        return True
    return False


def main() -> None:
    model = build_model()

    # 可动态调整的上下文限制配置
    config = {
        "window": DEFAULT_MODEL_CONTEXT_WINDOW,
        "max_tokens": DEFAULT_MAX_CONTEXT_TOKENS,
        "max_messages": DEFAULT_MAX_MESSAGES,
        "session_id": None,  # 由启动菜单决定
    }
    clamp_budget_to_window(config)

    # 启动菜单：列出历史会话，用户决定恢复哪个或开始新对话
    session_id, messages, ledger = choose_session()
    config["session_id"] = session_id
    if session_id:
        print(f"已恢复会话 {session_id}（{len(messages) - 1} 条历史）。当前会话: {session_id}")

    print(
        "开始对话（exit 退出）\n"
        "命令: /clear 清空记忆 | /config 查看占用 | /save 保存记忆 | /load 加载记忆 | "
        f"/window <n> 模型窗口(当前{config['window']}) | "
        f"/tokens <n> 调整token预算(当前{config['max_tokens']}) | "
        f"/msgs <n> 调整条数上限(当前{config['max_messages']}) | /conds 条件变更历史\n"
        "回复为流式实时输出；Ctrl+C 可中断本次回复（不写入记忆）。"
    )
    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            break
        if handle_command(user_input, config, messages, ledger):
            if config["session_id"]:
                save_session(messages, config["session_id"], ledger)
            continue

        # ── 条件账本：每轮版本化记录 + 一致性检查 ──
        ledger.advance_turn()
        if any(m in user_input for m in ConditionLedger.EXPLICIT_MARKERS):
            ledger.explicit_override(user_input)
        elif _looks_like_conflict(user_input, ledger):
            ledger.flag_conflict(user_input)
        else:
            ledger.record(user_input)

        messages.append(HumanMessage(content=user_input))
        trim_history(messages, config["max_tokens"], config["max_messages"])
        # 注入提示仅用于本次模型调用，不写入持久化会话历史
        call_messages = _inject_condition_hints(messages, ledger)

        # 流式输出：chunk 经回调逐字打印，同时聚合为完整回复
        print("\n模型: ", end="", flush=True)
        chunks: list[AIMessageChunk] = []
        try:
            for chunk in model.stream(call_messages):
                chunks.append(chunk)
        except KeyboardInterrupt:
            print("\n（已中断本次回复）")
            messages.pop()  # 撤回这条未得到回复的用户消息
            continue
        response = sum(chunks[1:], chunks[0]) if chunks else AIMessage(content="")
        messages.append(AIMessage(content=response.content))
        print()  # 流式输出已打印完毕，换行
        # 首次对话时生成会话 ID 并保存；此后每轮自动落盘
        if not config["session_id"]:
            config["session_id"] = _new_session_id()
            print(f"新会话已创建: {config['session_id']}")
        save_session(messages, config["session_id"], ledger)

        print(
            f"\n[上下文占用: {count_tokens(messages)}/{config['max_tokens']} tokens"
            f"（模型窗口 {config['window']}），{len(messages) - 1} 条历史]"
        )

    # 退出前保存当前会话（覆盖 Ctrl+C 中断等场景）
    if config["session_id"]:
        save_session(messages, config["session_id"], ledger)
        print(f"对话结束，会话 {config['session_id']} 已保存，再见！")
    else:
        print("对话结束（未产生任何对话内容，未创建会话），再见！")


if __name__ == "__main__":
    main()
