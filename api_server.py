# -*- coding: utf-8 -*-
"""对外 HTTP API：让外部系统通过 user_id 调用 GLM 对话，每个用户记忆独立。

复用 glm_demo 的模型构建、上下文裁剪与会话持久化逻辑，在其上叠加：
- 用户隔离：每个 user_id 的会话存为 sessions/api/<user_id>__<session_id>.json
- 会话索引：sessions/api/index.json 记录每个用户当前指向的会话
- 可选鉴权：设置环境变量 API_KEY 后，请求需携带 X-API-Key 头（/health 除外）
- 流式接口：POST /chat/stream 以 SSE 返回逐字增量

启动：python api_server.py（或 uvicorn api_server:app --port 8000）
详细用法见 API.md。
"""
import json
import os
import re
import threading
import time

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

import glm_demo

# ===== API 专用配置 =====
# API 会话与 CLI 会话分目录存放（glm_demo.list_sessions 只扫描其顶层，互不干扰）
glm_demo.DEFAULT_MEMORY_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "sessions", "api"
)

INDEX_PATH = os.path.join(glm_demo.DEFAULT_MEMORY_DIR, "index.json")
API_KEY = os.getenv("API_KEY", "").strip()  # 非空时启用鉴权

# 复用 glm_demo 的上下文限制默认值（环境变量 MODEL_CONTEXT_WINDOW /
# MAX_CONTEXT_TOKENS / MAX_MESSAGES 对 API 同样生效）
CONFIG = {
    "window": glm_demo.DEFAULT_MODEL_CONTEXT_WINDOW,
    "max_tokens": glm_demo.DEFAULT_MAX_CONTEXT_TOKENS,
    "max_messages": glm_demo.DEFAULT_MAX_MESSAGES,
}
glm_demo.clamp_budget_to_window(CONFIG)

# 模型实例：去掉 CLI 的逐字打印回调，避免服务端日志被回复内容刷屏
model = glm_demo.build_model()
model.callbacks = []

app = FastAPI(title="GLM 对话 API", version="1.0.0", description="按 user_id 隔离记忆的多轮对话接口")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== 并发控制：单进程内按 user_id 串行化，避免同用户并发写坏会话 =====
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()
_index_lock = threading.Lock()

_USER_RE = re.compile(r"[A-Za-z0-9_\-]{1,64}")


def _user_lock(user_id: str) -> threading.Lock:
    with _locks_guard:
        if user_id not in _locks:
            _locks[user_id] = threading.Lock()
        return _locks[user_id]


def _require_user(user_id: str) -> None:
    """user_id 用于拼文件路径，只允许安全的字符。"""
    if not _USER_RE.fullmatch(user_id):
        raise HTTPException(400, "user_id 只能包含字母、数字、下划线、连字符，长度 1-64")


def _session_file(user_id: str, session_id: str) -> str:
    # 双下划线分隔：user_id 不含下划线以外的冲突来源，且 user_id 可含下划线时也不会与前缀混淆
    return os.path.join(glm_demo.DEFAULT_MEMORY_DIR, f"{user_id}__{session_id}.json")


# ===== 用户 → 当前会话 索引 =====
def _load_index() -> dict:
    try:
        with open(INDEX_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_index(index: dict) -> None:
    os.makedirs(glm_demo.DEFAULT_MEMORY_DIR, exist_ok=True)
    tmp = INDEX_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    os.replace(tmp, INDEX_PATH)


def _dump_messages(messages: list) -> list[dict]:
    """LangChain 消息 → 可 JSON 序列化的结构（与 glm_demo.save_session 相同格式）。"""
    from langchain_core.messages import messages_to_dict
    return messages_to_dict(messages)


def _load_messages_file(user_id: str, session_id: str) -> tuple[list, "glm_demo.ConditionLedger"]:
    """读取 {user_id}__{session_id}.json；损坏抛 ValueError，由调用方处理。"""
    from langchain_core.messages import messages_from_dict
    with open(_session_file(user_id, session_id), encoding="utf-8") as f:
        data = json.load(f)
    messages = messages_from_dict(data["messages"])
    if not messages or type(messages[0]).__name__ != "SystemMessage":
        raise ValueError("缺少 SystemMessage")
    return messages, glm_demo.ConditionLedger.from_dict(data.get("conditions"))


def _save_messages_file(user_id: str, session_id: str, messages: list, ledger=None) -> str:
    """原子写入 {user_id}__{session_id}.json，返回文件路径。"""
    os.makedirs(glm_demo.DEFAULT_MEMORY_DIR, exist_ok=True)
    path = _session_file(user_id, session_id)
    data = {
        "version": 1,
        "session_id": session_id,
        "user_id": user_id,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "messages": _dump_messages(messages),
        "conditions": (ledger or glm_demo.ConditionLedger()).to_dict(),
    }
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)
    return path


def _get_current(user_id: str) -> str | None:
    return _load_index().get(user_id, {}).get("session_id")


def _set_current(user_id: str, session_id: str | None) -> None:
    with _index_lock:
        index = _load_index()
        if session_id is None:
            index.pop(user_id, None)
        else:
            index[user_id] = {"session_id": session_id, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        _save_index(index)


# ===== 对话核心（与 CLI 相同的 裁剪 → 调模型 → 入库 → 落盘 流程） =====
def _load_current_messages(user_id: str) -> tuple[str | None, list, "glm_demo.ConditionLedger"]:
    session_id = _get_current(user_id)
    if session_id and os.path.exists(_session_file(user_id, session_id)):
        try:
            messages, ledger = _load_messages_file(user_id, session_id)
            return session_id, messages, ledger
        except (json.JSONDecodeError, ValueError, OSError):
            pass  # 损坏则从头开始
    return None, glm_demo._fresh_messages(), glm_demo.ConditionLedger()


def _update_ledger(ledger: "glm_demo.ConditionLedger", text: str) -> None:
    """API 端与 CLI 相同的每轮账本更新：显式修正覆盖 / 矛盾打标 / 普通记录。"""
    ledger.advance_turn()
    if any(m in text for m in glm_demo.ConditionLedger.EXPLICIT_MARKERS):
        ledger.explicit_override(text)
    elif glm_demo._looks_like_conflict(text, ledger):
        ledger.flag_conflict(text)
    else:
        ledger.record(text)


def _friendly_messages(messages: list) -> list[dict]:
    role_map = {"SystemMessage": "system", "HumanMessage": "user", "AIMessage": "assistant"}
    return [{"role": role_map.get(type(m).__name__, type(m).__name__), "content": str(m.content)} for m in messages]


def _usage(messages: list) -> dict:
    return {
        "used_tokens": glm_demo.count_tokens(messages),
        "token_budget": CONFIG["max_tokens"],
        "model_window": CONFIG["window"],
        "history_count": len(messages) - 1,
    }


def _apply_auth(request: Request) -> None:
    if API_KEY and request.headers.get("X-API-Key") != API_KEY:
        raise HTTPException(401, "缺少或无效的 X-API-Key 请求头")


# ===== 请求/响应模型 =====
class ChatRequest(BaseModel):
    user_id: str = Field(..., examples=["user_001"])
    message: str = Field(..., min_length=1, max_length=32000, examples=["你好，介绍一下你自己"])


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    used_tokens: int
    token_budget: int
    model_window: int
    history_count: int


# ===== 接口 =====
@app.get("/health", tags=["系统"])
def health():
    """健康检查（无需鉴权）。"""
    return {"status": "ok", "model": "glm-5.3-flash", "api_sessions_dir": glm_demo.DEFAULT_MEMORY_DIR}


@app.post("/chat", response_model=ChatResponse, tags=["对话"])
def chat(req: ChatRequest, request: Request):
    """单轮对话：传 user_id 与 message，返回完整回复（自动带上该用户的历史记忆）。"""
    _apply_auth(request)
    _require_user(req.user_id)
    with _user_lock(req.user_id):
        session_id, messages, ledger = _load_current_messages(req.user_id)
        _update_ledger(ledger, req.message)
        messages.append(HumanMessage(content=req.message))
        glm_demo.trim_history(messages, CONFIG["max_tokens"], CONFIG["max_messages"])
        call_messages = glm_demo._inject_condition_hints(messages, ledger)
        try:
            reply = str(model.invoke(call_messages).content)
        except Exception:
            messages.pop()  # 撤回未得到回复的用户消息
            raise HTTPException(502, "调用模型失败，请稍后重试")
        if not reply:
            messages.pop()
            raise HTTPException(502, "模型返回了空回复")
        messages.append(AIMessage(content=reply))
        if session_id is None:
            session_id = glm_demo._new_session_id()
        _save_messages_file(req.user_id, session_id, messages, ledger)
        _set_current(req.user_id, session_id)
        usage = _usage(messages)
    return ChatResponse(session_id=session_id, reply=reply, **usage)


@app.post("/chat/stream", tags=["对话"])
def chat_stream(req: ChatRequest, request: Request):
    """流式对话：以 SSE（text/event-stream）返回。

    事件格式（JSON 行）：
    {"session_id": "..."}        开始（新会话时为 null，结束事件里给最终值）
    {"delta": "文字"}            逐段增量文本
    {"done": true, ...占用信息}  正常结束，完整回复已入记忆
    {"error": "..."}             出错（该轮不写入记忆）
    """
    _apply_auth(request)
    _require_user(req.user_id)

    def generate():
        with _user_lock(req.user_id):
            session_id, messages, ledger = _load_current_messages(req.user_id)
            _update_ledger(ledger, req.message)
            messages.append(HumanMessage(content=req.message))
            glm_demo.trim_history(messages, CONFIG["max_tokens"], CONFIG["max_messages"])
            call_messages = glm_demo._inject_condition_hints(messages, ledger)
            ok = False
            try:
                yield f"data: {json.dumps({'session_id': session_id}, ensure_ascii=False)}\n\n"
                chunks = []
                for chunk in model.stream(call_messages):
                    text = chunk.content if isinstance(chunk.content, str) else ""
                    if text:
                        chunks.append(chunk)
                        yield f"data: {json.dumps({'delta': text}, ensure_ascii=False)}\n\n"
                reply = "".join(str(c.content) for c in chunks) if chunks else ""
                if not reply:
                    yield f"data: {json.dumps({'error': '模型返回了空回复'}, ensure_ascii=False)}\n\n"
                    return
                messages.append(AIMessage(content=reply))
                if session_id is None:
                    session_id = glm_demo._new_session_id()
                _save_messages_file(req.user_id, session_id, messages, ledger)
                _set_current(req.user_id, session_id)
                ok = True
                usage = _usage(messages)
                done = {"done": True, "session_id": session_id, "reply": reply, **usage}
                yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': f'调用模型失败: {e}'}, ensure_ascii=False)}\n\n"
            finally:
                if not ok and messages and type(messages[-1]).__name__ == "HumanMessage":
                    messages.pop()  # 未完成的这一轮不写入记忆

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/sessions/{user_id}", tags=["会话管理"])
def list_user_sessions(user_id: str, request: Request):
    """列出该用户的所有 API 会话（含各自的消息预览），current 标记当前会话。"""
    _apply_auth(request)
    _require_user(user_id)
    items = []
    prefix = f"{user_id}__"
    for name in os.listdir(glm_demo.DEFAULT_MEMORY_DIR):
        if not name.endswith(".json") or not name.startswith(prefix):
            continue
        sid = name[len(prefix):-5]
        try:
            messages, _ledger = _load_messages_file(user_id, sid)
        except (json.JSONDecodeError, ValueError, OSError):
            continue  # 跳过损坏文件
        first_user = next((str(m.content) for m in messages if type(m).__name__ == "HumanMessage"), "")
        items.append({
            "session_id": sid,
            "history_count": len(messages) - 1,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(_session_file(user_id, sid)))),
            "preview": first_user[:30],
        })
    items.sort(key=lambda s: s["updated_at"], reverse=True)
    return {"user_id": user_id, "current": _get_current(user_id), "sessions": items}


class SwitchRequest(BaseModel):
    user_id: str = Field(..., examples=["user_001"])
    session_id: str = Field(..., examples=["20261021_124502_wkzd"])


@app.post("/sessions/switch", tags=["会话管理"])
def switch_session(req: SwitchRequest, request: Request):
    """把该用户的"当前会话"切换到指定的历史会话，后续 /chat 接着它继续聊。"""
    _apply_auth(request)
    _require_user(req.user_id)
    if not os.path.exists(_session_file(req.user_id, req.session_id)):
        raise HTTPException(404, f"会话 {req.session_id} 不存在（或不属于该用户）")
    _set_current(req.user_id, req.session_id)
    return {"user_id": req.user_id, "current": req.session_id}


@app.post("/sessions/new", tags=["会话管理"])
def new_session(req: ChatRequest, request: Request):
    """开新会话：清掉该用户的"当前会话"指针，下一次 /chat 自动创建新会话。"""
    _apply_auth(request)
    _require_user(req.user_id)
    _set_current(req.user_id, None)
    return {"user_id": req.user_id, "current": None}


@app.get("/history/{user_id}", tags=["会话管理"])
def get_history(user_id: str, request: Request):
    """查看该用户当前会话的完整消息历史（含角色与内容）。"""
    _apply_auth(request)
    _require_user(user_id)
    with _user_lock(user_id):
        session_id, messages, _ledger = _load_current_messages(user_id)
        if session_id is None:
            raise HTTPException(404, "该用户还没有会话，先调用 /chat 一次")
        return {"user_id": user_id, "session_id": session_id, "messages": _friendly_messages(messages)}


@app.delete("/sessions/{user_id}/{session_id}", tags=["会话管理"])
def delete_session(user_id: str, session_id: str, request: Request):
    """删除该用户的一个会话文件；若删的是当前会话，同时清掉指针。"""
    _apply_auth(request)
    _require_user(user_id)
    path = _session_file(user_id, session_id)
    if not os.path.exists(path):
        raise HTTPException(404, f"会话 {session_id} 不存在（或不属于该用户）")
    with _user_lock(user_id):
        os.remove(path)
        if _get_current(user_id) == session_id:
            _set_current(user_id, None)
    return {"deleted": session_id}


@app.exception_handler(HTTPException)
def http_error(request: Request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
