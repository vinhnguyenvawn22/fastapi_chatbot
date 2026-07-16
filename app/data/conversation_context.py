from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ConversationContext:
    thread_id: str | None = None
    original_question: str | None = None
    standalone_question: str | None = None
    history: list[dict] = field(default_factory=list)
    rewrite_debug: dict = field(default_factory=dict)
    history_message_count: int = 0
    history_chars: int = 0
    user_message_id: str | None = None
    assistant_message_id: str | None = None


_context: ContextVar[ConversationContext] = ContextVar(
    "conversation_context", default=ConversationContext()
)


def set_conversation_context(value: ConversationContext):
    return _context.set(value)


def reset_conversation_context(token) -> None:
    _context.reset(token)


def get_conversation_context() -> ConversationContext:
    return _context.get()
