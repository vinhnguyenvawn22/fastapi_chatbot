from app.data.elasticsearch_client import normalize_text


EXPLICIT_SV_TERMS = ("sinh vien", "nguoi hoc", "em la sinh vien")
EXPLICIT_CBGV_TERMS = ("giang vien", "can bo", "cbgv", "thay co", "toi la thay", "toi la co")
SV_BUSINESS_TERMS = (
    "xem diem", "tra cuu diem", "ket qua hoc tap", "lich hoc",
    "thoi khoa bieu", "hoc phi", "phuc khao", "dang ky hoc phan",
    "cham lai bai thi", "xem lai diem thi", "khieu nai diem",
    "diem thi sai", "gui yeu cau phuc khao", "don phuc khao",
    "ket qua bai thi", "thi lai", "hoan thi",
    "dang ky thi lai", "dang ki thi lai", "huy dang ky thi lai", "huy dang ki thi lai",
    "diem danh", "tra cuu diem danh", "chuyen can", "diem chuyen can",
    "so buoi vang", "so tiet vang", "ty le vang", "nghi co phep",
    "nghi khong phep", "ren luyen", "chuong trinh dao tao",
)
CBGV_BUSINESS_TERMS = (
    "lich day", "khoi luong giang day", "cong tac giang vien", "coi thi",
    "cham thi", "khoi luong cong tac", "khoi luong", "muon thiet bi phong hoc",
    "nhan su giang vien", "khoi luong coi thi", "khoi luong cham thi",
    "lop hoc phan giang vien", "lich coi thi", "minh chung kiem dinh",
    "ho so thu tuc hanh chinh", "muon thiet bi", "bao hong thiet bi",
)
PROCEDURE_SIGNALS = (
    "xem o dau", "o dau", "vao dau", "bam vao dau", "thuc hien the nao",
    "cac buoc", "duong dan", "link", "man hinh", "chuc nang", "tra cuu",
    "dang nhap", "dang ky tren he thong", "truy cap", "chon muc nao",
    "xem diem", "xem lich hoc", "thoi khoa bieu xem", "xem lop hoc phan",
    "cach xem", "cach tra cuu", "cach truy cap", "cach lam",
    "lam the nao de", "chon hoc ky", "theo hoc ky", "xem chi tiet",
    "truy cap nhanh", "lam the nao", "cach gui", "cach thuc hien",
    "gui yeu cau", "nop don", "thuc hien phuc khao", "huong dan",
    "thi lai", "hoan thi",
    "dang ky thi lai", "dang ki thi lai", "huy dang ky thi lai", "huy dang ki thi lai",
)
POLICY_SIGNALS = (
    "quy dinh", "quy che", "quyet dinh", "thong bao", "van ban",
    "dieu kien", "doi tuong", "thoi han", "duoc phep", "co bat buoc",
    "trach nhiem", "quyen", "nghia vu", "theo dieu", "theo muc",
    "theo chuong", "theo van ban", "co bi", "bi cam thi", "cam thi",
    "khong duoc thi", "diem chuyen can", "nghi hoc khong phep",
    "nghi hoc co phep", "nghi hoc tren", "so tiet vang", "ty le vang",
)


def _hits(text: str, terms) -> list[str]:
    return [term for term in terms if term in text]


def _explicit_audience(text: str) -> tuple[str, dict]:
    normalized = normalize_text(text)
    sv = _hits(normalized, EXPLICIT_SV_TERMS)
    cbgv = _hits(normalized, EXPLICIT_CBGV_TERMS)
    if sv and cbgv:
        audience = "mixed"
    elif sv:
        audience = "sv"
    elif cbgv:
        audience = "cbgv"
    else:
        audience = "unknown"
    return audience, {"sv": sv, "cbgv": cbgv}


def analyze_query_context(question: str, history: list[dict] | None = None) -> dict:
    normalized = normalize_text(question)
    audience, current_signals = _explicit_audience(question)
    audience_source = "explicit_question" if audience != "unknown" else "unknown"
    history_signals = {}

    if audience == "unknown":
        for message in reversed(history or []):
            if message.get("role") != "user":
                continue
            history_audience, signals = _explicit_audience(message.get("content", ""))
            if history_audience != "unknown":
                audience = history_audience
                audience_source = "conversation_history"
                history_signals = signals
                break

    inferred_sv = _hits(normalized, SV_BUSINESS_TERMS)
    inferred_cbgv = _hits(normalized, CBGV_BUSINESS_TERMS)
    if audience == "unknown" and (inferred_sv or inferred_cbgv):
        audience = "mixed" if inferred_sv and inferred_cbgv else "sv" if inferred_sv else "cbgv"
        audience_source = "business_inference"
    if audience == "unknown":
        audience = "sv"
        audience_source = "default_student"

    procedure = _hits(normalized, PROCEDURE_SIGNALS)
    policy = _hits(normalized, POLICY_SIGNALS)
    if procedure and policy:
        information_need = "mixed"
    elif procedure:
        information_need = "procedure_ui"
    elif policy:
        information_need = "policy_document"
    else:
        information_need = "unknown"

    confidence = {
        "explicit_question": 1.0,
        "conversation_history": 0.9,
        "business_inference": 0.7,
        "default_student": 0.35,
        "unknown": 0.0,
    }[audience_source]
    return {
        "audience_hint": audience,
        "audience_source": audience_source,
        "audience_confidence": confidence,
        "audience_signals": {
            "current": current_signals,
            "history": history_signals,
            "inferred_sv": inferred_sv,
            "inferred_cbgv": inferred_cbgv,
        },
        "information_need": information_need,
        "information_need_signals": {"procedure": procedure, "policy": policy},
    }
