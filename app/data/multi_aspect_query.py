import re
from typing import Any

from app.data.elasticsearch_client import get_keywords, normalize_text


MAX_ASPECTS = 5
DOCS_PER_ASPECT = 2

_CLAUSE_SEPARATOR = re.compile(
    r"\s*(?:[;?\n]+|,\s*(?:đồng thời|dong thoi|ngoài ra|ngoai ra|bên cạnh đó|ben canh do)\s+|"
    r"\b(?:và|va|đồng thời|dong thoi|ngoài ra|ngoai ra|bên cạnh đó|ben canh do)\b)\s*",
    flags=re.IGNORECASE,
)
_INFORMATION_NEED_MARKERS = (
    "la gi",
    "nhu the nao",
    "the nao",
    "bao nhieu",
    "may",
    "o dau",
    "khi nao",
    "vi sao",
    "tai sao",
    "dieu kien",
    "tieu chi",
    "quy dinh",
    "quy trinh",
    "thu tuc",
    "ho so",
    "cach",
    "lam sao",
    "lam the nao",
    "co duoc",
    "co can",
    "can gi",
    "can nhung gi",
    "gom nhung gi",
    "bao gom",
    "muc nao",
    "truong hop nao",
    "ai",
)
_ACTION_MARKERS = (
    "dang ky",
    "dang ki",
    "bao hong",
    "hoan",
    "huy",
    "rut",
    "nop",
    "gui",
    "tim",
    "xem",
    "tra cuu",
    "danh gia",
    "phe duyet",
    "cap",
    "doi",
    "xin",
    "kiem tra",
    "theo doi",
)
_SHARED_NEED_PREFIX = re.compile(
    r"^\s*(làm sao để|lam sao de|làm thế nào để|lam the nao de|"
    r"cách thức|cach thuc|cách|cach|thủ tục|thu tuc)\s+",
    flags=re.IGNORECASE,
)
_SHAREABLE_LEADING_ACTIONS = (
    "xem",
    "kiem tra",
    "tra cuu",
    "theo doi",
)
_COLLECTIVE_METRIC_MARKERS = (
    "so gio",
    "so tiet",
    "so luong",
    "tong so",
    "khoi luong",
)
_WEAK_WORDS = {
    "toi",
    "minh",
    "ban",
    "cho",
    "ve",
    "thi",
    "la",
    "gi",
    "co",
    "duoc",
    "can",
    "nhung",
    "cac",
    "mot",
}
_ELLIPTICAL_PREFIXES = (
    "cach ",
    "thu tuc ",
    "ho so ",
    "le phi ",
    "duoc ",
    "lam sao ",
    "lam the nao ",
    "sinh vien nop ",
    "thoi han ",
    "can ",
    "nop ",
    "phai nop ",
    "gui ",
    "xem ",
    "kiem tra ",
    "ket qua ",
    "tinh trang ",
    "trang thai ",
)
_SEMANTIC_GENERIC_KEYWORDS = {
    "sinh", "vien", "dieu", "kien", "truong", "hop", "nhu", "the", "nao",
    "cach", "lam", "sao", "duoc", "can", "nhung", "gom", "khi",
    "thoi", "gian", "toi", "minh", "la", "gi", "co", "khong", "may",
    "nhieu", "tin", "chi", "diem", "muc", "loai",
    "dang", "ky", "nop", "gui", "xem", "kiem", "tra", "thuc", "hien",
    "thu", "tuc", "ho", "so", "xin", "le", "phi", "dau", "da",
    "ngu", "lien", "quan", "phai", "tren",
}
_BROAD_ENTITY_KEYWORDS = {
    "yeu", "cau", "thong", "trang", "thai", "xu", "ly",
    "thong", "tin", "chuc", "nang", "noi", "dung",
}
_AUDIENCE_ROLES = {
    "sinh vien": "student",
    "hoc vien": "student",
    "nguoi hoc": "student",
    "giang vien": "staff",
    "can bo": "staff",
    "cbgv": "staff",
    "thay co": "staff",
}
_SHARED_PREDICATE_START = re.compile(
    r"^(?P<subjects>.+?)\s+"
    r"(?P<predicate>(?:xem|tra cứu|tra cuu|kiểm tra|kiem tra|tìm|tim|"
    r"gửi|gui|đăng ký|dang ky|dang ki|theo dõi|theo doi|"
    r"có thể|co the|có được|co duoc|được|duoc|có|co)\b.+)$",
    flags=re.IGNORECASE,
)
_COORDINATED_SUBJECT_SEPARATOR = re.compile(
    r"\s*(?:,|(?:\b(?:và|va|cùng với|cung voi)\b))\s*",
    flags=re.IGNORECASE,
)
_ROLE_PAIRED_TERMS = (
    {
        "pattern": re.compile(
            r"\bhọc tập\s*/\s*công tác\b",
            flags=re.IGNORECASE,
        ),
        "student": "học tập",
        "staff": "công tác",
    },
    {
        "pattern": re.compile(
            r"\bcông tác\s*/\s*học tập\b",
            flags=re.IGNORECASE,
        ),
        "student": "học tập",
        "staff": "công tác",
    },
    {
        "pattern": re.compile(
            r"\bhọc\s*/\s*dạy\b",
            flags=re.IGNORECASE,
        ),
        "student": "học",
        "staff": "dạy",
    },
    {
        "pattern": re.compile(
            r"\bdạy\s*/\s*học\b",
            flags=re.IGNORECASE,
        ),
        "student": "học",
        "staff": "dạy",
    },
)
_AUDIENCE_TITLES = {
    "sinh vien": "Sinh viên",
    "hoc vien": "Học viên",
    "nguoi hoc": "Người học",
    "giang vien": "Giảng viên",
    "can bo": "Cán bộ",
    "cbgv": "Cán bộ, giảng viên",
    "thay co": "Thầy cô",
}


def _content_keywords(text: str) -> list[str]:
    tokens = list(get_keywords(text))
    tokens.extend(re.findall(r"[a-z0-9]+", normalize_text(text)))
    return list(dict.fromkeys(
        token
        for token in tokens
        if token not in _WEAK_WORDS and len(token) > 1
    ))


def _has_information_need(clause: str) -> bool:
    normalized = normalize_text(clause)
    keywords = _content_keywords(clause)
    if len(keywords) < 2:
        return False
    if any(marker in normalized for marker in _INFORMATION_NEED_MARKERS):
        return True
    return (
        len(keywords) >= 3
        and any(marker in normalized for marker in _ACTION_MARKERS)
    )


def _deduplicate_clauses(clauses: list[str]) -> list[str]:
    deduplicated = []
    seen = set()
    for clause in clauses:
        cleaned = re.sub(r"\s+", " ", clause).strip(" ,.;?")
        key = normalize_text(cleaned)
        if not key or key in seen:
            continue
        seen.add(key)
        deduplicated.append(cleaned)
    return deduplicated


def _audience_role(text: str) -> str | None:
    return _AUDIENCE_ROLES.get(normalize_text(text))


def _specialize_predicate_for_audience(
    predicate: str,
    audience_role: str,
) -> str:
    specialized = predicate
    for pair in _ROLE_PAIRED_TERMS:
        specialized = pair["pattern"].sub(
            pair[audience_role],
            specialized,
        )
    return specialized


def _expand_coordinated_audience_need(
    question: str,
) -> tuple[list[str], list[str]] | None:
    """Duplicate a shared information need for each explicitly named audience."""
    match = _SHARED_PREDICATE_START.match(question.strip(" .;?"))
    if not match:
        return None

    subjects = _deduplicate_clauses(
        _COORDINATED_SUBJECT_SEPARATOR.split(match.group("subjects"))
    )
    if len(subjects) < 2 or len(subjects) > MAX_ASPECTS:
        return None

    roles = [_audience_role(subject) for subject in subjects]
    if any(role is None for role in roles) or len(set(roles)) < 2:
        return None

    predicate = match.group("predicate").strip()
    expanded = [
        f"{subject} {_specialize_predicate_for_audience(predicate, role)}"
        for subject, role in zip(subjects, roles)
    ]
    if not all(_has_information_need(clause) for clause in expanded):
        return None
    return _deduplicate_clauses(expanded), subjects


def _presentation_title(
    clause: str,
    audience_scope: str | None,
    prefer_audience: bool,
) -> str:
    if prefer_audience and audience_scope:
        return _AUDIENCE_TITLES.get(audience_scope, audience_scope.capitalize())

    title = re.sub(r"\s+", " ", str(clause or "")).strip(" .;:?")
    title = re.sub(
        r"\s+(?:là gì|la gi|như thế nào|nhu the nao|thế nào|the nao|"
        r"ở đâu|o dau|khi nào|khi nao|cần những gì|can nhung gi|"
        r"gồm những gì|gom nhung gi)\s*$",
        "",
        title,
        flags=re.IGNORECASE,
    ).strip(" .;:?")
    if not title:
        return "Thông tin cần biết"
    title = title[0].upper() + title[1:]
    return title[:100].rstrip()


def _inherit_shared_need_for_coordinated_actions(
    clauses: list[str],
) -> tuple[list[str], list[str]]:
    """Expand `how to A and B` so B becomes an independently answerable action."""
    if len(clauses) < 2:
        return clauses, clauses

    prefix_match = _SHARED_NEED_PREFIX.match(clauses[0])
    if not prefix_match:
        return clauses, clauses

    prefix = prefix_match.group(1)
    first_remainder = normalize_text(
        clauses[0][prefix_match.end():]
    )
    shared_action = next(
        (
            action
            for action in _SHAREABLE_LEADING_ACTIONS
            if first_remainder.startswith(f"{action} ")
        ),
        None,
    )
    expanded = [clauses[0]]
    for clause in clauses[1:]:
        normalized_clause = normalize_text(clause)
        is_action = any(marker in normalized_clause for marker in _ACTION_MARKERS)
        if not _has_information_need(clause) and is_action:
            expanded.append(f"{prefix} {clause}")
        elif not _has_information_need(clause) and shared_action:
            expanded.append(f"{prefix} {shared_action} {clause}")
        else:
            expanded.append(clause)
    return expanded, clauses


def _is_single_collective_metric_question(
    question: str,
    clauses: list[str],
) -> bool:
    """Keep `metric of A and B + one final question` as a single information need."""
    if len(clauses) != 2 or ";" in question or question.count("?") > 1:
        return False

    first = normalize_text(clauses[0])
    second = normalize_text(clauses[1])
    if not any(marker in first for marker in _COLLECTIVE_METRIC_MARKERS):
        return False
    if any(marker in first for marker in _INFORMATION_NEED_MARKERS):
        return False
    if not any(marker in second for marker in _INFORMATION_NEED_MARKERS):
        return False

    first_tokens = re.findall(r"[a-z0-9]+", first)
    second_tokens = set(re.findall(r"[a-z0-9]+", second))
    return bool(first_tokens and first_tokens[-1] in second_tokens)


def _retrieval_query_for_clause(
    clause: str,
    previous_clause: str | None,
) -> tuple[str, bool]:
    """Add prior context only when a clause is grammatically elliptical."""
    if not previous_clause:
        return clause, False

    normalized_clause = normalize_text(clause)
    if not normalized_clause.startswith(_ELLIPTICAL_PREFIXES):
        return clause, False

    current_keywords = _specific_topic_tokens(clause)
    if (
        any(
            normalized_clause.startswith(f"cach {action} ")
            for action in _SHAREABLE_LEADING_ACTIONS
        )
        and current_keywords
    ):
        return clause, False
    if len(current_keywords) >= 2:
        return clause, False

    previous_keywords = _specific_topic_tokens(previous_clause)
    if not previous_keywords:
        return clause, False

    topic_context = " ".join(previous_keywords)
    return f"{clause}. Ngữ cảnh chủ đề: {topic_context}", True


def _specific_topic_tokens(text: str) -> list[str]:
    normalized = normalize_text(text)
    raw_keywords = re.findall(r"[a-z0-9]+", normalized)
    keywords = [
        keyword
        for keyword in raw_keywords
        if len(keyword) >= 3 or keyword == "thi"
    ]
    selected = []
    for index, keyword in enumerate(keywords):
        previous_keyword = keywords[index - 1] if index else None
        next_keyword = keywords[index + 1] if index + 1 < len(keywords) else None
        if keyword == "bao" and next_keyword == "nhieu":
            continue
        if keyword == "canh" and previous_keyword == "ngu":
            continue
        if keyword == "thi" and not (
            previous_keyword in {"hoan", "lich", "qua", "ket"}
            or next_keyword in {"lai", "hoan"}
        ):
            continue
        preserve_compound_term = (
            keyword == "loai"
            and (previous_keyword == "xep" or next_keyword == "gioi")
        ) or (
            keyword == "diem"
            and next_keyword == "danh"
        ) or (
            keyword == "can"
            and previous_keyword == "chuyen"
        ) or (
            keyword == "truong"
            and previous_keyword == "chuyen"
        )
        if preserve_compound_term or (
            keyword not in _SEMANTIC_GENERIC_KEYWORDS
            and keyword not in _BROAD_ENTITY_KEYWORDS
        ):
            selected.append(keyword)
    return list(dict.fromkeys(selected))


def _specific_topic_keywords(text: str) -> set[str]:
    return set(_specific_topic_tokens(text))


def _specific_topic_bigrams(text: str, keywords: set[str]) -> set[str]:
    raw_tokens = re.findall(r"[a-z0-9]+", normalize_text(text))
    return {
        f"{raw_tokens[index]} {raw_tokens[index + 1]}"
        for index in range(len(raw_tokens) - 1)
        if (
            raw_tokens[index] in keywords
            and raw_tokens[index + 1] in keywords
        )
    }


def _focused_retrieval_query(clause: str, retrieval_query: str) -> str | None:
    primary_question = normalize_text(clause)
    query_parts = re.split(
        r"\.\s*ngu canh (?:lien quan|chu de):\s*",
        normalize_text(retrieval_query),
        maxsplit=1,
    )
    topic_tokens = _specific_topic_tokens(query_parts[-1])
    if not topic_tokens:
        topic_tokens = _specific_topic_tokens(primary_question)

    hints = []
    if any(
        marker in primary_question
        for marker in (
            "cach dang ky", "thu tuc", "lam sao", "lam the nao",
            "thuc hien nhu the nao", "cach gui", "cach nop",
        )
    ):
        hints.extend(("huong dan", "thu tuc", "don", "phu luc"))
        if "dang ky" in primary_question:
            hints.append("dang ky")
    if "ho so" in primary_question:
        if "thu tuc" not in hints:
            hints.extend(
                ("ho so", "don", "giay to", "bieu mau", "phu luc", "minh chung")
            )
    if any(
        marker in primary_question
        for marker in ("dieu kien", "khi nao", "truong hop nao")
    ):
        hints.extend(("dieu kien", "truong hop", "neu", "phai"))
    if "o dau" in primary_question:
        asks_lookup_location = any(
            marker in primary_question
            for marker in ("xem", "tra cuu", "kiem tra", "theo doi")
        )
        if asks_lookup_location:
            hints.extend(
                ("man hinh", "chuc nang", "duong dan", "truy cap", "he thong")
            )
        else:
            hints.extend(("noi nop", "don vi tiep nhan", "phong", "bo phan"))
    if "thoi han" in primary_question:
        hints.extend(("thoi han", "cham nhat", "trong vong"))
    if any(
        marker in primary_question
        for marker in ("bao nhieu", "toi da", "toi thieu")
    ):
        hints.extend(("muc", "so luong", "toi da", "toi thieu"))

    focused_terms = list(dict.fromkeys((*hints, *topic_tokens)))
    if not hints or not topic_tokens:
        return None
    return " ".join(focused_terms)


def _guidance_retrieval_query(clause: str, retrieval_query: str) -> str | None:
    primary_question = normalize_text(clause)
    if not any(
        marker in primary_question
        for marker in (
            "cach", "lam sao", "lam the nao", "thu tuc",
            "dang ky", "gui", "nop", "xem", "kiem tra", "tra cuu",
        )
    ):
        return None

    query_parts = re.split(
        r"\.\s*ngu canh (?:lien quan|chu de):\s*",
        normalize_text(retrieval_query),
        maxsplit=1,
    )
    topic_tokens = list(dict.fromkeys(
        token
        for part in (primary_question, *query_parts[1:])
        for token in _specific_topic_tokens(part)
    ))
    if not topic_tokens:
        return None

    action_terms = []
    for marker in (
        "dang ky", "gui", "nop", "xem trang thai", "kiem tra",
        "tra cuu", "bao hong", "hoan",
    ):
        if marker in primary_question:
            action_terms.extend(marker.split())
    return " ".join(dict.fromkeys(
        ("huong", "dan", *action_terms, *topic_tokens)
    ))


def _submission_retrieval_query(clause: str, retrieval_query: str) -> str | None:
    primary_question = normalize_text(clause)
    if not any(
        marker in primary_question
        for marker in ("ho so", "dang ky", "nop", "xin ")
    ):
        return None

    query_parts = re.split(
        r"\.\s*ngu canh (?:lien quan|chu de):\s*",
        normalize_text(retrieval_query),
        maxsplit=1,
    )
    topic_tokens = list(dict.fromkeys(
        token
        for part in (primary_question, *query_parts[1:])
        for token in _specific_topic_tokens(part)
    ))
    if not topic_tokens:
        return None
    return " ".join(("huong", "dan", "nop", *topic_tokens))


def _semantic_alias_retrieval_query(clause: str) -> str | None:
    normalized = normalize_text(clause)
    if "gpa" in normalized:
        return "diem trung binh tich luy duoc tinh nhu the nao"
    if "trang thai" in normalized and "phieu" in normalized:
        return "xem trang thai yeu cau danh gia thu tuc hanh chinh"
    if "diem danh" in normalized:
        return "huong dan xem diem danh"
    if "diem chu" in normalized and any(
        marker in normalized for marker in ("quy doi", "thang diem")
    ):
        return "tinh diem trung binh tich luy quy doi diem chu thang diem 4"
    return None


def decompose_multi_aspect_query(question: str) -> dict[str, Any]:
    """Detect independent information needs without using an LLM."""
    cleaned_question = re.sub(r"\s+", " ", str(question or "")).strip()
    if not cleaned_question:
        return {
            "is_multi_aspect": False,
            "method": "structural_rules_v1",
            "reason": "empty_question",
            "aspects": [],
        }

    coordinated_audience = _expand_coordinated_audience_need(cleaned_question)
    if coordinated_audience:
        clauses, raw_clauses = coordinated_audience
    else:
        clauses = _deduplicate_clauses(_CLAUSE_SEPARATOR.split(cleaned_question))
        raw_clauses = list(clauses)
    if _is_single_collective_metric_question(cleaned_question, clauses):
        return {
            "is_multi_aspect": False,
            "method": "structural_rules_v1",
            "reason": "single_collective_metric_need",
            "candidate_clauses": clauses,
            "aspects": [],
        }
    if not coordinated_audience:
        clauses, raw_clauses = _inherit_shared_need_for_coordinated_actions(
            clauses
        )
    valid_clauses = [clause for clause in clauses if _has_information_need(clause)]
    if len(clauses) < 2 or len(valid_clauses) != len(clauses):
        return {
            "is_multi_aspect": False,
            "method": "structural_rules_v1",
            "reason": "no_independent_information_needs",
            "candidate_clauses": clauses,
            "raw_candidate_clauses": raw_clauses,
            "aspects": [],
        }

    aspects = []
    normalized_full_question = normalize_text(cleaned_question)
    audience_terms = tuple(_AUDIENCE_ROLES)
    full_audiences = [
        audience
        for audience in audience_terms
        if audience in normalized_full_question
    ]
    shared_audience_scope = (
        full_audiences[0]
        if len(full_audiences) == 1
        else None
    )
    previous_topic_clause = None
    for index, clause in enumerate(valid_clauses[:MAX_ASPECTS], start=1):
        normalized_clause = normalize_text(clause)
        audience_scope = next(
            (
                audience
                for audience in audience_terms
                if audience in normalized_clause
            ),
            shared_audience_scope,
        )
        retrieval_query, context_inherited = _retrieval_query_for_clause(
            clause,
            previous_topic_clause,
        )
        focused_query = _focused_retrieval_query(clause, retrieval_query)
        guidance_query = _guidance_retrieval_query(clause, retrieval_query)
        submission_query = _submission_retrieval_query(clause, retrieval_query)
        alias_query = _semantic_alias_retrieval_query(clause)
        if audience_scope:
            focused_query = (
                f"{audience_scope} {focused_query}"
                if focused_query
                else None
            )
            guidance_query = (
                f"{audience_scope} {guidance_query}"
                if guidance_query
                else None
            )
            submission_query = (
                f"{audience_scope} {submission_query}"
                if submission_query
                else None
            )
        retrieval_queries = list(dict.fromkeys(
            query
            for query in (
                retrieval_query,
                focused_query,
                guidance_query,
                submission_query,
                alias_query,
            )
            if query
        ))
        use_alias_for_filter = (
            "gpa" in normalize_text(clause)
            or (
                "diem chu" in normalize_text(clause)
                and "quy doi" in normalize_text(clause)
            )
            or (
                "trang thai" in normalize_text(clause)
                and "phieu" in normalize_text(clause)
            )
        )
        semantic_query = (
            alias_query
            if alias_query and use_alias_for_filter
            else retrieval_query
        )
        if audience_scope:
            semantic_query = f"{semantic_query}. Đối tượng: {audience_scope}"
        aspects.append({
            "aspect_id": f"aspect_{index}",
            "question": clause,
            "presentation_title": _presentation_title(
                clause,
                audience_scope,
                prefer_audience=len(full_audiences) > 1,
            ),
            "retrieval_query": retrieval_query,
            "retrieval_queries": retrieval_queries,
            "focused_retrieval_query": focused_query,
            "guidance_retrieval_query": guidance_query,
            "submission_retrieval_query": submission_query,
            "alias_retrieval_query": alias_query,
            "audience_scope": audience_scope,
            "semantic_query": semantic_query,
            "context_inherited": context_inherited,
            "original_question": cleaned_question,
            "keywords": _content_keywords(retrieval_query),
        })
        if _specific_topic_tokens(clause):
            previous_topic_clause = clause

    specific_topic_count = len(_specific_topic_keywords(cleaned_question))
    needs_clarification = specific_topic_count < 2

    return {
        "is_multi_aspect": len(aspects) >= 2,
        "method": "structural_rules_v1",
        "reason": "independent_information_needs_detected",
        "needs_clarification": needs_clarification,
        "clarification_reason": (
            "missing_specific_request_topic"
            if needs_clarification
            else None
        ),
        "candidate_clauses": clauses,
        "raw_candidate_clauses": raw_clauses,
        "aspects": aspects,
    }


def filter_semantic_aspect_docs(
    aspect_question: str,
    docs: list[dict],
) -> tuple[list[dict], str]:
    """Filter chunks by topic and by the kind of information being requested."""
    normalized_question = normalize_text(aspect_question)
    semantic_body = re.sub(
        r"\.\s*doi tuong:\s*(?:sinh vien|hoc vien|nguoi hoc|"
        r"giang vien|can bo|cbgv|thay co)\s*$",
        "",
        normalized_question,
    )
    query_parts = re.split(
        r"\.\s*ngu canh (?:lien quan|chu de):\s*",
        semantic_body,
        maxsplit=1,
    )
    primary_question = query_parts[0]
    topic_groups = []
    for part in query_parts:
        keywords = _specific_topic_tokens(part)
        if keywords:
            keyword_set = set(keywords)
            topic_groups.append({
                "keywords": keyword_set,
                "bigrams": _specific_topic_bigrams(part, keyword_set),
            })

    accepted = []
    for doc in docs or []:
        title_text = normalize_text(
            " ".join(
                str(doc.get(field) or "")
                for field in ("title", "heading", "section_path")
            )
        )
        searchable = normalize_text(
            " ".join(
                str(doc.get(field) or "")
                for field in (
                    "title", "heading", "section_path", "content",
                    "doc_name", "relative_path",
                )
            )
        )
        topic_mismatch = False
        topic_match_total = 0
        for topic_group in topic_groups:
            topic_keywords = topic_group["keywords"]
            matched_count = sum(
                bool(re.search(
                    rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])",
                    searchable,
                ))
                for keyword in topic_keywords
            )
            required_count = min(2, len(topic_keywords))
            if matched_count < required_count:
                topic_mismatch = True
                break
            bigrams = topic_group["bigrams"]
            if bigrams and not any(
                re.search(
                    rf"(?<![a-z0-9]){re.escape(bigram)}(?![a-z0-9])",
                    searchable,
                )
                for bigram in bigrams
            ):
                topic_mismatch = True
                break
            topic_match_total += matched_count
        if topic_mismatch:
            continue

        asks_procedure = any(
            marker in primary_question
            for marker in (
                "cach dang ky", "thu tuc", "lam sao", "lam the nao",
                "thuc hien nhu the nao", "cach gui", "cach nop",
            )
        )
        asks_dossier = "ho so" in primary_question
        if asks_dossier:
            dossier_hits = sum(
                marker in searchable
                for marker in (
                    "ho so", "don ", "giay to", "bieu mau", "minh chung",
                    "ban sao", "ban chinh", "chung chi", "xac nhan",
                    "phu luc", "theo mau", "mau ",
                )
            )
            procedure_dossier = (
                asks_procedure
                and "thu tuc" in searchable
                and dossier_hits >= 2
            )
            if (
                "ho so" not in searchable
                and dossier_hits < 3
                and not procedure_dossier
            ):
                continue
            asks_dossier_contents = any(
                marker in primary_question
                for marker in ("gom nhung gi", "can nhung gi", "bao gom")
            )
            minimum_hits = 2 if asks_dossier_contents else 1
            if dossier_hits < minimum_hits:
                continue

        asks_student_scope = (
            "sinh vien" in primary_question
            or "hoc vien" in primary_question
            or "nguoi hoc" in primary_question
            or "doi tuong: sinh vien" in normalized_question
            or "doi tuong: hoc vien" in normalized_question
            or "doi tuong: nguoi hoc" in normalized_question
        )
        if asks_student_scope and not any(
            marker in searchable
            for marker in ("sinh vien", "hoc vien", "nguoi hoc")
        ):
            continue

        asks_staff_scope = any(
            marker in primary_question
            for marker in ("giang vien", "can bo", "cbgv", "thay co")
        ) or any(
            f"doi tuong: {marker}" in normalized_question
            for marker in ("giang vien", "can bo", "cbgv", "thay co")
        )
        if asks_staff_scope and not any(
            marker in searchable
            for marker in ("giang vien", "can bo", "cbgv", "thay co")
        ):
            continue

        asks_lookup_location = (
            "o dau" in primary_question
            and any(
                marker in primary_question
                for marker in ("xem", "tra cuu", "kiem tra", "theo doi")
            )
        )
        if asks_lookup_location:
            document_type = normalize_text(doc.get("document_type", ""))
            section_type = normalize_text(doc.get("section_type", ""))
            source_identity = normalize_text(
                " ".join(
                    str(doc.get(field) or "")
                    for field in (
                        "doc_name", "relative_path", "ten_van_ban",
                        "don_vi_ban_hanh",
                    )
                )
            )
            is_guidance_source = (
                document_type == "business_document"
                or section_type == "business_section"
                or "web support" in source_identity
            )
            ui_evidence_hits = sum(
                marker in searchable
                for marker in (
                    "man ", "man hinh", "chuc nang", "tra cuu",
                    "truy cap", "duong dan", "support.uneti",
                    "website", "click", "nhan nut",
                )
            )
            if not is_guidance_source or ui_evidence_hits < 1:
                continue

        if asks_procedure:
            action_hits = sum(
                marker in searchable
                for marker in (
                    "buoc ", "dang nhap", "truy cap", "chon ", "nhap ",
                    "gui ", "nop ", "dang ky", "website", "he thong",
                    "phong ", "bo phan",
                )
            )
            if action_hits < 2:
                continue

        asks_system_procedure = any(
            marker in primary_question
            for marker in (
                "tren he thong", "tren website", "truc tuyen",
                "support.uneti",
            )
        )
        if asks_system_procedure:
            has_platform_cue = any(
                marker in searchable
                for marker in (
                    "support.uneti", "website", "truc tuyen", "module",
                )
            )
            has_workflow_cue = any(
                marker in searchable
                for marker in (
                    "dang nhap", "truy cap", "click", "gui yeu cau",
                    "nhan nut", "chon module",
                )
            )
            if not has_platform_cue or not has_workflow_cue:
                continue

        asks_condition = any(
            marker in primary_question
            for marker in ("dieu kien", "khi nao", "truong hop nao")
        )
        condition_hits = sum(
            marker in searchable
            for marker in (
                "dieu kien", "neu ", "truong hop", "vi pham", "roi vao",
                "thuoc dien", "phai ", "duoc ", "khong duoc",
            )
        )
        if asks_condition and condition_hits == 0:
            continue

        asks_payment = any(
            marker in primary_question
            for marker in ("thanh toan", "dong le phi", "nop le phi")
        )
        payment_hits = sum(
            marker in searchable
            for marker in (
                "thanh toan", "le phi", "hoc phi", "nop ", "dong ", "thu ",
            )
        )
        if asks_payment and payment_hits < 2:
            continue

        if "o dau" in primary_question:
            location_markers = (
                (
                    "man ", "man hinh", "chuc nang", "tra cuu",
                    "truy cap", "duong dan", "support.uneti",
                    "website", "he thong",
                )
                if asks_lookup_location
                else (
                    "phong ", "bo phan", "mot cua", "tai ",
                    "noi nop", "don vi tiep nhan",
                )
            )
            if not any(marker in searchable for marker in location_markers):
                continue

        asks_deadline = (
            "thoi han" in primary_question
            or (
                "khi nao" in primary_question
                and any(
                    action in primary_question
                    for action in ("nop ", "gui ", "rut ")
                )
            )
        )
        deadline_actions = [
            action
            for action in ("nop", "gui", "dang ky", "rut")
            if re.search(
                rf"(?<![a-z0-9]){re.escape(action)}(?![a-z0-9])",
                primary_question,
            )
        ]
        deadline_hits = sum(
            marker in searchable
            for marker in (
                "thoi han", "cham nhat", "trong vong", "truoc ngay",
                "sau ngay", "ke tu ngay",
            )
        )
        deadline_near_action = bool(re.search(
            r"(?:nop|gui|dang ky|rut).{0,50}\d+\s*(?:ngay|tuan|thang|nam)"
            r"|\d+\s*(?:ngay|tuan|thang|nam).{0,50}(?:nop|gui|dang ky|rut)",
            searchable,
        ))
        if asks_deadline:
            if deadline_actions and not any(
                re.search(
                    rf"(?<![a-z0-9]){re.escape(action)}(?![a-z0-9])",
                    searchable,
                )
                for action in deadline_actions
            ):
                continue
            if deadline_hits == 0 and not deadline_near_action:
                continue

        asks_number = any(
            marker in primary_question
            for marker in ("bao nhieu", "may tin chi", "may diem")
        )
        if asks_number and not re.search(r"\d", searchable):
            continue

        title_topic_hits = sum(
            keyword in title_text
            for group in topic_groups
            for keyword in group["keywords"]
        )
        semantic_score = (
            topic_match_total
            + title_topic_hits * 2
            + min(condition_hits, 2)
            + (2 if asks_procedure else 0)
            + (2 if asks_dossier else 0)
            + (2 if asks_payment else 0)
            + (2 if asks_deadline else 0)
        )
        accepted.append((semantic_score, doc))

    if accepted:
        accepted.sort(
            key=lambda item: (
                item[0],
                float(item[1].get("rerank_score") or 0),
                float(item[1].get("vector_score") or 0),
            ),
            reverse=True,
        )
        return [doc for _, doc in accepted], "semantic_need_and_topic_passed"
    return [], "no_semantic_answer_for_aspect"


def _document_key(doc: dict) -> tuple:
    return (
        doc.get("source_type"),
        doc.get("relative_path") or doc.get("doc_name"),
        doc.get("chunk_index"),
        doc.get("title"),
    )


def merge_multi_aspect_results(
    base_docs: list[dict],
    aspect_results: list[dict],
    limit: int,
) -> tuple[list[dict], dict]:
    """Round-robin evidence so every detected aspect gets context space."""
    selected = []
    seen = {}
    aspect_counts = {}

    def add(doc: dict, aspect: dict | None = None) -> bool:
        key = _document_key(doc)
        aspect_id = aspect.get("aspect_id") if aspect else None
        if key in seen:
            existing = seen[key]
            if aspect_id:
                coverage = list(existing.get("coverage_aspects") or [])
                if aspect_id not in coverage:
                    coverage.append(aspect_id)
                    existing["coverage_aspects"] = coverage
                    aspect_counts[aspect_id] = aspect_counts.get(aspect_id, 0) + 1
            return False

        item = dict(doc)
        if aspect_id:
            item["evidence_aspect"] = aspect_id
            item["coverage_aspects"] = [aspect_id]
            item["sub_question"] = aspect.get("question")
            aspect_counts[aspect_id] = aspect_counts.get(aspect_id, 0) + 1
        seen[key] = item
        selected.append(item)
        return True

    for rank in range(DOCS_PER_ASPECT):
        for result in aspect_results:
            docs = result.get("docs") or []
            if rank >= len(docs) or len(selected) >= limit:
                continue
            add(docs[rank], result)

    for doc in base_docs:
        if len(selected) >= limit:
            break
        add(doc)

    covered = [
        result["aspect_id"]
        for result in aspect_results
        if aspect_counts.get(result["aspect_id"], 0) > 0
    ]
    missing = [
        result["aspect_id"]
        for result in aspect_results
        if result["aspect_id"] not in covered
    ]
    return selected[:limit], {
        "selected_count": min(len(selected), limit),
        "aspect_counts": aspect_counts,
        "covered_aspects": covered,
        "missing_aspects": missing,
        "coverage_complete": not missing,
    }


def _reports_missing_evidence(text: str) -> bool:
    normalized = normalize_text(text)
    return bool(
        re.search(r"khong tim thay.{0,30}(?:can cu|thong tin)", normalized)
        or re.search(
            r"tai lieu.{0,25}(?:chua|khong).{0,20}cung cap.{0,25}thong tin",
            normalized,
        )
        or re.search(r"(?:khong|chua) co.{0,20}thong tin", normalized)
        or "khong co tai lieu phu hop" in normalized
    )


def validate_multi_aspect_answer(answer: str, aspects: list[dict]) -> dict:
    """Validate the explicit output contract for a one-call multi-aspect answer."""
    issues = []
    covered = []
    for index, aspect in enumerate(aspects, start=1):
        marker = f"Y_{index}"
        match = re.search(
            rf"\[{marker}\](.*?)\[/{marker}\]",
            str(answer or ""),
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match:
            issues.append({
                "aspect_id": aspect.get("aspect_id"),
                "marker": marker,
                "reason": "missing_output_block",
            })
            continue

        section = match.group(1).strip()
        if not section:
            issues.append({
                "aspect_id": aspect.get("aspect_id"),
                "marker": marker,
                "reason": "empty_output_block",
            })
            continue
        if aspect.get("has_evidence") and _reports_missing_evidence(section):
            issues.append({
                "aspect_id": aspect.get("aspect_id"),
                "marker": marker,
                "reason": "reported_missing_despite_retrieved_evidence",
            })
            continue
        covered.append(aspect.get("aspect_id"))

    return {
        "valid": not issues,
        "covered_aspects": covered,
        "issues": issues,
    }


def clean_multi_aspect_answer(
    answer: str,
    aspects: list[dict] | None = None,
) -> str:
    """Remove aspect markers and join self-contained answer sections."""
    raw_answer = str(answer or "")
    matches = list(re.finditer(
        r"(?ims)^[ \t]*\[Y_(\d+)\][ \t]*\r?\n?"
        r"(.*?)"
        r"^[ \t]*\[/Y_\1\][ \t]*(?:\r?\n)?",
        raw_answer,
    ))
    if not matches:
        return re.sub(
            r"(?im)^[ \t]*\[/?Y_\d+\][ \t]*\r?\n?",
            "",
            raw_answer,
        ).strip()

    sections = []
    for match in matches:
        content = match.group(2).strip()
        if content:
            sections.append(content)

    suffix = raw_answer[matches[-1].end():].strip()
    if suffix:
        sections.append(suffix)
    return "\n\n".join(sections).strip()
