from app.data.multi_aspect_query import (
    clean_multi_aspect_answer,
    decompose_multi_aspect_query,
    filter_semantic_aspect_docs,
    merge_multi_aspect_results,
    validate_multi_aspect_answer,
)
from app.data.elasticsearch_client import normalize_text
from app.data.prompt_builder import build_prompt


def _doc(name: str, chunk: int) -> dict:
    return {
        "source_type": "local_file",
        "doc_name": name,
        "relative_path": name,
        "title": f"Chunk {chunk}",
        "chunk_index": chunk,
        "content": f"Noi dung {name} {chunk}",
    }


def test_detects_independent_information_needs_joined_by_and():
    result = decompose_multi_aspect_query(
        "Điều kiện tốt nghiệp là gì và điều kiện tốt nghiệp loại giỏi là gì?"
    )

    assert result["is_multi_aspect"] is True
    assert [item["question"] for item in result["aspects"]] == [
        "Điều kiện tốt nghiệp là gì",
        "điều kiện tốt nghiệp loại giỏi là gì",
    ]
    assert [item["retrieval_query"] for item in result["aspects"]] == [
        "Điều kiện tốt nghiệp là gì",
        "điều kiện tốt nghiệp loại giỏi là gì",
    ]
    assert all(item["original_question"] for item in result["aspects"])


def test_does_not_split_a_coordinated_noun_list():
    result = decompose_multi_aspect_query(
        "Cần chứng chỉ ngoại ngữ và tin học nào để tốt nghiệp?"
    )

    assert result["is_multi_aspect"] is False
    assert result["aspects"] == []


def test_splits_coordinated_audiences_with_shared_information_need():
    result = decompose_multi_aspect_query(
        "Sinh viên và giảng viên xem thông tin học tập/công tác cá nhân ở đâu?"
    )

    assert result["is_multi_aspect"] is True
    assert [item["question"] for item in result["aspects"]] == [
        "Sinh viên xem thông tin học tập cá nhân ở đâu",
        "giảng viên xem thông tin công tác cá nhân ở đâu",
    ]
    assert [item["audience_scope"] for item in result["aspects"]] == [
        "sinh vien",
        "giang vien",
    ]
    assert "Đối tượng: sinh vien" in result["aspects"][0]["semantic_query"]
    assert "Đối tượng: giang vien" in result["aspects"][1]["semantic_query"]
    assert result["aspects"][0]["focused_retrieval_query"].startswith(
        "sinh vien man hinh chuc nang duong dan truy cap he thong"
    )
    assert result["aspects"][1]["focused_retrieval_query"].startswith(
        "giang vien man hinh chuc nang duong dan truy cap he thong"
    )


def test_splits_other_shared_needs_for_student_and_staff():
    questions = (
        "Sinh viên và cán bộ tra cứu lịch cá nhân ở đâu?",
        "Học viên cùng với giảng viên có được xem thông tin cá nhân không?",
        "Giảng viên và sinh viên kiểm tra công tác/học tập ở đâu?",
    )

    for question in questions:
        result = decompose_multi_aspect_query(question)
        assert result["is_multi_aspect"] is True
        assert len(result["aspects"]) == 2
        assert {
            _audience
            for _audience in (
                item["audience_scope"] for item in result["aspects"]
            )
        } in (
            {"sinh vien", "can bo"},
            {"hoc vien", "giang vien"},
            {"giang vien", "sinh vien"},
        )


def test_does_not_split_unknown_subjects_just_because_of_and():
    result = decompose_multi_aspect_query(
        "Phòng học và thư viện xem thông tin ở đâu?"
    )

    assert result["is_multi_aspect"] is False


def test_inherits_shared_how_to_prefix_for_coordinated_actions():
    result = decompose_multi_aspect_query(
        "Làm sao để đăng ký thi lại và hoãn thi?"
    )

    assert result["is_multi_aspect"] is True
    assert result["raw_candidate_clauses"] == [
        "Làm sao để đăng ký thi lại",
        "hoãn thi",
    ]
    assert [item["question"] for item in result["aspects"]] == [
        "Làm sao để đăng ký thi lại",
        "Làm sao để hoãn thi",
    ]
    assert result["aspects"][1]["focused_retrieval_query"] == (
        "huong dan thu tuc don phu luc hoan thi"
    )


def test_inherits_shared_view_action_for_coordinated_objects():
    result = decompose_multi_aspect_query(
        "Cách xem thời khóa biểu và điểm danh?"
    )

    assert [item["question"] for item in result["aspects"]] == [
        "Cách xem thời khóa biểu",
        "Cách xem điểm danh",
    ]


def test_adds_semantic_alias_query_for_gpa():
    result = decompose_multi_aspect_query(
        "Điểm chữ được quy đổi ra sao và GPA được tính như thế nào?"
    )

    assert result["aspects"][1]["alias_retrieval_query"] == (
        "diem trung binh tich luy duoc tinh nhu the nao"
    )
    assert result["aspects"][1]["semantic_query"] == (
        "diem trung binh tich luy duoc tinh nhu the nao"
    )


def test_normalizes_status_ticket_to_administrative_request():
    result = decompose_multi_aspect_query(
        "Cách đánh giá thủ tục hành chính và xem trạng thái phiếu?"
    )

    assert result["aspects"][1]["semantic_query"] == (
        "xem trang thai yeu cau danh gia thu tuc hanh chinh"
    )


def test_preserves_attendance_as_a_compound_topic():
    result = decompose_multi_aspect_query(
        "Cách xem thời khóa biểu và điểm danh?"
    )

    assert "diem" in result["aspects"][1]["keywords"]
    assert "danh" in result["aspects"][1]["keywords"]


def test_adds_short_submission_query_for_graduation_procedure():
    result = decompose_multi_aspect_query(
        "Điều kiện tốt nghiệp là gì và cách đăng ký tốt nghiệp trên hệ thống?"
    )

    assert result["aspects"][1]["submission_retrieval_query"] == (
        "huong dan nop tot nghiep"
    )


def test_keeps_shared_exam_workload_metric_as_one_need():
    result = decompose_multi_aspect_query(
        "Tôi muốn kiểm tra số giờ coi thi và chấm thi của mình thì xem ở đâu?"
    )

    assert result["is_multi_aspect"] is False
    assert result["reason"] == "single_collective_metric_need"


def test_detects_two_explicit_questions_without_and():
    result = decompose_multi_aspect_query(
        "Điều kiện chuyển trường là gì? Hồ sơ cần những gì?"
    )

    assert result["is_multi_aspect"] is True
    assert len(result["aspects"]) == 2


def test_inherits_context_for_elliptical_credit_limit_clause():
    result = decompose_multi_aspect_query(
        "Sinh viên bị cảnh báo học tập khi nào và được đăng ký tối đa bao nhiêu tín chỉ?"
    )

    assert result["is_multi_aspect"] is True
    second = result["aspects"][1]
    assert second["context_inherited"] is True
    assert "canh bao hoc tap" in normalize_text(second["retrieval_query"])


def test_detects_case_question_after_simultaneously():
    result = decompose_multi_aspect_query(
        "Điều kiện được cấp bằng là gì, đồng thời trường hợp nào bị hạ hạng tốt nghiệp?"
    )

    assert result["is_multi_aspect"] is True
    assert [item["question"] for item in result["aspects"]] == [
        "Điều kiện được cấp bằng là gì",
        "trường hợp nào bị hạ hạng tốt nghiệp",
    ]


def test_marks_generic_request_without_topic_for_clarification():
    result = decompose_multi_aspect_query(
        "Cách gửi yêu cầu trên hệ thống và làm thế nào để kiểm tra trạng thái xử lý?"
    )

    assert result["needs_clarification"] is True


def test_semantic_filter_rejects_policy_chunk_for_procedure_question():
    policy_doc = {
        **_doc("policy.docx", 1),
        "content": "Sinh viên có điểm F phải học lại học phần bắt buộc.",
    }
    procedure_doc = {
        **_doc("procedure.docx", 1),
        "content": (
            "Sinh viên đăng ký học lại trực tuyến trên website của Trường, "
            "chọn học phần trên hệ thống đăng ký."
        ),
    }

    docs, reason = filter_semantic_aspect_docs(
        "Cách đăng ký như thế nào. Ngữ cảnh liên quan: Sinh viên được học lại khi nào",
        [policy_doc, procedure_doc],
    )

    assert reason == "semantic_need_and_topic_passed"
    assert [doc["doc_name"] for doc in docs] == ["procedure.docx"]


def test_lookup_location_requires_audience_guidance_source():
    policy_doc = {
        **_doc("student-policy.pdf", 1),
        "title": "Co van hoc tap",
        "content": (
            "Co van thong bao cho sinh vien dia diem lam viec va tu van "
            "cac van de hoc tap ca nhan."
        ),
        "document_type": "regulation",
    }
    student_guide = {
        **_doc("WEB SUPPORT SV.docx", 1),
        "title": "Tra cuu ket qua hoc tap",
        "content": (
            "Sinh vien truy cap chuc nang Ket qua hoc tap tren "
            "support.uneti.edu.vn de xem thong tin hoc tap ca nhan."
        ),
        "document_type": "business_document",
        "section_type": "business_section",
    }
    staff_guide = {
        **_doc("WEB SUPPORT CBGV.docx", 1),
        "title": "Tra cuu khoi luong cong tac giang vien",
        "content": (
            "Giang vien truy cap chuc nang Tra cuu de xem thong tin "
            "cong tac ca nhan."
        ),
        "document_type": "business_document",
        "section_type": "business_section",
    }

    student_docs, _ = filter_semantic_aspect_docs(
        "Sinh viên xem thông tin học tập cá nhân ở đâu. Đối tượng: sinh vien",
        [policy_doc, staff_guide, student_guide],
    )
    staff_docs, _ = filter_semantic_aspect_docs(
        "Giảng viên xem thông tin công tác cá nhân ở đâu. Đối tượng: giang vien",
        [policy_doc, student_guide, staff_guide],
    )

    assert [doc["doc_name"] for doc in student_docs] == [
        "WEB SUPPORT SV.docx"
    ]
    assert [doc["doc_name"] for doc in staff_docs] == [
        "WEB SUPPORT CBGV.docx"
    ]


def test_merge_reserves_context_for_every_aspect():
    aspect_results = [
        {
            "aspect_id": "aspect_1",
            "question": "Y thu nhat la gi",
            "docs": [_doc("a.docx", 1), _doc("a.docx", 2)],
        },
        {
            "aspect_id": "aspect_2",
            "question": "Y thu hai la gi",
            "docs": [_doc("b.docx", 1), _doc("b.docx", 2)],
        },
    ]

    docs, debug = merge_multi_aspect_results(
        [_doc("base.docx", 1)],
        aspect_results,
        limit=4,
    )

    assert [doc["doc_name"] for doc in docs] == [
        "a.docx",
        "b.docx",
        "a.docx",
        "b.docx",
    ]
    assert debug["coverage_complete"] is True
    assert debug["aspect_counts"] == {"aspect_1": 2, "aspect_2": 2}


def test_shared_document_counts_as_coverage_for_both_aspects():
    shared = _doc("shared.docx", 1)
    docs, debug = merge_multi_aspect_results(
        [],
        [
            {
                "aspect_id": "aspect_1",
                "question": "Y thu nhat",
                "docs": [shared],
            },
            {
                "aspect_id": "aspect_2",
                "question": "Y thu hai",
                "docs": [shared],
            },
        ],
        limit=4,
    )

    assert len(docs) == 1
    assert docs[0]["coverage_aspects"] == ["aspect_1", "aspect_2"]
    assert debug["coverage_complete"] is True


def test_prompt_contains_explicit_aspect_checklist():
    prompt = build_prompt(
        "Cau hoi hai y",
        "context",
        required_aspects=[
            {
                "aspect_id": "aspect_1",
                "question": "Y thu nhat la gi",
                "has_evidence": True,
                "sources": [{"title": "Dieu 20", "doc_name": "quy-che.docx"}],
            },
            {
                "aspect_id": "aspect_2",
                "question": "Y thu hai can gi",
                "has_evidence": False,
                "sources": [],
            },
        ],
    )

    assert "CAC Y CAN TRA LOI VA BAN DO NGUON:" in prompt
    assert "Y_1: Y thu nhat la gi" in prompt
    assert "Tieu de hien thi: Y thu nhat la gi" in prompt
    assert "Dieu 20 - quy-che.docx" in prompt
    assert "Y_2: Y thu hai can gi" in prompt
    assert "[Y_1]" in prompt
    assert "[/Y_2]" in prompt


def test_location_prompt_uses_concise_dynamic_response_mode():
    prompt = build_prompt(
        "Sinh viên và giảng viên xem thông tin cá nhân ở đâu?",
        "context",
        required_aspects=[
            {
                "aspect_id": "aspect_1",
                "question": "Sinh viên xem thông tin cá nhân ở đâu",
                "presentation_title": "Sinh viên",
                "has_evidence": True,
                "sources": [],
            },
            {
                "aspect_id": "aspect_2",
                "question": "Giảng viên xem thông tin cá nhân ở đâu",
                "presentation_title": "Giảng viên",
                "has_evidence": True,
                "sources": [],
            },
        ],
    )

    assert "CHE DO TRINH BAY: TRA CUU VI TRI" in prompt
    assert "Moi khoi toi da 3 gach dau dong ngan" in prompt
    assert "Khong mo ta toan bo du lieu hien thi" in prompt


def test_validates_and_cleans_single_call_multi_aspect_answer():
    aspects = [
        {
            "aspect_id": "aspect_1",
            "has_evidence": True,
            "presentation_title": "Cách tính điểm",
        },
        {
            "aspect_id": "aspect_2",
            "has_evidence": True,
            "presentation_title": "Xếp loại giỏi",
        },
    ]
    answer = (
        "[Y_1]\nĐiểm được tính theo công thức A.\n[/Y_1]\n"
        "[Y_2]\nLoại giỏi từ 3,20 đến 3,59.\n[/Y_2]\n"
        "(Nguồn: Điều 20 - quy-che.docx)"
    )

    validation = validate_multi_aspect_answer(answer, aspects)

    assert validation["valid"] is True
    assert validation["covered_aspects"] == ["aspect_1", "aspect_2"]
    cleaned = clean_multi_aspect_answer(answer, aspects)
    assert "[Y_1]" not in cleaned
    assert cleaned == (
        "Điểm được tính theo công thức A.\n\n"
        "Loại giỏi từ 3,20 đến 3,59.\n\n"
        "(Nguồn: Điều 20 - quy-che.docx)"
    )


def test_rejects_missing_block_and_false_no_evidence_claim():
    aspects = [
        {"aspect_id": "aspect_1", "has_evidence": True},
        {"aspect_id": "aspect_2", "has_evidence": True},
    ]
    answer = (
        "[Y_1]\nTài liệu hiện không cung cấp thông tin cụ thể cho ý này.\n[/Y_1]"
    )

    validation = validate_multi_aspect_answer(answer, aspects)

    assert validation["valid"] is False
    assert [item["reason"] for item in validation["issues"]] == [
        "reported_missing_despite_retrieved_evidence",
        "missing_output_block",
    ]
