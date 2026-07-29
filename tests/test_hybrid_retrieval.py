import asyncio
import os


os.environ.setdefault("GEMINI_API_KEY", "test-gemini-api-key")

import app.data.elasticsearch_client as retrieval
import app.data.ambiguity_analyzer as ambiguity
import app.data.hyde as hyde
import app.data.langchain_pipeline as langchain_pipeline
import app.data.query_expansion as query_expansion
import app.data.reranker as reranker
from rank_bm25 import BM25Okapi


def _doc(name: str, chunk_index: int, **scores):
    return {
        "doc_name": name,
        "relative_path": name,
        "title": f"Chunk {chunk_index}",
        "chunk_index": chunk_index,
        "content": f"Content from {name}",
        **scores,
    }


def test_query_expansion_skips_specific_and_long_queries():
    assert query_expansion.should_expand_query("van ban so 880")[0] is False
    assert query_expansion.should_expand_query("Điều 15 quy định gì")[0] is False
    assert query_expansion.should_expand_query("Mục 2 áp dụng thế nào")[0] is False
    assert query_expansion.should_expand_query(
        "quy định sử dụng thiết bị trong phòng học như thế nào"
    )[0] is False


def test_query_expansion_keeps_original_when_gemini_fails(monkeypatch):
    monkeypatch.setattr(
        query_expansion,
        "expand_query_with_gemini",
        lambda question: (_ for _ in ()).throw(RuntimeError("Gemini unavailable")),
    )

    queries, debug = query_expansion.build_query_variants("quên mật khẩu")

    assert queries == ["quên mật khẩu"]
    assert debug["attempted"] is True
    assert debug["reason"] == "gemini_error_fallback_original"
    assert "Gemini unavailable" in debug["error"]


def test_query_expansion_keeps_original_and_limits_variants(monkeypatch):
    monkeypatch.setattr(
        query_expansion,
        "expand_query_with_gemini",
        lambda question: [
            "khôi phục mật khẩu",
            "cấp lại mật khẩu",
            "đặt lại thông tin đăng nhập",
            "biến thể vượt giới hạn",
        ],
    )

    queries, debug = query_expansion.build_query_variants("quên mật khẩu")

    assert queries[0] == "quên mật khẩu"
    assert len(queries) == 1 + query_expansion.QUERY_EXPANSION_MAX_VARIANTS
    assert len(queries) <= 3
    assert debug["variant_count"] == query_expansion.QUERY_EXPANSION_MAX_VARIANTS


def test_rrf_deduplicates_same_chunk():
    shared = _doc("a.pdf", 1)
    other = _doc("b.pdf", 1)

    results = retrieval._merge_with_rrf(
        [
            [{**shared, "bm25_score": 10, "keyword_score": 10}],
            [{**shared, "score": 0.8, "distance": 0.2}, other],
        ],
        limit=10,
    )

    assert len(results) == 2
    assert results[0]["doc_name"] == "a.pdf"
    assert results[0]["rrf_score"] > results[1]["rrf_score"]
    assert results[0]["bm25_score"] == 10
    assert results[0]["vector_score"] == 0.8


def test_local_business_query_gives_ann_a_bounded_rrf_advantage():
    bm25_doc = _doc(
        "regulation.pdf",
        1,
        retrieval_branches=["bm25_original"],
    )
    ann_doc = _doc(
        "guide.docx",
        1,
        retrieval_branches=["ann_original"],
    )

    results = retrieval._merge_with_rrf(
        [[bm25_doc], [ann_doc]],
        limit=2,
        branch_weights=retrieval._local_rrf_branch_weights(
            "Trên màn hình thống kê một cửa, tôi xuất Excel thế nào?"
        ),
    )

    assert results[0]["doc_name"] == "guide.docx"
    assert retrieval._local_query_profile("Điều 15 khoản 2 quy định gì?") == "legal"
    assert retrieval._local_query_profile(
        "Tôi xem chi tiết điểm thành phần của môn học thế nào?"
    ) == "business"


def test_local_rank_uses_cross_encoder_as_primary_signal():
    strong = _doc(
        "regulation.pdf",
        1,
        document_type="regulation",
        rerank_score=8.0,
        rrf_score=0.02,
    )
    weak_business = _doc(
        "guide.docx",
        1,
        document_type="business_document",
        rerank_score=1.0,
        rrf_score=0.03,
    )

    results = retrieval._rank_local_documents(
        "Nhấn nút nào trên màn hình để xuất Excel?",
        [strong, weak_business],
        limit=2,
    )

    assert results[0]["doc_name"] == "regulation.pdf"
    assert results[0]["local_final_score"] > results[1]["local_final_score"]


def test_local_rank_uses_business_anchor_to_reject_semantic_neighbor():
    correct = {
        **_doc("guide.docx", 1),
        "title": "Man Bao hong",
        "content": "Nhap mo ta su co thiet bi, sau do nhan Gui yeu cau.",
        "document_type": "business_document",
        "rerank_score": -2.0,
        "rrf_score": 0.02,
    }
    wrong = {
        **_doc("graduation.docx", 1),
        "title": "Dang ky tot nghiep",
        "content": "Hoan tat ho so va gui yeu cau xet tot nghiep.",
        "document_type": "business_document",
        "rerank_score": -1.8,
        "rrf_score": 0.021,
    }

    results = retrieval._rank_local_documents(
        "Sau khi mo ta su co thiet bi, toi lam gi de gui yeu cau?",
        [wrong, correct],
        limit=2,
    )

    assert results[0]["doc_name"] == "guide.docx"


def test_local_rank_prefers_appeal_section_for_regrade_alias():
    appeal = {
        **_doc("web-support-sv.docx", 1),
        "title": "1.2. Phuc khao",
        "section_path": "Mot cua > Khao thi > Phuc khao",
        "content": "Mo man Phuc khao va chon Gui yeu cau.",
        "document_type": "business_document",
        "rerank_score": -3.0,
        "rrf_score": 0.018,
    }
    exam_retake = {
        **_doc("web-support-sv.docx", 2),
        "title": "1.4. Dang ky thi lai",
        "section_path": "Mot cua > Khao thi > Dang ky thi lai",
        "content": "Sinh vien dang ky thi lai va gui yeu cau.",
        "document_type": "business_document",
        "rerank_score": -2.0,
        "rrf_score": 0.021,
    }

    results = retrieval._rank_local_documents(
        "Tôi muốn chấm lại bài thi thì đăng ký ở đâu?",
        [exam_retake, appeal],
        limit=2,
    )

    assert results[0]["title"] == "1.2. Phuc khao"
    assert results[0]["semantic_section_match"] == 1.0


def test_local_legal_rank_prefers_matching_document_name():
    correct = {
        **_doc("Quy che dao tao dai hoc chinh quy 832.docx", 1),
        "title": "Dieu 15",
        "document_type": "regulation",
        "rerank_score": 1.0,
        "rrf_score": 0.02,
    }
    wrong = {
        **_doc("Quy che cong tac sinh vien dao tao dai hoc chinh quy.pdf", 1),
        "title": "Dieu 15",
        "document_type": "regulation",
        "rerank_score": 1.0,
        "rrf_score": 0.02,
    }

    results = retrieval._rank_local_documents(
        "Dieu 15 Quy che dao tao dai hoc chinh quy quy dinh gi?",
        [wrong, correct],
        limit=2,
    )

    assert results[0]["doc_name"] == correct["doc_name"]


def test_bm25_prioritizes_document_number_and_department(monkeypatch):
    weak = {
        **_doc("general.pdf", 1),
        "content": "Quy định sử dụng thiết bị chung.",
        "phong_ban": "Phòng Đào tạo",
    }
    strong = {
        **_doc("qd-880.pdf", 1),
        "content": "Quy định khai thác và bảo trì thiết bị phòng học.",
        "so_van_ban": "880",
        "so_van_ban_ngan": "880",
        "phong_ban": "Phòng Công nghệ thông tin",
    }
    unrelated = {
        **_doc("unrelated.pdf", 1),
        "content": "Hướng dẫn đăng ký học phần.",
        "phong_ban": "Phòng Công tác sinh viên",
    }
    chunks = [weak, strong, unrelated]
    corpus = [retrieval._bm25_document_tokens(doc) for doc in chunks]
    retrieval._INDEX_CACHE["bm25"] = BM25Okapi(corpus)
    monkeypatch.setattr(
        retrieval,
        "_load_document_index",
        lambda: (chunks, {}, len(chunks)),
    )

    results = retrieval._search_bm25_documents(
        "văn bản 880 phòng công nghệ thông tin",
        limit=2,
    )

    assert results[0]["doc_name"] == "qd-880.pdf"
    assert results[0]["bm25_score"] >= results[-1]["bm25_score"]


def test_cross_encoder_failure_falls_back_to_rrf(monkeypatch):
    docs = [
        _doc("a.pdf", 1, rrf_score=0.03),
        _doc("b.pdf", 1, rrf_score=0.02),
    ]
    monkeypatch.setattr(
        reranker,
        "get_cross_encoder",
        lambda: (_ for _ in ()).throw(RuntimeError("model unavailable")),
    )

    results, debug = reranker.rerank_documents("test query", docs, final_top_k=2)

    assert [doc["doc_name"] for doc in results] == ["a.pdf", "b.pdf"]
    assert debug["reason"] == "cross_encoder_error_rrf_fallback"
    assert "model unavailable" in debug["error"]


def test_cross_encoder_success_reorders_candidates(monkeypatch):
    class FakeCrossEncoder:
        def predict(self, pairs):
            return [0.1, 0.9]

    monkeypatch.setattr(reranker, "get_cross_encoder", lambda: FakeCrossEncoder())
    docs = [
        _doc("a.pdf", 1, rrf_score=0.04),
        _doc("b.pdf", 1, rrf_score=0.03),
    ]

    results, debug = reranker.rerank_documents("test query", docs, final_top_k=2)

    assert [doc["doc_name"] for doc in results] == ["b.pdf", "a.pdf"]
    assert debug["reason"] == "cross_encoder_success"
    assert results[0]["rerank_score"] == 0.9


def test_attendance_exam_policy_query_uses_expanded_query_and_wider_top_k():
    question = "nghi hoc khong phep co bi cam thi khong"
    expanded = retrieval._academic_policy_retrieval_query(question)

    assert "diem chuyen can" in expanded
    assert "nghi hoc tren 50" in expanded
    assert retrieval._policy_query_profile(question) == "attendance_exam_eligibility"
    assert retrieval._effective_final_top_k(question, "official_document") >= 8


def test_absence_permission_comparison_is_not_final_exam_profile():
    question = "nghi hoc khong phep va nghi hoc co phep khac nhau nhung gi"
    expanded = retrieval._academic_policy_retrieval_query(question)

    assert retrieval._policy_query_profile(question) == "absence_permission_comparison"
    assert "diem chuyen can" in expanded
    assert "dieu 13" in expanded
    assert "ky thi phu" not in expanded
    assert retrieval._effective_final_top_k(question, "official_document") >= 8


def test_exam_retake_policy_query_does_not_expand_to_hoc_lai():
    question = "huong dan dang ky thi lai"
    expanded = retrieval._academic_policy_retrieval_query(question)

    assert retrieval._policy_query_profile(question) == "exam_retake"
    assert "dang ky thi lai" in expanded
    assert "hoc lai" not in expanded
    assert "hoc cai thien" not in expanded
    assert "dieu 11" not in expanded


def test_policy_profile_does_not_match_inside_business_words():
    assert retrieval._policy_query_profile(
        "video huong dan nghiep vu chuyen mon"
    ) is None
    assert retrieval._policy_query_profile(
        "mo ta su co thiet bi roi hoan tat gui yeu cau"
    ) is None
    assert retrieval._policy_query_profile(
        "du lieu lop hoc phan da duoc dong bo"
    ) is None


def test_gpa_query_expands_to_cumulative_average_terms():
    expanded = retrieval._academic_policy_retrieval_query("gpa la gi")

    assert retrieval._policy_query_profile("gpa la gi") == "grade_average"
    assert "diem trung binh tich luy" in retrieval.normalize_text(expanded)
    assert "diem trung binh hoc tap" in retrieval.normalize_text(expanded)
    assert "tinh diem trung binh" in retrieval.normalize_text(expanded)
    assert "dieu 20" in retrieval.normalize_text(expanded)


def test_projected_gpa_query_targets_student_simulation_screen():
    question = "Có chỗ nào nhập điểm giả định để xem GPA dự kiến không?"
    expanded = retrieval.normalize_text(
        retrieval._academic_policy_retrieval_query(question)
    )

    assert retrieval._policy_query_profile(question) == "projected_grade_ui"
    assert retrieval._local_query_profile(question) == "business"
    assert "du kien ket qua hoc tap" in expanded
    assert "nhap diem mong muon" in expanded
    assert "diem tong ket du kien" in expanded
    assert "support uneti" in expanded
    assert "dieu 20" not in expanded
    assert "quy che dao tao dai hoc chinh quy" not in expanded


def test_projected_grade_intent_handles_everyday_variants_without_hijacking_gpa_policy():
    projected_questions = (
        "Tôi thử nhập điểm mong muốn để tính điểm tích lũy ở đâu?",
        "Có chức năng mô phỏng GPA nếu kỳ này tôi đạt điểm B không?",
        "Làm sao ước tính xếp loại học lực với điểm dự kiến?",
    )

    for question in projected_questions:
        assert retrieval._policy_query_profile(question) == "projected_grade_ui"
        assert retrieval._local_query_profile(question) == "business"

    assert retrieval._policy_query_profile(
        "GPA được tính như thế nào?"
    ) == "grade_average"


def test_regrade_everyday_query_expands_to_appeal_ui_terms():
    question = "Tôi muốn chấm lại bài thi thì đăng ký ở đâu?"

    expanded = retrieval.normalize_text(
        retrieval._academic_policy_retrieval_query(question)
    )

    assert retrieval._local_query_profile(question) == "business"
    assert "phuc khao" in expanded
    assert "man phuc khao" in expanded
    assert "mot cua" in expanded
    assert "khao thi" in expanded
    assert "gui yeu cau" in expanded


def test_graduation_classification_profile_targets_article_25():
    question = "dieu kien tot nghiep la gi va dieu kien tot nghiep loai gioi"

    assert retrieval._policy_query_profile(question) == "graduation_classification"

    expanded = retrieval.normalize_text(
        retrieval._academic_policy_retrieval_query(question)
    )
    assert "hang tot nghiep" in expanded
    assert "hoc lai vuot qua 5 phan tram" in expanded
    assert "dieu kien xet tot nghiep" in expanded
    assert "dieu 24" in expanded
    assert retrieval._effective_final_top_k(question, "local_file") >= 8

    article_24 = {
        "title": "Dieu 24. Dieu kien xet tot nghiep",
        "content": "Sinh vien duoc cong nhan tot nghiep khi du cac dieu kien.",
        "doc_name": "Quy che dao tao dai hoc chinh quy.docx",
        "dieu": 24,
    }
    article_25 = {
        "title": "Dieu 25. Cap bang tot nghiep",
        "content": (
            "Hang tot nghiep duoc xac dinh theo diem trung binh tich luy. "
            "Loai gioi tu 3,20 den 3,59; hoc lai vuot qua 5% thi giam di mot muc."
        ),
        "doc_name": "Quy che dao tao dai hoc chinh quy.docx",
        "dieu": 25,
    }

    assert retrieval._policy_result_priority(question, article_25) > (
        retrieval._policy_result_priority(question, article_24)
    )


def test_exam_defer_how_to_query_uses_business_local_ranking():
    question = "toi muon hoan thi thi lam the nao"
    procedure_doc = {
        "doc_name": "2026.03.25.AI_HDSD TREN WEB SUPPORT SV.docx",
        "title": "1.5. Hoan thi",
        "content": (
            "Huong dan sinh vien de nghi hoan thi. Buoc 1 dang nhap he thong. "
            "Buoc 2 chon Thu tuc hanh chinh, Mot cua - Khao thi, Hoan thi. "
            "Buoc 3 dien du lieu. Buoc 4 gui yeu cau."
        ),
        "document_type": "business_document",
        "source_type": "local_file",
        "relative_path": "nghiep_vu/web-support-sv.docx",
        "chunk_index": 17,
        "rerank_score": -1.0,
        "rrf_score": 0.02,
    }
    policy_doc = {
        "doc_name": "Quy che dao tao dai hoc chinh quy.docx",
        "title": "Dieu 16. Cach tinh diem hoc phan",
        "content": "Diem I duoc ap dung khi sinh vien duoc phep hoan thi.",
        "document_type": "regulation",
        "source_type": "local_file",
        "relative_path": "quy-che.docx",
        "chunk_index": 31,
        "rerank_score": 0.0,
        "rrf_score": 0.03,
    }
    master_doc = {
        **policy_doc,
        "doc_name": "Quy che dao tao trinh do thac si.docx",
        "relative_path": "quy-che-thac-si.docx",
        "title": "Dieu 21. Thi, kiem tra, danh gia",
        "chunk_index": 36,
        "rerank_score": 0.5,
    }

    assert retrieval._policy_query_profile(question) == "exam_defer"
    assert retrieval._local_query_profile(question) == "business"

    ranked = retrieval._rank_local_documents(
        question,
        [policy_doc, procedure_doc, master_doc],
        limit=3,
    )
    assert ranked[0]["title"] == "1.5. Hoan thi"
    assert ranked[-1]["doc_name"] == "Quy che dao tao trinh do thac si.docx"


def test_exam_defer_multi_aspect_queries_choose_different_local_profiles():
    condition_question = "Dieu kien hoan thi la gi"
    procedure_question = "thu tuc xin hoan thi thuc hien nhu the nao"

    assert retrieval._local_query_profile(condition_question) == "neutral"
    assert retrieval._policy_query_profile(condition_question) == "exam_defer"
    assert retrieval._local_query_profile(procedure_question) == "business"
    assert retrieval._policy_query_profile(procedure_question) == "exam_defer"


def test_course_registration_change_query_expands_to_article_10_terms():
    question = "cach huy hoc phan da dang ky"
    expanded = retrieval._academic_policy_retrieval_query(question)
    normalized = retrieval.normalize_text(expanded)

    assert retrieval._policy_query_profile(question) == "course_registration_change"
    assert "huy dang ky hoc phan" in normalized
    assert "rut bot hoc phan" in normalized
    assert "dang ky khoi luong hoc tap" in normalized
    assert "dieu 10" in normalized
    assert "dieu 9" in normalized


def test_course_registration_change_priority_prefers_registration_article():
    question = "cach huy hoc phan da dang ky"
    direct_doc = {
        "doc_name": "DHKTKTCN_PDT_QD_2025_12_09_Quy che dao tao dai hoc chinh quy.docx",
        "title": "Dieu 10. Rut bot hoc phan da dang ky",
        "content": "Sinh vien duoc rut bot hoc phan trong thoi gian dang ky hoc phan.",
        "dieu": 10,
        "keyword_score": 10,
    }
    noisy_doc = {
        "doc_name": "TTNNTH_Quy doi chung chi tieng Anh.docx",
        "title": "Quy doi chung chi tieng Anh",
        "content": "Hoc phan da dang ky va chung chi tieng Anh TOEIC IELTS.",
        "keyword_score": 90,
    }

    results = retrieval._prioritize_policy_results(question, [noisy_doc, direct_doc])

    assert results[0]["title"] == "Dieu 10. Rut bot hoc phan da dang ky"


def test_credit_load_warning_query_expands_to_article_9_terms():
    question = "em dang bi canh bao hoc tap thi toi da duoc dang ky bao nhieu tin chi"
    expanded = retrieval._academic_policy_retrieval_query(question)
    normalized = retrieval.normalize_text(expanded)

    assert retrieval._policy_query_profile(question) == "credit_load_warning"
    assert "canh bao hoc tap" in normalized
    assert "dang ky khoi luong hoc tap" in normalized
    assert "16 tin chi" in normalized
    assert "dieu 9" in normalized
    assert "120 tin chi" not in normalized
    assert "150 tin chi" not in normalized


def test_credit_load_warning_priority_prefers_training_regulation():
    question = "em dang bi canh bao hoc tap thi toi da duoc dang ky bao nhieu tin chi"
    direct_doc = {
        "doc_name": "DHKTKTCN_PDT_QD_2025_12_09_Quy che dao tao dai hoc chinh quy.docx",
        "title": "Dieu 9. Dang ky khoi luong hoc tap",
        "content": "Sinh vien bi canh bao hoc tap khong duoc dang ky qua 16 tin chi.",
        "dieu": 9,
        "keyword_score": 10,
    }
    noisy_doc = {
        "doc_name": "2026.03.25.AI_HDSD TREN WEB SUPPORT SV.docx",
        "title": "Muc I -> 2 -> 2.2",
        "content": "Thoi khoa bieu lich hoc lich thi dang ky hoc tap tin chi.",
        "keyword_score": 90,
    }

    results = retrieval._prioritize_policy_results(question, [noisy_doc, direct_doc])

    assert results[0]["title"] == "Dieu 9. Dang ky khoi luong hoc tap"


def test_credit_load_warning_priority_prefers_specific_16_credit_clause():
    question = "em dang bi canh bao hoc tap thi toi da duoc dang ky bao nhieu tin chi"
    generic_clause = {
        "doc_name": "DHKTKTCN_PDT_QD_2025_12_09_Quy che dao tao dai hoc chinh quy.docx",
        "title": "Dieu 9. Dang ky khoi luong hoc tap (1)",
        "content": (
            "Khoi luong hoc tap toi da la 3/2 so tin chi trung binh mot hoc ky. "
            "Sinh vien vua bi canh bao hoc tap o hoc ky truoc do."
        ),
        "dieu": 9,
        "keyword_score": 100,
    }
    specific_clause = {
        "doc_name": "DHKTKTCN_PDT_QD_2025_12_09_Quy che dao tao dai hoc chinh quy.docx",
        "title": "Dieu 9. Dang ky khoi luong hoc tap (2)",
        "content": (
            "Sinh vien dang trong thoi gian bi canh bao ket qua hoc tap chi duoc "
            "dang ky khoi luong hoc tap khong qua 16 tin chi cho moi hoc ky."
        ),
        "dieu": 9,
        "keyword_score": 80,
    }

    results = retrieval._prioritize_policy_results(question, [generic_clause, specific_clause])

    assert results[0]["title"] == "Dieu 9. Dang ky khoi luong hoc tap (2)"


def test_transfer_school_query_expands_to_article_28_terms():
    question = "toi muon chuyen truong khong phai chuyen chuong trinh dao tao"
    expanded = retrieval._academic_policy_retrieval_query(question)
    normalized = retrieval.normalize_text(expanded)

    assert retrieval._policy_query_profile(question) == "transfer_school"
    assert "chuyen truong" in normalized
    assert "hieu truong" in normalized
    assert "dieu 28" in normalized


def test_transfer_school_priority_rejects_master_regulation():
    question = "toi muon chuyen truong khong phai chuyen chuong trinh dao tao"
    undergraduate_doc = {
        "doc_name": "DHKTKTCN_PDT_QD_2025_12_09_Quy che dao tao dai hoc chinh quy_832_20092023.docx",
        "title": "Dieu 28. Chuyen truong",
        "content": "Sinh vien duoc xet chuyen truong khi co dong y cua Hieu truong va cung nganh dao tao.",
        "dieu": 28,
        "keyword_score": 30,
    }
    master_doc = {
        "doc_name": "DHKTKTCN_PDT_QD_2025_12_09_Quy che tuyen sinh va dao tao trinh do thac si_834_20092023.docx",
        "title": "Dieu 20. Chuyen truong doi voi hoc vien thac si",
        "content": "Hoc vien thac si duoc xet chuyen truong theo quy che dao tao trinh do thac si.",
        "keyword_score": 100,
    }

    results = retrieval._prioritize_policy_results(question, [master_doc, undergraduate_doc])

    assert results[0]["title"] == "Dieu 28. Chuyen truong"


def test_elective_failed_course_query_expands_to_article_11_terms():
    question = "neu em bi F mon tu chon thi co the chon mon khac thay the khong"
    expanded = retrieval._academic_policy_retrieval_query(question)
    normalized = retrieval.normalize_text(expanded)

    assert retrieval._policy_query_profile(question) == "elective_failed_course"
    assert "hoc phan tu chon" in normalized
    assert "hoc doi" in normalized
    assert "tuong duong" in normalized
    assert "dieu 11" in normalized


def test_elective_failed_course_priority_prefers_article_11():
    question = "neu em bi F mon tu chon thi co the chon mon khac thay the khong"
    direct_doc = {
        "doc_name": "DHKTKTCN_PDT_QD_2025_12_09_Quy che dao tao dai hoc chinh quy_832_20092023.docx",
        "title": "Dieu 11. Hoc lai, hoc cai thien diem",
        "content": "Hoc phan tu chon bi diem F F+ thi sinh vien co the hoc doi sang hoc phan khac tuong duong.",
        "dieu": 11,
        "keyword_score": 30,
    }
    weak_doc = {
        "doc_name": "2026.03.25.AI_HDSD TREN WEB SUPPORT SV.docx",
        "title": "Lop hoc phan",
        "content": "Sinh vien xem lop hoc phan va so tin chi tren web support.",
        "keyword_score": 100,
    }

    results = retrieval._prioritize_policy_results(question, [weak_doc, direct_doc])

    assert results[0]["title"] == "Dieu 11. Hoc lai, hoc cai thien diem"


def test_f_grade_comparison_query_expands_to_articles_16_and_11():
    question = "diem F+ va F khac nhau nhu the nao co phai hoc lai ca hai khong"
    expanded = retrieval._academic_policy_retrieval_query(question)
    normalized = retrieval.normalize_text(expanded)

    assert retrieval._policy_query_profile(question) == "f_grade_comparison"
    assert "diem chu" in normalized
    assert "dieu 16" in normalized
    assert "dieu 11" in normalized


def test_f_grade_comparison_priority_keeps_grade_and_retake_articles():
    question = "diem F+ va F khac nhau nhu the nao co phai hoc lai ca hai khong"
    grade_doc = {
        "doc_name": "DHKTKTCN_PDT_QD_2025_12_09_Quy che dao tao dai hoc chinh quy_832_20092023.docx",
        "title": "Dieu 16. Thang diem danh gia",
        "content": "Diem hoc phan duoc quy doi sang diem chu F+ va F theo thang diem.",
        "dieu": 16,
        "keyword_score": 20,
    }
    retake_doc = {
        "doc_name": "DHKTKTCN_PDT_QD_2025_12_09_Quy che dao tao dai hoc chinh quy_832_20092023.docx",
        "title": "Dieu 11. Hoc lai, hoc doi hoc phan",
        "content": "Hoc phan bat buoc khong dat phai hoc lai; hoc phan tu chon co the hoc doi hoc phan tuong duong.",
        "dieu": 11,
        "keyword_score": 20,
    }
    noisy_doc = {
        "doc_name": "DHKTKTCN_PDT_QD_2025_12_09_Quy che tuyen sinh va dao tao trinh do thac si_834_20092023.docx",
        "title": "Diem hoc vien thac si",
        "content": "Hoc vien thac si co diem F.",
        "keyword_score": 200,
    }

    results = retrieval._prioritize_policy_results(question, [noisy_doc, retake_doc, grade_doc])

    assert {results[0]["dieu"], results[1]["dieu"]} == {11, 16}


def test_credit_definition_query_expands_to_article_2_terms():
    question = "mot tin chi tuong duong voi bao nhieu tiet hoc ly thuyet va thuc hanh"
    expanded = retrieval._academic_policy_retrieval_query(question)
    normalized = retrieval.normalize_text(expanded)

    assert retrieval._policy_query_profile(question) == "credit_definition"
    assert "15 tiet" in normalized
    assert "30 tiet" in normalized
    assert "45 60 gio" in normalized
    assert "dieu 2" in normalized


def test_credit_definition_priority_prefers_article_2_full_definition():
    question = "mot tin chi tuong duong voi bao nhieu tiet hoc ly thuyet va thuc hanh"
    direct_doc = {
        "doc_name": "DHKTKTCN_PDT_QD_2025_12_09_Quy che dao tao dai hoc chinh quy_832_20092023.docx",
        "title": "Dieu 2. Tin chi",
        "content": "Mot tin chi bang 15 tiet ly thuyet, 30 tiet thuc hanh, 30 40 gio thuc tap, 45 60 gio lam tieu luan do an.",
        "dieu": 2,
        "keyword_score": 30,
    }
    noisy_doc = {
        "doc_name": "DHKTKTCN_PDT_QD_2025_12_09_Quy dinh xay dung chuong trinh dao tao_833_20092023.docx",
        "title": "Khoi luong kien thuc",
        "content": "Chuong trinh cu nhan co 120 tin chi, ky su co 150 tin chi.",
        "keyword_score": 120,
    }

    results = retrieval._prioritize_policy_results(question, [noisy_doc, direct_doc])

    assert results[0]["title"] == "Dieu 2. Tin chi"


def test_policy_priority_prefers_direct_attendance_exam_source():
    question = "nghi hoc khong phep co bi cam thi khong"
    direct_doc = {
        "doc_name": "DHKTKTCN_PDT_QD_2025_12_09_Quy che dao tao dai hoc chinh quy.docx",
        "title": "Dieu 13. Danh gia hoc phan",
        "content": "Sinh vien nghi hoc tren 50% so tiet trong chuong trinh se bi cam thi.",
        "keyword_score": 10,
    }
    generic_doc = {
        "doc_name": "DHKTKTCN_PCTCTSV_Quy che Cong tac sinh vien.pdf",
        "title": "Dieu 26. To chuc thuc hien",
        "content": "Nghi hoc dai ngay khong ly do bi xu ly ky luat.",
        "keyword_score": 50,
    }

    results = retrieval._prioritize_policy_results(question, [generic_doc, direct_doc])

    assert results[0]["doc_name"].startswith("DHKTKTCN_PDT")


def test_policy_priority_rejects_final_exam_for_absence_comparison():
    question = "nghi hoc khong phep va nghi hoc co phep khac nhau nhung gi"
    attendance_doc = {
        "doc_name": "DHKTKTCN_PDT_QD_2025_12_09_Quy che dao tao dai hoc chinh quy.docx",
        "title": "Dieu 13. Danh gia hoc phan",
        "content": "Diem chuyen can can cu so tiet vang, nghi hoc trong chuong trinh hoc tap tren lop.",
        "dieu": 13,
        "keyword_score": 10,
    }
    final_exam_doc = {
        "doc_name": "DHKTKTCN_PDT_QD_2025_12_09_Quy che dao tao dai hoc chinh quy.docx",
        "title": "Dieu 15. Thi ket thuc hoc phan",
        "content": "Sinh vien vang mat trong ky thi ket thuc hoc phan phai du thi ky thi phu.",
        "dieu": 15,
        "keyword_score": 80,
    }

    results = retrieval._prioritize_policy_results(question, [final_exam_doc, attendance_doc])

    assert results[0]["title"] == "Dieu 13. Danh gia hoc phan"


def test_hybrid_pipeline_traces_all_steps_and_ann_fallback(monkeypatch):
    bm25_doc = _doc("bm25.pdf", 1, bm25_score=12, keyword_score=12, score=12)
    monkeypatch.setattr(retrieval, "_current_document_signature", lambda: ("sig",))
    monkeypatch.setattr(retrieval, "_get_search_cache", lambda key: None)
    monkeypatch.setattr(retrieval, "_set_search_cache", lambda key, value: None)
    monkeypatch.setattr(
        retrieval,
        "_search_metadata_documents",
        lambda question, limit: ([], {}),
    )
    monkeypatch.setattr(
        retrieval,
        "_search_bm25_documents",
        lambda question, limit, source_type_filter=None: [bm25_doc],
    )
    monkeypatch.setattr(
        retrieval,
        "search_similar_chunks",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("embedding failed")),
    )
    monkeypatch.setattr(
        retrieval,
        "rerank_documents",
        lambda question, docs, final_top_k=None: (
            docs[:1],
            {"reason": "cross_encoder_error_rrf_fallback"},
        ),
    )

    debug = {}
    results = asyncio.run(
        retrieval.search_documents(
            "quên mật khẩu",
            debug=debug,
            ambiguity_decision={
                "action": ambiguity.DIRECT_RETRIEVAL,
                "reason": "test",
            },
        )
    )

    assert results[0]["doc_name"] == "bm25.pdf"
    assert debug["expanded_queries"] == ["quên mật khẩu"]
    assert len(debug["bm25_results"]) == 1
    assert len(debug["ann_results"]) == 1
    assert len(debug["vector_errors"]) == 1
    assert debug["rrf_results"]
    assert debug["reranking"]["reason"] == "cross_encoder_error_rrf_fallback"
    assert debug["final_sources"][0]["doc_name"] == "bm25.pdf"


def test_bm25_failure_falls_back_to_ann(monkeypatch):
    ann_doc = _doc(
        "ann.pdf",
        1,
        score=0.85,
        distance=0.15,
    )
    monkeypatch.setattr(retrieval, "_current_document_signature", lambda: ("sig",))
    monkeypatch.setattr(retrieval, "_get_search_cache", lambda key: None)
    monkeypatch.setattr(retrieval, "_set_search_cache", lambda key, value: None)
    monkeypatch.setattr(
        retrieval,
        "_search_metadata_documents",
        lambda question, limit: ([], {}),
    )
    monkeypatch.setattr(
        retrieval,
        "_search_bm25_documents",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("BM25 failed")),
    )
    monkeypatch.setattr(
        retrieval,
        "search_similar_chunks",
        lambda *args, **kwargs: [ann_doc],
    )
    monkeypatch.setattr(
        retrieval,
        "rerank_documents",
        lambda question, docs, final_top_k=None: (
            docs[:1],
            {"reason": "cross_encoder_success"},
        ),
    )

    debug = {}
    results = asyncio.run(retrieval.search_documents("quy định email", debug=debug))

    assert results[0]["doc_name"] == "ann.pdf"
    assert debug["bm25_errors"] == [
        {"query": "quy định email", "error": "BM25 failed"}
    ]
    assert debug["ann_results"][0]["results"]


def test_langchain_emits_separate_hybrid_trace_steps():
    names = []

    def callback(name, output, input_data=None):
        names.append(name)

    langchain_pipeline._trace_hybrid_retrieval(
        {"trace_callback": callback},
        {
            "ambiguity": {
                "action": ambiguity.HYDE_RETRIEVAL,
                "topic": "camera",
            },
            "hyde": {"status": "success"},
            "bm25_results": [],
            "ann_results": [],
            "rrf_results": [],
            "reranking": {"reason": "cross_encoder_success"},
        },
    )

    assert names == [
        "ambiguity_detection",
        "probe_retrieval",
        "probe_evidence_decision",
        "hyde_generation",
        "grounded_hyde_generation",
        "bm25_retrieval",
        "ann_retrieval",
        "rrf_fusion",
        "cross_encoder_rerank",
    ]


def test_missing_exact_document_number_never_falls_back(monkeypatch):
    monkeypatch.setattr(retrieval, "_current_document_signature", lambda: ("sig",))
    monkeypatch.setattr(retrieval, "_get_search_cache", lambda key: None)
    monkeypatch.setattr(retrieval, "_set_search_cache", lambda key, value: None)
    monkeypatch.setattr(
        retrieval,
        "_search_metadata_documents",
        lambda question, limit: ([], {"so_van_ban": "999"}),
    )
    debug = {}
    results = asyncio.run(retrieval.search_documents("văn bản số 999", debug=debug))

    assert results == []
    assert debug["reranking"]["reason"] == "no_exact_document"


def test_rule_ambiguity_routes_known_topic_to_hyde():
    decision = ambiguity._rule_decision("camera ai quản lý")

    assert decision.action == ambiguity.HYDE_RETRIEVAL
    assert decision.topic == "camera"


def test_rule_ambiguity_routes_short_eligible_question_to_hyde():
    decision = ambiguity._rule_decision("ra trường cần gì")

    assert decision.action == ambiguity.HYDE_RETRIEVAL
    assert decision.topic == "tot_nghiep"
    assert decision.reason == "known_topic_hyde"


def test_rule_ambiguity_routes_specific_query_direct():
    decision = ambiguity._rule_decision("văn bản số 877 quy định gì")

    assert decision.action == ambiguity.DIRECT_RETRIEVAL


def test_rule_ambiguity_routes_article_and_section_directly():
    article = ambiguity._rule_decision("Điều 16 quy định trách nhiệm gì?")
    section = ambiguity._rule_decision("Mục 2 áp dụng cho đối tượng nào?")

    assert article.action == ambiguity.DIRECT_RETRIEVAL
    assert section.action == ambiguity.DIRECT_RETRIEVAL


def test_rule_ambiguity_routes_garbled_query_to_probe():
    decision = ambiguity._rule_decision("xtet đầu ra ta4 kiểu gì")

    assert decision.action == ambiguity.PROBE_RETRIEVAL
    assert decision.reason == "garbled_query_requires_probe"


def test_rule_ambiguity_routes_unknown_eligible_query_to_hyde():
    decision = ambiguity._rule_decision(
        "nội dung hoàn toàn mới chưa thuộc danh sách chủ đề viết tay hiện tại"
    )

    assert decision.action == ambiguity.HYDE_RETRIEVAL
    assert decision.reason == "eligible_query_hyde"


def test_hyde_cache_avoids_repeated_gemini_calls(monkeypatch):
    class FakeResponse:
        text = (
            "Quy định quản lý hệ thống camera giám sát mô tả trách nhiệm "
            "quản lý, vận hành, khai thác và bảo vệ dữ liệu hình ảnh."
        )

    calls = []

    def fake_generate_content(**kwargs):
        calls.append(kwargs)
        return FakeResponse()

    hyde.clear_hyde_cache()
    monkeypatch.setattr(hyde._client.models, "generate_content", fake_generate_content)

    first = hyde.generate_hyde_document("camera ai quản lý")
    second = hyde.generate_hyde_document("camera ai quản lý")

    assert first["status"] == "success"
    assert second["cache_hit"] is True
    assert len(calls) == 1


def test_ambiguity_cache_avoids_repeated_llm_calls(monkeypatch):
    calls = []
    ambiguity.clear_ambiguity_cache()
    monkeypatch.setattr(ambiguity, "_rule_decision", lambda question: None)

    def fake_llm(question):
        calls.append(question)
        return ambiguity.AmbiguityDecision(
            ambiguity.DIRECT_RETRIEVAL,
            "custom_topic",
            0.8,
            "mock_llm",
            analyzer="llm",
        )

    monkeypatch.setattr(ambiguity, "_analyze_with_llm", fake_llm)

    first = ambiguity.analyze_ambiguity("câu hỏi chưa có rule")
    second = ambiguity.analyze_ambiguity("câu hỏi chưa có rule")

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert len(calls) == 1


def test_hyde_need_clarification_stops_before_retrieval(monkeypatch):
    monkeypatch.setattr(retrieval, "_current_document_signature", lambda: ("sig",))
    monkeypatch.setattr(retrieval, "_get_search_cache", lambda key: None)
    monkeypatch.setattr(retrieval, "_set_search_cache", lambda key, value: None)
    monkeypatch.setattr(
        retrieval,
        "_search_metadata_documents",
        lambda question, limit: ([], {}),
    )
    monkeypatch.setattr(
        retrieval,
        "generate_hyde_document",
        lambda question: {
            "text": "",
            "attempted": True,
            "status": "need_clarification",
            "text_hash": None,
            "char_count": 0,
            "word_count": 0,
            "error": None,
            "cache_hit": False,
        },
    )
    monkeypatch.setattr(
        retrieval,
        "_search_bm25_documents",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("BM25 must not run")
        ),
    )
    monkeypatch.setattr(
        retrieval,
        "search_similar_chunks",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("ANN must not run")
        ),
    )

    debug = {}
    results = asyncio.run(
        retrieval.search_documents(
            "camera gì đó",
            debug=debug,
            ambiguity_decision={"action": ambiguity.HYDE_RETRIEVAL},
        )
    )

    assert results == []
    assert debug["fallback_reason"] == "hyde_requested_clarification"


def test_hyde_error_falls_back_to_direct_retrieval(monkeypatch):
    bm25_doc = _doc("direct.pdf", 1, bm25_score=6, keyword_score=6, score=6)
    monkeypatch.setattr(retrieval, "_current_document_signature", lambda: ("sig",))
    monkeypatch.setattr(retrieval, "_get_search_cache", lambda key: None)
    monkeypatch.setattr(retrieval, "_set_search_cache", lambda key, value: None)
    monkeypatch.setattr(
        retrieval,
        "_search_metadata_documents",
        lambda question, limit: ([], {}),
    )
    monkeypatch.setattr(
        retrieval,
        "generate_hyde_document",
        lambda question: {
            "text": "",
            "attempted": True,
            "status": "error_direct_fallback",
            "text_hash": None,
            "char_count": 0,
            "word_count": 0,
            "error": "Gemini unavailable",
            "cache_hit": False,
        },
    )
    monkeypatch.setattr(
        retrieval,
        "_search_bm25_documents",
        lambda *args, **kwargs: [bm25_doc],
    )
    monkeypatch.setattr(retrieval, "search_similar_chunks", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        retrieval,
        "rerank_documents",
        lambda question, docs, final_top_k=None: (
            docs,
            {"reason": "cross_encoder_success"},
        ),
    )

    debug = {}
    results = asyncio.run(
        retrieval.search_documents(
            "camera ai quản lý",
            debug=debug,
            ambiguity_decision={"action": ambiguity.HYDE_RETRIEVAL},
        )
    )

    assert results[0]["doc_name"] == "direct.pdf"
    assert debug["fallback_reason"] == "hyde_error_direct_retrieval"


def test_hyde_ann_is_fused_but_original_question_is_used_for_rerank(monkeypatch):
    bm25_doc = _doc("original.pdf", 1, bm25_score=5, keyword_score=5, score=5)
    hyde_doc = _doc("hyde.pdf", 1, score=0.9, distance=0.1)
    ann_queries = []
    rerank_questions = []

    monkeypatch.setattr(retrieval, "_current_document_signature", lambda: ("sig",))
    monkeypatch.setattr(retrieval, "_get_search_cache", lambda key: None)
    monkeypatch.setattr(retrieval, "_set_search_cache", lambda key, value: None)
    monkeypatch.setattr(
        retrieval,
        "_search_metadata_documents",
        lambda question, limit: ([], {}),
    )
    monkeypatch.setattr(
        retrieval,
        "_search_bm25_documents",
        lambda question, limit, source_type_filter=None: [bm25_doc],
    )
    monkeypatch.setattr(
        retrieval,
        "generate_hyde_document",
        lambda question: {
            "text": "hypothetical camera administrative document",
            "attempted": True,
            "status": "success",
            "text_hash": "hash",
            "char_count": 43,
            "word_count": 4,
            "error": None,
            "cache_hit": False,
        },
    )

    def fake_ann(query, **kwargs):
        ann_queries.append(query)
        return [hyde_doc] if query.startswith("hypothetical") else []

    def fake_rerank(question, docs, final_top_k=None):
        rerank_questions.append(question)
        return docs, {"reason": "cross_encoder_success"}

    monkeypatch.setattr(retrieval, "search_similar_chunks", fake_ann)
    monkeypatch.setattr(retrieval, "rerank_documents", fake_rerank)

    debug = {}
    results = asyncio.run(
        retrieval.search_documents(
            "camera ai quản lý",
            debug=debug,
            ambiguity_decision={
                "action": ambiguity.HYDE_RETRIEVAL,
                "topic": "camera",
                "confidence": 0.8,
            },
        )
    )

    assert ann_queries == [
        "camera ai quản lý",
        "hypothetical camera administrative document",
    ]
    assert rerank_questions == ["camera ai quản lý"]
    assert debug["ann_hyde_results"]
    assert debug["hyde"]["text_preview"] == "hypothetical camera administrative document"
    assert all("hypothetical" not in doc.get("content", "") for doc in results)
    assert "text" not in debug["hyde"]


def test_probe_unknown_topic_uses_grounded_hyde(monkeypatch):
    bm25_doc = _doc(
        "advisor.pdf",
        1,
        title="Tiêu chuẩn của cố vấn học tập",
        content="Cố vấn học tập phải đáp ứng các tiêu chuẩn theo quy định.",
        bm25_score=8,
        keyword_score=8,
        score=8,
    )
    grounded_doc = _doc(
        "advisor.pdf",
        2,
        title="Nhiệm vụ của cố vấn học tập",
        score=0.88,
        distance=0.12,
    )
    ann_queries = []

    monkeypatch.setattr(retrieval, "_current_document_signature", lambda: ("sig",))
    monkeypatch.setattr(retrieval, "_get_search_cache", lambda key: None)
    monkeypatch.setattr(retrieval, "_set_search_cache", lambda key, value: None)
    monkeypatch.setattr(
        retrieval,
        "_search_metadata_documents",
        lambda question, limit: ([], {}),
    )
    monkeypatch.setattr(
        retrieval,
        "_search_bm25_documents",
        lambda *args, **kwargs: [dict(bm25_doc)],
    )
    monkeypatch.setattr(
        retrieval,
        "generate_grounded_hyde_document",
        lambda question, evidence: {
            "text": "Tiêu chuẩn, nhiệm vụ và trách nhiệm của cố vấn học tập.",
            "attempted": True,
            "status": "success",
            "text_hash": "grounded-hash",
            "char_count": 58,
            "word_count": 10,
            "error": None,
            "cache_hit": False,
            "grounding_hash": "evidence-hash",
            "grounding_source_count": len(evidence),
        },
    )

    def fake_ann(query, **kwargs):
        ann_queries.append(query)
        if query.startswith("Tiêu chuẩn, nhiệm vụ"):
            return [dict(grounded_doc)]
        return [dict(bm25_doc, score=0.7, distance=0.3)]

    monkeypatch.setattr(retrieval, "search_similar_chunks", fake_ann)
    monkeypatch.setattr(
        retrieval,
        "rerank_documents",
        lambda question, docs, final_top_k=None: (
            docs[:2],
            {"reason": "cross_encoder_success"},
        ),
    )

    debug = {}
    results = asyncio.run(
        retrieval.search_documents(
            "Tiêu chuẩn của cố vấn học tập",
            debug=debug,
            ambiguity_decision={"action": ambiguity.PROBE_RETRIEVAL},
        )
    )

    assert len(ann_queries) == 2
    assert debug["probe_retrieval"]["has_confident_evidence"] is True
    assert debug["grounded_hyde"]["status"] == "success"
    assert debug["ann_grounded_hyde_results"]
    assert results


def test_probe_garbled_query_can_use_strong_document_evidence(monkeypatch):
    evidence = _doc(
        "hoc-phan.pdf",
        1,
        title="Đăng ký học phần",
        bm25_score=3,
        keyword_score=3,
        score=0.8,
        distance=0.2,
    )
    monkeypatch.setattr(retrieval, "_current_document_signature", lambda: ("sig",))
    monkeypatch.setattr(retrieval, "_get_search_cache", lambda key: None)
    monkeypatch.setattr(retrieval, "_set_search_cache", lambda key, value: None)
    monkeypatch.setattr(
        retrieval,
        "_search_metadata_documents",
        lambda question, limit: ([], {}),
    )
    monkeypatch.setattr(
        retrieval,
        "_search_bm25_documents",
        lambda *args, **kwargs: [dict(evidence)],
    )
    monkeypatch.setattr(
        retrieval,
        "search_similar_chunks",
        lambda *args, **kwargs: [dict(evidence)],
    )
    monkeypatch.setattr(
        retrieval,
        "generate_grounded_hyde_document",
        lambda question, docs: {
            "text": "Quy trình đăng ký học phần trong hệ thống đào tạo.",
            "attempted": True,
            "status": "success",
            "text_hash": "hash",
            "char_count": 49,
            "word_count": 9,
            "error": None,
            "cache_hit": False,
        },
    )
    monkeypatch.setattr(
        retrieval,
        "rerank_documents",
        lambda question, docs, final_top_k=None: (
            docs[:1],
            {"reason": "cross_encoder_success"},
        ),
    )

    debug = {}
    results = asyncio.run(
        retrieval.search_documents(
            "dk hp sao",
            debug=debug,
            ambiguity_decision={"action": ambiguity.PROBE_RETRIEVAL},
        )
    )

    assert results
    assert debug["probe_retrieval"]["has_confident_evidence"] is True
    assert debug["grounded_hyde"]["attempted"] is True


def test_probe_without_evidence_requests_clarification(monkeypatch):
    monkeypatch.setattr(retrieval, "_current_document_signature", lambda: ("sig",))
    monkeypatch.setattr(retrieval, "_get_search_cache", lambda key: None)
    monkeypatch.setattr(retrieval, "_set_search_cache", lambda key, value: None)
    monkeypatch.setattr(
        retrieval,
        "_search_metadata_documents",
        lambda question, limit: ([], {}),
    )
    monkeypatch.setattr(
        retrieval,
        "_search_bm25_documents",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(retrieval, "search_similar_chunks", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        retrieval,
        "generate_grounded_hyde_document",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Grounded HyDE must not run without evidence")
        ),
    )

    debug = {}
    results = asyncio.run(
        retrieval.search_documents(
            "xyz abc không xác định",
            debug=debug,
            ambiguity_decision={"action": ambiguity.PROBE_RETRIEVAL},
        )
    )

    assert results == []
    assert debug["fallback_reason"] == "probe_insufficient_evidence"
    assert debug["probe_retrieval"]["decision"] == "clarification_needed"


def test_cached_empty_probe_keeps_clarification_decision(monkeypatch):
    monkeypatch.setattr(retrieval, "_current_document_signature", lambda: ("sig",))
    monkeypatch.setattr(retrieval, "_get_search_cache", lambda key: [])

    debug = {}
    results = asyncio.run(
        retrieval.search_documents(
            "xyz abc không xác định",
            debug=debug,
            ambiguity_decision={"action": ambiguity.PROBE_RETRIEVAL},
        )
    )

    assert results == []
    assert debug["cache_hit"] is True
    assert debug["fallback_reason"] == "probe_insufficient_evidence"


def test_grounded_hyde_error_falls_back_to_original_retrieval(monkeypatch):
    original = _doc(
        "advisor.pdf",
        1,
        title="Cố vấn học tập",
        bm25_score=5,
        keyword_score=5,
        score=0.75,
        distance=0.25,
    )
    monkeypatch.setattr(retrieval, "_current_document_signature", lambda: ("sig",))
    monkeypatch.setattr(retrieval, "_get_search_cache", lambda key: None)
    monkeypatch.setattr(retrieval, "_set_search_cache", lambda key, value: None)
    monkeypatch.setattr(
        retrieval,
        "_search_metadata_documents",
        lambda question, limit: ([], {}),
    )
    monkeypatch.setattr(
        retrieval,
        "_search_bm25_documents",
        lambda *args, **kwargs: [dict(original)],
    )
    monkeypatch.setattr(
        retrieval,
        "search_similar_chunks",
        lambda *args, **kwargs: [dict(original)],
    )
    monkeypatch.setattr(
        retrieval,
        "generate_grounded_hyde_document",
        lambda *args, **kwargs: {
            "text": "",
            "attempted": True,
            "status": "error_direct_fallback",
            "text_hash": None,
            "char_count": 0,
            "word_count": 0,
            "error": "Gemini unavailable",
            "cache_hit": False,
        },
    )
    monkeypatch.setattr(
        retrieval,
        "rerank_documents",
        lambda question, docs, final_top_k=None: (
            docs[:1],
            {"reason": "cross_encoder_success"},
        ),
    )

    debug = {}
    results = asyncio.run(
        retrieval.search_documents(
            "cố vấn hỗ trợ gì",
            debug=debug,
            ambiguity_decision={"action": ambiguity.PROBE_RETRIEVAL},
        )
    )

    assert results
    assert debug["fallback_reason"] == "grounded_hyde_error_original_retrieval"


def test_probe_embedding_error_can_fall_back_to_bm25(monkeypatch):
    bm25_doc = _doc(
        "advisor.pdf",
        1,
        title="Tiêu chuẩn cố vấn học tập",
        bm25_score=8,
        keyword_score=8,
        score=8,
    )
    monkeypatch.setattr(retrieval, "_current_document_signature", lambda: ("sig",))
    monkeypatch.setattr(retrieval, "_get_search_cache", lambda key: None)
    monkeypatch.setattr(retrieval, "_set_search_cache", lambda key, value: None)
    monkeypatch.setattr(
        retrieval,
        "_search_metadata_documents",
        lambda question, limit: ([], {}),
    )
    monkeypatch.setattr(
        retrieval,
        "_search_bm25_documents",
        lambda *args, **kwargs: [dict(bm25_doc)],
    )
    monkeypatch.setattr(
        retrieval,
        "search_similar_chunks",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("embedding unavailable")
        ),
    )
    monkeypatch.setattr(
        retrieval,
        "generate_grounded_hyde_document",
        lambda *args, **kwargs: {
            "text": "Tiêu chuẩn của cố vấn học tập.",
            "attempted": True,
            "status": "success",
            "text_hash": "hash",
            "char_count": 34,
            "word_count": 6,
            "error": None,
            "cache_hit": False,
        },
    )
    monkeypatch.setattr(
        retrieval,
        "rerank_documents",
        lambda question, docs, final_top_k=None: (
            docs[:1],
            {"reason": "cross_encoder_success"},
        ),
    )

    debug = {}
    results = asyncio.run(
        retrieval.search_documents(
            "tiêu chuẩn cố vấn",
            debug=debug,
            ambiguity_decision={"action": ambiguity.PROBE_RETRIEVAL},
        )
    )

    assert results[0]["doc_name"] == "advisor.pdf"
    assert len(debug["vector_errors"]) == 2
