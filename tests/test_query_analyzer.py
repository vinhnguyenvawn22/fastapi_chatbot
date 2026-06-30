import os


os.environ.setdefault("GEMINI_API_KEY", "test-gemini-api-key")

from app.data.query_analyzer import QueryIntent, classify_query


def test_weather_question_is_out_of_scope():
    analysis = classify_query("nhiệt độ hôm nay là gì")

    assert analysis.intent == QueryIntent.OUT_OF_SCOPE
    assert analysis.reason == "out_of_scope_terms"


def test_document_question_still_routes_to_internal_documents():
    analysis = classify_query("điều kiện học bổng")

    assert analysis.intent == QueryIntent.INTERNAL_DOCUMENT
