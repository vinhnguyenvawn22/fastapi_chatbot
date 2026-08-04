# FastAPI Chatbot RAG UNETI

## 1. Tổng Quan

Dự án này là một chatbot RAG xây dựng bằng FastAPI. Ứng dụng nhận câu hỏi của người dùng, phân loại ý định, truy xuất nguồn phù hợp từ tài liệu nội bộ, tài liệu nghiệp vụ hoặc website UNETI, sau đó dùng Gemini để tạo câu trả lời kèm nguồn tham chiếu.

Các nhóm chức năng chính:

- Chat tổng hợp: tự chọn nguồn phù hợp giữa tài liệu nội bộ, tài liệu nghiệp vụ và website UNETI.
- Chat theo nguồn: ép hệ thống chỉ hỏi tài liệu nội bộ, nghiệp vụ hoặc website.
- Tra cứu nghiệp vụ không nhất thiết gọi LLM nếu câu hỏi khớp FAQ mapping.
- Quản lý hội thoại ẩn danh bằng cookie session và SQLite.
- Upload, đọc, chunk và index tài liệu PDF/DOCX.
- Hybrid retrieval: metadata search, BM25, vector search ChromaDB, RRF fusion, HyDE, query expansion, ambiguity detection và cross-encoder rerank.
- Giao diện web cơ bản ở `/` và `/chat-ui`.

## 2. Công Nghệ Sử Dụng

- Python, FastAPI, Uvicorn.
- Pydantic cho request/response schema.
- Gemini qua `google-genai`.
- Google Cloud Discovery Engine cho tìm kiếm website UNETI nếu có cấu hình.
- ChromaDB làm vector store local.
- Sentence Transformers để tạo embedding.
- `rank-bm25` cho lexical/BM25 retrieval.
- `pypdf` và `python-docx` để đọc tài liệu.
- SQLite để lưu phiên chat, thread, message và idempotency request.
- Pytest và FastAPI TestClient cho test.

## 3. Cấu Trúc Thư Mục Thực Tế

```text
fastapi_chatbot/
├── app/
│   ├── main.py
│   ├── controller/
│   │   ├── business_controller.py
│   │   ├── chatbot_controller.py
│   │   ├── document_controller.py
│   │   └── website_controller.py
│   ├── core/
│   │   ├── config.py
│   │   └── constants.py
│   ├── data/
│   │   ├── ambiguity_analyzer.py
│   │   ├── business_knowledge.py
│   │   ├── business_mapping_store.py
│   │   ├── contextualizer.py
│   │   ├── conversation_context.py
│   │   ├── conversation_repository.py
│   │   ├── elasticsearch_client.py
│   │   ├── embedding_client.py
│   │   ├── gemini_client.py
│   │   ├── hyde.py
│   │   ├── langchain_pipeline.py
│   │   ├── preload.py
│   │   ├── prompt_builder.py
│   │   ├── query_analyzer.py
│   │   ├── query_context.py
│   │   ├── query_expander.py
│   │   ├── query_expansion.py
│   │   ├── reranker.py
│   │   ├── trace_logger.py
│   │   ├── vector_store.py
│   │   └── website_search_client.py
│   ├── routers/
│   │   ├── business_router.py
│   │   ├── chat_router.py
│   │   ├── document_router.py
│   │   ├── health_router.py
│   │   ├── page_router.py
│   │   └── website_router.py
│   ├── schemas/
│   │   ├── business_schema.py
│   │   ├── chat_schema.py
│   │   ├── document_schema.py
│   │   └── website_schema.py
│   ├── services/
│   │   ├── __init__.py
│   │   └── conversation_service.py
│   ├── static/
│   │   └── uneti-ai-hero.png
│   └── templates/
│       ├── chat_ui.html
│       └── landing.html
├── documents/
│   └── nghiep_vu/
├── scripts/
│   ├── benchmark_retrieval.py
│   ├── extract_pcntt_mapping.py
│   └── reindex_documents.py
├── storage/
│   ├── business_knowledge_index/
│   │   └── index.json
│   ├── business_mapping/
│   │   └── pcntt_mapping.json
│   └── document_index/
│       └── index.json
├── tests/
├── uploads/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

Ngoài ra repo hiện có một số file log cũ như `uvicorn-8501.*.log` và file tài liệu nguồn `PCNTT_MAPPING_FILE.docx`.

## 4. Vai Trò Các Module Chính

### `app/main.py`

Entrypoint FastAPI. File này:

- Tạo app với lifespan.
- Khởi tạo `ConversationRepository` dùng SQLite.
- Preload các thành phần RAG nếu bật cấu hình.
- Mount static tại `/static`.
- Gắn middleware reset bộ đếm Gemini mỗi request.
- Gắn middleware tạo cookie session ẩn danh và `chat_owner_id`.
- Include các router: health, chat, nghiệp vụ, website, documents và page.

### `app/routers`

Router chỉ nhận request, gọi controller/service và trả response:

- `chat_router.py`: các endpoint chat, trace và thread.
- `business_router.py`: API hỏi/tra cứu nghiệp vụ.
- `website_router.py`: API tra cứu website UNETI.
- `document_router.py`: upload, liệt kê và đọc text tài liệu.
- `page_router.py`: trả HTML landing và chat UI.
- `health_router.py`: kiểm tra server.

### `app/controller`

Điều phối nghiệp vụ:

- `chatbot_controller.py`: luồng chat chính, phân loại câu hỏi, chọn nguồn, gọi retrieval, kiểm tra evidence, build sources, finalize trace.
- `business_controller.py`: hỏi đáp nghiệp vụ theo mapping và search nguồn nghiệp vụ.
- `website_controller.py`: tra cứu website UNETI, chuẩn hóa kết quả và trace.
- `document_controller.py`: xử lý upload, validate file, đọc PDF/DOCX, tách chunk và metadata.

### `app/data`

Data layer và RAG logic:

- `elasticsearch_client.py`: tên file hơi lịch sử, hiện xử lý index tài liệu local, metadata search, BM25, vector search, RRF, HyDE/probe và cache.
- `vector_store.py`: ChromaDB persistent collection, index và search vector chunks.
- `embedding_client.py`: tải/cache SentenceTransformer và tạo embedding.
- `reranker.py`: cross-encoder rerank.
- `hyde.py`: sinh HyDE và grounded HyDE bằng Gemini.
- `query_analyzer.py`: phân loại intent và trích metadata như số văn bản, điều, mục, chương, ngày.
- `query_context.py`: nhận diện đối tượng hỏi như sinh viên/CBGV và loại nhu cầu như thủ tục UI hay văn bản chính sách.
- `ambiguity_analyzer.py`: quyết định direct retrieval, HyDE, probe retrieval hoặc hỏi làm rõ.
- `query_expander.py` và `query_expansion.py`: mở rộng truy vấn bằng rule/cache/Gemini.
- `business_knowledge.py`: index và search tài liệu nghiệp vụ, FAQ mapping, guided retrieval theo location/keyword/vector và fallback generic hybrid.
- `business_mapping_store.py`: search file mapping JSON `pcntt_mapping.json`.
- `website_search_client.py`: tìm kiếm website UNETI qua Discovery Engine, fallback HTML/sitemap/category, tải nội dung trang/attachment, index website chunks.
- `prompt_builder.py`: ghép context thành `<NGUON>...</NGUON>` và tạo prompt cho tài liệu/website.
- `langchain_pipeline.py`: pipeline retrieve + prompt + generate answer, có trace từng bước.
- `conversation_repository.py`: SQLite repository cho session/thread/message/chat_requests.
- `contextualizer.py`: viết lại câu hỏi phụ thuộc lịch sử hội thoại.
- `conversation_context.py`: contextvar giữ thông tin hội thoại trong request.
- `trace_logger.py`: ghi trace JSON vào `storage/traces`.
- `preload.py`: preload model/index theo cấu hình.

### `app/schemas`

Định nghĩa Pydantic schema:

- `ChatRequest`, `ChatResponse`, `ChatSource`, thread/message/trace response.
- `BusinessAskRequest`, `BusinessAskResponse`, `BusinessSearchRequest`, `BusinessSearchResponse`.
- `WebsiteSearchRequest`, `WebsiteSearchResponse`.
- `DocumentResponse`.

### `app/services`

- `conversation_service.py`: tạo session token, hash session, validate UUID, quản lý lifecycle chat request. Service này hỗ trợ idempotency bằng `request_id`, lưu user/assistant message, rewrite câu hỏi theo history, xử lý replay response nếu request trùng.

### `scripts`

- `reindex_documents.py`: xóa vector cũ và index lại toàn bộ tài liệu được hỗ trợ.
- `extract_pcntt_mapping.py`: đọc `PCNTT_MAPPING_FILE.docx` và xuất `storage/business_mapping/pcntt_mapping.json`.
- `benchmark_retrieval.py`: benchmark retrieval hoặc benchmark synthetic song song.

### `tests`

Test bao phủ:

- App, docs, OpenAPI, chat router.
- Chat endpoints, prompt, routing, fallback, trace.
- Hội thoại/thread/session/idempotency/soft delete.
- Business API, FAQ mapping, guided retrieval, audience routing.
- Hybrid retrieval, query expansion, ambiguity, HyDE, rerank.
- Website search API với mock.

## 5. Luồng Xử Lý Chat/RAG

### Chat tổng hợp `/api/chat/`

1. Middleware đọc hoặc tạo cookie session ẩn danh.
2. `ConversationService.chat()` nhận `question`, `request_id`, `thread_id`.
3. Service claim request trong SQLite để chống gửi trùng.
4. Lấy lịch sử hội thoại gần nhất, nếu cần thì gọi `contextualizer` để viết lại câu hỏi độc lập.
5. `chatbot_controller.handle_chat()` tạo trace và phân loại câu hỏi bằng `query_analyzer`.
6. Hệ thống chọn hướng xử lý:
   - Website UNETI nếu câu hỏi có tín hiệu website/tin tức/link.
   - Tài liệu nghiệp vụ nếu câu hỏi liên quan Web Support, thủ tục, tra cứu nghiệp vụ.
   - Tài liệu nội bộ nếu câu hỏi hỏi quy định, quyết định, điều, mục, chương, số văn bản.
   - Ngoài phạm vi hoặc general advice sẽ trả câu trả lời an toàn theo rule.
7. Retrieval lấy nguồn liên quan:
   - Nội bộ: metadata/BM25/vector/HyDE/RRF/cross-encoder.
   - Nghiệp vụ: FAQ mapping, guided source search, generic hybrid.
   - Website: Discovery Engine hoặc fallback crawling/search, sau đó có thể index vào vector store với `source_type=website_uneti`.
8. Controller kiểm tra nguồn có đủ tin cậy không.
9. `prompt_builder` ghép context và prompt.
10. `gemini_client.ask_gemini()` gọi Gemini.
11. Response được làm sạch, gắn nguồn, ghi trace vào `storage/traces`, lưu message vào SQLite và trả về client.

### Upload và index tài liệu

1. `POST /api/documents/upload` nhận file PDF.
2. `document_controller` validate tên file, MIME type, kích thước và nội dung PDF.
3. File được lưu vào `DOCUMENTS_DIR`.
4. Cache document index bị clear.
5. Tài liệu được tách chunk, trích metadata và index vào ChromaDB.

## 6. API Endpoint Quan Trọng

Các prefix được khai báo trong `app/main.py`.

| Method | Endpoint | Mục đích |
| --- | --- | --- |
| GET | `/` | Trang landing HTML |
| GET | `/chat-ui` | Giao diện chat HTML |
| GET | `/api/health/` | Health check |
| POST | `/api/chat/` | Chat tổng hợp, tự chọn nguồn |
| POST | `/api/chat/business` | Chat chỉ dùng nguồn nghiệp vụ |
| POST | `/api/chat/internal` | Chat chỉ dùng tài liệu nội bộ |
| POST | `/api/chat/website` | Chat chỉ dùng website UNETI |
| GET | `/api/chat/traces/{trace_id}` | Xem trace debug của một câu hỏi |
| POST | `/api/chat/threads` | Tạo thread chat |
| GET | `/api/chat/threads` | Danh sách thread của session hiện tại |
| GET | `/api/chat/threads/{thread_id}` | Chi tiết thread |
| GET | `/api/chat/threads/{thread_id}/messages` | Danh sách message trong thread |
| DELETE | `/api/chat/threads/{thread_id}` | Soft delete thread |
| POST | `/api/nghiep-vu/ask` | Hỏi đáp theo FAQ mapping nghiệp vụ |
| POST | `/api/nghiep-vu/search` | Search nguồn nghiệp vụ, không bắt buộc gọi LLM |
| POST | `/api/website/search` | Tra cứu trực tiếp website UNETI |
| POST | `/api/documents/upload` | Upload PDF và index |
| GET | `/api/documents/` | Liệt kê tài liệu |
| GET | `/api/documents/{file_name}/text` | Đọc text đã trích xuất từ PDF/DOCX |

Ví dụ request chat:

```json
{
  "question": "Sinh viên xem điểm ở đâu?",
  "request_id": "uuid-hoac-id-duy-nhat",
  "thread_id": null
}
```

`request_id` là bắt buộc và dùng để chống xử lý trùng. Nếu gửi lại cùng `request_id` với cùng nội dung, hệ thống có thể replay response đã lưu.

## 7. Biến Môi Trường

Tạo file `.env` từ `.env.example`. Không commit `.env` thật.

Biến cấu hình mô hình local:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:4b-instruct
OLLAMA_TIMEOUT_SECONDS=180
```

Qwen chạy trên máy qua Ollama nên không cần API key và không phát sinh phí theo token.

Biến thường dùng:

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3:4b
DOCUMENTS_DIR=uploads/Tong hop van ban AI
DOCUMENT_INDEX_CACHE_ENABLED=true
DOCUMENT_INDEX_CACHE_FILE=storage/document_index/index.json
BUSINESS_DOCUMENTS_DIR=documents/nghiep_vu
BUSINESS_INDEX_CACHE_ENABLED=true
BUSINESS_INDEX_CACHE_FILE=storage/business_knowledge_index/index.json
BUSINESS_MAPPING_FILE=storage/business_mapping/pcntt_mapping.json

SEARCH_TOP_K=3
MAX_CONTEXT_CHUNKS=3
MAX_CONTEXT_CHARS=12000
CHUNK_SIZE=2000
CHUNK_OVERLAP=500

EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
VECTOR_STORE_DIR=storage/chroma_db
VECTOR_COLLECTION_NAME=document_chunks
VECTOR_MAX_DISTANCE=0.75

CHAT_DATABASE_FILE=storage/chat_history.sqlite3
CHAT_SESSION_COOKIE_NAME=chat_session
CHAT_SESSION_MAX_AGE_SECONDS=2592000
CHAT_COOKIE_SECURE=false
CHAT_COOKIE_SAMESITE=lax
CHAT_HISTORY_MAX_MESSAGES=10
CHAT_HISTORY_MAX_CHARS=6000
CHAT_QUESTION_MAX_CHARS=4000
```

Biến cho website UNETI/Discovery Engine:

```env
UNETI_WEBSITE_DOMAIN=uneti.edu.vn
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
DISCOVERY_PROJECT_NUMBER=your_google_cloud_project_number
DISCOVERY_LOCATION=global
DISCOVERY_COLLECTION_ID=default_collection
DISCOVERY_ENGINE_ID=your_discovery_engine_id
DISCOVERY_SERVING_CONFIG_ID=default_search
WEBSITE_SEARCH_TOP_K=10
WEBSITE_RERANK_TOP_K=2
WEBSITE_MIN_SOURCE_SCORE=50
```

Nếu thiếu cấu hình Discovery Engine, website search có cơ chế fallback trong code, nhưng chất lượng/phạm vi có thể khác tùy truy vấn và kết nối mạng.

## 8. Cài Đặt Và Chạy Dự Án

### Tạo môi trường Python

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### Cài thư viện

```bash
pip install -r requirements.txt
```

### Tạo file môi trường

```bash
cp .env.example .env
```

Trên Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

### Cài Ollama và tải Qwen

Cài Ollama cho Windows, mở Ollama rồi chạy:

```powershell
ollama run qwen3:4b-instruct
```

Lệnh này tự tải model ở lần chạy đầu tiên. Sau khi model trả lời được trong terminal, có thể thoát phiên chat bằng `/bye` rồi chạy server FastAPI.

### Chạy server

```bash
uvicorn app.main:app --reload
```

Mặc định mở:

- App: `http://127.0.0.1:8000`
- Chat UI: `http://127.0.0.1:8000/chat-ui`
- Swagger: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

### Chạy test

```bash
pytest
```

Nhiều test tự set `GEMINI_API_KEY=test-gemini-api-key` và mock các phần gọi ngoài, nên không nhất thiết gọi Gemini thật.

## 9. Dữ Liệu, Tài Liệu Và Index

- `documents/nghiep_vu/`: tài liệu nghiệp vụ gốc, gồm DOCX/PDF hướng dẫn Web Support, quy chế/quy định.
- `PCNTT_MAPPING_FILE.docx`: file mapping nguồn FAQ PCNTT ở root.
- `storage/business_mapping/pcntt_mapping.json`: mapping FAQ đã trích xuất từ file DOCX.
- `storage/business_knowledge_index/index.json`: cache index nghiệp vụ.
- `storage/document_index/index.json`: cache index tài liệu nội bộ.
- `storage/chroma_db/`: vector store ChromaDB local, có thể được tạo khi index.
- `storage/chat_history.sqlite3`: SQLite lưu session/thread/message, có thể được tạo khi chạy.
- `storage/traces/`: JSON trace debug từng câu hỏi, có thể được tạo khi chat.
- `uploads/`: nơi lưu tài liệu upload theo `DOCUMENTS_DIR`.

## 10. Script Hỗ Trợ

Tạo lại mapping FAQ PCNTT:

```bash
python scripts/extract_pcntt_mapping.py
```

Index lại toàn bộ tài liệu vào ChromaDB:

```bash
python scripts/reindex_documents.py
```

Benchmark retrieval:

```bash
python scripts/benchmark_retrieval.py --rounds 1
```

Benchmark synthetic song song:

```bash
python scripts/benchmark_retrieval.py --synthetic
```

## 11. Lưu Ý Khi Phát Triển

- Không commit `.env`, API key, service account hoặc credential thật.
- `app/core/config.py` hiện raise lỗi nếu thiếu `GEMINI_API_KEY`, nên cần set biến môi trường trước khi import app.
- Một số file/comment cũ có dấu hiệu lỗi encoding mojibake. Khi sửa, nên lưu file mới bằng UTF-8.
- `elasticsearch_client.py` không nhất thiết đang dùng Elasticsearch thật; tên file là lịch sử, logic hiện tại là retrieval local/hybrid.
- Khi thêm endpoint, giữ pattern router mỏng, controller điều phối, data layer xử lý retrieval/model.
- Khi thay đổi schema chat, cần kiểm tra lại tests liên quan `test_chat_langchain.py`, `test_conversations.py` và frontend trong `chat_ui.html`.
- Khi thêm tài liệu mới thủ công, cần reindex nếu muốn vector search thấy ngay.
- Website search phụ thuộc cấu hình Google Discovery Engine và fallback network. Khi test nên mock để tránh gọi dịch vụ thật.
- `request_id` trong `ChatRequest` là bắt buộc để đảm bảo idempotency.
- Cookie session là ẩn danh, không nên log hoặc expose token gốc; hệ thống chỉ lưu hash session.

## 12. Context Cho AI Ở Chat Mới

Bạn đang làm trong dự án `fastapi_chatbot`, một chatbot RAG FastAPI cho UNETI. Entrypoint là `app/main.py`. App có middleware tạo session chat ẩn danh bằng cookie, lưu hội thoại vào SQLite qua `ConversationRepository`, và expose các route `/api/chat`, `/api/nghiep-vu`, `/api/website`, `/api/documents`, `/api/health`, `/`, `/chat-ui`.

Luồng chat chính nằm ở `app/controller/chatbot_controller.py`, gọi pipeline trong `app/data/langchain_pipeline.py`. Retrieval nội bộ nằm chủ yếu ở `app/data/elasticsearch_client.py`, vector Chroma ở `app/data/vector_store.py`, embedding ở `app/data/embedding_client.py`, prompt ở `app/data/prompt_builder.py`, Gemini ở `app/data/gemini_client.py`. Nghiệp vụ Web Support/PCNTT nằm ở `app/data/business_knowledge.py` và `app/data/business_mapping_store.py`. Website UNETI nằm ở `app/data/website_search_client.py`. Quản lý hội thoại nằm ở `app/services/conversation_service.py` và `app/data/conversation_repository.py`.

Trước khi sửa code hãy đọc file liên quan bằng `rg --files` và `Get-Content`. Không đọc/ghi lộ `.env` thật. Nếu chạy app dùng `uvicorn app.main:app --reload`; nếu chạy test dùng `pytest`. Giữ thay đổi nhỏ, đúng module, không tự bịa chức năng ngoài code hiện có.
