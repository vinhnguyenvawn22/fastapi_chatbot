# FastAPI Chatbot Metadata

## 1. Giới thiệu dự án

Dự án xây dựng một chatbot sử dụng FastAPI, có khả năng nhận câu hỏi từ người dùng, tìm kiếm nội dung liên quan trong tài liệu đã xử lý metadata, sau đó trả lời kèm nguồn tài liệu.

Mục tiêu chính của dự án:

* Xây dựng API chatbot bằng FastAPI.
* Chia code rõ ràng theo từng module: `routers`, `controller`, `data`, `schemas`, `core`.
* Có giao diện web chatbot đơn giản để người dùng nhập câu hỏi và nhận câu trả lời.
* Mô phỏng hoặc triển khai luồng RAG: nhận câu hỏi, tìm tài liệu liên quan, xây dựng context, tạo prompt và trả lời kèm nguồn.
* Hỗ trợ vector search bằng ChromaDB để cải thiện khả năng tìm kiếm tài liệu.
* Làm việc nhóm bằng GitHub theo quy trình: clone repo, tạo branch, commit, push và tạo Pull Request.
* Không code trực tiếp trên nhánh `main`.

---

## 2. Thành viên nhóm và phân công công việc

Nhóm gồm 4 thành viên. Mỗi thành viên phụ trách một phần riêng để tránh sửa trùng file và hạn chế conflict khi làm việc nhóm.

| STT | Thành viên         | Branch phụ trách      | Folder/File chính                                                      | Nhiệm vụ                                                                                            |
| --- | ------------------ | --------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| 1   | Nguyễn Thị Hải Yến | `feature/routers`     | `app/routers`                                                          | Tạo các API endpoint, nhận request từ người dùng và gọi sang controller                             |
| 2   | Nguyễn Văn Vinh    | `feature/controllers` | `app/controller`                                                       | Điều phối logic chatbot, xử lý luồng nghiệp vụ chính                                                |
| 3   | Phương Thảo        | `feature/data-layer`  | `app/data`                                                             | Xử lý tìm kiếm dữ liệu, metadata, context, prompt và gọi model                                      |
| 4   | Nguyễn Chính Nghĩa | `docs-and-tests`      | `README.md`, `tests`, `.gitignore`, `.env.example`, `requirements.txt` | Viết tài liệu hướng dẫn, test API, kiểm tra workflow GitHub và đảm bảo project có thể chạy lại được |

### 2.1. Chi tiết công việc từng thành viên

#### Thành viên 1: Routers

Phụ trách các file:

* `app/routers/chat_router.py`
* `app/routers/document_router.py`
* `app/routers/health_router.py`
* `app/routers/page_router.py`

Nhiệm vụ:

* Tạo endpoint nhận câu hỏi chatbot.
* Tạo endpoint kiểm tra server.
* Tạo endpoint upload tài liệu.
* Tạo route giao diện web chatbot tại `/`.
* Router chỉ nên nhận request, gọi controller và trả response.
* Không viết logic xử lý dài trong router.

---

#### Thành viên 2: Controllers

Phụ trách các file:

* `app/controller/chatbot_controller.py`
* `app/controller/document_controller.py`

Nhiệm vụ:

* Điều phối luồng xử lý chatbot.
* Nhận dữ liệu từ router.
* Gọi data layer để tìm kiếm tài liệu.
* Gọi prompt builder để tạo prompt.
* Gọi Gemini hoặc hàm xử lý model.
* Trả response đúng định dạng cho router.

---

#### Thành viên 3: Data Layer

Phụ trách các file:

* `app/data/elasticsearch_client.py`
* `app/data/embedding_client.py`
* `app/data/vector_store.py`
* `app/data/prompt_builder.py`
* `app/data/gemini_client.py`

Nhiệm vụ:

* Tạo hàm tìm kiếm tài liệu liên quan.
* Tạo embedding cho câu hỏi và nội dung tài liệu.
* Lưu và tìm kiếm vector bằng ChromaDB.
* Xây dựng context từ tài liệu tìm được.
* Xây dựng prompt cho chatbot.
* Tạo hàm gọi mô hình Gemini.
* Đảm bảo câu trả lời có kèm nguồn tài liệu.

---

#### Thành viên 4: Docs and Tests

Phụ trách các file:

* `README.md`
* `tests/`
* `.gitignore`
* `.env.example`
* `requirements.txt`

Nhiệm vụ:

* Viết hướng dẫn cài đặt và chạy project.
* Viết hướng dẫn Git workflow cho nhóm.
* Kiểm tra API trên Swagger.
* Viết test cơ bản cho web chatbot và API.
* Cập nhật danh sách thư viện cần cài trong `requirements.txt`.
* Đảm bảo không đẩy file nhạy cảm như `.env`, API key, credential, thư mục `venv`, file upload thật hoặc database local lên GitHub.

---

## 3. Cấu trúc thư mục dự án

```text
fastapi_chatbot/
├── app/
│   ├── main.py
│   │
│   ├── routers/
│   │   ├── page_router.py
│   │   ├── chat_router.py
│   │   ├── document_router.py
│   │   └── health_router.py
│   │
│   ├── controller/
│   │   ├── chatbot_controller.py
│   │   └── document_controller.py
│   │
│   ├── data/
│   │   ├── elasticsearch_client.py
│   │   ├── embedding_client.py
│   │   ├── vector_store.py
│   │   ├── prompt_builder.py
│   │   └── gemini_client.py
│   │
│   ├── schemas/
│   │   ├── chat_schema.py
│   │   └── document_schema.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   └── constants.py
│   │
│   └── templates/
│       └── chat_ui.html
│
├── scripts/
│   └── reindex_documents.py
│
├── storage/
│   └── chroma_db/
│
├── uploads/
├── tests/
│   └── test_app.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 4. Vai trò từng thư mục

| Thư mục          | Vai trò                                                                            |
| ---------------- | ---------------------------------------------------------------------------------- |
| `app/routers`    | Nhận request từ web hoặc Swagger, gọi controller và trả response                   |
| `app/controller` | Điều phối luồng xử lý nghiệp vụ                                                    |
| `app/data`       | Xử lý dữ liệu, embedding, vector search, build context, build prompt và gọi Gemini |
| `app/schemas`    | Định nghĩa request/response bằng Pydantic                                          |
| `app/core`       | Chứa cấu hình chung, biến môi trường và hằng số                                    |
| `app/templates`  | Chứa giao diện web chatbot                                                         |
| `scripts`        | Chứa script hỗ trợ, ví dụ reindex tài liệu                                         |
| `storage`        | Chứa dữ liệu sinh ra khi chạy vector database local                                |
| `uploads`        | Chứa tài liệu upload khi chạy project                                              |
| `tests`          | Chứa các file kiểm thử API                                                         |

---

## 5. Hướng dẫn cho người mới vào nhóm

Phần này dành cho thành viên mới khi bắt đầu tham gia project.

### Bước 1: Clone project về máy

Mở Git Bash hoặc terminal, sau đó chạy:

```bash
git clone https://github.com/vinhnguyenvawn22/fastapi_chatbot.git
cd fastapi_chatbot
```

### Bước 2: Kiểm tra các branch hiện có

```bash
git branch -a
```

Nếu thấy có `main` và `develop` là đúng.

### Bước 3: Chuyển sang nhánh `develop`

Không code trực tiếp trên `main`.

```bash
git checkout develop
git pull origin develop
```

Nếu repository chưa có nhánh `develop`, có thể làm tạm trên nhánh riêng được tạo từ `main`.

### Bước 4: Tạo branch riêng để làm việc

Mỗi thành viên tạo một branch riêng từ `develop`.

```bash
git checkout -b ten-branch
```

Ví dụ:

```bash
git checkout -b feature/routers
git checkout -b feature/controllers
git checkout -b feature/data-layer
git checkout -b docs-and-tests
```

### Bước 5: Kiểm tra đang ở branch nào

```bash
git branch
```

Nếu thấy dấu `*` ở branch của mình là đúng.

Ví dụ:

```text
  develop
* docs-and-tests
  main
```

---

## 6. Quy tắc đặt tên branch

| Loại công việc                    | Cách đặt tên            | Ví dụ                     |
| --------------------------------- | ----------------------- | ------------------------- |
| Thêm chức năng mới                | `feature/ten-chuc-nang` | `feature/chat-api`        |
| Sửa lỗi                           | `fix/ten-loi`           | `fix/search-empty-result` |
| Viết tài liệu                     | `docs/noi-dung`         | `docs/update-readme`      |
| Viết test                         | `test/noi-dung`         | `test/chat-api`           |
| Tài liệu và test của thành viên 4 | `docs-and-tests`        | `docs-and-tests`          |

---

## 7. Quy trình làm việc với Git

### Bước 1: Trước khi code, luôn cập nhật code mới nhất

```bash
git checkout develop
git pull origin develop
```

Sau đó tạo branch mới:

```bash
git checkout -b feature/ten-chuc-nang
```

Ví dụ:

```bash
git checkout -b docs-and-tests
```

### Bước 2: Code phần việc của mình

Mỗi thành viên chỉ nên sửa các file thuộc phần mình phụ trách.

Ví dụ:

* Người làm routers sửa trong `app/routers`.
* Người làm controllers sửa trong `app/controller`.
* Người làm data layer sửa trong `app/data`.
* Người làm docs/tests sửa `README.md`, `.gitignore`, `.env.example`, `requirements.txt`, `tests`.

### Bước 3: Kiểm tra file đã thay đổi

```bash
git status
```

Nếu có file màu đỏ hoặc xanh nghĩa là có thay đổi chưa commit.

### Bước 4: Thêm file vào Git

Nên add từng file cụ thể để tránh đẩy nhầm file không cần thiết:

```bash
git add README.md
git add .env.example
git add .gitignore
git add requirements.txt
git add tests/test_app.py
```

Không nên dùng `git add .` nếu chưa kiểm tra kỹ vì có thể add nhầm `.env`, `venv`, `storage` hoặc `uploads`.

### Bước 5: Commit code

Cú pháp:

```bash
git commit -m "loai: mo ta ngan gon noi dung da lam"
```

Ví dụ:

```bash
git commit -m "docs: update setup guide and add basic tests"
```

Một số mẫu commit nên dùng:

```bash
git commit -m "feat: add chat api"
git commit -m "feat: add document upload router"
git commit -m "fix: handle empty search result"
git commit -m "docs: update readme git workflow"
git commit -m "test: add chat api test"
```

Ý nghĩa một số loại commit:

| Loại       | Ý nghĩa                                      |
| ---------- | -------------------------------------------- |
| `feat`     | Thêm chức năng mới                           |
| `fix`      | Sửa lỗi                                      |
| `docs`     | Sửa tài liệu                                 |
| `test`     | Thêm hoặc sửa test                           |
| `refactor` | Sửa lại code nhưng không đổi chức năng       |
| `chore`    | Công việc phụ như cấu hình, cài đặt, dọn dẹp |

### Bước 6: Push branch lên GitHub

Lần đầu push branch mới:

```bash
git push -u origin ten-branch
```

Ví dụ:

```bash
git push -u origin docs-and-tests
```

Những lần sau, nếu đang ở đúng branch, chỉ cần:

```bash
git push
```

---

## 8. Tạo Pull Request trên GitHub

Sau khi push branch lên GitHub, cần tạo Pull Request để merge code vào `develop`.

Các bước:

1. Vào repository trên GitHub.
2. Bấm nút **Compare & pull request**.
3. Chọn:

   * Base branch: `develop`
   * Compare branch: branch của mình, ví dụ `docs-and-tests`
4. Ghi tiêu đề Pull Request.
5. Ghi mô tả Pull Request.
6. Bấm **Create pull request**.
7. Nhờ ít nhất 1 thành viên trong nhóm review.
8. Sau khi review xong mới merge vào `develop`.

Lưu ý:

* Không tự ý merge code khi chưa được review.
* Không tạo Pull Request trực tiếp vào `main` khi đang phát triển.
* Code ổn định ở `develop` rồi mới merge sang `main`.

---

## 9. Mẫu nội dung Pull Request

Có thể copy mẫu này khi tạo Pull Request:

````md
## Đã làm

- Cập nhật README hướng dẫn cài đặt, cấu hình `.env`, chạy server và mở web chatbot.
- Bổ sung `.env.example` để người mới biết cần cấu hình `GEMINI_API_KEY`.
- Cập nhật `.gitignore` để tránh đẩy `.env`, `venv/`, `uploads/`, dữ liệu ChromaDB và cache Python.
- Cập nhật `requirements.txt` để tránh thiếu thư viện khi clone project.
- Thêm test cơ bản cho `/`, `/docs`, `/openapi.json` và `/api/chat/`.

## Cần review

- Kiểm tra lại danh sách thư viện trong `requirements.txt`.
- Kiểm tra nội dung README đã đúng với luồng chạy thực tế chưa.
- Kiểm tra test `/api/chat/` đã phù hợp với schema hiện tại chưa.

## Cách test

```bash
pip install -r requirements.txt
python -m pytest -q
python -m uvicorn app.main:app --reload
````

Mở:

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/docs
```

````

---

## 10. Cài đặt project

### Bước 1: Tạo môi trường ảo

```bash
python -m venv venv
````

### Bước 2: Kích hoạt môi trường ảo

Trên Windows PowerShell:

```bash
venv\Scripts\activate
```

Trên Git Bash:

```bash
source venv/Scripts/activate
```

Trên macOS/Linux:

```bash
source venv/bin/activate
```

Khi kích hoạt thành công, terminal sẽ có dạng:

```text
(venv) PS ...\fastapi_chatbot>
```

### Bước 3: Cài thư viện

```bash
pip install -r requirements.txt
```

Nếu thiếu thư viện trong quá trình chạy, có thể cài lại các thư viện chính:

```bash
pip install fastapi uvicorn pydantic python-dotenv jinja2 chromadb google-generativeai python-multipart pypdf pytest httpx
```

---

## 11. Cấu hình biến môi trường

Project cần file `.env` để lưu cấu hình chạy local, đặc biệt là `GEMINI_API_KEY`.

### Bước 1: Copy file mẫu

Trên Windows PowerShell:

```bash
copy .env.example .env
```

Trên macOS/Linux:

```bash
cp .env.example .env
```

### Bước 2: Điền API key thật vào file `.env`

Mở file `.env` và cấu hình theo mẫu:

```env
GEMINI_API_KEY=your_real_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
VECTOR_STORE_PATH=storage/chroma_db
RETRIEVAL_TOP_K=5
SIMILARITY_THRESHOLD=0.3
```

Lưu ý:

* Không commit file `.env` lên GitHub.
* Không ghi API key thật vào `.env.example`.
* Nếu lỡ đẩy API key lên GitHub, cần đổi key ngay.

---

## 12. Chạy project

Chạy server FastAPI:

```bash
python -m uvicorn app.main:app --reload
```

Hoặc:

```bash
uvicorn app.main:app --reload
```

Nếu chạy thành công, terminal sẽ hiện dạng:

```text
Uvicorn running on http://127.0.0.1:8000
Application startup complete.
```

---

## 13. Mở giao diện web chatbot

Sau khi server chạy thành công, mở trình duyệt tại:

```text
http://127.0.0.1:8000/
```

Đây là giao diện chatbot được load từ file:

```text
app/templates/chat_ui.html
```

Giao diện này sẽ gửi câu hỏi đến API:

```text
POST /api/chat/
```

Nếu vào `/` bị lỗi `404 Not Found`, cần kiểm tra lại:

* `app/routers/page_router.py` có route `@router.get("/")` chưa.
* `app/main.py` đã include `page_router` chưa.
* File `app/templates/chat_ui.html` có tồn tại đúng vị trí không.

---

## 14. Mở Swagger UI

Swagger UI dùng để kiểm tra API trực tiếp trên trình duyệt.

Mở:

```text
http://127.0.0.1:8000/docs
```

OpenAPI JSON:

```text
http://127.0.0.1:8000/openapi.json
```

Một số endpoint thường dùng:

| Endpoint                                       | Phương thức | Mục đích                                        |
| ---------------------------------------------- | ----------- | ----------------------------------------------- |
| `/`                                            | GET         | Mở giao diện web chatbot                        |
| `/api/chat/`                                   | POST        | Gửi câu hỏi cho chatbot                         |
| `/api/documents/` hoặc `/api/documents/upload` | POST        | Upload hoặc xử lý tài liệu nếu router có hỗ trợ |
| `/docs`                                        | GET         | Mở Swagger UI                                   |
| `/openapi.json`                                | GET         | Xem OpenAPI schema                              |

Lưu ý: Tên endpoint tài liệu có thể thay đổi tùy theo nội dung trong `document_router.py`. Nên kiểm tra chính xác trong Swagger tại `/docs`.

---

## 15. Test API chatbot

Endpoint chính:

```text
POST /api/chat/
```

Body mẫu:

```json
{
  "question": "Phòng Tổ hợp STUDIO ở đâu?"
}
```

Response mẫu:

```json
{
  "question": "Phòng Tổ hợp STUDIO ở đâu?",
  "answer": "Nội dung câu trả lời từ chatbot",
  "source": "Tên tài liệu nguồn"
}
```

Nếu API trả lời được và terminal hiện:

```text
POST /api/chat/ HTTP/1.1" 200 OK
```

nghĩa là endpoint chatbot đã hoạt động.

---

## 16. Luồng xử lý chatbot

Luồng xử lý chính của chatbot:

```text
Người dùng nhập câu hỏi trên giao diện web
→ Frontend gọi API POST /api/chat/
→ chat_router nhận request
→ chatbot_controller điều phối xử lý
→ data layer tìm tài liệu hoặc chunk liên quan
→ prompt_builder tạo prompt từ nội dung tìm được
→ gemini_client gọi Gemini để sinh câu trả lời
→ API trả answer và source về frontend
→ Web hiển thị câu trả lời và nguồn tài liệu
```

Giải thích ngắn gọn:

| Thành phần                | Vai trò                                        |
| ------------------------- | ---------------------------------------------- |
| `app/main.py`             | Khởi tạo FastAPI và include các router         |
| `page_router.py`          | Trả về giao diện web chatbot tại `/`           |
| `chat_router.py`          | Nhận câu hỏi từ frontend hoặc Swagger          |
| `chatbot_controller.py`   | Điều phối quá trình xử lý câu hỏi              |
| `elasticsearch_client.py` | Lớp retrieval, tìm tài liệu liên quan          |
| `embedding_client.py`     | Tạo vector embedding cho câu hỏi hoặc tài liệu |
| `vector_store.py`         | Lưu và tìm kiếm vector bằng ChromaDB           |
| `prompt_builder.py`       | Tạo context và prompt để gửi cho Gemini        |
| `gemini_client.py`        | Gọi Gemini để sinh câu trả lời                 |
| `chat_schema.py`          | Chuẩn hóa request/response                     |
| `chat_ui.html`            | Giao diện web chatbot                          |

---

## 17. Metadata cần lưu

| Trường        | Kiểu dữ liệu | Ý nghĩa                                     |
| ------------- | ------------ | ------------------------------------------- |
| `doc_name`    | String       | Tên tài liệu nguồn                          |
| `title`       | String       | Tiêu đề, mục hoặc điều khoản trong tài liệu |
| `content`     | Text         | Nội dung đoạn tài liệu                      |
| `chunk_index` | Number       | Thứ tự đoạn đã tách                         |
| `file_path`   | String       | Đường dẫn file gốc                          |
| `source_type` | String       | Loại nguồn tài liệu                         |
| `is_active`   | Boolean      | Trạng thái tài liệu còn hiệu lực hay không  |
| `score`       | Float        | Điểm liên quan hoặc độ tương đồng của chunk |
| `page_start`  | Number       | Trang bắt đầu nếu trích xuất từ PDF         |
| `page_end`    | Number       | Trang kết thúc nếu trích xuất từ PDF        |

Ví dụ metadata:

```json
{
  "doc_name": "quy-che-dao-tao.pdf",
  "title": "Điều 2. Hoãn thi",
  "chunk_index": 1,
  "content": "Sinh viên có thể xin hoãn thi nếu có lý do chính đáng.",
  "file_path": "uploads/ChatAI/quy-che-dao-tao.pdf",
  "source_type": "official_document",
  "is_active": true,
  "score": 0.87,
  "page_start": 3,
  "page_end": 4
}
```

---

## 18. Chạy test

Cài thư viện test nếu chưa có:

```bash
pip install pytest httpx
```

Chạy test:

```bash
python -m pytest -q
```

Kết quả mong đợi:

```text
4 passed
```

Các test cơ bản:

| Test                          | Mục đích                                           |
| ----------------------------- | -------------------------------------------------- |
| `test_home_page_returns_html` | Kiểm tra web chatbot tại `/`                       |
| `test_docs_page_available`    | Kiểm tra Swagger `/docs`                           |
| `test_openapi_json_available` | Kiểm tra OpenAPI schema                            |
| `test_chat_api_with_mock`     | Kiểm tra API `/api/chat/` mà không gọi Gemini thật |

---

## 19. Quy tắc không được đẩy lên GitHub

Không đẩy các file hoặc thư mục sau lên GitHub:

* `.env`
* API key
* Token
* File credential
* File service account
* Thư mục môi trường ảo `venv/`
* Thư mục upload thật `uploads/`
* Dữ liệu vector database local `storage/chroma_db/`
* File cache Python `__pycache__/`
* File cache test `.pytest_cache/`
* File log `*.log`

Nếu lỡ add nhầm file `.env`, cần gỡ khỏi Git:

```bash
git restore --staged .env
```

Nếu file `.env` đã từng bị commit, cần xóa khỏi Git tracking:

```bash
git rm --cached .env
git add .gitignore
git commit -m "fix: remove env file from repository"
git push
```

Nếu đã đẩy API key thật lên GitHub, cần đổi API key ngay.

---

## 20. Một số lỗi thường gặp và cách xử lý

### Lỗi 1: Thiếu `GEMINI_API_KEY`

Thông báo lỗi có dạng:

```text
ValueError: Thiếu GEMINI_API_KEY trong file .env
```

Cách xử lý:

* Kiểm tra đã có file `.env` chưa.
* Kiểm tra `.env` có nằm ngang hàng với `app/` không.
* Kiểm tra đã có dòng `GEMINI_API_KEY=...` chưa.
* Không đặt tên nhầm thành `.env.txt`.

---

### Lỗi 2: Thiếu thư viện `chromadb`

Thông báo lỗi có dạng:

```text
ModuleNotFoundError: No module named 'chromadb'
```

Cách xử lý:

```bash
pip install chromadb
```

Sau đó cập nhật lại `requirements.txt` nếu cần.

---

### Lỗi 3: Vào `/` bị `404 Not Found`

Nguyên nhân thường gặp:

* Chưa có route `GET /`.
* Chưa include `page_router` trong `app/main.py`.
* File `chat_ui.html` đặt sai vị trí.

Cách kiểm tra:

```text
app/templates/chat_ui.html
app/routers/page_router.py
app/main.py
```

Route web cần trả về giao diện chatbot tại:

```text
http://127.0.0.1:8000/
```

---

### Lỗi 4: Gõ sai đường dẫn `/docs`

Đường dẫn đúng:

```text
http://127.0.0.1:8000/docs
```

Nếu gõ nhầm `/dosc` thì sẽ bị `404 Not Found`.

---

### Lỗi 5: API chat không kết nối từ giao diện web

Cần kiểm tra trong file `chat_ui.html` có đúng API URL:

```javascript
const API_URL = "/api/chat/";
```

Nếu backend dùng endpoint khác, cần sửa URL cho khớp với router thực tế.

---

### Lỗi 6: Không push được branch mới

Nếu gặp lỗi vì branch chưa tồn tại trên GitHub, chạy:

```bash
git push -u origin ten-branch
```

Ví dụ:

```bash
git push -u origin docs-and-tests
```

---

### Lỗi 7: Đang ở nhầm branch

Kiểm tra branch hiện tại:

```bash
git branch
```

Chuyển về branch đúng:

```bash
git checkout docs-and-tests
```

---

### Lỗi 8: Bị conflict khi merge Pull Request

Cách xử lý cơ bản:

```bash
git checkout develop
git pull origin develop
git checkout ten-branch
git merge develop
```

Sau đó mở file bị conflict, sửa thủ công, rồi commit lại:

```bash
git add .
git commit -m "fix: resolve merge conflict"
git push
```

---

## 21. Checklist trước khi nộp bài

* [ ] Có repository GitHub chung cho nhóm.
* [ ] Có nhánh `main`.
* [ ] Có nhánh `develop` nếu nhóm sử dụng quy trình develop.
* [ ] Mỗi thành viên có branch riêng.
* [ ] Không code trực tiếp trên `main`.
* [ ] Không code trực tiếp trên `develop`.
* [ ] Có Pull Request trước khi merge.
* [ ] Có review trước khi merge.
* [ ] Project chạy được bằng lệnh `python -m uvicorn app.main:app --reload`.
* [ ] Mở được web chatbot tại `/`.
* [ ] Mở được Swagger UI tại `/docs`.
* [ ] Test được API `/api/chat/`.
* [ ] Có đủ folder `app/routers`, `app/controller`, `app/data`, `app/schemas`, `app/core`.
* [ ] Có file giao diện `app/templates/chat_ui.html`.
* [ ] Router không chứa logic nghiệp vụ quá dài.
* [ ] Controller điều phối luồng xử lý rõ ràng.
* [ ] Data layer có xử lý metadata hoặc vector search.
* [ ] Câu trả lời chatbot có kèm nguồn.
* [ ] Có file `.gitignore`.
* [ ] Có file `.env.example`.
* [ ] Có file `README.md` hướng dẫn rõ ràng.
* [ ] Có file `requirements.txt` cập nhật đủ thư viện.
* [ ] Có test cơ bản trong thư mục `tests/`.
* [ ] Chạy được `python -m pytest -q`.
* [ ] Không đẩy `.env`, API key hoặc credential lên GitHub.
* [ ] Không đẩy `venv/`, `uploads/`, `storage/chroma_db/`, `__pycache__/` lên GitHub.

---



## Reindex tai lieu RAG

Pipeline tai lieu doc recursive cac file PDF trong `DOCUMENTS_DIR`. Cau hinh khuyen nghi:

```env
DOCUMENTS_DIR=uploads/Tổng hợp văn bản AI
```

Cac file `.doc`, `.docx`, `.xlsx`, `.json`, `.png`, `.rar` duoc bo qua o giai doan nay; he thong chi index `.pdf`.

Cai dependency va reindex:

```bash
pip install -r requirements.txt
python scripts/reindex_documents.py
```

Ket qua reindex bao so PDF phat hien, so file index thanh cong, tong so chunk, `vector_count` trong ChromaDB va danh sach file loi neu co. Moi chunk co metadata `phong_ban`, `relative_path`, `source_root` de giam lay nham nguon khi hoi theo phong ban hoac chu de rong.


## Tra cuu website UNETI bang Vertex AI Search

Luồng website không crawl HTML trực tiếp. Hệ thống gọi Google Vertex AI Search/Discovery Engine, lọc domain `uneti.edu.vn`, loại trùng URL, rerank kết quả và chỉ đưa 1-2 nguồn tốt nhất sang Gemini.

Cấu hình `.env` cần có:

```env
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
DISCOVERY_PROJECT_NUMBER=your_google_cloud_project_number
DISCOVERY_LOCATION=global
DISCOVERY_COLLECTION_ID=default_collection
DISCOVERY_ENGINE_ID=your_discovery_engine_id
DISCOVERY_SERVING_CONFIG_ID=default_search
UNETI_WEBSITE_DOMAIN=uneti.edu.vn
WEBSITE_SEARCH_TOP_K=10
WEBSITE_RERANK_TOP_K=2
```

Các câu hỏi có dấu hiệu như `website`, `trang web`, `uneti.edu.vn`, `tin tức`, `bài viết`, `link` sẽ được route sang intent `website_uneti`. Các câu hỏi về quy định, quyết định, điều khoản trong tài liệu nội bộ vẫn dùng pipeline RAG PDF.

## Debug pipeline bang trace_id

Moi cau hoi gui vao API chat se tra ve `trace_id` trong response:

```bash
POST /api/chat/
```

Dung `trace_id` do de tra cuu chi tiet cac buoc xu ly:

```bash
GET /api/chat/traces/{trace_id}
```

Endpoint trace tra ve cac thong tin chinh:

- `trace_id`, `question`, `created_at`, `updated_at`
- `steps`: cac buoc nhu `classify_query`, `retrieval`, `website_search`, `context_builder`, `llm_call`
- `response`: cau tra loi cuoi cung va nguon da tra ve cho nguoi dung

Endpoint nay chi nhan `trace_id` dang UUID hop le va chi doc file trong `storage/traces/{trace_id}.json`; neu trace khong ton tai se tra ve HTTP 404.
