# FastAPI Chatbot Metadata

## 1. Giới thiệu dự án

Dự án xây dựng một chatbot sử dụng FastAPI, có khả năng nhận câu hỏi từ người dùng, tìm kiếm nội dung liên quan trong tài liệu đã xử lý metadata, sau đó trả lời kèm nguồn tài liệu.

Mục tiêu chính của dự án:

- Xây dựng API chatbot bằng FastAPI.
- Chia code rõ ràng theo từng module: `routers`, `controller`, `data`, `schemas`, `core`.
- Mô phỏng luồng xử lý chatbot: nhận câu hỏi, tìm tài liệu liên quan, xây dựng prompt và trả lời kèm nguồn.
- Làm việc nhóm bằng GitHub theo quy trình: clone repo, tạo branch, commit, push và tạo Pull Request.
- Không code trực tiếp trên nhánh `main`.

---

## 2. Thành viên nhóm và phân công công việc

Nhóm gồm 4 thành viên. Mỗi thành viên phụ trách một phần riêng để tránh sửa trùng file và hạn chế conflict khi làm việc nhóm.

| STT | Thành viên | Branch phụ trách | Folder/File chính | Nhiệm vụ |
|---|---|---|---|---|
| 1 | Nguyễn Thị Hải Yến | `feature/routers` | `app/routers` | Tạo các API endpoint, nhận request từ người dùng và gọi sang controller |
| 2 | Nguyễn Văn Vinh | `feature/controllers` | `app/controller` | Điều phối logic chatbot, xử lý luồng nghiệp vụ chính |
| 3 | Phương Thảo | `feature/data-layer` | `app/data` | Xử lý tìm kiếm dữ liệu, metadata, context, prompt và gọi model |
| 4 | Nguyễn Chính Nghĩa | `docs-and-tests` | `README.md`, `tests`, `.gitignore`, `.env.example` | Viết tài liệu hướng dẫn, test API và kiểm tra workflow GitHub |
### 2.1. Chi tiết công việc từng thành viên

#### Thành viên 1: Routers

Phụ trách các file:

- `app/routers/chat_router.py`
- `app/routers/document_router.py`
- `app/routers/health_router.py`

Nhiệm vụ:

- Tạo endpoint `POST /chat/`.
- Tạo endpoint kiểm tra server `GET /health/`.
- Tạo endpoint upload tài liệu `POST /documents/upload`.
- Router chỉ nên nhận request, gọi controller và trả response.
- Không viết logic xử lý dài trong router.

---

#### Thành viên 2: Controllers

Phụ trách các file:

- `app/controller/chatbot_controller.py`
- `app/controller/document_controller.py`

Nhiệm vụ:

- Điều phối luồng xử lý chatbot.
- Nhận dữ liệu từ router.
- Gọi data layer để tìm kiếm tài liệu.
- Gọi prompt builder để tạo prompt.
- Trả response đúng định dạng cho router.

---

#### Thành viên 3: Data Layer

Phụ trách các file:

- `app/data/elasticsearch_client.py`
- `app/data/prompt_builder.py`
- `app/data/gemini_client.py`

Nhiệm vụ:

- Tạo hàm tìm kiếm tài liệu liên quan.
- Xây dựng context từ tài liệu tìm được.
- Xây dựng prompt cho chatbot.
- Tạo hàm gọi mô hình AI hoặc bản giả lập để test.
- Đảm bảo câu trả lời có kèm nguồn tài liệu.

---

#### Thành viên 4: Docs and Tests

Phụ trách các file:

- `README.md`
- `tests/`
- `.gitignore`
- `.env.example`

Nhiệm vụ:

- Viết hướng dẫn cài đặt và chạy project.
- Viết hướng dẫn Git workflow cho nhóm.
- Kiểm tra API trên Swagger.
- Viết test cơ bản nếu cần.
- Đảm bảo không đẩy file nhạy cảm như `.env`, API key hoặc credential lên GitHub.

---

## 3. Cấu trúc thư mục dự án

```text
fastapi-chatbot/
├── app/
│   ├── main.py
│   │
│   ├── routers/
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
│   │   ├── prompt_builder.py
│   │   └── gemini_client.py
│   │
│   ├── schemas/
│   │   ├── chat_schema.py
│   │   └── document_schema.py
│   │
│   └── core/
│       ├── config.py
│       └── constants.py
│
├── tests/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 4. Vai trò từng thư mục

| Thư mục | Vai trò |
|---|---|
| `app/routers` | Nhận request từ client, gọi controller và trả response |
| `app/controller` | Điều phối luồng xử lý nghiệp vụ |
| `app/data` | Xử lý dữ liệu, search tài liệu, build context, build prompt và gọi model |
| `app/schemas` | Định nghĩa request/response bằng Pydantic |
| `app/core` | Chứa cấu hình chung, biến môi trường và hằng số |
| `tests` | Chứa các file kiểm thử API |

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

---

### Bước 3: Chuyển sang nhánh `develop`

Không code trực tiếp trên `main`.

Chạy:

```bash
git checkout develop
```

Sau đó cập nhật code mới nhất:

```bash
git pull origin develop
```

### Bước 4: Tạo branch riêng để làm việc

Mỗi thành viên tạo một branch riêng từ `develop`.

Cú pháp chung:

```bash
git checkout -b ten-branch
```

Ví dụ nếu làm phần router:

```bash
git checkout -b feature/routers
```

Nếu làm phần controller:

```bash
git checkout -b feature/controllers
```

Nếu làm phần data layer:

```bash
git checkout -b feature/data-layer
```

Nếu làm tài liệu và test:

```bash
git checkout -b docs-and-tests
```

---

### Bước 5: Kiểm tra đang ở branch nào

```bash
git branch
```

Nếu thấy dấu `*` ở branch của mình là đúng.

Ví dụ:

```text
  develop
* feature/routers
  main
```

---

## 6. Quy tắc đặt tên branch

| Loại công việc | Cách đặt tên | Ví dụ |
|---|---|---|
| Thêm chức năng mới | `feature/ten-chuc-nang` | `feature/chat-api` |
| Sửa lỗi | `fix/ten-loi` | `fix/search-empty-result` |
| Viết tài liệu | `docs/noi-dung` | `docs/update-readme` |
| Viết test | `test/noi-dung` | `test/chat-api` |

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
git checkout -b feature/chat-api
```

---

### Bước 2: Code phần việc của mình

Mỗi thành viên chỉ nên sửa các file thuộc phần mình phụ trách.

Ví dụ:

- Người làm routers sửa trong `app/routers`.
- Người làm controllers sửa trong `app/controller`.
- Người làm data layer sửa trong `app/data`.
- Người làm docs/tests sửa `README.md`, `.gitignore`, `.env.example`, `tests`.

---

### Bước 3: Kiểm tra file đã thay đổi

```bash
git status
```

Nếu có file màu đỏ hoặc xanh nghĩa là có thay đổi chưa commit.

---

### Bước 4: Thêm file vào Git

```bash
git add .
```

Hoặc chỉ add một file cụ thể:

```bash
git add README.md
```

---

### Bước 5: Commit code

Cú pháp:

```bash
git commit -m "loai: mo ta ngan gon noi dung da lam"
```

Ví dụ:

```bash
git commit -m "feat: add chat router"
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

| Loại | Ý nghĩa |
|---|---|
| `feat` | Thêm chức năng mới |
| `fix` | Sửa lỗi |
| `docs` | Sửa tài liệu |
| `test` | Thêm hoặc sửa test |
| `refactor` | Sửa lại code nhưng không đổi chức năng |
| `chore` | Công việc phụ như cấu hình, cài đặt, dọn dẹp |

---

### Bước 6: Push branch lên GitHub

Lần đầu push branch mới:

```bash
git push -u origin ten-branch
```

Ví dụ:

```bash
git push -u origin feature/routers
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
   - Base branch: `develop`
   - Compare branch: branch của mình, ví dụ `feature/routers`
4. Ghi tiêu đề Pull Request.
5. Ghi mô tả Pull Request.
6. Bấm **Create pull request**.
7. Nhờ ít nhất 1 thành viên trong nhóm review.
8. Sau khi review xong mới merge vào `develop`.

Lưu ý:

- Không tự ý merge code khi chưa được review.
- Không tạo Pull Request trực tiếp vào `main` khi đang phát triển.
- Code ổn định ở `develop` rồi mới merge sang `main`.

---

## 9. Mẫu nội dung Pull Request

Có thể copy mẫu này khi tạo Pull Request:

```md
## Đã làm

- Thêm API POST /chat
- Gọi sang chatbot_controller
- Trả về question, answer và source

## Cần review

- Kiểm tra tên route
- Kiểm tra format response
- Kiểm tra xử lý khi không tìm thấy tài liệu

## Cách test

- Chạy server bằng lệnh:
  uvicorn app.main:app --reload

- Mở Swagger:
  http://127.0.0.1:8000/docs

- Test endpoint:
  POST /chat/
```

---

## 10. Cài đặt project

### Bước 1: Tạo môi trường ảo

```bash
python -m venv venv
```

---

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

---

### Bước 3: Cài thư viện

```bash
pip install -r requirements.txt
```

Nếu chưa có `requirements.txt`, có thể cài tạm các thư viện chính:

```bash
pip install fastapi uvicorn pydantic python-dotenv
```

---

## 11. Chạy project

Chạy server FastAPI:

```bash
uvicorn app.main:app --reload
```

Nếu chạy thành công, terminal sẽ hiện dạng:

```text
Uvicorn running on http://127.0.0.1:8000
```

Mở Swagger UI tại:

```text
http://127.0.0.1:8000/docs
```


## 14. Luồng xử lý chatbot

Luồng xử lý chính của chatbot:

```text
Người dùng đặt câu hỏi
→ API /chat nhận request
→ Router gọi sang controller
→ Controller gọi data layer để tìm tài liệu
→ Data layer trả về tài liệu liên quan
→ Build context từ tài liệu
→ Build prompt
→ Gọi model AI hoặc bản giả lập
→ Trả về câu trả lời kèm nguồn
```

---

## 15. Metadata cần lưu

| Trường | Kiểu dữ liệu | Ý nghĩa |
|---|---|---|
| `doc_name` | String | Tên tài liệu nguồn |
| `title` | String | Tiêu đề, mục hoặc điều khoản trong tài liệu |
| `content` | Text | Nội dung đoạn tài liệu |
| `chunk_index` | Number | Thứ tự đoạn đã tách |
| `file_path` | String | Đường dẫn file gốc |
| `source_type` | String | Loại nguồn tài liệu |
| `is_active` | Boolean | Trạng thái tài liệu còn hiệu lực hay không |

Ví dụ metadata:

```json
{
  "doc_name": "quy-che-dao-tao.pdf",
  "title": "Điều 2. Hoãn thi",
  "chunk_index": 1,
  "content": "Sinh viên có thể xin hoãn thi nếu có lý do chính đáng.",
  "file_path": "uploads/ChatAI/quy-che-dao-tao.pdf",
  "source_type": "official_document",
  "is_active": true
}
```
## 16. Quy tắc không được đẩy lên GitHub

Không đẩy các file sau lên GitHub:

- `.env`
- API key
- Token
- File credential
- File service account
- Thư mục môi trường ảo `venv/`
- File upload thật trong `uploads/`
- File cache Python `__pycache__/`


## 17. Một số lỗi Git thường gặp

### Lỗi 1: Không push được branch mới

Nếu gặp lỗi vì branch chưa tồn tại trên GitHub, chạy:

```bash
git push -u origin ten-branch
```

Ví dụ:

```bash
git push -u origin feature/routers
```

---

### Lỗi 2: Đang ở nhầm branch

Kiểm tra branch hiện tại:

```bash
git branch
```

Chuyển về branch đúng:

```bash
git checkout ten-branch
```

Ví dụ:

```bash
git checkout develop
```

---

### Lỗi 3: Bị conflict khi merge Pull Request

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

### Lỗi 4: Lỡ commit nhầm file `.env`

Cần xóa file khỏi Git và thêm vào `.gitignore`:

```bash
git rm --cached .env
git add .gitignore
git commit -m "fix: remove env file from repository"
git push
```

Nếu đã đẩy API key lên GitHub, cần đổi API key ngay.


## 18. Checklist trước khi nộp bài

- [ ] Có repository GitHub chung cho nhóm.
- [ ] Có nhánh `main`.
- [ ] Có nhánh `develop`.
- [ ] Mỗi thành viên có branch riêng.
- [ ] Không code trực tiếp trên `main`.
- [ ] Không code trực tiếp trên `develop`.
- [ ] Có Pull Request trước khi merge.
- [ ] Có review trước khi merge.
- [ ] Project chạy được bằng lệnh `uvicorn app.main:app --reload`.
- [ ] Mở được Swagger UI tại `/docs`.
- [ ] Test được API `/chat/`.
- [ ] Có đủ folder `app/routers`, `app/controller`, `app/data`, `app/schemas`, `app/core`.
- [ ] Router không chứa logic nghiệp vụ quá dài.
- [ ] Controller điều phối luồng xử lý rõ ràng.
- [ ] Data layer có xử lý metadata hoặc bản giả lập để test.
- [ ] Câu trả lời chatbot có kèm nguồn.
- [ ] Có file `.gitignore`.
- [ ] Có file `.env.example`.
- [ ] Có file `README.md` hướng dẫn rõ ràng.
- [ ] Không đẩy `.env`, API key hoặc credential lên GitHub.



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
