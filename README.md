# FastAPI Chatbot Metadata

## 1. Giới thiệu
Dự án chatbot đọc tài liệu nội bộ, xử lý metadata và trả lời kèm nguồn.

## 2. Thành viên nhóm
- Bạn 1: Routers
- Bạn 2: Controllers
- Bạn 3: Data layer
- Bạn 4: Docs + Tests

## 3. Cấu trúc project
Mô tả cây thư mục app/routers, app/controller, app/data.

## 4. Cài đặt
pip install -r requirements.txt

## 5. Chạy project
uvicorn app.main:app --reload

## 6. API endpoints
- GET /health
- POST /chat
- POST /documents/upload

## 7. Git workflow
Không code trực tiếp trên main. Mỗi chức năng dùng branch riêng và tạo Pull Request.

## 8. Luồng xử lý chatbot
Upload -> Extract -> Split -> Search -> Build Context -> AI Response -> Source

