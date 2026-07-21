# Báo Cáo Tổng Quan Chatbot RAG UNETI

## 1. Mục Đích Hệ Thống

Chatbot được xây dựng nhằm hỗ trợ người dùng tra cứu thông tin liên quan đến Trường Đại học Kinh tế - Kỹ thuật Công nghiệp (UNETI), đặc biệt là các nhóm thông tin như quy chế, quy định, điều kiện học vụ, thủ tục hành chính, hướng dẫn sử dụng Web Support và một số thông tin website của trường.

Hệ thống không chỉ trả lời dựa trên kiến thức có sẵn của mô hình ngôn ngữ, mà sử dụng kiến trúc RAG (Retrieval-Augmented Generation). Nghĩa là trước khi sinh câu trả lời, chatbot sẽ tìm kiếm các đoạn tài liệu liên quan trong kho dữ liệu nội bộ, tài liệu nghiệp vụ hoặc website UNETI, sau đó đưa các đoạn này vào prompt để Gemini trả lời có căn cứ.

Mục tiêu chính:

- Trả lời câu hỏi dựa trên tài liệu thật.
- Có nguồn tham chiếu ở cuối câu trả lời.
- Hạn chế bịa thông tin ngoài tài liệu.
- Hỗ trợ cả câu hỏi quy định/chính sách và câu hỏi thao tác nghiệp vụ.
- Ghi trace chi tiết để debug quá trình tìm nguồn và sinh câu trả lời.

## 2. Công Nghệ Sử Dụng

Các thành phần chính của hệ thống:

- FastAPI: xây dựng API backend.
- Uvicorn: chạy server ASGI.
- Pydantic: định nghĩa schema request/response.
- Gemini API: sinh câu trả lời cuối và một số bước hỗ trợ như rewrite câu hỏi hoặc retrieval plan.
- Sentence Transformers: tạo embedding cho câu hỏi và tài liệu.
- ChromaDB: lưu vector embedding của các chunk tài liệu.
- BM25/rank-bm25: tìm kiếm theo từ khóa.
- Cross-encoder reranker: xếp hạng lại các chunk sau retrieval.
- pypdf: đọc nội dung PDF.
- python-docx/đọc XML DOCX: trích xuất nội dung file Word.
- SQLite: lưu session, thread, message, request idempotency.
- Pytest: kiểm thử các luồng chat, retrieval, routing, business API.

## 3. Kiến Trúc Tổng Thể

Hệ thống được chia thành các lớp chính:

- `app/main.py`: entrypoint FastAPI, khởi tạo middleware, session, router và preload RAG.
- `app/routers`: khai báo endpoint HTTP.
- `app/controller`: điều phối nghiệp vụ, ví dụ chat tổng, upload tài liệu, business search, website search.
- `app/data`: xử lý dữ liệu, retrieval, vector store, prompt, Gemini, trace, query analyzer.
- `app/schemas`: định nghĩa request/response model.
- `app/services`: quản lý hội thoại, session, thread, idempotency.
- `documents`: nơi chứa tài liệu nguồn.
- `storage`: nơi lưu index, mapping, trace và dữ liệu phụ trợ.
- `tests`: test tự động.

Endpoint quan trọng nhất là:

- `POST /api/chat/`: API chat tổng hợp, tự quyết định tìm trong tài liệu nội bộ, tài liệu nghiệp vụ, website hoặc kết hợp nhiều nguồn.
- `POST /api/chat/internal`: chỉ dùng tài liệu nội bộ.
- `POST /api/chat/business`: chỉ dùng tài liệu nghiệp vụ.
- `POST /api/chat/website`: chỉ dùng website UNETI.
- `GET /api/chat/traces/{trace_id}`: xem trace debug của một lượt hỏi.

## 4. Các Nhóm Dữ Liệu

### 4.1. Tài Liệu Nội Bộ

Tài liệu nội bộ thường là quy chế, quy định, quyết định, thông báo, hướng dẫn chính thức. Các tài liệu này được lưu trong thư mục `documents` và được index thành các chunk có metadata.

Mỗi chunk có thể chứa các trường:

- `doc_name`: tên file nguồn.
- `relative_path`: đường dẫn tương đối.
- `title`: tiêu đề đoạn, điều, mục hoặc chương.
- `dieu`, `muc`, `chuong`: metadata điều khoản nếu trích xuất được.
- `content`: nội dung chunk.
- `source_type`: thường là `official_document`.
- `so_van_ban`, `ngay_ban_hanh`, `ngay_hieu_luc`, `loai_van_ban`, `don_vi_ban_hanh`: metadata văn bản.
- `content_hash`, `chunk_index`, `updated_at`: định danh và quản lý index.

### 4.2. Tài Liệu Nghiệp Vụ

Tài liệu nghiệp vụ nằm chủ yếu trong `documents/nghiep_vu`, ví dụ hướng dẫn Web Support cho sinh viên, cán bộ giảng viên và file mapping PCNTT.

Nhóm này phục vụ các câu hỏi kiểu:

- Cách xem kết quả học tập.
- Cách đăng ký/hoãn/thi lại nếu có trong Web Support.
- Cách truy cập chức năng xử lý hồ sơ thủ tục hành chính.
- Cách phê duyệt/trình duyệt hồ sơ.
- Các thao tác trên hệ thống.

Nguồn nghiệp vụ có thể có `source_type` là:

- `business_document`.
- `business_faq_mapping`.

Trong đó `business_faq_mapping` được xây dựng từ bảng mapping câu hỏi - câu trả lời - vị trí trong file gốc - từ khóa - đối tượng sử dụng.

### 4.3. Website UNETI

Website UNETI được xử lý qua `website_search_client.py`. Hệ thống có thể tìm kiếm thông tin website, tải nội dung trang hoặc attachment, sau đó chunk và index vào vector store với `source_type=website_uneti`.

Điểm cần lưu ý: vì website chunks dùng chung ChromaDB với tài liệu khác, luồng retrieval nội bộ cần filter `source_type="official_document"` khi chỉ muốn lấy văn bản chính thức, tránh lấy nhầm nội dung website.

## 5. Tiền Xử Lý Dữ Liệu Đầu Vào Khi Index Tài Liệu

Tiền xử lý tài liệu là bước biến file PDF/DOCX/XLSX thành các đoạn nhỏ có thể tìm kiếm.

### 5.1. Kiểm Tra File

Khi upload hoặc index tài liệu, hệ thống kiểm tra:

- Tên file có hợp lệ không.
- File có nằm trong thư mục tài liệu được phép không.
- Định dạng có được hỗ trợ không.
- Với upload qua API hiện tại chủ yếu hỗ trợ PDF.
- Với scan/index tài liệu nội bộ có hỗ trợ PDF và DOCX; một số luồng nghiệp vụ có đọc thêm XLSX.
- File tạm của Office như `~$...` sẽ bị bỏ qua.

Tên file được làm sạch để tránh ký tự nguy hiểm như:

- `< > : " \ | ? *`
- ký tự điều khiển.
- path traversal như `../`.

### 5.2. Trích Xuất Text

Với PDF:

- Dùng `PdfReader`.
- Đọc từng trang.
- Mỗi trang được ghép vào text tổng, có đánh dấu trang.

Với DOCX:

- Đọc paragraph.
- Đọc cả table.
- Nội dung table được ghép theo hàng/cột bằng dấu phân cách.

Với tài liệu nghiệp vụ:

- Một số file DOCX được đọc trực tiếp từ XML trong file zip DOCX.
- Bảng mapping được parse thành các dòng FAQ gồm câu hỏi, câu trả lời chuẩn, vị trí trong file gốc, từ khóa và đối tượng.

### 5.3. Chuẩn Hóa Text Và Metadata

Hệ thống có hàm `normalize_text()` để:

- Chuẩn hóa Unicode.
- Chuyển về dạng không dấu.
- Chuyển lowercase.
- Loại bỏ/chuẩn hóa ký tự đặc biệt.
- Giúp so khớp tiếng Việt ổn định hơn giữa câu hỏi và tài liệu.

Metadata được trích xuất từ header, filename và nội dung đầu văn bản:

- Số văn bản.
- Số văn bản ngắn.
- Ngày ban hành.
- Ngày hiệu lực.
- Loại văn bản.
- Đơn vị ban hành.
- Tên văn bản.
- Phòng ban hoặc thư mục nguồn.

Ví dụ, nếu filename có mã quyết định hoặc số văn bản, hệ thống ưu tiên dùng thông tin đó để hỗ trợ tìm kiếm chính xác.

### 5.4. Tách Chunk

Sau khi có text, hệ thống tách thành chunk.

Chiến lược tách:

- Ưu tiên tách theo cấu trúc văn bản: `Chương`, `Mục`, `Điều`.
- Nếu không nhận diện được cấu trúc, fallback sang tách theo độ dài.
- Tách đệ quy theo các separator: đoạn trắng, xuống dòng, câu, dấu chấm phẩy, dấu phẩy, khoảng trắng.
- Có overlap giữa các chunk để tránh mất ngữ cảnh ở ranh giới.

Mỗi chunk được gắn:

- `title`: tiêu đề điều/mục/chương hoặc đoạn fallback.
- `dieu`, `muc`, `chuong`: nếu nhận diện được.
- `chunk_index`: thứ tự chunk.
- `content`: nội dung đoạn.
- metadata văn bản.

### 5.5. Tạo Embedding Và Lưu Vector

Sau khi tạo chunk:

- Nội dung chunk được đưa vào SentenceTransformer để tạo embedding.
- Embedding được normalize.
- Chunk và metadata được upsert vào ChromaDB.
- ID chunk ổn định theo `content_hash:chunk_index`, giúp reindex không tạo trùng dữ liệu.

Vector store chỉ nhận metadata kiểu đơn giản, nên các trường `None`, list, dict được lọc bỏ trước khi lưu.

## 6. Tiền Xử Lý Câu Hỏi Người Dùng

Khi người dùng gọi `/api/chat/`, câu hỏi không được đưa ngay vào Gemini. Hệ thống xử lý trước qua nhiều bước.

### 6.1. Quản Lý Session Và Lịch Sử

Middleware tạo hoặc đọc cookie session ẩn danh.

`ConversationService`:

- Nhận `question`, `thread_id`, `request_id`.
- Chống gửi trùng bằng idempotency.
- Lấy lịch sử hội thoại gần nhất.
- Nếu câu hỏi phụ thuộc ngữ cảnh, có thể gọi contextualizer để viết lại thành câu hỏi độc lập.

Ví dụ:

- Người dùng hỏi: "cái đó làm như thế nào?"
- Hệ thống có thể rewrite thành câu hỏi đầy đủ dựa trên đoạn chat trước.

Tuy nhiên, nếu câu hỏi đã độc lập thì không cần rewrite để tiết kiệm Gemini.

### 6.2. Chuẩn Hóa Và Phân Tích Ý Định

`query_analyzer.py` và `query_context.py` phân tích câu hỏi:

- Câu hỏi thuộc website, tài liệu nội bộ, nghiệp vụ hay ngoài phạm vi.
- Có nhắc quy chế, điều, mục, chương, số văn bản không.
- Có hỏi thao tác hệ thống/Web Support không.
- Đối tượng hỏi là sinh viên hay cán bộ giảng viên.
- Nhu cầu là chính sách/quy định hay thủ tục UI.

Ví dụ:

- "cần chứng chỉ gì để ra trường" nên ưu tiên tài liệu nội bộ/quy định.
- "cách xem kết quả học tập theo kì" nên ưu tiên tài liệu nghiệp vụ/Web Support.
- "nghỉ học không phép có bị cấm thi không" nên ưu tiên quy chế nội bộ.
- "cách truy cập chức năng xử lý hồ sơ thủ tục hành chính" nên ưu tiên nghiệp vụ CBGV/Web Support.

### 6.3. Query Decomposition

Với câu hỏi phức tạp hoặc cần so sánh, hệ thống có bước tách câu hỏi thành nhiều khía cạnh.

Ví dụ câu:

`nghỉ học không phép và nghỉ học có phép khác nhau những gì`

có thể được tách thành các khía cạnh:

- Hệ thống điểm danh ghi nhận nghỉ có phép/không phép.
- Nghỉ học tạm thời/bảo lưu.
- Hoãn thi có lý do.
- Nghỉ không phép/bỏ học/bỏ kiểm tra/bỏ thi không lý do.
- Điểm chuyên cần, số tiết vắng, cấm thi.
- Bảng so sánh cuối.

Điểm quan trọng: decomposition hiện ưu tiên rule-based, không nhất thiết gọi Gemini, nhằm giảm quota và tăng tính ổn định.

## 7. Luồng RAG Của `/api/chat/`

Luồng tổng quát:

1. Nhận câu hỏi từ người dùng.
2. Chuẩn hóa và phân tích intent/context.
3. Nếu cần, rewrite câu hỏi theo lịch sử hội thoại.
4. Nếu câu hỏi phức tạp, tách thành nhiều sub-question.
5. Retrieve tài liệu liên quan từ nhiều nguồn.
6. Gộp, dedupe và xếp hạng evidence.
7. Chọn các chunk đủ mạnh, đủ phủ ý.
8. Build context dạng `<NGUON>...</NGUON>`.
9. Build prompt với quy tắc trả lời và trích nguồn.
10. Gọi Gemini để sinh câu trả lời cuối.
11. Nếu Gemini lỗi quota hoặc trả lời yếu, dùng fallback có căn cứ từ evidence.
12. Ghi trace và trả response.

## 8. Retrieval Tài Liệu Nội Bộ

Tài liệu nội bộ dùng hybrid retrieval trong `elasticsearch_client.py`.

Các nhánh tìm kiếm chính:

- Metadata search: tìm theo số văn bản, điều, mục, chương, ngày, tên văn bản.
- BM25: tìm theo từ khóa.
- Vector search: tìm ngữ nghĩa bằng ChromaDB.
- Probe retrieval: kiểm tra xem câu hỏi mơ hồ có đủ evidence không.
- HyDE/Grounded HyDE: có thể sinh mô tả giả định để tìm kiếm tốt hơn, nhưng tốn Gemini.
- RRF: gộp nhiều danh sách kết quả theo Reciprocal Rank Fusion.
- Cross-encoder rerank: xếp hạng lại các chunk tốt nhất.

Kết quả cuối thường có các trường debug:

- `score`.
- `keyword_score`.
- `bm25_score`.
- `vector_score`.
- `rrf_score`.
- `rerank_score`.
- `retrieval_branches`.

Với câu hỏi quy chế/quy định, hệ thống có các rule ưu tiên tài liệu chính thức, ví dụ điều kiện tốt nghiệp, thi lại, hoãn thi, nghỉ học, điểm chuyên cần, cấm thi.

## 9. Retrieval Tài Liệu Nghiệp Vụ

Tài liệu nghiệp vụ dùng `business_knowledge.py`.

Các cơ chế chính:

- FAQ mapping search: tìm trong bảng mapping câu hỏi - trả lời chuẩn.
- Keyword/location matching: dùng từ khóa, vị trí trong file gốc, đối tượng sử dụng.
- Business retrieval plan: có thể dùng Gemini để viết lại truy vấn nghiệp vụ, xác định intent/domain/must/avoid.
- Generic hybrid search: tìm trong text tài liệu nghiệp vụ nếu mapping không đủ.
- Audience routing: phân biệt sinh viên, CBGV hoặc đối tượng chung.

Với câu hỏi thao tác hệ thống, nguồn nghiệp vụ thường đáng tin hơn nguồn quy chế, vì nó chỉ ra đường dẫn, menu, mục chức năng và cách thực hiện.

## 10. Multi-Hop RAG Và Chọn Nguồn Theo Độ Phủ

Hệ thống đã được cải thiện theo hướng multi-hop:

- Không chỉ search một lần bằng câu gốc.
- Với câu phức tạp, tạo nhiều sub-question.
- Mỗi sub-question có thể retrieve cả nguồn nội bộ và nghiệp vụ.
- Kết quả được gắn `evidence_aspect` và `coverage_aspects`.
- Khi chọn evidence, hệ thống không chỉ lấy top score chung mà cố gắng phủ đủ các khía cạnh quan trọng.

Ví dụ với câu hỏi so sánh nghỉ có phép/không phép, câu trả lời tốt cần ghép:

- Nguồn nghiệp vụ: Web Support ghi nhận nghỉ có phép/không phép trong điểm danh.
- Nguồn quy chế: quy định điểm chuyên cần, tỷ lệ vắng, cấm thi.
- Kết luận cẩn trọng: nếu tài liệu không nêu chế tài riêng giữa có phép và không phép thì phải nói rõ.

Đây là kỹ thuật gần với:

- Query Decomposition.
- Multi-hop Retrieval.
- Coverage-based Evidence Selection.
- Synthesis Prompting.

## 11. Build Context Và Prompt

`prompt_builder.py` biến danh sách docs thành context có thẻ:

```text
<NGUON id="1" ten_tai_lieu="..." dieu_khoan="..." chunk_index="..." source_type="..." khia_canh="...">
Nội dung chunk
</NGUON>
```

Mỗi nguồn có metadata:

- Tên tài liệu.
- Điều khoản/tiêu đề.
- Chunk index.
- Điểm liên quan.
- Số văn bản/ngày ban hành/ngày hiệu lực.
- Đường dẫn tương đối hoặc URL.
- Loại nguồn.
- Khía cạnh evidence.

Prompt yêu cầu Gemini:

- Chỉ trả lời dựa trên context.
- Không bịa thông tin ngoài tài liệu.
- Nếu thiếu căn cứ thì nói không tìm thấy căn cứ đủ rõ.
- Nếu chỉ có căn cứ một phần thì trả lời phần có căn cứ và nói rõ phần thiếu.
- Với câu hỏi nhiều khía cạnh, dùng nhiều nguồn thay vì chỉ nguồn đầu.
- Dùng nguồn nghiệp vụ cho thao tác hệ thống.
- Dùng nguồn chính thức cho quy định, điều kiện, chế tài.
- Kết thúc bằng dòng nguồn.

## 12. Cơ Chế Gọi Gemini Và Xoay API Key

`gemini_client.py` load nhiều API key từ `.env` và gọi theo kiểu round-robin/failover.

Cơ chế tổng quát:

- Mỗi lần gọi Gemini, hệ thống lấy danh sách key bắt đầu từ một vị trí luân phiên.
- Nếu key bị quota/rate limit hoặc unavailable, thử key tiếp theo.
- Nếu tất cả key đều lỗi, trả về thông báo lỗi chuẩn hóa.

Một câu hỏi không nhất thiết chỉ gọi Gemini một lần. Tùy luồng, có thể gọi:

- Rewrite câu hỏi theo lịch sử.
- Business retrieval plan.
- HyDE/Grounded HyDE.
- Mapping judge.
- Sinh câu trả lời cuối.

Trong cấu hình tiết kiệm quota, nên hướng tới:

- Rule-based decomposition: 0 lần Gemini.
- Retrieval bằng keyword/vector/BM25/rerank: 0 lần Gemini.
- Chỉ dùng Gemini cho final answer: 1 lần Gemini/câu hỏi.

## 13. Fallback Khi Gemini Lỗi Hoặc Hết Quota

Khi Gemini trả lỗi quota/rate limit, hệ thống có fallback:

- Nếu có docs, tạo câu trả lời extractive từ các đoạn evidence.
- Với một số câu hỏi nghiệp vụ, dùng fallback từ FAQ/source nghiệp vụ.
- Nếu không có evidence đủ rõ, trả thông báo không tìm thấy căn cứ.

Fallback giúp hệ thống vẫn trả lời được một phần khi retrieval đã đúng nhưng LLM không sinh được.

## 14. Trace Và Debug

Mỗi lượt hỏi có thể ghi trace trong `storage/traces`.

Trace giúp kiểm tra:

- Câu hỏi gốc.
- Câu hỏi sau rewrite.
- Intent/route được chọn.
- Có gọi Gemini ở bước nào không.
- Retrieval lấy được bao nhiêu nguồn.
- Top sources gồm doc name, title, chunk index, score.
- Có bị quota/rate limit không.
- Prompt/context đã chọn.
- Câu trả lời cuối.

Endpoint:

```text
GET /api/chat/traces/{trace_id}
```

Trace rất quan trọng để debug các lỗi như:

- Chọn sai nguồn.
- Lấy nhầm tài liệu website vào luồng nội bộ.
- Câu hỏi học lại bị hiểu nhầm thành thi lại.
- Chỉ dùng một nguồn trong khi câu hỏi cần nhiều nguồn.
- Gemini hết quota nhưng response không thể hiện rõ.

## 15. Ưu Điểm Hiện Tại

- Có kiến trúc RAG khá đầy đủ: metadata, BM25, vector, RRF, rerank.
- Có phân biệt nguồn nội bộ, nghiệp vụ và website.
- Có trace debug chi tiết.
- Có cơ chế multi-hop cho câu hỏi phức tạp.
- Có coverage-based evidence selection để tránh chỉ lấy một chunk đầu.
- Có FAQ mapping nghiệp vụ, giúp nhiều câu thao tác trả lời nhanh và ít cần LLM.
- Có fallback khi Gemini lỗi.
- Có test cho routing, retrieval, business API và chat.

## 16. Hạn Chế Và Rủi Ro

- Một số file hoặc README cũ có dấu hiệu lỗi encoding, cần chuẩn hóa UTF-8.
- Gemini có thể bị gọi nhiều lần trong một câu hỏi nếu bật rewrite, HyDE, business retrieval plan.
- Nếu nhiều API key cùng project, quota vẫn có thể hết nhanh vì quota tính theo project/model.
- `gemini_call_count` trong response có thể chưa phản ánh đúng số lần gọi thực tế do một số call chạy trong thread.
- Phân tách nội bộ/nghiệp vụ giúp kiểm soát nguồn nhưng làm routing phức tạp.
- Nếu prompt nhận context yếu hoặc thiếu, Gemini vẫn có thể trả "không tìm thấy căn cứ".
- Fallback extractive có thể đúng nguồn nhưng câu trả lời chưa mượt.
- Nếu ChromaDB chứa chung official, business và website chunks, cần filter `source_type` cẩn thận.

## 17. Đề Xuất Phát Triển Tiếp

Các hướng nên ưu tiên:

1. Giảm số lần gọi Gemini:
   - Chỉ rewrite khi câu hỏi phụ thuộc lịch sử.
   - Tắt hoặc hạn chế HyDE.
   - Không gọi business retrieval plan cho từng sub-question nếu search thường đã đủ tốt.
   - Cache retrieval plan lâu hơn.

2. Chuẩn hóa counter Gemini:
   - Ghi số lần gọi Gemini vào request state hoặc trace logger thay vì chỉ dùng ContextVar.

3. Cải thiện fallback:
   - Với FAQ nghiệp vụ, trả lời trực tiếp từ `faq_answer`.
   - Với quy chế, trích câu liên quan nhất và gắn nguồn rõ ràng.

4. Làm route tổng quát hơn:
   - Thay vì chọn một route thắng sớm, có thể search song song official + business cho nhiều câu.
   - Sau đó evidence selector quyết định nguồn nào đủ mạnh.

5. Kiểm soát nguồn theo loại:
   - Internal retrieval filter `official_document`.
   - Business retrieval chỉ lấy `business_document`/`business_faq_mapping`.
   - Website retrieval chỉ lấy `website_uneti`.

6. Chuẩn hóa dữ liệu:
   - Reindex sau khi sửa encoding hoặc thêm tài liệu mới.
   - Kiểm tra metadata `source_type`, `doc_name`, `title`, `chunk_index`.
   - Tách rõ tài liệu đại học, thạc sĩ, sinh viên, CBGV.

## 18. Kết Luận

Chatbot hiện là một hệ thống RAG tương đối hoàn chỉnh, có khả năng tìm kiếm tài liệu theo nhiều chiến lược và tổng hợp câu trả lời từ nhiều nguồn. Điểm mạnh lớn nhất là hệ thống không phụ thuộc hoàn toàn vào Gemini, mà có lớp retrieval, metadata, vector search, rerank và trace debug.

Với các câu hỏi đơn giản, hệ thống có thể tìm nguồn và gọi Gemini một lần để trả lời. Với các câu hỏi phức tạp, hệ thống có thể tách thành nhiều khía cạnh, tìm nhiều nguồn và tổng hợp lại. Tuy nhiên, để vận hành ổn định hơn, cần tiếp tục tối ưu số lần gọi Gemini, cải thiện fallback và đảm bảo routing giữa nguồn nội bộ, nghiệp vụ, website không bị chồng chéo.

Nếu được tối ưu đúng hướng, chatbot có thể tiến gần cách hoạt động của chatbot trường: không chỉ lấy một chunk đầu tiên, mà biết chia câu hỏi, tìm nhiều căn cứ, chọn nguồn theo độ phủ và tổng hợp câu trả lời có cấu trúc.
