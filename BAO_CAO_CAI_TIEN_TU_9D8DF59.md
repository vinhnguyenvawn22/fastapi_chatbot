# Báo cáo cải tiến chatbot RAG UNETI từ commit 9d8df59050ff9124ee46f267b910cf295ccb623d

## 1. Phạm vi so sánh

Báo cáo này tổng hợp các thay đổi của chatbot từ commit:

`9d8df59050ff9124ee46f267b910cf295ccb623d`

đến trạng thái hiện tại của nhánh `develop`.

Tổng quan thay đổi:

- 9 file có thay đổi.
- Khoảng 1878 dòng được bổ sung.
- Các nhóm file chính được cải tiến:
  - `app/controller/chatbot_controller.py`
  - `app/data/elasticsearch_client.py`
  - `app/data/business_knowledge.py`
  - `app/data/query_context.py`
  - Các file test trong `tests/`

## 2. Mục tiêu cải tiến

Trước mốc cải tiến, chatbot gặp một số vấn đề chính:

- Một số câu hỏi học vụ bị chọn sai nguồn, ví dụ lấy tài liệu thạc sĩ cho câu hỏi sinh viên đại học.
- Câu hỏi chính sách nội bộ đôi khi bị kéo sang nguồn nghiệp vụ hoặc website.
- Một số câu hỏi dùng từ đời thường không khớp với từ trong quy chế, làm retrieval lạc nguồn.
- Câu hỏi cần căn cứ rõ nhưng chatbot vẫn trả lời "không tìm thấy căn cứ".
- Luồng multi-hop có nguy cơ gọi Gemini quá nhiều lần.
- Chưa có đủ test regression cho các lỗi thực tế đã phát hiện.

Mục tiêu của đợt cải tiến là:

- Tăng độ chính xác khi truy xuất tài liệu nội bộ.
- Ưu tiên đúng Quy chế đào tạo đại học chính quy 832 cho nhóm câu hỏi sinh viên đại học.
- Giảm chọn nhầm tài liệu thạc sĩ, tuyển sinh, website, web support khi câu hỏi là quy định học vụ.
- Cho phép câu hỏi đời thường được mở rộng sang thuật ngữ trong văn bản.
- Giảm số lần gọi Gemini ở các bước phụ.
- Bổ sung test để tránh lỗi cũ tái diễn.

## 3. Cải tiến routing và nhận diện ngữ cảnh câu hỏi

File chính:

- `app/data/query_context.py`
- `app/controller/chatbot_controller.py`

Đã bổ sung thêm các tín hiệu nhận diện câu hỏi sinh viên và chính sách học vụ, ví dụ:

- `cảnh báo học tập`
- `khối lượng học tập`
- `chuyển trường`
- `học cải thiện`
- `học phần tự chọn`
- `học cùng lúc hai chương trình`
- `hủy đăng ký học phần`
- `rút bớt học phần`
- `điểm F`, `F+`
- `tín chỉ tương đương`

Ý nghĩa:

- Chatbot nhận ra tốt hơn câu hỏi thuộc nhóm quy chế đào tạo.
- Các câu hỏi về tín chỉ, học phần, điểm, cảnh báo học tập, chuyển trường được ưu tiên luồng tài liệu nội bộ.
- Hạn chế việc đẩy câu hỏi chính sách sang Web Support hoặc website.

Ví dụ cải thiện:

- "Em đang bị cảnh báo học tập thì tối đa được đăng ký bao nhiêu tín chỉ?"
- "Tôi muốn chuyển trường, không phải chuyển chương trình đào tạo"
- "F+ và F khác nhau thế nào?"
- "Cách hủy học phần đã đăng ký"

## 4. Cải tiến query expansion cho câu hỏi đời thường

File chính:

- `app/data/elasticsearch_client.py`

Đã bổ sung nhiều mapping mở rộng truy vấn để đưa câu hỏi đời thường về đúng thuật ngữ trong quy chế.

Ví dụ:

- `gpa` được mở rộng thành:
  - `điểm trung bình tích lũy`
  - `điểm trung bình học tập`
  - `điểm trung bình chung tích lũy`
  - `tính điểm trung bình`
  - `điểm hệ 4`

- `hủy học phần` được mở rộng thành:
  - `hủy đăng ký học phần`
  - `rút bớt học phần`
  - `đăng ký khối lượng học tập`
  - `Điều 10`
  - `Điều 9`

- `chuyển trường` được mở rộng thành:
  - `điều kiện chuyển trường`
  - `Hiệu trưởng`
  - `cùng ngành`
  - `nơi cư trú`
  - `hoàn cảnh khó khăn`
  - `Quy chế đào tạo đại học chính quy`
  - `Điều 28`

- `cảnh báo học tập` được mở rộng thành:
  - `khối lượng học tập`
  - `đăng ký khối lượng học tập`
  - `không quá 16 tín chỉ`
  - `Điều 9`

Ý nghĩa:

- Người dùng không cần hỏi đúng từng chữ trong văn bản.
- Retrieval có thêm keyword pháp quy để tìm đúng điều khoản.
- BM25, vector search và reranker có tín hiệu tốt hơn để chọn chunk đúng.

## 5. Cải tiến policy profile cho các nhóm câu hỏi học vụ

File chính:

- `app/data/elasticsearch_client.py`
- `app/controller/chatbot_controller.py`

Đã bổ sung hoặc cải thiện các policy profile:

- `grade_average`: câu hỏi về GPA, điểm trung bình, điểm tích lũy.
- `course_registration_change`: câu hỏi hủy/rút học phần đã đăng ký.
- `credit_load_warning`: câu hỏi sinh viên bị cảnh báo học tập được đăng ký tối đa bao nhiêu tín chỉ.
- `transfer_school`: câu hỏi chuyển trường.
- `elective_failed_course`: câu hỏi học phần tự chọn bị F/F+.
- `f_grade_comparison`: câu hỏi so sánh F+ và F.
- `credit_definition`: câu hỏi một tín chỉ tương đương bao nhiêu tiết/giờ.

Mỗi profile có bộ từ khóa, điều khoản và rule ưu tiên riêng.

Ý nghĩa:

- Câu hỏi không chỉ được xử lý bằng độ giống vector chung.
- Hệ thống có thêm lớp hiểu nghiệp vụ để ưu tiên đúng điều/mục.
- Giảm lỗi lấy chunk có từ giống nhau nhưng sai chủ đề.

## 6. Cải tiến chấm điểm và ưu tiên nguồn

File chính:

- `app/data/elasticsearch_client.py`
- `app/controller/chatbot_controller.py`

Đã bổ sung scoring theo hướng:

- Cộng điểm mạnh cho nguồn đúng điều khoản.
- Cộng điểm cho Quy chế đào tạo đại học chính quy 832 khi câu hỏi thuộc nhóm sinh viên đại học.
- Cộng điểm cho keyword bắt buộc như `16 tín chỉ`, `cảnh báo học tập`, `chuyển trường`, `học phần tự chọn`, `F+`, `F`.
- Trừ điểm nguồn sai đối tượng như thạc sĩ nếu câu hỏi không nhắc thạc sĩ.
- Trừ điểm nguồn nhiễu như web support, thiết bị, phòng học, nghiên cứu khoa học khi câu hỏi là quy chế học vụ.

Ví dụ:

- Câu "cảnh báo học tập tối đa bao nhiêu tín chỉ" được ưu tiên Điều 9, khoản có nội dung "không quá 16 tín chỉ".
- Câu "chuyển trường" được ưu tiên Điều 28 của QD 832, không nhầm sang chuyển chương trình đào tạo hoặc quy chế thạc sĩ.
- Câu "F môn tự chọn" được ưu tiên Điều 11 về học lại/học đổi.

## 7. Sửa lỗi fallback nhầm sang website

File chính:

- `app/controller/chatbot_controller.py`

Đã phát hiện lỗi với câu:

`tôi muốn chuyển trường, không phải chuyển chương trình đào tạo`

Trace cho thấy hệ thống đã tìm được đúng Điều 28 của Quy chế đào tạo đại học chính quy 832, nhưng sau đó rule lọc nguồn loại nhầm toàn bộ nguồn nội bộ. Vì không còn nguồn sau lọc, API tổng fallback sang website và trả:

`Không tìm thấy thông tin phù hợp trên website UNETI.`

Nguyên nhân:

- Rule cũ loại nguồn nếu từ `tuyển sinh` xuất hiện trong toàn bộ nội dung chunk.
- Chunk Điều 28 đúng có nhắc tới từ `tuyển sinh` trong nội dung, dù tài liệu không phải quy chế tuyển sinh.

Cách sửa:

- Chỉ loại `tuyển sinh` khi xuất hiện trong metadata/tên tài liệu.
- Không loại chunk đúng chỉ vì nội dung điều khoản có nhắc tới từ này.

Kết quả:

- Retrieval giữ lại đúng 2 chunk Điều 28 của QD 832.
- Luồng aggregate không còn rơi sang website fallback cho câu hỏi chuyển trường.

## 8. Cải tiến câu trả lời deterministic cho cảnh báo học tập

File chính:

- `app/controller/chatbot_controller.py`

Đã bổ sung cơ chế trả lời deterministic cho câu hỏi:

`Em đang bị cảnh báo học tập thì tối đa được đăng ký bao nhiêu tín chỉ?`

Khi hệ thống tìm thấy căn cứ có cụm:

- `cảnh báo học tập`
- `không quá 16 tín chỉ`
- `đăng ký khối lượng học tập`

thì backend có thể trả lời trực tiếp theo căn cứ, thay vì để Gemini suy luận sai sang mức tối đa chung `3/2 số tín chỉ trung bình`.

Ý nghĩa:

- Tránh câu trả lời sai với các điều khoản có con số rõ ràng.
- Ưu tiên điều khoản riêng cho sinh viên bị cảnh báo học tập.
- Giảm rủi ro Gemini bỏ qua chi tiết quan trọng trong context.

## 9. Giảm số lần gọi Gemini trong multi-hop

File chính:

- `app/data/business_knowledge.py`
- `app/controller/chatbot_controller.py`

Trước đó, khi dùng multi-hop, mỗi sub-question có thể kích hoạt thêm Gemini để tạo business retrieval plan hoặc mapping judge. Điều này làm mỗi câu hỏi có thể tốn nhiều lượt gọi Gemini.

Đã bổ sung flag:

`skip_retrieval_plan_llm`

Khi retrieve cho sub-question trong multi-hop:

- Không gọi Gemini để tạo retrieval plan nghiệp vụ.
- Không gọi LLM judge mapping ở bước phụ.
- Chỉ dùng rule/keyword/vector search.

Ý nghĩa:

- Giảm chi phí và quota Gemini.
- Hạn chế tình trạng mới hỏi vài câu đã hết quota.
- Multi-hop vẫn có thể tìm nhiều nguồn, nhưng không nhân số lần gọi Gemini theo số sub-question.

## 10. Cải thiện phân biệt nghiệp vụ và nội bộ

File chính:

- `app/controller/chatbot_controller.py`
- `app/data/business_knowledge.py`

Đã cải thiện cách API tổng chọn giữa:

- Tài liệu nội bộ chính thức: quy chế, quy định, quyết định.
- Tài liệu nghiệp vụ: hướng dẫn thao tác Web Support, quy trình hệ thống.
- Website UNETI.

Nguyên tắc sau cải tiến:

- Câu hỏi quy định/chính sách/học vụ ưu tiên nguồn `official_document`.
- Câu hỏi thao tác hệ thống ưu tiên `business_document` hoặc `business_faq_mapping`.
- Website chỉ là fallback khi không có căn cứ nội bộ/nghiệp vụ phù hợp.
- Không để website chunk lẫn vào luồng nội bộ khi đang cần văn bản chính thức.

Ví dụ:

- "Cách xem kết quả học tập theo kì" phù hợp nguồn nghiệp vụ/Web Support.
- "Cảnh báo học tập tối đa bao nhiêu tín chỉ" phù hợp quy chế 832.
- "Tôi muốn chuyển trường" phù hợp Điều 28 quy chế 832.

## 11. Bổ sung test regression

Các file test được bổ sung:

- `tests/test_hybrid_retrieval.py`
- `tests/test_chat_langchain.py`
- `tests/test_business_faq_mapping.py`

Nhóm test mới bao phủ:

- Query expansion cho GPA.
- Query expansion cho hủy/rút học phần.
- Query expansion và priority cho chuyển trường.
- Query expansion và priority cho cảnh báo học tập.
- Query expansion và priority cho học phần tự chọn bị F/F+.
- Query expansion và priority cho so sánh F+ và F.
- Query expansion và priority cho định nghĩa tín chỉ.
- Chặn lấy nhầm quy chế thạc sĩ.
- Chặn nguồn web support khi câu hỏi là quy chế học vụ.
- Đảm bảo multi-hop sub-question không gọi Gemini retrieval plan.
- Đảm bảo câu cảnh báo học tập trả đúng 16 tín chỉ.

Một số test đã chạy thành công:

- Nhóm chuyển trường: 3 passed.
- Nhóm cảnh báo học tập: 3 passed.
- Nhóm policy profile mở rộng: 12 passed.

## 12. Giá trị đạt được

Sau cải tiến, chatbot tốt hơn ở các điểm:

- Tìm đúng nguồn hơn cho câu hỏi học vụ.
- Giảm nhầm giữa học lại, thi lại, hoãn thi, rút học phần, chuyển trường.
- Giảm nhầm giữa sinh viên đại học và học viên thạc sĩ.
- Giảm nhầm giữa quy chế nội bộ và hướng dẫn Web Support.
- Có khả năng xử lý câu hỏi dùng từ đời thường tốt hơn.
- Có cơ chế bảo vệ các câu hỏi có căn cứ số liệu rõ ràng.
- Giảm số lần gọi Gemini trong multi-hop.
- Có thêm test tự động để kiểm soát chất lượng sau mỗi lần sửa.

## 13. Hạn chế còn lại

Một số điểm vẫn cần tiếp tục theo dõi:

- Các rule học vụ đang tăng dần theo lỗi thực tế, nên cần tiếp tục tổng quát hóa để tránh quá nhiều rule chồng chéo.
- Một số câu hỏi phức tạp vẫn phụ thuộc vào chất lượng chunk và metadata của tài liệu gốc.
- Nếu index tài liệu bị cũ hoặc metadata sai, retrieval vẫn có thể chọn sai nguồn.
- File `storage/document_index/index.json` có thay đổi do quá trình chạy/index, cần kiểm soát khi commit để tránh đưa thay đổi cache không cần thiết.
- Cần test thực tế thêm với bộ ground truth đầy đủ để đo điểm trước/sau.

## 14. Kết luận

Từ commit `9d8df59050ff9124ee46f267b910cf295ccb623d`, chatbot đã được cải tiến theo hướng thực dụng: không chỉ dựa vào semantic search chung, mà bổ sung thêm query expansion, policy profile, scoring theo nguồn, rule lọc sai đối tượng và test regression.

Nhóm cải tiến quan trọng nhất là làm cho API tổng `/api/chat/` chọn đúng tài liệu chính thức trong các câu hỏi học vụ, đồng thời giảm số lần gọi Gemini ở các bước phụ. Điều này giúp chatbot trả lời ổn định hơn, tiết kiệm quota hơn và dễ debug hơn thông qua trace.
