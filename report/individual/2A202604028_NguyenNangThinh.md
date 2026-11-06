# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202604028
- **Họ và tên:** Nguyễn Năng Thịnh
- **SEED** (in ở ô đầu notebook): 1766

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính
Bảng dưới đây so sánh hiệu suất của 8 phương pháp lọc tin nhắn spam theo các tiêu chí: Precision, Recall, F1 trên tập test, F1 trên tập SUB 40, thời gian huấn luyện, tốc độ suy luận, độ trễ p50/p95 và chi phí trên 1.000 tin nhắn.

Mô hình	Precision	Recall	F1 (test)	F1 (SUB 40)	Train (s)	Tin/s	p50 (ms/tin)	p95 (ms/tin)	USD/1k tin
1. BoW + LogReg	1.000	0.760	0.864	0.750	3.114	16311.522	0.045	0.210	0.000
2. BoW + MLP	1.000	0.833	0.909	0.919	7.320	20055.798	0.306	1.219	0.000
3a. MiniLM + kNN	0.893	0.958	0.925	0.947	2.291	129.071	16.229	21.169	0.000
3b. MiniLM + LogReg	0.968	0.938	0.952	0.947	2.640	1861.691	6.686	7.257	0.000
3c. MiniLM + MLP head	0.967	0.927	0.947	0.947	3.340	2536.406	6.797	8.165	0.000
4a. LLM nhỏ zero-shot	0.000	0.000	0.000	0.000	0.000	1.024	934.284	1484.982	0.008
4b. LLM lớn zero-shot	0.000	0.000	0.000	0.000	0.000	0.916	1073.822	1331.455	0.078
4c. LLM nhỏ few-shot	0.000	0.000	0.000	0.000	0.000	1.109	835.443	1221.851	0.019

Nhận xét:

* MiniLM + LogReg đạt F1 test cao nhất (0.952).
* MiniLM + LogReg có p95 chỉ 7.257 ms/tin, thấp hơn giới hạn 50 ms/tin.
* BoW + MLP có tốc độ suy luận cao nhất (20.055 tin/s theo số liệu hiển thị là 20.055,798 tin/s).


## 3. Dự đoán và thực tế

1. Machine Learning: Logistic Regression đạt F1 test 0.864. Mô hình có precision 1.0 nhưng recall chỉ 0.76, nghĩa là những tin được dự đoán là spam đều chính xác, nhưng vẫn bỏ sót một số tin spam.
2. Deep Learning: MLP đạt F1 0.909, cao hơn Logistic Regression. Tuy nhiên, đường validation loss bắt đầu tăng sau epoch 5 trong khi training loss tiếp tục giảm. Đây là dấu hiệu overfitting.
3. Embedding: Encoder + Logistic Regression đạt F1 0.952, vượt cả MLP trên BoW. Kết quả này cho thấy chất lượng biểu diễn đầu vào có ảnh hưởng rất lớn đến hiệu quả phân loại.

## 4. Phân tích 50 case khó

So sánh các mô hình:

Mô hình	F1 thông thường	F1 50 case khó	Mức giảm
BoW + LogReg	0.750	0.333	0.417
BoW + MLP	0.919	0.286	0.633
Emb + kNN	0.947	0.737	0.210
Emb + LogReg	0.947	0.717	0.230
Emb + MLP head	0.947	0.667	0.280
LLM nhỏ zero-shot	0.000	0.000	0.000

BoW + MLP là mô hình tụt hiệu suất nhiều nhất. F1 giảm từ 0.919 trên tập SUB thông thường xuống còn 0.286 trên 50 case khó, tức giảm 0.633.

Dựa trên bảng phân tích theo nhóm, BoW + MLP gặp khó khăn đặc biệt ở:

* Spam giọng tự nhiên: 0% – Mô hình không nhận diện được những tin nhắn spam có cách diễn đạt giống hội thoại thông thường.
* Né từ khóa: 0% – Khi nội dung spam sử dụng từ đồng nghĩa hoặc cách diễn đạt khác với những từ khóa đã học, mô hình dễ phân loại sai.
* Thao túng / lừa đảo: 12% – Mô hình khó nhận diện ý đồ lừa đảo khi nội dung không chứa những từ khóa spam quen thuộc.

Hai ví dụ về cơ chế mô hình phân loại sai:

Ví dụ 1 – Spam sử dụng ngôn ngữ tự nhiên:

Một tin nhắn quảng cáo hoặc lừa đảo có thể được viết giống như một cuộc trò chuyện bình thường, chẳng hạn: “Hey, I have something interesting to share with you. Let me know when you’re free.”

BoW chỉ quan tâm đến sự xuất hiện của các từ riêng lẻ, không hiểu được ngữ cảnh và ý định thực sự của người gửi. Vì vậy, những từ ngữ mang tính hội thoại có thể khiến mô hình nhầm tin nhắn spam thành ham.

Ví dụ 2 – Spam né từ khóa:

Một tin nhắn quảng cáo có thể sử dụng cách diễn đạt như “Get your exclusive reward today” thay vì những cụm từ spam quen thuộc như “free”, “prize” hoặc “winner”.

Do BoW biểu diễn văn bản dựa trên sự xuất hiện của các từ trong từ điển, mô hình khó nhận diện những cách diễn đạt mới có ý nghĩa tương tự. Đây là hạn chế của phương pháp biểu diễn văn bản không có khả năng hiểu ngữ nghĩa.

Lưu ý: Hai ví dụ trên minh họa cơ chế lỗi theo nhóm, không phải nguyên văn hai tin nhắn được xác nhận từ tập 50 case.

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật

Với yêu cầu xử lý 10 triệu SMS/ngày và độ trễ p95 < 50 ms/tin, tôi lựa chọn MiniLM Embedding + Logistic Regression (3b).

Cơ sở lựa chọn

Dựa trên kết quả thực nghiệm:

Tiêu chí	BoW + LogReg	BoW + MLP	MiniLM + LogReg	Yêu cầu
F1 test	0.864	0.909	0.952	Càng cao càng tốt
F1 (SUB 40)	0.750	0.919	0.947	Càng cao càng tốt
F1 (50 case khó)	0.333	0.286	0.717	Càng cao càng tốt
Tốc độ suy luận (tin/s)	16,311.522	20,055.798	1,861.691	≥ 116
p95 (ms/tin)	0.210	1.219	7.257	< 50
Chi phí/1.000 tin	$0	$0	$0	Càng thấp càng tốt
Đáp ứng độ trễ	Có	Có	Có	Bắt buộc
Đánh giá	Nhanh, nhưng yếu ở case khó	F1 thông thường tốt, tổng quát hóa kém ở case khó	Cân bằng tốt giữa độ chính xác và khả năng xử lý case khó	

Mô hình có p95 = 7.257 ms/tin, thấp hơn đáng kể so với giới hạn 50 ms/tin. F1 test đạt 0.952, đồng thời duy trì hiệu suất tương đối tốt trên 50 case khó với F1 = 0.717.

Khả năng đáp ứng quy mô 10 triệu SMS/ngày

Trung bình hệ thống cần xử lý:

10.000.000 / 86.400 ≈ 116 tin/giây.

Trong khi đó, tốc độ suy luận đo được là 1,861.691 tin/giây, cao hơn khoảng 16 lần so với nhu cầu trung bình.

Vì vậy, xét trên kết quả benchmark, phương án này có tiềm năng đáp ứng yêu cầu về thông lượng và độ trễ.

Những đánh đổi cần chấp nhận

* Tài nguyên tính toán: MiniLM cần tài nguyên để chuyển văn bản thành embedding, trong khi Logistic Regression có chi phí tính toán rất thấp.
* Độ phức tạp triển khai: Hệ thống cần duy trì encoder MiniLM và mô hình Logistic Regression thay vì chỉ một mô hình đơn giản.
* Khả năng nhận diện spam mới: Mô hình vẫn có thể thất bại với những kiểu spam mới hoặc nội dung lừa đảo sử dụng ngôn ngữ tự nhiên. F1 trên 50 case khó chỉ đạt 0.717.
* Khả năng mở rộng: Cần kiểm thử thêm khi xử lý đồng thời nhiều yêu cầu, vì benchmark hiện tại chưa chứng minh được p95 dưới tải thực tế.

## 7. AI log và điều đã học

Hỏi gì	Nhận gì	tôi sửa gì
Viết các câu predict	AI đưa ra các con số dự đoán khá cụ thể.	Viết lại theo cách diễn đạt của tôi, sử dụng các ước lượng thay vì khẳng định những con số chính xác khi chưa chạy thực tế.
Thử thách #5	AI đề xuất 4 tin nhắn và dự đoán rằng embedding sẽ hoạt động tốt hơn.	Sau khi chạy thực nghiệm, kết quả thực tế ngược lại với dự đoán. tôi tự phân tích và giải thích lại trong mục 5.

Qua bài thực hành, tôi nhận ra rằng điểm số cao trên tập test thông thường không đồng nghĩa với việc mô hình hoạt động tốt trong mọi tình huống. Khi chuyển sang dữ liệu khác biệt, chẳng hạn 50 case khó hoặc những tin nhắn tự viết, thứ hạng giữa các mô hình có thể thay đổi đáng kể. Điều này cho thấy mô hình vẫn có những giới hạn nhất định và có thể chỉ hoạt động tốt trong phạm vi dữ liệu mà nó đã học.

tôi cũng hiểu rằng khi so sánh hai mô hình, cần phân biệt rõ sự khác biệt đến từ kiến trúc mô hình hay từ cách tối ưu và huấn luyện. Ngoài ra, với tập dữ liệu nhỏ như SUB 40, chỉ một vài dự đoán thay đổi cũng có thể làm F1 biến động đáng kể. Vì vậy, không nên chỉ dựa vào một chỉ số hoặc một tập dữ liệu nhỏ để kết luận mô hình nào tốt hơn.

