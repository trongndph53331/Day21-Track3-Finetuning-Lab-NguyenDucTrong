# Reflection — Lab 21

**1. Điều gì làm bạn ngạc nhiên nhất?**
Prompt tối ưu đưa target từ 0 lên 0.765 mà chưa cập nhật trọng số. `attn_only` với rank khớp ngân sách cũng đạt 0.965 như all-linear dù train loss thấp hơn.

**2. Bạn mất nhiều thời gian nhất ở đâu?**
Bốn lượt huấn luyện NB3/NB4 trên T4 là phần lâu nhất; pipeline gần 90 phút. Tôi không dự đoán cấu hình 4-bit lại chậm hơn fp16 trên GPU này.

**3. Niềm tin nào đã thay đổi?**
Train loss thấp nhất không đồng nghĩa model tốt nhất. `attn_only` có loss tốt hơn nhưng target chỉ hòa; model target tốt nhất vẫn FAILED vì regression.

**4. Bạn dùng AI assistant vào việc gì? Chỗ nào nó sai?**
Tôi dùng AI assistant để kiểm tra môi trường, chạy notebook, đối chiếu artifact và tổng hợp báo cáo. Không được suy diễn output baseline theo mẫu khi notebook không lưu chúng; báo cáo ghi rõ giới hạn đó thay vì bịa dữ liệu.

**5. Nếu fine-tune cho khách hàng thật, bước đầu tiên là gì?**
Tôi sẽ đóng băng bộ target và regression đại diện, rồi thống nhất cổng triển khai và dung sai trước khi train. Sau đó mới kiểm tra template, mask, token và baseline prompt mạnh.
