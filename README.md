# **GIỚI THIỆU VỀ LINEAR REGRESSION**

Trong lĩnh vực **Trí tuệ Nhân tạo** (AI) và **Học Máy** (Machine Learning), bài toán hồi quy đóng vai trò quan trọng trong việc dự đoán các giá trị liên tục. Hồi quy tuyến tính, một trong những kỹ thuật cơ bản nhất, là nền tảng cho nhiều thuật toán phức tạp hơn.

Ví dụ thực tế về bài toán hồi quy:

1. **Dự đoán giá nhà**: Dựa vào diện tích, số phòng ngủ, vị trí địa lý, tuổi của ngôi nhà.
2. **Dự báo doanh số**: Dựa vào chi phí quảng cáo, mùa trong năm, xu hướng thị trường.
3. **Ước tính mức tiêu thụ nhiên liệu của xe**: Dựa vào trọng lượng xe, công suất động cơ, số km đã đi.
4. **Dự đoán nhiệt độ**: Dựa vào độ ẩm, tốc độ gió, áp suất khí quyển.

## **1. Cơ sở của thuật toán Linear Regression**

### **1.1 Định nghĩa**

Hồi quy tuyến tính (Linear Regression) là một phương pháp thống kê để mô hình hóa mối quan hệ tuyến tính giữa một biến phụ thuộc (thường ký hiệu là $Y$) và một hoặc nhiều biến độc lập (thường ký hiệu là $X$).

![Bài toán hồi quy đơn giản](../assets/Linear-Regression/images/SimpleLinearRegression.png)

### **1.2 Mục đích chính**

- Dự đoán: Ước tính giá trị của biến phụ thuộc dựa trên các biến độc lập.
- Giải thích: Xác định mức độ ảnh hưởng của từng biến độc lập đến biến phụ thuộc.
- Kiểm tra giả thuyết: Xác nhận hoặc bác bỏ các giả thuyết về mối quan hệ giữa các biến.

## **2. Các loại và Công thức của Hồi quy Tuyến tính**

### **2.1 Hồi quy Tuyến tính Đơn giản**

Công thức: $Y = mX + b$

Trong đó:

- $Y$: Biến phụ thuộc
- $X$: Biến độc lập
- $m$: Hệ số góc (slope)
- $b$: Hệ số chặn (intercept)

Ví dụ: Dự đoán giá nhà ($Y$) dựa vào diện tích ($X$)
$Y = 1000X + 50000$ (đơn vị: nghìn đồng)

Giải thích: Mỗi mét vuông tăng thêm làm tăng giá nhà 1 triệu đồng, và giá khởi điểm là 50 triệu đồng.

### **2.2 Hồi quy Tuyến tính Đa biến**

Công thức: $Y = m_{1}X_{1} + m_{2}X_{2} + ... + m_{n}X_{n} + b$

Trong đó:

- $Y$: Biến phụ thuộc
- $X_{1}, X_{2}, ..., X_{n}$: Các biến độc lập
- $m_{1}, m_{2}, ..., m_{n}$: Các hệ số tương ứng
- $b$: Hệ số chặn

Ví dụ: Dự đoán giá nhà (Y) dựa vào diện tích ($X_{1}$), số phòng ngủ ($X_{2}$), và khoảng cách đến trung tâm ($X_{3}$)
$Y = 1000X_{1} + 50000X_{2} - 5000X_{3} + 100000$ (đơn vị: nghìn đồng)

Giải thích:

- Mỗi mét vuông tăng thêm làm tăng giá 1 triệu đồng
- Mỗi phòng ngủ thêm làm tăng giá 50 triệu đồng
- Mỗi km xa trung tâm làm giảm giá 5 triệu đồng
- Giá khởi điểm là 100 triệu đồng

## **3. Triển khai và Tối ưu hóa**

### **3.1 Hàm Chi phí (Cost Function)**

Hàm chi phí đo lường mức độ "sai lệch" của mô hình. Mục tiêu là tối thiểu hóa hàm này.

Mean Squared Error (MSE):

$$J = \frac{1}{n} * \sum(y_{i} - \hat{y_{i}})^{2}$$

Trong đó:

- $J$: Giá trị của hàm chi phí
- $n$: Số lượng mẫu
- $y_{i}$: Giá trị thực tế
- $\hat{y_{i}}$: Giá trị dự đoán

### **3.2 Gradient Descent**

Đây là thuật toán tối ưu để tìm giá trị tối thiểu của hàm chi phí.

Các bước:

1. Khởi tạo các tham số ($m$ và $b$) với giá trị bất kỳ.
2. Tính gradient (đạo hàm) của hàm chi phí theo từng tham số.
3. Cập nhật các tham số: $\alpha$ là tốc độ học (learning rate).
$$m_{new} = m_{old} - \alpha * \displaystyle \frac{\partial J}{\partial m}$$
$$b_{new} = b_{old} - \alpha * \displaystyle \frac{\partial J}{\partial b}$$
4. Lặp lại bước 2 và 3 cho đến khi hội tụ.

**Ví dụ cụ thể:**

Giả sử ta có dữ liệu:

```python
X = [1, 2, 3]

Y = [2, 4, 5]
```

Khởi tạo: $m = 0$, $b = 0$, $\alpha = 0.01$

Lặp lại:

1. Tính $\hat{y} = mX + b$
2. Tính gradient:
$$\displaystyle \frac{\partial J}{\partial m} = (\frac{2}{n}) * \sum (\hat{y_{i}} - y_{i}) * x_{i}$$
$$\displaystyle \frac{\partial J}{\partial b} = (\frac{2}{n}) * \sum (\hat{y_{i}} - y_{i})$$
3. Cập nhật m và b Sau nhiều lần lặp, ta sẽ có $m \approx 1.5$ và $b \approx 0.5$

## **4. Đánh giá Mô hình**

### **4.1 R-squared $R^{2}$**

$R^{2}$ đo lường phần trăm biến thiên của biến phụ thuộc được giải thích bởi mô hình.

$$R^{2} = 1 - (\frac{RSS}{TSS})$$

Trong đó:

- $R_{2}$: hệ số xác định (coefficient of determination)
- RSS: Tổng bình phương phần dư (sum of squares of residuals)
- TSS: Tổng bình phương tổng (total sum of squares)

> $R_{2}$ nằm trong khoảng $[0, 1]$. Giá trị càng gần $1$ càng tốt.

### **Mean Absolute Error (MAE)**

$$MAE = \frac{1}{n} * \sum |y_{i} - \hat{y_{i}}|$$

MAE dễ hiểu và ít nhạy cảm với outliers hơn MSE.

### **4.3 Root Mean Squared Error (RMSE)**

$$RMSE = \sqrt{MSE}$$

RMSE có đơn vị giống với biến phụ thuộc, giúp dễ dàng diễn giải.

## **5. Kết luận**

Hồi quy tuyến tính, mặc dù đơn giản, vẫn là một công cụ mạnh mẽ trong phân tích dữ liệu và học máy. Nó cung cấp cơ sở cho nhiều kỹ thuật phức tạp hơn như hồi quy đa thức, hồi quy logistic, và mạng neural.

Tuy nhiên, cần lưu ý rằng hồi quy tuyến tính có những hạn chế:

- Giả định tuyến tính có thể không phù hợp với mọi dữ liệu thực tế.
- Nhạy cảm với outliers.
- Không thể mô hình hóa các mối quan hệ phi tuyến tính phức tạp.

Để khắc phục những hạn chế này, có thể cân nhắc sử dụng các kỹ thuật nâng cao như hồi quy phi tuyến, mô hình cây quyết định, hoặc mạng neural, tùy thuộc vào bản chất của dữ liệu và bài toán cụ thể.
