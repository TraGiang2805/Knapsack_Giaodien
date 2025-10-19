# 🎒 Knapsack Problem Solver with PyQt5 GUI

Ứng dụng giải bài toán Knapsack với giao diện đồ họa PyQt5, hỗ trợ so sánh nhiều thuật toán tối ưu hóa.

## ✨ Tính năng

### 🔧 Thuật toán được hỗ trợ
- **Simulated Annealing (SA)** - Thuật toán mô phỏng luyện kim
- **Bee Colony Optimization (BCO)** - Thuật toán tối ưu đàn ong
- **Genetic Algorithm (GA)** - Thuật toán di truyền

### 📊 Giao diện người dùng
- **Tab Nhập Dữ Liệu**: Nhập thủ công hoặc tạo dữ liệu ngẫu nhiên
- **Tab Chạy Thuật Toán**: Chạy từng thuật toán hoặc so sánh tất cả
- **Tab Kết Quả & Biểu Đồ**: Hiển thị kết quả và biểu đồ so sánh trực quan

### 📈 Biểu đồ so sánh
- **Biểu đồ giá trị**: So sánh giá trị tối ưu của các thuật toán
- **Biểu đồ thời gian**: So sánh thời gian thực thi
- **Biểu đồ hiệu suất**: Tính toán hiệu suất (giá trị/thời gian)

## 🚀 Cài đặt

### Yêu cầu hệ thống
- Python 3.7+
- PyQt5
- NumPy
- Matplotlib
- SimpleAI

### Cài đặt dependencies
```bash
pip install PyQt5 numpy matplotlib simpleai
```

## 🎯 Cách sử dụng

1. **Chạy ứng dụng**:
   ```bash
   python knapsack_pyqt.py
   ```

2. **Nhập dữ liệu**:
   - Chuyển sang tab "📊 Nhập Dữ Liệu"
   - Nhập số lượng vật phẩm và dữ liệu
   - Hoặc nhấn "🎲 Tạo Dữ Liệu Ngẫu Nhiên"

3. **Chạy thuật toán**:
   - Chuyển sang tab "⚙️ Chạy Thuật Toán"
   - Chọn thuật toán muốn chạy
   - Hoặc nhấn "🚀 So Sánh Tất Cả" để chạy cả 3 thuật toán

4. **Xem kết quả**:
   - Chuyển sang tab "📈 Kết Quả & Biểu Đồ"
   - Nhấn các nút biểu đồ để xem so sánh trực quan

## 📋 Cấu trúc dự án

```
knapsack_pyqt.py          # File chính chứa toàn bộ ứng dụng
├── KnapsackProblem       # Class định nghĩa bài toán Knapsack
├── Algorithm Functions   # Các hàm thuật toán (SA, BCO, GA)
├── AlgorithmWorker       # Worker thread để chạy thuật toán
├── MplCanvas            # Widget matplotlib để vẽ biểu đồ
└── KnapsackGUI          # Class giao diện chính
```

## 🔬 Thuật toán

### Simulated Annealing (SA)
- Sử dụng thư viện SimpleAI
- 5000 iterations
- Penalty cho vi phạm ràng buộc trọng lượng

### Bee Colony Optimization (BCO)
- 30 bees, 200 iterations
- Probabilistic selection dựa trên fitness
- Random mutation cho mỗi bee

### Genetic Algorithm (GA)
- 30 individuals, 100 generations
- Tournament selection
- Single-point crossover
- Mutation rate: 10%

## 🎨 Giao diện

- **Thiết kế hiện đại** với màu sắc đẹp mắt
- **Responsive layout** với tabs và groups
- **Progress indicators** khi chạy thuật toán
- **Error handling** với thông báo rõ ràng
- **Biểu đồ tương tác** với matplotlib

## 📊 Screenshots

Ứng dụng có giao diện thân thiện với:
- Bảng nhập dữ liệu trực quan
- Nút bấm màu sắc phân biệt thuật toán
- Biểu đồ so sánh chi tiết
- Thông báo kết quả đầy đủ

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Hãy tạo issue hoặc pull request.

## 📄 License

Dự án này được phát hành dưới MIT License.

## 👨‍💻 Tác giả

Được phát triển với ❤️ bằng Python và PyQt5