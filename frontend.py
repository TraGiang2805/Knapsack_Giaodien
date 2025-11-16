import sys
import threading
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QTabWidget, QLabel, QLineEdit,
                            QPushButton, QTableWidget, QTableWidgetItem,
                            QTextEdit, QProgressBar, QGroupBox, QGridLayout,
                            QMessageBox, QHeaderView, QFrame, QSplitter)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


# Import logic nghiệp vụ từ Backend
# Đảm bảo file knapsack_backend.py nằm cùng thư mục
try:
   from backend import run_SA, run_BCO, run_GA, random_dataset
except ImportError:
   print("Lỗi: Không tìm thấy file knapsack_backend.py. Vui lòng đảm bảo đã tạo file đó.")
   sys.exit(1)




# -----------------------------
# Helper Widgets (Frontend components)
# -----------------------------
class MplCanvas(FigureCanvas):
   """Widget để nhúng Matplotlib vào PyQt5"""
   def __init__(self, parent=None, width=5, height=4, dpi=100):
       self.fig = Figure(figsize=(width, height), dpi=dpi)
       super().__init__(self.fig)
       self.setParent(parent)


# -----------------------------
# Worker Thread (Connection Layer - Bridge between GUI and Backend Logic)
# -----------------------------
class AlgorithmWorker(QThread):
   """Luồng làm việc để chạy các thuật toán Knapsack (gọi Backend)"""
   finished = pyqtSignal(str, object)  # algorithm_name, result
   error = pyqtSignal(str)             # error message
  
   def __init__(self, algorithm, weights, values, capacity):
       super().__init__()
       self.algorithm = algorithm
       self.weights = weights
       self.values = values
       self.capacity = capacity
      
   def run(self):
       """Chạy logic nghiệp vụ (run_XXX) trong luồng nền"""
       try:
           if self.algorithm == "SA":
               result = run_SA(self.weights, self.values, self.capacity)
               self.finished.emit("Simulated Annealing", result)
           elif self.algorithm == "BCO":
               result = run_BCO(self.weights, self.values, self.capacity)
               self.finished.emit("Bee Colony Optimization", result)
           elif self.algorithm == "GA":
               result = run_GA(self.weights, self.values, self.capacity)
               self.finished.emit("Genetic Algorithm", result)
           elif self.algorithm == "ALL":
               sa_result = run_SA(self.weights, self.values, self.capacity)
               bco_result = run_BCO(self.weights, self.values, self.capacity)
               ga_result = run_GA(self.weights, self.values, self.capacity)
               self.finished.emit("ALL", (sa_result, bco_result, ga_result))
       except Exception as e:
           # Phát tín hiệu lỗi ra giao diện
           self.error.emit(str(e))


# -----------------------------
# Main Application Window (View/Controller)
# -----------------------------
class KnapsackGUI(QMainWindow):
   """Cửa sổ chính của ứng dụng Knapsack Solver"""
   def __init__(self):
       super().__init__()
       # Data Model (Tối thiểu, giữ ở Frontend để cập nhật giao diện)
       self.weights = []
       self.values = []
       self.capacity = 0
       self.worker = None
       self.last_results = None
      
       self.init_ui()
      
   # --- GUI Setup Methods ---
   def init_ui(self):
       self.setWindowTitle("Bài Toán Knapsack - PyQt5 GUI")
       self.setGeometry(100, 100, 1400, 900)
      
       self.setStyleSheet("""... (CSS Styles đã bị rút gọn) ...""") # Giữ nguyên style
       self.setStyleSheet("""
           QMainWindow { background-color: #f5f5f5; }
           QTabWidget::pane { border: 1px solid #c0c0c0; background-color: white; }
           QTabBar::tab { background-color: #e0e0e0; padding: 8px 16px; margin-right: 2px; }
           QTabBar::tab:selected { background-color: white; border-bottom: 2px solid #2196F3; }
           QPushButton { background-color: #2196F3; color: white; border: none; padding: 8px 16px; border-radius: 4px; font-weight: bold; }
           QPushButton:hover { background-color: #1976D2; }
           QPushButton:pressed { background-color: #0D47A1; }
           QGroupBox { font-weight: bold; border: 2px solid #cccccc; border-radius: 5px; margin-top: 1ex; padding-top: 10px; }
           QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; }
       """)
      
       central_widget = QWidget()
       self.setCentralWidget(central_widget)
      
       self.tab_widget = QTabWidget()
      
       self.create_data_input_tab()
       self.create_algorithm_tab()
       self.create_results_tab()
      
       self.tab_widget.addTab(self.data_tab, "Nhập Dữ Liệu")
       self.tab_widget.addTab(self.algorithm_tab, "Chạy Thuật Toán")
       self.tab_widget.addTab(self.results_tab, "Kết Quả & Biểu Đồ")
      
       main_layout = QVBoxLayout()
       main_layout.addWidget(self.tab_widget)
       central_widget.setLayout(main_layout)
      
   def create_data_input_tab(self):
       # ... (Toàn bộ logic tạo giao diện cho tab Nhập Dữ Liệu) ...
       self.data_tab = QWidget()
       layout = QVBoxLayout()
       title_label = QLabel("BÀI TOÁN KNAPSACK - NHẬP DỮ LIỆU")
       title_label.setAlignment(Qt.AlignCenter)
       title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #1976D2; margin: 10px;")
       layout.addWidget(title_label)
      
       manual_group = QGroupBox("Nhập Dữ Liệu Thủ Công")
       manual_layout = QGridLayout()
       manual_layout.addWidget(QLabel("Số lượng vật phẩm:"), 0, 0)
       self.num_items_input = QLineEdit()
       self.num_items_input.setText("5")
       self.num_items_input.setMaximumWidth(100)
       manual_layout.addWidget(self.num_items_input, 0, 1)
      
       self.create_form_btn = QPushButton("Tạo Form Nhập")
       self.create_form_btn.clicked.connect(self.create_input_form)
       manual_layout.addWidget(self.create_form_btn, 0, 2)
      
       self.items_table = QTableWidget()
       self.items_table.setColumnCount(3)
       self.items_table.setHorizontalHeaderLabels(["Vật phẩm", "Trọng lượng", "Giá trị"])
       header = self.items_table.horizontalHeader()
       header.setSectionResizeMode(QHeaderView.Stretch)
       manual_layout.addWidget(self.items_table, 1, 0, 1, 3)
      
       manual_layout.addWidget(QLabel("Sức chứa ba lô:"), 2, 0)
       self.capacity_input = QLineEdit()
       self.capacity_input.setMaximumWidth(150)
       manual_layout.addWidget(self.capacity_input, 2, 1)
      
       self.save_data_btn = QPushButton("Lưu Dữ Liệu")
       self.save_data_btn.clicked.connect(self.save_manual_data)
       manual_layout.addWidget(self.save_data_btn, 2, 2)
      
       manual_group.setLayout(manual_layout)
       layout.addWidget(manual_group)
      
       random_group = QGroupBox("Tạo Dữ Liệu Ngẫu Nhiên")
       random_layout = QVBoxLayout()
       self.random_data_btn = QPushButton("Tạo Dữ Liệu Ngẫu Nhiên")
       self.random_data_btn.clicked.connect(self.generate_random_data)
       self.random_data_btn.setStyleSheet("background-color: #4CAF50; font-size: 14px; padding: 10px;")
       random_layout.addWidget(self.random_data_btn)
       random_group.setLayout(random_layout)
       layout.addWidget(random_group)
      
       display_group = QGroupBox("Dữ Liệu Hiện Tại")
       display_layout = QVBoxLayout()
       self.data_display = QTextEdit()
       self.data_display.setReadOnly(True)
       self.data_display.setMaximumHeight(200)
       display_layout.addWidget(self.data_display)
       display_group.setLayout(display_layout)
       layout.addWidget(display_group)
       self.data_tab.setLayout(layout)
      
   def create_algorithm_tab(self):
       # ... (Toàn bộ logic tạo giao diện cho tab Chạy Thuật Toán) ...
       self.algorithm_tab = QWidget()
       layout = QVBoxLayout()
       title_label = QLabel("CHẠY THUẬT TOÁN KNAPSACK")
       title_label.setAlignment(Qt.AlignCenter)
       title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #1976D2; margin: 10px;")
       layout.addWidget(title_label)
      
       selection_group = QGroupBox("Chọn Thuật Toán")
       selection_layout = QHBoxLayout()
       self.sa_btn = QPushButton("Simulated Annealing")
       self.sa_btn.clicked.connect(lambda: self.run_algorithm("SA"))
       self.sa_btn.setStyleSheet("background-color: #FF9800; font-size: 12px; padding: 8px;")
       self.bco_btn = QPushButton("Bee Colony Optimization")
       self.bco_btn.clicked.connect(lambda: self.run_algorithm("BCO"))
       self.bco_btn.setStyleSheet("background-color: #9C27B0; font-size: 12px; padding: 8px;")
       self.ga_btn = QPushButton("Genetic Algorithm")
       self.ga_btn.clicked.connect(lambda: self.run_algorithm("GA"))
       self.ga_btn.setStyleSheet("background-color: #4CAF50; font-size: 12px; padding: 8px;")
       self.all_btn = QPushButton("So Sánh Tất Cả")
       self.all_btn.clicked.connect(lambda: self.run_algorithm("ALL"))
       self.all_btn.setStyleSheet("background-color: #F44336; font-size: 12px; padding: 8px;")
       selection_layout.addWidget(self.sa_btn)
       selection_layout.addWidget(self.bco_btn)
       selection_layout.addWidget(self.ga_btn)
       selection_layout.addWidget(self.all_btn)
       selection_group.setLayout(selection_layout)
       layout.addWidget(selection_group)
      
       progress_group = QGroupBox("Tiến Trình")
       progress_layout = QVBoxLayout()
       self.progress_label = QLabel("Sẵn sàng chạy thuật toán...")
       self.progress_label.setAlignment(Qt.AlignCenter)
       progress_layout.addWidget(self.progress_label)
       self.progress_bar = QProgressBar()
       self.progress_bar.setVisible(False)
       progress_layout.addWidget(self.progress_bar)
       progress_group.setLayout(progress_layout)
       layout.addWidget(progress_group)
      
       results_group = QGroupBox("Kết Quả")
       results_layout = QVBoxLayout()
       self.results_display = QTextEdit()
       self.results_display.setReadOnly(True)
       results_layout.addWidget(self.results_display)
       results_group.setLayout(results_layout)
       layout.addWidget(results_group)
       self.algorithm_tab.setLayout(layout)
      
   def create_results_tab(self):
       # ... (Toàn bộ logic tạo giao diện cho tab Kết Quả & Biểu Đồ) ...
       self.results_tab = QWidget()
       layout = QVBoxLayout()
       title_label = QLabel("KẾT QUẢ VÀ BIỂU ĐỒ")
       title_label.setAlignment(Qt.AlignCenter)
       title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #1976D2; margin: 10px;")
       layout.addWidget(title_label)
      
       control_layout = QHBoxLayout()
       self.plot_values_btn = QPushButton("Biểu Đồ Giá Trị")
       self.plot_values_btn.clicked.connect(self.plot_values_comparison)
       self.plot_values_btn.setStyleSheet("background-color: #4CAF50; font-size: 12px; padding: 8px;")
       self.plot_time_btn = QPushButton("Biểu Đồ Thời Gian")
       self.plot_time_btn.clicked.connect(self.plot_time_comparison)
       self.plot_time_btn.setStyleSheet("background-color: #FF9800; font-size: 12px; padding: 8px;")
       self.plot_efficiency_btn = QPushButton("Biểu Đồ Hiệu Suất")
       self.plot_efficiency_btn.clicked.connect(self.plot_efficiency_comparison)
       self.plot_efficiency_btn.setStyleSheet("background-color: #9C27B0; font-size: 12px; padding: 8px;")
       self.clear_plot_btn = QPushButton("Xóa Biểu Đồ")
       self.clear_plot_btn.clicked.connect(self.clear_plot)
       self.clear_plot_btn.setStyleSheet("background-color: #F44336; font-size: 12px; padding: 8px;")
       control_layout.addWidget(self.plot_values_btn)
       control_layout.addWidget(self.plot_time_btn)
       control_layout.addWidget(self.plot_efficiency_btn)
       control_layout.addWidget(self.clear_plot_btn)
       layout.addLayout(control_layout)
      
       self.canvas = MplCanvas(self, width=12, height=8, dpi=100)
       layout.addWidget(self.canvas)
      
       instructions = QLabel("Chạy thuật toán để xem biểu đồ so sánh kết quả")
       instructions.setAlignment(Qt.AlignCenter)
       instructions.setStyleSheet("font-size: 14px; color: #666; margin: 20px;")
       layout.addWidget(instructions)
       self.results_tab.setLayout(layout)


   # --- Data Handling (Frontend/Input Validation Logic) ---
   def create_input_form(self):
       # ... (Logic tạo bảng nhập) ...
       try:
           num_items = int(self.num_items_input.text())
           if num_items <= 0:
               QMessageBox.warning(self, "Lỗi", "Số lượng vật phẩm phải lớn hơn 0!")
               return
       except ValueError:
           QMessageBox.warning(self, "Lỗi", "Vui lòng nhập số lượng vật phẩm hợp lệ!")
           return
       self.items_table.setRowCount(num_items)
       for i in range(num_items):
           item_name = QTableWidgetItem(f"Vật {i+1}")
           item_name.setFlags(Qt.ItemIsEnabled)
           self.items_table.setItem(i, 0, item_name)
           self.items_table.setItem(i, 1, QTableWidgetItem(""))
           self.items_table.setItem(i, 2, QTableWidgetItem(""))


   def save_manual_data(self):
       # ... (Logic lưu dữ liệu và validate) ...
       try:
           weights, values = [], []
           for i in range(self.items_table.rowCount()):
               weight_text = self.items_table.item(i, 1).text()
               value_text = self.items_table.item(i, 2).text()
               if not weight_text or not value_text:
                   QMessageBox.warning(self, "Lỗi", f"Vui lòng nhập đầy đủ dữ liệu cho vật phẩm {i+1}!")
                   return
               weight = float(weight_text)
               value = float(value_text)
               if weight <= 0 or value <= 0:
                   QMessageBox.warning(self, "Lỗi", f"Dữ liệu vật phẩm {i+1} không hợp lệ!")
                   return
               weights.append(weight)
               values.append(value)
          
           capacity = float(self.capacity_input.text())
           if capacity <= 0:
               QMessageBox.warning(self, "Lỗi", "Sức chứa ba lô phải lớn hơn 0!")
               return
              
           self.weights = weights
           self.values = values
           self.capacity = capacity
           self.update_data_display()
           QMessageBox.information(self, "Thành công", "Dữ liệu đã được lưu!")
          
       except ValueError:
           QMessageBox.warning(self, "Lỗi", "Vui lòng nhập dữ liệu số hợp lệ!")


   def generate_random_data(self):
       # Gọi hàm backend để tạo dữ liệu ngẫu nhiên
       self.weights, self.values, self.capacity = random_dataset()
       self.update_data_display()
       QMessageBox.information(self, "Thành công", "Dữ liệu ngẫu nhiên đã được tạo!")


   def update_data_display(self):
       # ... (Logic hiển thị dữ liệu) ...
       self.data_display.clear()
       if not self.weights:
           self.data_display.append("Chưa có dữ liệu. Vui lòng nhập dữ liệu hoặc tạo dữ liệu ngẫu nhiên.")
           return
          
       self.data_display.append(f"Số vật phẩm: {len(self.weights)}")
       self.data_display.append(f"Sức chứa ba lô: {self.capacity}")
       self.data_display.append("")
       self.data_display.append(f"{'Vật phẩm':<10}{'Trọng lượng':<15}{'Giá trị':<15}{'Tỷ lệ V/W':<15}")
       self.data_display.append("-" * 60)
      
       for i, (w, v) in enumerate(zip(self.weights, self.values)):
           ratio = v/w if w > 0 else 0
           self.data_display.append(f"{f'Vật {i+1}':<10}{w:<15.2f}{v:<15.2f}{ratio:<15.2f}")


   # --- Algorithm Execution & Result Handling (Controller Logic) ---
   def run_algorithm(self, algorithm):
       if not self.weights:
           QMessageBox.warning(self, "Lỗi", "Vui lòng nhập dữ liệu trước khi chạy thuật toán!")
           return
          
       self.sa_btn.setEnabled(False)
       self.bco_btn.setEnabled(False)
       self.ga_btn.setEnabled(False)
       self.all_btn.setEnabled(False)
      
       self.progress_bar.setVisible(True)
       self.progress_bar.setRange(0, 0)
       self.progress_label.setText(f"Đang chạy thuật toán {algorithm}...")
      
       # Khởi tạo và chạy luồng Worker
       self.worker = AlgorithmWorker(algorithm, self.weights, self.values, self.capacity)
       self.worker.finished.connect(self.on_algorithm_finished)
       self.worker.error.connect(self.on_algorithm_error)
       self.worker.start()
      
   def on_algorithm_finished(self, algorithm_name, result):
       self.sa_btn.setEnabled(True)
       self.bco_btn.setEnabled(True)
       self.ga_btn.setEnabled(True)
       self.all_btn.setEnabled(True)
      
       self.progress_bar.setVisible(False)
       self.progress_label.setText("Hoàn thành!")
      
       if algorithm_name == "ALL":
           self.display_comparison_results(result)
       else:
           self.display_single_result(algorithm_name, result)
          
   def on_algorithm_error(self, error_msg):
       self.sa_btn.setEnabled(True)
       self.bco_btn.setEnabled(True)
       self.ga_btn.setEnabled(True)
       self.all_btn.setEnabled(True)
      
       self.progress_bar.setVisible(False)
       self.progress_label.setText("Có lỗi xảy ra!")
      
       QMessageBox.critical(self, "Lỗi", f"Có lỗi xảy ra: {error_msg}")
      
   def display_single_result(self, algorithm_name, result):
       # ... (Logic hiển thị kết quả chi tiết) ...
       state, value, weight, time_taken, complexity = result
       self.results_display.clear()
       self.results_display.append(f"KẾT QUẢ {algorithm_name.upper()}")
       self.results_display.append("=" * 50)
       self.results_display.append(f"Giá trị tối ưu: {value:.2f}")
       self.results_display.append(f"Trọng lượng: {weight:.2f}")
       self.results_display.append(f"Thời gian thực thi: {time_taken:.4f} giây")
       self.results_display.append(f"Độ phức tạp: {complexity} lần lặp")
       self.results_display.append("\nTrạng thái giải pháp:")
       self.results_display.append(f"{state}")
       self.results_display.append("\nCác vật phẩm được chọn:")
       self.results_display.append("-" * 30)
       selected_items = [f"Vật {i+1}: TL={self.weights[i]:.2f}, GT={self.values[i]:.2f}" for i, selected in enumerate(state) if selected]
       if selected_items:
           self.results_display.append("\n".join(f"• {item}" for item in selected_items))
       else:
           self.results_display.append("Không có vật phẩm nào được chọn.")
          
   def display_comparison_results(self, results):
       # ... (Logic hiển thị bảng so sánh) ...
       sa_result, bco_result, ga_result = results
       self.last_results = results # Lưu kết quả để vẽ biểu đồ
      
       self.results_display.clear()
       self.results_display.append("BẢNG SO SÁNH KẾT QUẢ")
       self.results_display.append("=" * 80)
       self.results_display.append(f"{'Thuật toán':<25}{'Giá trị':<12}{'Trọng lượng':<15}{'Thời gian (s)':<15}{'Độ phức tạp':<15}")
       self.results_display.append("-" * 80)
      
       algorithms = [("Simulated Annealing", sa_result), ("Bee Colony Optimization", bco_result), ("Genetic Algorithm", ga_result)]
      
       for name, result in algorithms:
           state, value, weight, time_taken, complexity = result
           self.results_display.append(f"{name:<25}{value:<12.2f}{weight:<15.2f}{time_taken:<15.4f}{complexity:<15}")
          
       self.results_display.append("=" * 80)
       best_value = max(sa_result[1], bco_result[1], ga_result[1])
       best_algo = next(name for name, result in algorithms if result[1] == best_value)
       self.results_display.append(f"\nThuật toán tốt nhất: **{best_algo}** với giá trị **{best_value:.2f}**")
       self.results_display.append("\n Mẹo: Chuyển sang tab 'Kết Quả & Biểu Đồ' để xem biểu đồ so sánh!")
      
   # --- Plotting Methods (View Logic) ---
   def plot_values_comparison(self):
       # ... (Logic vẽ biểu đồ giá trị) ...
       if not self.last_results:
           QMessageBox.warning(self, "Cảnh báo", "Vui lòng chạy thuật toán trước khi vẽ biểu đồ!")
           return
       sa_result, bco_result, ga_result = self.last_results
       algorithms = ['Simulated\nAnnealing', 'Bee Colony\nOptimization', 'Genetic\nAlgorithm']
       values = [sa_result[1], bco_result[1], ga_result[1]]
      
       self.canvas.fig.clear()
       ax = self.canvas.fig.add_subplot(111)
       bars = ax.bar(algorithms, values, color=['#FF9800', '#9C27B0', '#4CAF50'], alpha=0.8)
      
       for bar, value in zip(bars, values):
           height = bar.get_height()
           ax.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
          
       ax.set_title('So Sánh Giá Trị Tối Ưu Của Các Thuật Toán', fontsize=16, fontweight='bold')
       ax.set_ylabel('Giá Trị Tối Ưu', fontsize=12)
       ax.set_xlabel('Thuật Toán', fontsize=12)
       ax.grid(True, alpha=0.3)
       bars[values.index(max(values))].set_color('#FFD700')
      
       self.canvas.fig.tight_layout()
       self.canvas.draw()
      
   def plot_time_comparison(self):
       # ... (Logic vẽ biểu đồ thời gian) ...
       if not self.last_results:
           QMessageBox.warning(self, "Cảnh báo", "Vui lòng chạy thuật toán trước khi vẽ biểu đồ!")
           return
       sa_result, bco_result, ga_result = self.last_results
       algorithms = ['Simulated\nAnnealing', 'Bee Colony\nOptimization', 'Genetic\nAlgorithm']
       times = [sa_result[3], bco_result[3], ga_result[3]]
      
       self.canvas.fig.clear()
       ax = self.canvas.fig.add_subplot(111)
       bars = ax.bar(algorithms, times, color=['#FF9800', '#9C27B0', '#4CAF50'], alpha=0.8)
      
       for bar, time_val in zip(bars, times):
           height = bar.get_height()
           ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01, f'{time_val:.4f}s', ha='center', va='bottom', fontweight='bold')
          
       ax.set_title('So Sánh Thời Gian Thực Thi Của Các Thuật Toán', fontsize=16, fontweight='bold')
       ax.set_ylabel('Thời Gian (giây)', fontsize=12)
       ax.set_xlabel('Thuật Toán', fontsize=12)
       ax.grid(True, alpha=0.3)
       bars[times.index(min(times))].set_color('#00BCD4')
      
       self.canvas.fig.tight_layout()
       self.canvas.draw()


   def plot_efficiency_comparison(self):
       # ... (Logic vẽ biểu đồ hiệu suất) ...
       if not self.last_results:
           QMessageBox.warning(self, "Cảnh báo", "Vui lòng chạy thuật toán trước khi vẽ biểu đồ!")
           return
       sa_result, bco_result, ga_result = self.last_results
       algorithms = ['Simulated\nAnnealing', 'Bee Colony\nOptimization', 'Genetic\nAlgorithm']
       efficiency = []
       for result in [sa_result, bco_result, ga_result]:
           efficiency.append(result[1] / result[3] if result[3] > 0 else 0)
      
       self.canvas.fig.clear()
       ax = self.canvas.fig.add_subplot(111)
       bars = ax.bar(algorithms, efficiency, color=['#FF9800', '#9C27B0', '#4CAF50'], alpha=0.8)
      
       for bar, eff in zip(bars, efficiency):
           height = bar.get_height()
           ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01, f'{eff:.2f}', ha='center', va='bottom', fontweight='bold')
          
       ax.set_title('So Sánh Hiệu Suất Của Các Thuật Toán\n(Giá Trị/Thời Gian)', fontsize=16, fontweight='bold')
       ax.set_ylabel('Hiệu Suất (Giá Trị/Thời Gian)', fontsize=12)
       ax.set_xlabel('Thuật Toán', fontsize=12)
       ax.grid(True, alpha=0.3)
       bars[efficiency.index(max(efficiency))].set_color('#8BC34A')
      
       self.canvas.fig.tight_layout()
       self.canvas.draw()
      
   def clear_plot(self):
       # ... (Logic xóa biểu đồ) ...
       self.canvas.fig.clear()
       ax = self.canvas.fig.add_subplot(111)
       ax.text(0.5, 0.5, 'Biểu đồ đã được xóa\nChạy thuật toán để tạo biểu đồ mới',
               ha='center', va='center', fontsize=14, transform=ax.transAxes, alpha=0.7)
       ax.set_xticks([])
       ax.set_yticks([])
       ax.set_title('Sẵn sàng tạo biểu đồ mới', fontsize=16, fontweight='bold')
       self.canvas.fig.tight_layout()
       self.canvas.draw()