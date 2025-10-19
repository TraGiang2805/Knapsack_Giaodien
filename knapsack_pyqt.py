import sys
import numpy as np
import time
from simpleai.search import SearchProblem, simulated_annealing
import random
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

# -----------------------------
# Knapsack Problem Class
# -----------------------------
class KnapsackProblem(SearchProblem):
    def __init__(self, weights, values, capacity):
        self.weights = weights
        self.values = values
        self.capacity = capacity
        initial_state = tuple([0 for _ in range(len(weights))])
        super().__init__(initial_state)

    def actions(self, state):
        return list(range(len(state)))

    def result(self, state, action):
        new_state = list(state)
        new_state[action] = 1 - new_state[action]
        return tuple(new_state)

    def value(self, state):
        total_weight = sum(w * s for w, s in zip(self.weights, state))
        total_value = sum(v * s for v, s in zip(self.values, state))
        if total_weight > self.capacity:
            total_value -= (total_weight - self.capacity) * 10
        return total_value

# -----------------------------
# Algorithm Functions
# -----------------------------
def run_SA(weights, values, capacity):
    problem = KnapsackProblem(weights, values, capacity)
    start_time = time.time()
    result = simulated_annealing(problem, iterations_limit=5000)
    elapsed = time.time() - start_time

    best_state = result.state
    total_weight = sum(w * s for w, s in zip(weights, best_state))
    total_value = sum(v * s for v, s in zip(values, best_state))
    return best_state, total_value, total_weight, elapsed, 5000

def run_BCO(weights, values, capacity, num_bees=30, num_iterations=200):
    start_time = time.time()
    n = len(weights)
    population = np.random.randint(0, 2, (num_bees, n))

    def fitness(state):
        w = np.sum(np.array(weights) * state)
        v = np.sum(np.array(values) * state)
        if w > capacity:
            v -= (w - capacity) * 10
        return v

    best_solution = None
    best_fitness = float("-inf")

    for _ in range(num_iterations):
        fitness_values = np.array([fitness(s) for s in population])
        probs = (fitness_values - fitness_values.min() + 1e-6)
        probs = probs / probs.sum()

        new_population = []
        for _ in range(num_bees):
            j = np.random.choice(range(num_bees), p=probs)
            candidate = population[j].copy()
            flip = np.random.randint(0, n)
            candidate[flip] = 1 - candidate[flip]
            new_population.append(candidate)
        population = np.array(new_population)

        for s in population:
            f = fitness(s)
            if f > best_fitness:
                best_fitness = f
                best_solution = s.copy()

    elapsed = time.time() - start_time
    total_weight = np.sum(np.array(weights) * best_solution)
    return best_solution, best_fitness, total_weight, elapsed, num_bees * num_iterations

def run_GA(weights, values, capacity, pop_size=30, generations=100, mutation_rate=0.1):
    start_time = time.time()
    n = len(weights)
    population = np.random.randint(0, 2, (pop_size, n))

    def fitness(state):
        w = np.sum(np.array(weights) * state)
        v = np.sum(np.array(values) * state)
        if w > capacity:
            v -= (w - capacity) * 10
        return v

    best_solution = None
    best_fitness = float("-inf")

    for _ in range(generations):
        fitness_values = np.array([fitness(s) for s in population])
        parents_idx = np.argsort(fitness_values)[-pop_size // 2:]
        parents = population[parents_idx]

        children = []
        for _ in range(pop_size // 2):
            p1, p2 = parents[np.random.randint(0, len(parents), 2)]
            point = np.random.randint(1, n - 1)
            child = np.concatenate([p1[:point], p2[point:]])
            if np.random.rand() < mutation_rate:
                flip = np.random.randint(0, n)
                child[flip] = 1 - child[flip]
            children.append(child)

        population = np.vstack((parents, children))

        for s in population:
            f = fitness(s)
            if f > best_fitness:
                best_fitness = f
                best_solution = s.copy()

    elapsed = time.time() - start_time
    total_weight = np.sum(np.array(weights) * best_solution)
    return best_solution, best_fitness, total_weight, elapsed, generations * pop_size

def random_dataset():
    n = random.randint(5, 12)
    weights = [random.randint(5, 40) for _ in range(n)]
    values = [random.randint(10, 100) for _ in range(n)]
    capacity = random.randint(sum(weights)//3, sum(weights)//2)
    return weights, values, capacity

# -----------------------------
# Worker Thread for Algorithm Execution
# -----------------------------
class AlgorithmWorker(QThread):
    finished = pyqtSignal(str, object)  # algorithm_name, result
    error = pyqtSignal(str)  # error message
    
    def __init__(self, algorithm, weights, values, capacity):
        super().__init__()
        self.algorithm = algorithm
        self.weights = weights
        self.values = values
        self.capacity = capacity
        
    def run(self):
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
            self.error.emit(str(e))

# -----------------------------
# Matplotlib Canvas Widget
# -----------------------------
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

# -----------------------------
# Main Application Window
# -----------------------------
class KnapsackGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.weights = []
        self.values = []
        self.capacity = 0
        self.worker = None
        
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("🎒 Bài Toán Knapsack - PyQt5 GUI")
        self.setGeometry(100, 100, 1400, 900)
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QTabWidget::pane {
                border: 1px solid #c0c0c0;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid #2196F3;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create tabs
        self.create_data_input_tab()
        self.create_algorithm_tab()
        self.create_results_tab()
        
        # Add tabs to tab widget
        self.tab_widget.addTab(self.data_tab, "📊 Nhập Dữ Liệu")
        self.tab_widget.addTab(self.algorithm_tab, "⚙️ Chạy Thuật Toán")
        self.tab_widget.addTab(self.results_tab, "📈 Kết Quả & Biểu Đồ")
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.tab_widget)
        central_widget.setLayout(main_layout)
        
    def create_data_input_tab(self):
        self.data_tab = QWidget()
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("🎒 BÀI TOÁN KNAPSACK - NHẬP DỮ LIỆU")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #1976D2; margin: 10px;")
        layout.addWidget(title_label)
        
        # Manual input section
        manual_group = QGroupBox("Nhập Dữ Liệu Thủ Công")
        manual_layout = QGridLayout()
        
        # Number of items
        manual_layout.addWidget(QLabel("Số lượng vật phẩm:"), 0, 0)
        self.num_items_input = QLineEdit()
        self.num_items_input.setText("5")
        self.num_items_input.setMaximumWidth(100)
        manual_layout.addWidget(self.num_items_input, 0, 1)
        
        self.create_form_btn = QPushButton("Tạo Form Nhập")
        self.create_form_btn.clicked.connect(self.create_input_form)
        manual_layout.addWidget(self.create_form_btn, 0, 2)
        
        # Items table
        self.items_table = QTableWidget()
        self.items_table.setColumnCount(3)
        self.items_table.setHorizontalHeaderLabels(["Vật phẩm", "Trọng lượng", "Giá trị"])
        header = self.items_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        manual_layout.addWidget(self.items_table, 1, 0, 1, 3)
        
        # Capacity input
        manual_layout.addWidget(QLabel("Sức chứa ba lô:"), 2, 0)
        self.capacity_input = QLineEdit()
        self.capacity_input.setMaximumWidth(150)
        manual_layout.addWidget(self.capacity_input, 2, 1)
        
        self.save_data_btn = QPushButton("💾 Lưu Dữ Liệu")
        self.save_data_btn.clicked.connect(self.save_manual_data)
        manual_layout.addWidget(self.save_data_btn, 2, 2)
        
        manual_group.setLayout(manual_layout)
        layout.addWidget(manual_group)
        
        # Random data section
        random_group = QGroupBox("Tạo Dữ Liệu Ngẫu Nhiên")
        random_layout = QVBoxLayout()
        
        self.random_data_btn = QPushButton("🎲 Tạo Dữ Liệu Ngẫu Nhiên")
        self.random_data_btn.clicked.connect(self.generate_random_data)
        self.random_data_btn.setStyleSheet("background-color: #4CAF50; font-size: 14px; padding: 10px;")
        random_layout.addWidget(self.random_data_btn)
        
        random_group.setLayout(random_layout)
        layout.addWidget(random_group)
        
        # Current data display
        display_group = QGroupBox("Dữ Liệu Hiện Tại")
        display_layout = QVBoxLayout()
        
        self.data_display = QTextEdit()
        self.data_display.setReadOnly(True)
        self.data_display.setMaximumHeight(200)
        display_layout.addWidget(self.data_display)
        
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        self.data_tab.setLayout(layout)
        
    def create_input_form(self):
        try:
            num_items = int(self.num_items_input.text())
            if num_items <= 0:
                QMessageBox.warning(self, "Lỗi", "Số lượng vật phẩm phải lớn hơn 0!")
                return
        except ValueError:
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập số lượng vật phẩm hợp lệ!")
            return
            
        # Set up table
        self.items_table.setRowCount(num_items)
        
        for i in range(num_items):
            # Item name
            item_name = QTableWidgetItem(f"Vật {i+1}")
            item_name.setFlags(Qt.ItemIsEnabled)
            self.items_table.setItem(i, 0, item_name)
            
            # Weight input
            weight_item = QTableWidgetItem("")
            self.items_table.setItem(i, 1, weight_item)
            
            # Value input
            value_item = QTableWidgetItem("")
            self.items_table.setItem(i, 2, value_item)
            
    def save_manual_data(self):
        try:
            weights = []
            values = []
            
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
        self.weights, self.values, self.capacity = random_dataset()
        self.update_data_display()
        QMessageBox.information(self, "Thành công", "Dữ liệu ngẫu nhiên đã được tạo!")
        
    def update_data_display(self):
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
            
    def create_algorithm_tab(self):
        self.algorithm_tab = QWidget()
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("⚙️ CHẠY THUẬT TOÁN KNAPSACK")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #1976D2; margin: 10px;")
        layout.addWidget(title_label)
        
        # Algorithm selection
        selection_group = QGroupBox("Chọn Thuật Toán")
        selection_layout = QHBoxLayout()
        
        self.sa_btn = QPushButton("🌡️ Simulated Annealing")
        self.sa_btn.clicked.connect(lambda: self.run_algorithm("SA"))
        self.sa_btn.setStyleSheet("background-color: #FF9800; font-size: 12px; padding: 8px;")
        
        self.bco_btn = QPushButton("🐝 Bee Colony Optimization")
        self.bco_btn.clicked.connect(lambda: self.run_algorithm("BCO"))
        self.bco_btn.setStyleSheet("background-color: #9C27B0; font-size: 12px; padding: 8px;")
        
        self.ga_btn = QPushButton("🧬 Genetic Algorithm")
        self.ga_btn.clicked.connect(lambda: self.run_algorithm("GA"))
        self.ga_btn.setStyleSheet("background-color: #4CAF50; font-size: 12px; padding: 8px;")
        
        self.all_btn = QPushButton("🚀 So Sánh Tất Cả")
        self.all_btn.clicked.connect(lambda: self.run_algorithm("ALL"))
        self.all_btn.setStyleSheet("background-color: #F44336; font-size: 12px; padding: 8px;")
        
        selection_layout.addWidget(self.sa_btn)
        selection_layout.addWidget(self.bco_btn)
        selection_layout.addWidget(self.ga_btn)
        selection_layout.addWidget(self.all_btn)
        
        selection_group.setLayout(selection_layout)
        layout.addWidget(selection_group)
        
        # Progress section
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
        
        # Results display
        results_group = QGroupBox("Kết Quả")
        results_layout = QVBoxLayout()
        
        self.results_display = QTextEdit()
        self.results_display.setReadOnly(True)
        results_layout.addWidget(self.results_display)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        self.algorithm_tab.setLayout(layout)
        
    def run_algorithm(self, algorithm):
        if not self.weights:
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập dữ liệu trước khi chạy thuật toán!")
            return
            
        # Disable buttons during execution
        self.sa_btn.setEnabled(False)
        self.bco_btn.setEnabled(False)
        self.ga_btn.setEnabled(False)
        self.all_btn.setEnabled(False)
        
        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.progress_label.setText(f"Đang chạy thuật toán {algorithm}...")
        
        # Start worker thread
        self.worker = AlgorithmWorker(algorithm, self.weights, self.values, self.capacity)
        self.worker.finished.connect(self.on_algorithm_finished)
        self.worker.error.connect(self.on_algorithm_error)
        self.worker.start()
        
    def on_algorithm_finished(self, algorithm_name, result):
        # Re-enable buttons
        self.sa_btn.setEnabled(True)
        self.bco_btn.setEnabled(True)
        self.ga_btn.setEnabled(True)
        self.all_btn.setEnabled(True)
        
        # Hide progress
        self.progress_bar.setVisible(False)
        self.progress_label.setText("Hoàn thành!")
        
        # Display results
        if algorithm_name == "ALL":
            self.display_comparison_results(result)
        else:
            self.display_single_result(algorithm_name, result)
            
    def on_algorithm_error(self, error_msg):
        # Re-enable buttons
        self.sa_btn.setEnabled(True)
        self.bco_btn.setEnabled(True)
        self.ga_btn.setEnabled(True)
        self.all_btn.setEnabled(True)
        
        # Hide progress
        self.progress_bar.setVisible(False)
        self.progress_label.setText("Có lỗi xảy ra!")
        
        QMessageBox.critical(self, "Lỗi", f"Có lỗi xảy ra: {error_msg}")
        
    def display_single_result(self, algorithm_name, result):
        state, value, weight, time_taken, complexity = result
        
        self.results_display.clear()
        self.results_display.append(f"🎯 KẾT QUẢ {algorithm_name.upper()}")
        self.results_display.append("=" * 50)
        self.results_display.append("")
        
        self.results_display.append(f"Giá trị tối ưu: {value:.2f}")
        self.results_display.append(f"Trọng lượng: {weight:.2f}")
        self.results_display.append(f"Thời gian thực thi: {time_taken:.4f} giây")
        self.results_display.append(f"Độ phức tạp: {complexity} lần lặp")
        self.results_display.append("")
        
        self.results_display.append("Trạng thái giải pháp:")
        self.results_display.append(f"{state}")
        self.results_display.append("")
        
        # Show selected items
        self.results_display.append("Các vật phẩm được chọn:")
        self.results_display.append("-" * 30)
        selected_items = []
        for i, selected in enumerate(state):
            if selected:
                selected_items.append(f"Vật {i+1}: Trọng lượng={self.weights[i]:.2f}, Giá trị={self.values[i]:.2f}")
        
        if selected_items:
            for item in selected_items:
                self.results_display.append(f"• {item}")
        else:
            self.results_display.append("Không có vật phẩm nào được chọn.")
            
    def display_comparison_results(self, results):
        sa_result, bco_result, ga_result = results
        
        # Lưu kết quả để vẽ biểu đồ
        self.last_results = results
        
        self.results_display.clear()
        self.results_display.append("📊 BẢNG SO SÁNH KẾT QUẢ")
        self.results_display.append("=" * 80)
        self.results_display.append("")
        
        # Header
        self.results_display.append(f"{'Thuật toán':<25}{'Giá trị':<12}{'Trọng lượng':<15}{'Thời gian (s)':<15}{'Độ phức tạp':<15}")
        self.results_display.append("-" * 80)
        
        # Results
        algorithms = [
            ("Simulated Annealing", sa_result),
            ("Bee Colony Optimization", bco_result),
            ("Genetic Algorithm", ga_result)
        ]
        
        for name, result in algorithms:
            state, value, weight, time_taken, complexity = result
            self.results_display.append(f"{name:<25}{value:<12.2f}{weight:<15.2f}{time_taken:<15.4f}{complexity:<15}")
            
        self.results_display.append("=" * 80)
        self.results_display.append("")
        
        # Find best result
        best_value = max(sa_result[1], bco_result[1], ga_result[1])
        best_algo = ""
        if sa_result[1] == best_value:
            best_algo = "Simulated Annealing"
        elif bco_result[1] == best_value:
            best_algo = "Bee Colony Optimization"
        else:
            best_algo = "Genetic Algorithm"
            
        self.results_display.append(f"🏆 Thuật toán tốt nhất: {best_algo} với giá trị {best_value:.2f}")
        self.results_display.append("")
        self.results_display.append("💡 Mẹo: Chuyển sang tab 'Kết Quả & Biểu Đồ' để xem biểu đồ so sánh!")
        
    def create_results_tab(self):
        self.results_tab = QWidget()
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("📈 KẾT QUẢ VÀ BIỂU ĐỒ")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #1976D2; margin: 10px;")
        layout.addWidget(title_label)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.plot_values_btn = QPushButton("📊 Biểu Đồ Giá Trị")
        self.plot_values_btn.clicked.connect(self.plot_values_comparison)
        self.plot_values_btn.setStyleSheet("background-color: #4CAF50; font-size: 12px; padding: 8px;")
        
        self.plot_time_btn = QPushButton("⏱️ Biểu Đồ Thời Gian")
        self.plot_time_btn.clicked.connect(self.plot_time_comparison)
        self.plot_time_btn.setStyleSheet("background-color: #FF9800; font-size: 12px; padding: 8px;")
        
        self.plot_efficiency_btn = QPushButton("⚡ Biểu Đồ Hiệu Suất")
        self.plot_efficiency_btn.clicked.connect(self.plot_efficiency_comparison)
        self.plot_efficiency_btn.setStyleSheet("background-color: #9C27B0; font-size: 12px; padding: 8px;")
        
        self.clear_plot_btn = QPushButton("🗑️ Xóa Biểu Đồ")
        self.clear_plot_btn.clicked.connect(self.clear_plot)
        self.clear_plot_btn.setStyleSheet("background-color: #F44336; font-size: 12px; padding: 8px;")
        
        control_layout.addWidget(self.plot_values_btn)
        control_layout.addWidget(self.plot_time_btn)
        control_layout.addWidget(self.plot_efficiency_btn)
        control_layout.addWidget(self.clear_plot_btn)
        
        layout.addLayout(control_layout)
        
        # Create matplotlib canvas
        self.canvas = MplCanvas(self, width=12, height=8, dpi=100)
        layout.addWidget(self.canvas)
        
        # Instructions
        instructions = QLabel("Chạy thuật toán để xem biểu đồ so sánh kết quả")
        instructions.setAlignment(Qt.AlignCenter)
        instructions.setStyleSheet("font-size: 14px; color: #666; margin: 20px;")
        layout.addWidget(instructions)
        
        # Store results for plotting
        self.last_results = None
        
        self.results_tab.setLayout(layout)
    
    def plot_values_comparison(self):
        """Vẽ biểu đồ so sánh giá trị tối ưu của các thuật toán"""
        if not self.last_results:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chạy thuật toán trước khi vẽ biểu đồ!")
            return
            
        sa_result, bco_result, ga_result = self.last_results
        
        algorithms = ['Simulated\nAnnealing', 'Bee Colony\nOptimization', 'Genetic\nAlgorithm']
        values = [sa_result[1], bco_result[1], ga_result[1]]
        
        # Tạo biểu đồ
        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)
        
        bars = ax.bar(algorithms, values, color=['#FF9800', '#9C27B0', '#4CAF50'], alpha=0.8)
        
        # Thêm giá trị trên mỗi cột
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('So Sánh Giá Trị Tối Ưu Của Các Thuật Toán', fontsize=16, fontweight='bold')
        ax.set_ylabel('Giá Trị Tối Ưu', fontsize=12)
        ax.set_xlabel('Thuật Toán', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Tìm thuật toán tốt nhất
        best_idx = values.index(max(values))
        bars[best_idx].set_color('#FFD700')  # Màu vàng cho thuật toán tốt nhất
        
        self.canvas.fig.tight_layout()
        self.canvas.draw()
    
    def plot_time_comparison(self):
        """Vẽ biểu đồ so sánh thời gian thực thi"""
        if not self.last_results:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chạy thuật toán trước khi vẽ biểu đồ!")
            return
            
        sa_result, bco_result, ga_result = self.last_results
        
        algorithms = ['Simulated\nAnnealing', 'Bee Colony\nOptimization', 'Genetic\nAlgorithm']
        times = [sa_result[3], bco_result[3], ga_result[3]]
        
        # Tạo biểu đồ
        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)
        
        bars = ax.bar(algorithms, times, color=['#FF9800', '#9C27B0', '#4CAF50'], alpha=0.8)
        
        # Thêm giá trị trên mỗi cột
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{time_val:.4f}s', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('So Sánh Thời Gian Thực Thi Của Các Thuật Toán', fontsize=16, fontweight='bold')
        ax.set_ylabel('Thời Gian (giây)', fontsize=12)
        ax.set_xlabel('Thuật Toán', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Tìm thuật toán nhanh nhất
        fastest_idx = times.index(min(times))
        bars[fastest_idx].set_color('#00BCD4')  # Màu xanh cho thuật toán nhanh nhất
        
        self.canvas.fig.tight_layout()
        self.canvas.draw()
    
    def plot_efficiency_comparison(self):
        """Vẽ biểu đồ so sánh hiệu suất (giá trị/thời gian)"""
        if not self.last_results:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chạy thuật toán trước khi vẽ biểu đồ!")
            return
            
        sa_result, bco_result, ga_result = self.last_results
        
        algorithms = ['Simulated\nAnnealing', 'Bee Colony\nOptimization', 'Genetic\nAlgorithm']
        
        # Tính hiệu suất (giá trị/thời gian)
        efficiency = []
        for result in [sa_result, bco_result, ga_result]:
            if result[3] > 0:  # Tránh chia cho 0
                efficiency.append(result[1] / result[3])
            else:
                efficiency.append(0)
        
        # Tạo biểu đồ
        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)
        
        bars = ax.bar(algorithms, efficiency, color=['#FF9800', '#9C27B0', '#4CAF50'], alpha=0.8)
        
        # Thêm giá trị trên mỗi cột
        for bar, eff in zip(bars, efficiency):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{eff:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('So Sánh Hiệu Suất Của Các Thuật Toán\n(Giá Trị/Thời Gian)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Hiệu Suất (Giá Trị/Thời Gian)', fontsize=12)
        ax.set_xlabel('Thuật Toán', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Tìm thuật toán hiệu suất cao nhất
        best_idx = efficiency.index(max(efficiency))
        bars[best_idx].set_color('#8BC34A')  # Màu xanh lá cho thuật toán hiệu suất cao nhất
        
        self.canvas.fig.tight_layout()
        self.canvas.draw()
    
    def clear_plot(self):
        """Xóa biểu đồ hiện tại"""
        self.canvas.fig.clear()
        self.canvas.draw()
        
        # Hiển thị thông báo
        ax = self.canvas.fig.add_subplot(111)
        ax.text(0.5, 0.5, 'Biểu đồ đã được xóa\nChạy thuật toán để tạo biểu đồ mới', 
                ha='center', va='center', fontsize=14, 
                transform=ax.transAxes, alpha=0.7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Sẵn sàng tạo biểu đồ mới', fontsize=16, fontweight='bold')
        
        self.canvas.fig.tight_layout()
        self.canvas.draw()

def main():
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Knapsack Problem Solver")
    app.setApplicationVersion("1.0")
    
    # Create and show main window
    window = KnapsackGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
