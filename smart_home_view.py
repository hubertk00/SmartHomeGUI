import sys
import torch
from PyQt5.QtWidgets import (QApplication, QWidget, QGridLayout, QVBoxLayout, 
                             QLabel, QFrame, QPushButton, QProgressBar, QHBoxLayout, QSizePolicy,
                             QListWidget, QListWidgetItem, QGraphicsDropShadowEffect)
from PyQt5.QtCore import pyqtSlot, Qt, QTimer, QTime, QDate, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QColor, QLinearGradient, QPalette, QBrush

from smart_home_model import SmartHomeState
from audio_worker import AudioWorker

STYLESHEET = """
    QWidget {
        font-family: 'Segoe UI', 'San Francisco', sans-serif;
    }
    QProgressBar {
        border: none;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
        height: 8px;
    }
    QProgressBar::chunk {
        background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #4facfe, stop:1 #00f2fe);
        border-radius: 4px;
    }
"""

class ModernButton(QPushButton):
    def __init__(self, text, color_start="#4facfe", color_end="#00f2fe", is_danger=False):
        super().__init__(text)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(45)
        
        if is_danger:
            color_start, color_end = "#ff416c", "#ff4b2b"
        elif color_start == "gray":
            color_start, color_end = "#485563", "#29323c"
            
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 {color_start}, stop:1 {color_end});
                color: white;
                border-radius: 22px;
                font-weight: 600;
                font-size: 14px;
                padding: 0 20px;
                border: 1px solid rgba(255,255,255,0.1);
            }}
            QPushButton:hover {{
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 {color_end}, stop:1 {color_start});
                padding-top: 2px;
                border-radius: 22px;
            }}
            QPushButton:pressed {{
                background-color: {color_start};
                border-radius: 22px;
            }}
            QPushButton:disabled {{
                background-color: rgba(255, 255, 255, 0.1);
                color: rgba(255, 255, 255, 0.3);
                border: none;
                border-radius: 22px;
            }}
        """)
        
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)

class DeviceTile(QFrame):
    def __init__(self, name):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(160, 120)      
        self.setMaximumHeight(220)
        
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        
        self.lbl_icon = QLabel(name[0].upper())
        self.lbl_icon.setAlignment(Qt.AlignCenter)
        self.lbl_icon.setStyleSheet("font-size: 32px; color: rgba(255,255,255,0.8); margin-bottom: 5px;")
        
        self.lbl_name = QLabel(name)
        self.lbl_name.setAlignment(Qt.AlignCenter)
        self.lbl_name.setStyleSheet("color: white; font-weight: 600; font-size: 16px;")
        
        self.lbl_status = QLabel("OFF")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("color: rgba(255, 255, 255, 0.5); font-size: 12px; margin-top: 5px;")
        
        layout.addWidget(self.lbl_icon)
        layout.addWidget(self.lbl_name)
        layout.addWidget(self.lbl_status)
        self.setLayout(layout)
        
        self.default_style = """
            QFrame {
                background-color: rgba(255, 255, 255, 0.05);
                border-radius: 20px;
                border: 1px solid rgba(255, 255, 255, 0.05);
            }
            QFrame:hover {
                background-color: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
        """
        self.setStyleSheet(self.default_style)

        self.shadow = QGraphicsDropShadowEffect()
        self.shadow.setBlurRadius(20)
        self.shadow.setColor(QColor(0, 0, 0, 50))
        self.shadow.setOffset(0, 5)
        self.setGraphicsEffect(self.shadow)

    def update_style(self, is_on, brightness=100):
        if is_on:
            r, g, b = 255, 255, 255
            opacity = max(0.6, brightness/100.0)
            
            self.setStyleSheet(f"""
                QFrame {{
                    background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(34, 197, 94, {opacity}), stop:1 rgba(16, 185, 129, {opacity}));
                    border-radius: 20px;
                    border: 1px solid rgba(255, 255, 255, 0.3);
                }}
            """)
            status_text = f"ON {brightness}%" if brightness < 100 else "ON"
            self.lbl_status.setText(status_text)
            self.lbl_status.setStyleSheet("color: rgba(255, 255, 255, 0.9); font-weight: bold;")
            self.lbl_icon.setStyleSheet("font-size: 32px; color: white;")
        else:
            self.setStyleSheet(self.default_style)
            self.lbl_status.setText("OFF")
            self.lbl_status.setStyleSheet("color: rgba(255, 255, 255, 0.5);")
            self.lbl_icon.setStyleSheet("font-size: 32px; color: rgba(255,255,255,0.8);")

class InfoPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSizeConstraint(QVBoxLayout.SetMinAndMaxSize)
        
        self.lbl_time = QLabel("00:00")
        self.lbl_time.setAlignment(Qt.AlignCenter)
        self.lbl_time.setStyleSheet("font-size: 90px; font-weight: 200; color: white; margin-bottom: -10px;")
        
        self.lbl_date = QLabel("Loading date...")
        self.lbl_date.setAlignment(Qt.AlignCenter)
        self.lbl_date.setStyleSheet("font-size: 20px; color: rgba(255,255,255,0.7); font-weight: 400; letter-spacing: 1px; margin-bottom: 30px;")
        
        self.reading_frame = QFrame()
        self.reading_frame.setFixedSize(900, 350)
        self.reading_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(0, 0, 0, 0.4);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 25px;
            }
        """)
        read_shadow = QGraphicsDropShadowEffect()
        read_shadow.setBlurRadius(50)
        read_shadow.setColor(QColor(0,0,0,100))
        self.reading_frame.setGraphicsEffect(read_shadow)
        
        reading_layout = QVBoxLayout(self.reading_frame)
        reading_layout.setContentsMargins(40, 40, 40, 40)
        reading_layout.setSpacing(20)
        
        self.lbl_reading_instruction = QLabel("PRZECZYTAJ")
        self.lbl_reading_instruction.setAlignment(Qt.AlignCenter)
        self.lbl_reading_instruction.setStyleSheet("color: #4facfe; font-size: 14px; font-weight: 700; letter-spacing: 2px;")
        
        self.lbl_reading_text = QLabel()
        self.lbl_reading_text.setWordWrap(True)
        self.lbl_reading_text.setAlignment(Qt.AlignCenter)
        self.lbl_reading_text.setStyleSheet("""
            font-size: 28px; 
            line-height: 1.4; 
            color: rgba(255,255,255,0.95); 
            font-weight: 300;
        """)
        
        reading_layout.addWidget(self.lbl_reading_instruction)
        reading_layout.addWidget(self.lbl_reading_text)
        
        self.log_list = QListWidget()
        self.log_list.setFocusPolicy(Qt.NoFocus)
        self.log_list.setStyleSheet("""
            QListWidget {
                background-color: transparent;
                border: none;
                color: rgba(255, 255, 255, 0.6);
                font-size: 13px;
            }
            QListWidget::item {
                padding: 4px;
                background: transparent;
            }
        """)
        self.log_list.setMaximumHeight(150)
        self.log_list.setMaximumWidth(600)
        
        layout.addWidget(self.lbl_time)
        layout.addWidget(self.lbl_date)
        layout.addWidget(self.reading_frame)
        layout.addWidget(self.log_list)
        
        self.reading_frame.hide()
        self.setLayout(layout)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)
        self.update_time()

    def update_time(self):
        current_time = QTime.currentTime().toString("HH:mm")
        current_date = QDate.currentDate().toString("dddd, d MMMM") 
        self.lbl_time.setText(current_time)
        self.lbl_date.setText(current_date.upper())

    def add_log(self, text):
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M")
        item = QListWidgetItem(f"{timestamp} • {text}")
        item.setTextAlignment(Qt.AlignCenter)
        self.log_list.insertItem(0, item)
        if self.log_list.count() > 5:
            self.log_list.takeItem(5)

    def show_reading_mode(self, text_to_read):
        self.lbl_date.hide()
        self.log_list.hide()
        self.lbl_reading_text.setText(text_to_read)
        self.reading_frame.show()

    def hide_reading_mode(self):
        self.reading_frame.hide()
        self.lbl_date.show()
        self.log_list.show()
        
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Home Controller")
        self.resize(1280, 850)
        
        self.setAutoFillBackground(True)
        palette = self.palette()
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, QColor("#1e2024"))
        gradient.setColorAt(1.0, QColor("#0b0c10")) 
        palette.setBrush(QPalette.Window, QBrush(gradient))
        self.setPalette(palette)
        
        self.setStyleSheet(STYLESHEET)
        
        self.logic = SmartHomeState()
        
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(40, 40, 40, 40)
        self.main_layout.setSpacing(20)
        
        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(20)
        self.tiles = {} 
        self.light_brightness = 100

        excluded_commands = ["Wróciłem", "Wychodzę", "Jaśniej", "Ciemniej"]
        valid_devices = [name for name in self.logic.devices if name not in excluded_commands]
        
        columns = 4
        row, col = 0, 0
        for name in valid_devices:
            tile = DeviceTile(name)
            self.tiles[name] = tile
            self.grid_layout.addWidget(tile, row, col)
            col += 1
            if col >= columns:
                col = 0
                row += 1
            
        self.main_layout.addLayout(self.grid_layout)
        
        self.main_layout.addStretch(1) 
        
        self.info_panel = InfoPanel()
        self.main_layout.addWidget(self.info_panel, alignment=Qt.AlignCenter)
        
        self.main_layout.addStretch(1)
        
        self.lbl_info = QLabel("Skonfiguruj własne słowo wybudzające")
        self.lbl_info.setAlignment(Qt.AlignCenter)
        self.lbl_info.setStyleSheet("""
            color: rgba(255, 255, 255, 0.4); 
            font-size: 24px; 
            font-weight: 500; 
            letter-spacing: 1px;
            text-transform: uppercase;
        """)
        self.main_layout.addWidget(self.lbl_info)
        self.main_layout.addSpacing(15)

        self.controls_panel = QFrame()
        self.controls_panel.setStyleSheet("""
            QFrame {
                background-color: rgba(30, 30, 30, 0.8);
                border-radius: 30px;
                border: 0px solid rgba(255, 255, 255, 0.1);
            }
        """)
        self.controls_panel.setFixedHeight(80) 
        bar_shadow = QGraphicsDropShadowEffect()
        bar_shadow.setBlurRadius(30)
        bar_shadow.setColor(QColor(0,0,0,150))
        self.controls_panel.setGraphicsEffect(bar_shadow)
        
        controls_layout = QHBoxLayout(self.controls_panel)
        controls_layout.setContentsMargins(30, 10, 30, 10)
        controls_layout.setSpacing(20)

        self.lbl_audio_status = QLabel("Gotowy")
        self.lbl_audio_status.setStyleSheet("color: rgba(255,255,255,0.5); font-weight: 600; min-width: 150px;")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(100)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        
        self.btn_record = ModernButton("NAGRAJ PRÓBKĘ", is_danger=True)
        self.btn_record.clicked.connect(self.on_record_click)

        controls_layout.addWidget(self.lbl_audio_status)
        controls_layout.addWidget(self.progress_bar)
        controls_layout.addStretch(1) 
        controls_layout.addWidget(self.btn_record)

        self.review_widget = QWidget()
        self.review_layout = QHBoxLayout(self.review_widget)
        self.review_layout.setContentsMargins(0,0,0,0)
        
        self.btn_play = ModernButton("ODSŁUCHAJ", color_start="#4facfe", color_end="#00f2fe")
        self.btn_ok = ModernButton("ZATWIERDŹ", color_start="#43e97b", color_end="#38f9d7")
        self.btn_bad = ModernButton("ODRZUĆ", is_danger=True)

        self.review_layout.addWidget(self.btn_play)
        self.review_layout.addWidget(self.btn_ok)
        self.review_layout.addWidget(self.btn_bad)
        
        controls_layout.addWidget(self.review_widget)
        self.review_widget.hide()

        self.phase2_widget = QWidget()
        self.phase2_layout = QHBoxLayout(self.phase2_widget)
        self.phase2_layout.setContentsMargins(0,0,0,0)
        
        self.btn_improve = ModernButton("DOKŁADNOŚĆ +", color_start="#667eea", color_end="#764ba2")
        self.btn_finish_basic = ModernButton("POMIŃ", color_start="gray")
        
        self.phase2_layout.addWidget(self.btn_improve)
        self.phase2_layout.addWidget(self.btn_finish_basic)
        
        controls_layout.addWidget(self.phase2_widget)
        self.phase2_widget.hide()

        self.main_layout.addWidget(self.controls_panel)
        self.setLayout(self.main_layout)

        self.btn_play.clicked.connect(lambda: self.worker.play_recording())
        self.btn_ok.clicked.connect(lambda: self.worker.accept_recording())
        self.btn_bad.clicked.connect(lambda: self.worker.discard_recording())
        self.btn_improve.clicked.connect(self.start_accuracy_training)
        self.btn_finish_basic.clicked.connect(self.finish_basic_enrollment)

        self.init_audio_worker()

    def resizeEvent(self, event):
        palette = self.palette()
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, QColor("#141e30"))
        gradient.setColorAt(1.0, QColor("#243b55"))
        palette.setBrush(QPalette.Window, QBrush(gradient))
        self.setPalette(palette)
        super().resizeEvent(event)

    def init_audio_worker(self):
        class SystemArgs: 
             def __init__(self): 
                 self.rate=16000; self.n_mfcc=20; self.window_size=16000; self.chunk=2048; 
                 self.vad_threshold=0.005; self.listen_duration=4.0; self.cmd_threshold=0.8; 
                 self.wake_threshold=0.85; self.cooldown=1.0

        device = torch.device("cpu") 
        wake_path = r"models\matchboxnet_wakeword.pth"
        cmd_path = r"models\matchboxnet_commands.pth"

        self.worker = AudioWorker(
            wake_model_path=wake_path, 
            cmd_model_path=cmd_path, 
            args=SystemArgs(), 
            device=device
        )
        
        self.worker.cmd_signal.connect(self.process_voice_command)
        self.worker.status_signal.connect(self.lbl_audio_status.setText)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.wake_signal.connect(self.on_wake_detected)
        self.worker.review_request_signal.connect(self.show_review_panel)
        self.worker.enrollment_finished_signal.connect(self.reset_ui_state)
        self.worker.enrollment_phase1_finished_signal.connect(self.on_phase1_finished)

        self.worker.start()

    @pyqtSlot(str, float)
    def process_voice_command(self, command_name, probability):
        print(f"Cmd: {command_name}")
        self.info_panel.add_log(f"Rozpoznano: {command_name} ({int(probability*100)}%)")

        if command_name == "Wychodzę":
            self.logic.save()
            self.logic.set_all(False)
            self.refresh_all_tiles()
        elif command_name == "Wróciłem":
            if self.logic.restore(): self.refresh_all_tiles()
        elif command_name == "Jaśniej":
            if "Światło" in self.tiles:
                self.light_brightness = min(100, self.light_brightness + 20) 
                if self.logic.devices.get("Światło"):
                    self.tiles["Światło"].update_style(True, self.light_brightness)
        elif command_name == "Ciemniej":
            if "Światło" in self.tiles:
                self.light_brightness = max(20, self.light_brightness - 20) 
                if self.logic.devices.get("Światło"):
                    self.tiles["Światło"].update_style(True, self.light_brightness)
        elif command_name in self.logic.devices:
            new_state = self.logic.toggle(command_name)
            self.tiles[command_name].update_style(new_state)

    def refresh_all_tiles(self):
        for name, state in self.logic.devices.items():
            if name in self.tiles:
                if name == "Światło":
                    self.tiles[name].update_style(state, self.light_brightness)
                else:
                    self.tiles[name].update_style(state, 100)

    def on_record_click(self):
        self.btn_record.hide()
        self.worker.start_manual_recording()

    def show_review_panel(self):
        self.btn_record.hide()
        self.review_widget.show()
        self.lbl_audio_status.setText("Weryfikacja próbki...")

    def reset_ui_state(self):
        self.review_widget.hide()
        self.phase2_widget.hide()    
        self.info_panel.hide_reading_mode()        
        self.btn_record.show()
        self.btn_record.setEnabled(True)
        self.lbl_audio_status.setText("Gotowy")

    def on_wake_detected(self, prob):
        self.controls_panel.setStyleSheet("""
            QFrame {
                background-color: rgba(30, 30, 30, 0.9);
                border-radius: 30px;
                border: 2px solid #4facfe;
            }
        """)
        QTimer.singleShot(400, lambda: self.controls_panel.setStyleSheet("""
            QFrame {
                background-color: rgba(30, 30, 30, 0.8);
                border-radius: 30px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
        """))

    def on_phase1_finished(self):
        self.review_widget.hide()
        self.phase2_widget.show()
        self.lbl_audio_status.setText("Etap 1 zakończony")

    def start_accuracy_training(self):
        self.phase2_widget.hide()
        text = "Domen Prevc został zwycięzcą sylwestrowych kwalifikacji do noworocznego konkursu 74. " \
               "Turnieju Czterech Skoczni na Grosse Olympiaschanze (HS142) w Garmisch-Partenkirchen."
        self.info_panel.show_reading_mode(text)        
        self.worker.record_background_noise() 

    def finish_basic_enrollment(self):
        self.worker.save_enrollment_without_noise()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())