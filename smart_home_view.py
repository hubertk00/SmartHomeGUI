import sys
import torch
from PyQt5.QtWidgets import (QApplication, QWidget, QGridLayout, QVBoxLayout, 
                             QLabel, QFrame, QPushButton, QProgressBar, QHBoxLayout, QSizePolicy,
                             QListWidget, QListWidgetItem)
from PyQt5.QtCore import pyqtSlot, Qt, QTimer, QTime, QDate
from PyQt5.QtGui import QFont

from smart_home_model import SmartHomeState
from audio_worker import AudioWorker

class SystemArgs:
    def __init__(self):
        self.rate = 16000
        self.n_mfcc = 20
        self.window_size = 16000
        self.chunk = 2048
        self.vad_threshold = 0.005
        self.listen_duration = 4.0
        self.cmd_threshold = 0.8
        self.wake_threshold = 0.85
        self.cooldown = 1.0

class DeviceTile(QFrame):
    def __init__(self, name):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(150, 100)      
        self.setMaximumHeight(300)  
        self.setStyleSheet("background-color: #333; border-radius: 8px;")
        layout = QVBoxLayout()
        self.lbl_name = QLabel(name)
        self.lbl_name.setAlignment(Qt.AlignCenter)
        self.lbl_name.setStyleSheet("color: white; font-weight: bold;")
        
        self.lbl_status = QLabel("OFF")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("color: #777;")
        
        layout.addWidget(self.lbl_name)
        layout.addWidget(self.lbl_status)
        self.setLayout(layout)

    def update_style(self, is_on, brightness=100):
        if is_on:
            factor = max(0.2, brightness/100.0)
            r = int(46 * factor)
            g = int(125 * factor)
            b = int(50 * factor)
            self.setStyleSheet(f"background-color: rgb({r}, {g}, {b}); border-radius: 8px;")
            if brightness < 100:
                self.lbl_status.setText(f"ON ({brightness}%)")
            else:
                self.lbl_status.setText("ON")
            self.lbl_status.setStyleSheet("color: white;")
        else:
            self.setStyleSheet("background-color: #333; border-radius: 8px;") 
            self.lbl_status.setText("OFF")
            self.lbl_status.setStyleSheet("color: #777;")

class InfoPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: transparent;") 
        
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        
        self.lbl_time = QLabel("00:00")
        self.lbl_time.setAlignment(Qt.AlignCenter)
        self.lbl_time.setStyleSheet("font-size: 60px; font-weight: bold; color: #eeeeee; font-family: 'Segoe UI', sans-serif;")
        
        self.lbl_date = QLabel("Poniedziałek, 1 Stycznia")
        self.lbl_date.setAlignment(Qt.AlignCenter)
        self.lbl_date.setStyleSheet("font-size: 18px; color: #aaaaaa; margin-bottom: 20px;")
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)
        self.update_time() 
        
        self.log_list = QListWidget()
        self.log_list.setFocusPolicy(Qt.NoFocus)
        self.log_list.setStyleSheet("""
            QListWidget {
                background-color: rgba(0, 0, 0, 0.3); 
                border-radius: 10px; 
                border: 1px solid #333;
                color: #03a9f4;
                font-size: 14px;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #222;
            }
        """)
        self.log_list.setMaximumHeight(200)
        self.log_list.setMaximumWidth(600)  
        
        layout.addWidget(self.lbl_time)
        layout.addWidget(self.lbl_date)
        layout.addWidget(self.log_list)
        
        self.setLayout(layout)

    def update_time(self):
        current_time = QTime.currentTime().toString("HH:mm")
        current_date = QDate.currentDate().toString("dddd, d MMMM yyyy") 
        self.lbl_time.setText(current_time)
        self.lbl_date.setText(current_date)

    def add_log(self, text):
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        item = QListWidgetItem(f"[{timestamp}] {text}")
        item.setTextAlignment(Qt.AlignCenter)
        self.log_list.insertItem(0, item)
        
        if self.log_list.count() > 5:
            self.log_list.takeItem(5)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Home RPi Controller")
        self.setStyleSheet("background-color: #121212; color: white;")
        self.resize(1200, 800)        
        self.logic = SmartHomeState()
        
        self.main_layout = QVBoxLayout()
        
        self.grid_layout = QGridLayout()
        self.tiles = {} 
        
        self.light_brightness = 100

        columns = 4
        row, col = 0, 0

        excluded_commands = ["Wróciłem", "Wychodzę", "Jaśniej", "Ciemniej"]
        valid_devices = [name for name in self.logic.devices if name not in excluded_commands]
        total = len(valid_devices)

        for i, name in enumerate(valid_devices):
            tile = DeviceTile(name)
            self.tiles[name] = tile
            self.grid_layout.addWidget(tile, row, col)
            col += 1
            if col >= columns:
                col = 0
                row += 1
            
        self.main_layout.addLayout(self.grid_layout)
        self.main_layout.addSpacing(20)

        self.main_layout.addStretch(1) 
        
        self.info_panel = InfoPanel()
        self.main_layout.addWidget(self.info_panel, alignment=Qt.AlignCenter)
        
        self.main_layout.addStretch(1)
        controls_panel = QFrame()
        controls_panel.setStyleSheet("background-color: #1e1e1e; border-radius: 10px; padding: 10px;")
        controls_layout = QVBoxLayout(controls_panel)

        self.lbl_audio_status = QLabel("Inicjalizacja...")
        self.lbl_audio_status.setAlignment(Qt.AlignCenter)
        self.lbl_audio_status.setFont(QFont("Arial", 12))
        self.lbl_audio_status.setStyleSheet("color: #03a9f4;")
        controls_layout.addWidget(self.lbl_audio_status)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("QProgressBar {border: 0px; background-color: #333; height: 5px;} QProgressBar::chunk {background-color: #03a9f4;}")
        controls_layout.addWidget(self.progress_bar)

        self.btn_record = QPushButton("Nagraj próbkę")
        self.btn_record.setMinimumHeight(40)
        self.btn_record.setStyleSheet("background-color: #d84315; color: white; font-weight: bold; border-radius: 5px;")
        self.btn_record.clicked.connect(self.on_record_click)
        controls_layout.addWidget(self.btn_record)

        self.review_panel = QFrame()
        self.review_layout = QHBoxLayout(self.review_panel)
        self.review_layout.setContentsMargins(0,0,0,0)

        self.btn_play = QPushButton("Odsłuchaj")
        self.btn_play.setStyleSheet("background-color: #1976d2; color: white; font-weight: bold; border-radius: 5px;")
        
        self.btn_ok = QPushButton("Dobrze")
        self.btn_ok.setStyleSheet("background-color: #388e3c; color: white; font-weight: bold; border-radius: 5px;")
        
        self.btn_bad = QPushButton("Źle")
        self.btn_bad.setStyleSheet("background-color: #d32f2f; color: white; font-weight: bold; border-radius: 5px;")

        self.review_layout.addWidget(self.btn_play)
        self.review_layout.addWidget(self.btn_ok)
        self.review_layout.addWidget(self.btn_bad)
        
        controls_layout.addWidget(self.review_panel)
        self.review_panel.hide()

        self.main_layout.addWidget(controls_panel)
        self.setLayout(self.main_layout)

        self.init_audio_worker()
        
        self.btn_play.clicked.connect(lambda: self.worker.play_recording())
        self.btn_ok.clicked.connect(lambda: self.worker.accept_recording())
        self.btn_bad.clicked.connect(lambda: self.worker.discard_recording())

    def init_audio_worker(self):
        args = SystemArgs()
        device = torch.device("cpu") 

        wake_path = r"models\matchboxnet_wakeword.pth"
        cmd_path = r"models\matchboxnet_commands.pth"

        self.worker = AudioWorker(
            wake_model_path=wake_path, 
            cmd_model_path=cmd_path, 
            args=args, 
            device=device,
            wake_arch='matchboxnet',
            cmd_arch='matchboxnet'
        )
        
        self.worker.cmd_signal.connect(self.process_voice_command)
        self.worker.status_signal.connect(self.lbl_audio_status.setText)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.wake_signal.connect(self.on_wake_detected)
        self.worker.review_request_signal.connect(self.show_review_panel)
        self.worker.enrollment_finished_signal.connect(self.reset_ui_state)

        self.worker.start()

    @pyqtSlot(str, float)
    def process_voice_command(self, command_name, probability):
        print(f"Komenda: {command_name} ({probability:.2f})")
        self.info_panel.add_log(f"Rozpoznano: {command_name} ({int(probability*100)}%)")

        if command_name == "Wychodzę":
            self.logic.save()
            self.logic.set_all(False)
            self.refresh_all_tiles()

        elif command_name == "Wróciłem":
            if self.logic.restore():
                self.refresh_all_tiles()
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
        self.btn_record.setEnabled(False)
        self.btn_record.setText("Nagrywanie...")
        self.btn_record.setStyleSheet("background-color: #e64a19; color: white; font-weight: bold; border-radius: 5px;")
        self.worker.start_manual_recording()

    def show_review_panel(self):
        self.btn_record.hide()
        self.review_panel.show()
        self.lbl_audio_status.setText("Czy nagranie jest wyraźne?")

    def reset_ui_state(self):
        self.review_panel.hide()
        self.btn_record.show()
        self.btn_record.setEnabled(True)
        self.btn_record.setText("Nagraj próbkę")
        self.btn_record.setStyleSheet("background-color: #d84315; color: white; font-weight: bold; border-radius: 5px;")

    def on_wake_detected(self, prob):
        self.setStyleSheet("background-color: #0d47a1;")
        QTimer.singleShot(300, lambda: self.setStyleSheet("background-color: #121212; color: white;"))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())