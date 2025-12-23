import sys
import os
import numpy
import torch

from PyQt5.QtWidgets import (QApplication, QWidget, QGridLayout, QVBoxLayout, 
                             QLabel, QFrame, QPushButton, QProgressBar, QHBoxLayout)
from PyQt5.QtCore import pyqtSlot, Qt, QTimer
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
        self.setFixedSize(150, 100)
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

    def update_style(self, is_on):
        if is_on:
            self.setStyleSheet("background-color: #2e7d32; border-radius: 8px;") 
            self.lbl_status.setText("ON")
            self.lbl_status.setStyleSheet("color: white;")
        else:
            self.setStyleSheet("background-color: #333; border-radius: 8px;") 
            self.lbl_status.setText("OFF")
            self.lbl_status.setStyleSheet("color: #777;")

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Home RPi Controller")
        self.resize(800, 500)
        self.setStyleSheet("background-color: #121212; color: white;")

        self.logic = SmartHomeState()
        
        self.main_layout = QVBoxLayout()
        
        self.grid_layout = QGridLayout()
        self.tiles = {} 

        row, col = 0, 0
        for name in self.logic.devices:
            tile = DeviceTile(name)
            self.grid_layout.addWidget(tile, row, col)
            self.tiles[name] = tile
            
            col += 1
            if col > 2:
                col = 0
                row += 1
        
        self.main_layout.addLayout(self.grid_layout)
        self.main_layout.addSpacing(20)

        controls_panel = QFrame()
        controls_panel.setStyleSheet("background-color: #1e1e1e; border-radius: 10px; padding: 10px;")
        controls_layout = QVBoxLayout(controls_panel)

        self.lbl_audio_status = QLabel("Inicjalizacja systemu...")
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

        self.btn_enroll = QPushButton("Nagraj nowe słowo wybudzające")
        self.btn_enroll.setMinimumHeight(40)
        self.btn_enroll.setStyleSheet("""
            QPushButton {background-color: #d84315; color: white; font-weight: bold; border-radius: 5px;}
            QPushButton:hover {background-color: #bf360c;}
            QPushButton:disabled {background-color: #424242; color: #757575;}
        """)
        self.btn_enroll.clicked.connect(self.start_enrollment_ui)
        controls_layout.addWidget(self.btn_enroll)

        self.main_layout.addWidget(controls_panel)
        self.setLayout(self.main_layout)

        self.init_audio_worker()

    def init_audio_worker(self):
        args = SystemArgs()
        device = torch.device("cpu") 

        wake_path = "models/wake_model.pth"
        cmd_path = "models/cmd_model.pth"

        self.worker = AudioWorker(
            wake_model_path=wake_path, 
            cmd_model_path=cmd_path, 
            args=args, 
            device=device
        )
        
        self.worker.cmd_signal.connect(self.process_voice_command)
        
        self.worker.status_signal.connect(self.lbl_audio_status.setText)
        
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        
        self.worker.enrollment_finished_signal.connect(self.on_enrollment_finished)
        
        self.worker.wake_signal.connect(self.on_wake_detected)

        self.worker.start()

    @pyqtSlot(str)
    def process_voice_command(self, command_name):
        print(f"Przetwarzam komendę: {command_name}")
            
        if command_name == "Wychodze":
            self.logic.save()
            self.logic.set_all(False)
            self.refresh_all_tiles()

        elif command_name == "Wrocilem":
            if self.logic.restore():
                self.refresh_all_tiles()
            
        elif command_name in self.logic.devices:
            new_state = self.logic.toggle(command_name)
            self.tiles[command_name].update_style(new_state)

    def refresh_all_tiles(self):
        for name, state in self.logic.devices.items():
            if name in self.tiles:
                self.tiles[name].update_style(state)
    
    def start_enrollment_ui(self):
        self.btn_enroll.setEnabled(False)
        self.btn_enroll.setText("Postępuj zgodnie z instrukcjami...")
        self.worker.start_enrollment()

    def on_enrollment_finished(self):
        self.btn_enroll.setEnabled(True)
        self.btn_enroll.setText("Nagraj nowe słowo wybudzające")
        self.progress_bar.setValue(0)

    def on_wake_detected(self, prob):
        self.setStyleSheet("background-color: #0d47a1;")
        QTimer.singleShot(300, lambda: self.setStyleSheet("background-color: #121212; color: white;"))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()

    window.show()
    sys.exit(app.exec_())