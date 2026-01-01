from PyQt5.QtCore import QThread, pyqtSignal
import torch
import torch.nn as nn
import torchaudio.transforms as T
import sounddevice as sd
import numpy as np
import queue
import threading
import time
import argparse
import sys
import os
import psutil 
import csv     

import onnxruntime as ort

from neuralnet.MatchboxNet import MatchboxNet
from neuralnet.Resnet import ResNet8, ResNet14
from neuralnet.CRNN import CRNN

COMMANDS = ['Ciemniej', 'Jaśniej', 'Muzyka', 'Rolety', 'Światło', 'Telewizor', 'Wróciłem', 'Wychodzę', 'Tlo']

def get_rms(audio_chunk):
    return np.sqrt(np.mean(audio_chunk**2)) + 1e-7

class AudioWorker(QThread):
    log_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    wake_signal = pyqtSignal(float)
    cmd_signal = pyqtSignal(str, float)
    enrollment_finished_signal = pyqtSignal()
    review_request_signal = pyqtSignal()
    enrollment_phase1_finished_signal = pyqtSignal()

    def __init__(self, wake_model_path, cmd_model_path, args, device, wake_arch='matchboxnet', cmd_arch='matchboxnet'):        
        super().__init__()
        self.wake_model_path = wake_model_path
        self.cmd_model_path = cmd_model_path
        self.args = args
        self.device = device

        self.wake_arch = wake_arch 
        self.cmd_arch = cmd_arch

        self.running = True
        self.mode = "listening" 
        self.q = queue.Queue()
        
        self.sample_rate = 16000
        self.block_size = 2048
        self.window_size = 16000
        self.device = torch.device("cpu")

        self.mfcc_transform = T.MFCC(
            sample_rate=self.sample_rate, n_mfcc=20, 
            melkwargs={"n_fft": 512, "n_mels": 40, "hop_length": 160, "mel_scale": "htk"}
        )

        self.wake_model_net = self._load_full_model(self.wake_model_path, self.wake_arch, num_classes=1)
        
        self.extractor, self.classifier_head = self._load_split_model(self.cmd_model_path, self.cmd_arch, num_classes=9)
        
        self.custom_wake_path = "custom_wake.pt"
        self.reference_embedding = None
        self.negative_embedding = None
        self.use_custom_wake = False
        self._check_wake_mode() 

        self.audio_window = np.zeros(self.window_size, dtype=np.float32)
        self.is_awake = False
        self.awake_timer = 0

        self.enrollment_samples = []
        self.enrollment_state = "IDLE" 
        self.rec_buffer_emb = []    
        self.temp_raw_audio = None  
        self.temp_mean_emb = None   
        self.manual_rec_counter = 0

    def _load_full_model(self, path, arch, num_classes):
        if not path: return None
        self.log_signal.emit(f"Ładowanie modelu Wake Word (Hugo): {arch}...")
        model = self._create_arch(arch, num_classes)
        try:
            ckpt = torch.load(path, map_location=self.device)
            state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            model.load_state_dict(state)
            model.eval()
            return model
        except Exception as e:
            self.log_signal.emit(f"Błąd modelu Wake: {e}")
            return None

    def _load_split_model(self, path, arch, num_classes):
        self.log_signal.emit(f"Ładowanie modelu Komend: {arch}...")
        model = self._create_arch(arch, num_classes)
        try:
            ckpt = torch.load(path, map_location=self.device)
            state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            model.load_state_dict(state)
            model.eval()
        except Exception as e:
            self.log_signal.emit(f"Błąd modelu Komend: {e}")
            return None, None

        head = None
        if hasattr(model, 'fc'):
            head = model.fc
            model.fc = nn.Identity()
        elif hasattr(model, 'epilogue_conv3'): 
            head = model.epilogue_conv3
            model.epilogue_conv3 = nn.Identity()
        elif hasattr(model, 'classifier'):
            head = model.classifier
            model.classifier = nn.Identity()
        elif hasattr(model, 'linear'):
            head = model.linear
            model.linear = nn.Identity()
        else:
            head = nn.Identity()

        return model, head
    
    def _create_arch(self, arch, num_classes):
        input_channels = 20
        if arch == 'matchboxnet':
            return MatchboxNet(input_channels=input_channels, num_classes=num_classes, B=3, R=1, C=64)
        elif arch == 'resnet14':
            return ResNet14(input_channels=input_channels, num_classes=num_classes, k=1.5)
        elif arch == 'resnet8':
             return ResNet8(input_channels=input_channels, num_classes=num_classes, k=1.5)
        else:
            return ResNet14(input_channels=input_channels, num_classes=num_classes, k=1.5)

    def _check_wake_mode(self):
        if os.path.exists(self.custom_wake_path):
            try:
                data = torch.load(self.custom_wake_path, map_location=self.device)
                if isinstance(data, dict):
                    self.reference_embedding = data['positive']
                    self.negative_embedding = data.get('negative')
                else:
                    self.reference_embedding = data
                    self.negative_embedding = None
                self.use_custom_wake = True
                self.log_signal.emit("Tryb custom wake word")
            except:
                self.use_custom_wake = False
        else:
            self.use_custom_wake = False
            self.log_signal.emit("Tryb domyślny wake word")

    def audio_callback(self, indata, frames, time, status):
        self.q.put(indata.copy().flatten())

    def get_mfcc_tensor(self, audio_window):
        waveform = torch.from_numpy(audio_window).float()
        if waveform.ndim == 1: waveform = waveform.unsqueeze(0)
        return self.mfcc_transform(waveform)

    def get_embedding(self, mfcc_tensor):
        with torch.no_grad():
            if mfcc_tensor.ndim == 2: mfcc_tensor = mfcc_tensor.unsqueeze(0)
        return self.extractor(mfcc_tensor)

    def run(self):
        with sd.InputStream(callback=self.audio_callback, channels=1, samplerate=self.sample_rate, blocksize=self.block_size, dtype='float32'):
            
            while self.running:
                try:
                    data = self.q.get(timeout=0.5)
                    self.audio_window = np.roll(self.audio_window, -len(data))
                    self.audio_window[-len(data):] = data
                    
                    rms = get_rms(self.audio_window)

                    if self.mode == "enrollment":
                        self._logic_enrollment()
                    elif self.mode == "background_enrollment":
                        self._logic_background()
                    else:
                        self._inference(rms)
                        
                except queue.Empty:
                    pass
                except Exception as e:
                    self.log_signal.emit(f"Error: {e}")

    def _inference(self, rms):
        if rms < 0.005: return
        current_time = time.time()

        mfcc = self.get_mfcc_tensor(self.audio_window)

        if not self.is_awake:
            if self.use_custom_wake:
                emb = self.get_embedding(mfcc)
                norm_emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                if self.reference_embedding is not None:
                    score_pos = torch.nn.functional.cosine_similarity(norm_emb, self.reference_embedding)
                
                if self.negative_embedding is not None:
                    score_neg = torch.nn.functional.cosine_similarity(norm_emb, self.negative_embedding)
                    if score_pos.item() > 0.85 and score_pos.item() > (score_neg.item()+0.1):
                        self._trigger_wake(score_pos.item())
                    else:
                        if score_pos.item() > 0.85:
                            self._trigger_wake(score_pos.item())
            else:
                if self.wake_model_net is not None:
                    if mfcc.ndim == 2: mfcc = mfcc.unsqueeze(0)
                    
                    output = self.wake_model_net(mfcc)
                    prob = torch.sigmoid(output).item()
                    
                    if prob > 0.85:
                        self._trigger_wake(prob)

        else:
            if current_time - self.awake_timer > 4.0:
                self.is_awake = False
                self.status_signal.emit("Czuwanie")
                return

            emb = self.get_embedding(mfcc)
            
            if self.cmd_arch == 'matchboxnet':
                 output = self.classifier_head(emb.unsqueeze(2)) 
                 output = output.view(1, -1)
            else:
                 output = self.classifier_head(emb)

            probs = torch.softmax(output, dim=1)
            max_prob, idx = torch.max(probs, dim=1)
            cmd = COMMANDS[idx.item()]

            if max_prob.item() > 0.8:
                if cmd != 'Tlo':
                    self.cmd_signal.emit(cmd, max_prob.item())
                    self.is_awake = False
                    self.status_signal.emit("Wykonano")
                    self.audio_window[:] = 0
                    time.sleep(1.0)

    def start_manual_recording(self):
        self.mode = "enrollment"
        self.enrollment_state = "RECORDING"
        self.rec_buffer_emb = []
        self.manual_rec_counter = 0
        self.status_signal.emit("Nagrywam")
        self.audio_window[:] = 0
    
    def _logic_enrollment(self):
        try:
            if self.enrollment_state == "RECORDING":
                mfcc = self.get_mfcc_tensor(self.audio_window)
                emb = self.get_embedding(mfcc)
                self.rec_buffer_emb.append(emb)

                self.manual_rec_counter += 1

                if self.manual_rec_counter >= 12:
                    stack = torch.cat(self.rec_buffer_emb, dim=0)
                    self.temp_mean_emb = torch.mean(stack, dim=0, keepdim=True)
                    
                    self.temp_raw_audio = self.audio_window.copy()
                    
                    self.enrollment_state = "WAIT_FOR_DECISION"
                    self.status_signal.emit("Odsłuchaj i zatwierdź.")
                    self.review_request_signal.emit()
                elif self.enrollment_state == "WAIT_FOR_DECISION":
                    pass
        except Exception as e:
            self.log_signal.emit(f"Enrollment error: {e}")
            self.enrollment_state = "IDLE"
            self.mode = "listening"

    def _trigger_wake(self, score):
        self.is_awake = True
        self.awake_timer = time.time()
        self.wake_signal.emit(score)
        self.status_signal.emit(f"SŁUCHAM! ({score:.2f})")
        self.audio_window[:] = 0
        time.sleep(0.2)    
        
    def play_recording(self):
        if self.temp_raw_audio is not None:
            sd.play(self.temp_raw_audio, self.sample_rate)
    
    def accept_recording(self):
        if self.temp_mean_emb is not None:
            self.enrollment_samples.append(self.temp_mean_emb)
            count = len(self.enrollment_samples)

            self.progress_signal.emit(count*10)
            self.log_signal.emit(f"Zatwierdzono próbkę {count}/10")

            if count >= 10:
                self.enrollment_state = "PHASE_1_COMPLETE"
                stacked = torch.cat(self.enrollment_samples, dim=0)
                mean_emb = torch.mean(stacked, dim=0, keepdim=True)
                self.temp_positive_embedding = torch.nn.functional.normalize(mean_emb, p=2, dim=1)
                
                self.status_signal.emit("Etap 1 zakończony.")
                self.enrollment_phase1_finished_signal.emit()            
            else:
                self.enrollment_state = "IDLE"
                self.status_signal.emit(f"Gotowe ({count}/10). Kliknij 'Nagraj")
                self.enrollment_finished_signal.emit()
    
    def discard_recording(self):
        self.log_signal.emit("Odrzucono próbkę")
        self.enrollment_state = "IDLE"
        self.status_signal.emit("Spróbuj ponownie")
        self.enrollment_finished_signal.emit()
    
    def _logic_background(self):
        try:
            mfcc = self.get_mfcc_tensor(self.audio_window)
            emb = self.get_embedding(mfcc)
            self.rec_buffer_emb.append(emb)

            self.manual_rec_counter += 1
            
            target_blocks = 80 
            progress = int((self.manual_rec_counter / target_blocks) * 100)
            self.progress_signal.emit(progress)

            if self.manual_rec_counter >= target_blocks:
                self._finalize_background()
                
        except Exception as e:
            self.log_signal.emit(f"Background enrollment error: {e}")
            self.mode = "listening"

    def record_background_noise(self):
            self.mode = "background_enrollment"
            self.rec_buffer_emb = [] 
            self.manual_rec_counter = 0 
            self.status_signal.emit("CZYTAJ TEKST...")

    def _finalize_enrollment(self):
        stacked = torch.cat(self.enrollment_samples, dim=0)
        mean_emb = torch.mean(stacked, dim=0, keepdim=True)
        self.reference_embedding = torch.nn.functional.normalize(mean_emb, p=2, dim=1)
        data_to_save = {
            'positive': self.reference_embedding,
            'negative': self.negative_embedding
        }
        torch.save(data_to_save, self.custom_wake_path)
        
        self.use_custom_wake = True 
        self.log_signal.emit("Przełączono na custom Wake Word.")
        
        self.mode = "listening"
        self.enrollment_state = "IDLE"
        self.enrollment_samples = [] 
        self.enrollment_finished_signal.emit()
        self.status_signal.emit("Zapisano nowy wzorzec.")        

    def _finalize_background(self):
        try:
            if not self.rec_buffer_emb:
                self.log_signal.emit("Pusty bufor tła")
                self.mode = "listening"
                return
            stacked = torch.cat(self.rec_buffer_emb, dim=0)
            mean_neg = torch.mean(stacked, dim=0, keepdim=True)
            self.negative_embedding = torch.nn.functional.normalize(mean_neg, p=2, dim=1)
            self.reference_embedding = self.temp_positive_embedding

            self._save_to_file()

        except Exception as e:
            self.log_signal.emit(f"Błąd finalizacji tła: {e}")
        finally:
            self._reset_after_enrollment()

    def save_enrollment_without_noise(self):
        self.reference_embedding = self.temp_positive_embedding
        self.negative_embedding = None
        
        self._save_to_file()
        self.log_signal.emit("Zapisano wzorzec podstawowy.")
        self._reset_after_enrollment()

    def _save_to_file(self):
        data_to_save = {
            'positive': self.reference_embedding,
            'negative': self.negative_embedding
        }
        torch.save(data_to_save, self.custom_wake_path)
        self.use_custom_wake = True

    def _reset_after_enrollment(self):
        self.mode = "listening"
        self.enrollment_state = "IDLE"
        self.enrollment_samples = []
        self.rec_buffer_emb = []
        self.temp_positive_embedding = None
        
        self.enrollment_finished_signal.emit()
        self.status_signal.emit("Gotowy do pracy.")