#!/usr/bin/env python3
"""
🧠 Brain State Calibration GUI - Simple Training Interface
"""

import sys
import time
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class BrainStateCalibrator(QMainWindow):
    """Simple brain state calibration interface."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 Brain State Calibrator")
        self.setGeometry(100, 100, 1000, 700)
        self.setStyleSheet("background-color: #1a1a1a; color: white;")
        
        # Training data
        self.training_data = {}
        self.current_class = None
        self.samples_collected = 0
        self.target_samples = 50
        
        # OpenBCI connection
        self.streamer = None
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("🧠 BRAIN STATE CALIBRATION")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #00ffff; margin: 20px;")
        layout.addWidget(title)
        
        # Status
        self.status_label = QLabel("Ready to start calibration...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 16px; color: #ffff00; margin: 10px;")
        layout.addWidget(self.status_label)
        
        # Progress section
        progress_frame = QFrame()
        progress_frame.setStyleSheet("background-color: #2a2a2a; border: 2px solid #ff6b35; border-radius: 10px; margin: 10px; padding: 20px;")
        progress_layout = QVBoxLayout(progress_frame)
        
        # Current class
        self.class_label = QLabel("No class selected")
        self.class_label.setAlignment(Qt.AlignCenter)
        self.class_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #ff6b35; margin: 10px;")
        progress_layout.addWidget(self.class_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #666666;
                border-radius: 10px;
                text-align: center;
                background-color: #333333;
                color: white;
                font-weight: bold;
                height: 30px;
            }
            QProgressBar::chunk {
                background-color: #ff6b35;
                border-radius: 8px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)
        
        # Samples counter
        self.samples_label = QLabel("0 / 50 samples")
        self.samples_label.setAlignment(Qt.AlignCenter)
        self.samples_label.setStyleSheet("font-size: 14px; color: #cccccc; margin: 5px;")
        progress_layout.addWidget(self.samples_label)
        
        layout.addWidget(progress_frame)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Connect button
        self.connect_button = QPushButton("🔌 CONNECT")
        self.connect_button.setStyleSheet("""
            QPushButton {
                background-color: #00ff00;
                color: black;
                border: none;
                border-radius: 10px;
                padding: 15px 25px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #33ff33; }
        """)
        self.connect_button.clicked.connect(self.connect_device)
        controls_layout.addWidget(self.connect_button)
        
        # Start training button
        self.train_button = QPushButton("🎬 START TRAINING")
        self.train_button.setStyleSheet("""
            QPushButton {
                background-color: #ff6b35;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 15px 25px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #ff8555; }
        """)
        self.train_button.clicked.connect(self.start_training)
        self.train_button.setEnabled(False)
        controls_layout.addWidget(self.train_button)
        
        layout.addLayout(controls_layout)
        
        # Brain states to train
        states_frame = QFrame()
        states_frame.setStyleSheet("background-color: #2a2a2a; border: 2px solid #00ff00; border-radius: 10px; margin: 10px; padding: 20px;")
        states_layout = QVBoxLayout(states_frame)
        
        states_title = QLabel("🧠 Brain States to Train")
        states_title.setAlignment(Qt.AlignCenter)
        states_title.setStyleSheet("font-size: 18px; color: #00ff00; margin: 10px;")
        states_layout.addWidget(states_title)
        
        # Brain state buttons
        self.state_buttons = {}
        states = [
            ("🤚 Left Hand", "left_hand"),
            ("👍 Right Hand", "right_hand"), 
            ("😌 Rest", "rest"),
            ("🔢 Math", "math"),
            ("🎵 Music", "music"),
            ("😊 Face", "face"),
            ("📝 Word", "word")
        ]
        
        states_grid = QGridLayout()
        for i, (name, key) in enumerate(states):
            button = QPushButton(name)
            button.setStyleSheet("""
                QPushButton {
                    background-color: #666666;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 10px;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover { background-color: #888888; }
                QPushButton:checked { background-color: #9b59b6; }
            """)
            button.setCheckable(True)
            button.clicked.connect(lambda checked, k=key: self.select_brain_state(k))
            button.setEnabled(False)
            
            states_grid.addWidget(button, i // 3, i % 3)
            self.state_buttons[key] = button
        
        states_layout.addLayout(states_grid)
        layout.addWidget(states_frame)
        
        # Log area
        self.log_area = QTextEdit()
        self.log_area.setStyleSheet("""
            QTextEdit {
                background-color: #333333;
                border: 2px solid #666666;
                border-radius: 10px;
                color: white;
                font-family: monospace;
                font-size: 11px;
                padding: 10px;
            }
        """)
        self.log_area.setMaximumHeight(150)
        self.log("🚀 Brain State Calibrator initialized")
        layout.addWidget(self.log_area)
        
        # Timer for data collection
        self.collection_timer = QTimer()
        self.collection_timer.timeout.connect(self.collect_sample)
    
    def log(self, message):
        """Add message to log area."""
        self.log_area.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())
    
    def connect_device(self):
        """Connect to OpenBCI device."""
        self.log("🔌 Connecting to OpenBCI...")
        
        try:
            from openbci_stream import OpenBCIStreamer
            
            self.streamer = OpenBCIStreamer(0, "COM3", "./calibration.csv")
            
            if self.streamer.setup_board():
                self.streamer.start_stream()
                time.sleep(2)  # Let buffer fill
                
                self.log("✅ Connected successfully!")
                self.status_label.setText("✅ Connected - Ready to train")
                self.connect_button.setText("✅ CONNECTED")
                self.connect_button.setEnabled(False)
                self.train_button.setEnabled(True)
                
                # Enable state buttons
                for button in self.state_buttons.values():
                    button.setEnabled(True)
            else:
                raise Exception("Failed to setup board")
                
        except Exception as e:
            self.log(f"❌ Connection failed: {e}")
            self.status_label.setText("❌ Connection failed")
    
    def start_training(self):
        """Start the training sequence."""
        if not self.streamer:
            self.log("❌ Not connected to device")
            return
        
        self.log("🎬 Starting training sequence...")
        self.status_label.setText("🎬 Select a brain state to train")
    
    def select_brain_state(self, state_key):
        """Select and start training a brain state."""
        # Uncheck all other buttons
        for key, button in self.state_buttons.items():
            if key != state_key:
                button.setChecked(False)
        
        self.current_class = state_key
        self.samples_collected = 0
        
        # Initialize training data for this class
        if state_key not in self.training_data:
            self.training_data[state_key] = []
        
        self.class_label.setText(f"Training: {state_key.upper()}")
        self.update_progress()
        
        self.log(f"🎯 Starting {state_key} training...")
        self.log(f"🤚 Please think about: {state_key}")
        
        # Start data collection
        self.collection_timer.start(500)  # Collect every 500ms
    
    def collect_sample(self):
        """Collect a training sample."""
        if not self.streamer or not self.current_class:
            return
        
        try:
            # Get EEG data
            new_eeg_data, timestamps = self.streamer.get_new_data()
            
            if new_eeg_data is not None and new_eeg_data.shape[1] >= self.streamer.sampling_rate:
                # Process data
                window_data = new_eeg_data[:, -self.streamer.sampling_rate:]
                
                from openbci_stream import apply_filters, compute_band_powers
                filtered_data = apply_filters(window_data, self.streamer.sampling_rate)
                band_powers = compute_band_powers(filtered_data, self.streamer.sampling_rate)
                
                # Store sample
                self.training_data[self.current_class].append(band_powers)
                self.samples_collected += 1
                
                self.log(f"📊 Sample {self.samples_collected}: {self.current_class}")
                self.update_progress()
                
                # Check if done with this class
                if self.samples_collected >= self.target_samples:
                    self.collection_timer.stop()
                    self.state_buttons[self.current_class].setChecked(False)
                    self.state_buttons[self.current_class].setText(f"✅ {self.current_class}")
                    
                    self.log(f"✅ Completed {self.current_class} training!")
                    self.class_label.setText("Select next brain state")
                    self.current_class = None
                    
                    # Check if all states are trained
                    total_states = len([s for s in self.training_data.values() if len(s) >= self.target_samples])
                    if total_states >= 3:  # Need at least 3 states
                        self.train_classifier()
            
        except Exception as e:
            self.log(f"❌ Error collecting sample: {e}")
    
    def update_progress(self):
        """Update progress display."""
        progress = int((self.samples_collected / self.target_samples) * 100)
        self.progress_bar.setValue(progress)
        self.samples_label.setText(f"{self.samples_collected} / {self.target_samples} samples")
    
    def train_classifier(self):
        """Train the brain state classifier."""
        self.log("🧠 Training neural network...")
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            import joblib
            
            # Prepare training data
            X_train = []
            y_train = []
            
            for class_idx, (state, samples) in enumerate(self.training_data.items()):
                if len(samples) >= self.target_samples:
                    for sample in samples[:self.target_samples]:
                        X_train.append(sample)
                        y_train.append(class_idx)
            
            if len(X_train) > 0:
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                
                # Train model
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Calculate accuracy
                y_pred = model.predict(X_train)
                accuracy = np.mean(y_pred == y_train)
                
                # Save model
                joblib.dump(model, 'brain_state_model.pkl')
                
                self.log(f"🎉 Training completed! Accuracy: {accuracy:.1%}")
                self.status_label.setText(f"🎉 Training completed! Accuracy: {accuracy:.1%}")
            
        except Exception as e:
            self.log(f"❌ Training failed: {e}")
    
    def closeEvent(self, event):
        """Clean up when closing."""
        if self.streamer:
            self.streamer.cleanup()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    calibrator = BrainStateCalibrator()
    calibrator.show()
    sys.exit(app.exec_()) 