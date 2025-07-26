#!/usr/bin/env python3
"""Real Alpha Wave Mind Control with EEG Integration!"""

import sys
import time
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from openbci_stream import OpenBCIStreamer
from scipy.signal import welch
import threading

class RealAlphaController(QThread):
    """Thread to handle real EEG data processing."""
    
    alpha_updated = pyqtSignal(float)  # Signal to update alpha power
    status_updated = pyqtSignal(str)   # Signal to update status
    
    def __init__(self, port="COM3"):
        super().__init__()
        self.port = port
        self.running = False
        self.streamer = None
        
    def run(self):
        """Main thread execution for real-time alpha monitoring."""
        try:
            # Setup EEG streamer
            self.streamer = OpenBCIStreamer(2, self.port, './alpha_control.csv')
            
            if not self.streamer.setup_board():
                self.status_updated.emit("❌ Failed to connect to EEG device")
                return
                
            self.streamer.start_stream()
            self.status_updated.emit("✅ EEG Connected! Monitoring alpha waves...")
            time.sleep(3)  # Let buffer fill
            
            self.running = True
            
            while self.running:
                # Collect 2 seconds of data
                data_buffer = []
                start_time = time.time()
                
                while time.time() - start_time < 2.0 and self.running:
                    new_eeg_data, timestamps = self.streamer.get_new_data()
                    if new_eeg_data.shape[1] > 0:
                        data_buffer.append(new_eeg_data)
                    time.sleep(0.1)
                
                if data_buffer and self.running:
                    # Concatenate data
                    full_data = np.concatenate(data_buffer, axis=1)
                    
                    # Calculate alpha power for motor channels
                    alpha_power = self.calculate_alpha_power(full_data)
                    
                    # Emit alpha power update
                    self.alpha_updated.emit(alpha_power)
                    
        except Exception as e:
            self.status_updated.emit(f"❌ Error: {str(e)}")
        finally:
            if self.streamer:
                self.streamer.cleanup()
    
    def calculate_alpha_power(self, eeg_data):
        """Calculate alpha power from EEG data."""
        motor_channels = [8, 9, 10]  # C3, Cz, C4
        alpha_powers = []
        
        for ch in motor_channels:
            if ch < eeg_data.shape[0]:
                # Power spectral density
                f, psd = welch(eeg_data[ch], fs=125, nperseg=256)
                
                # Alpha band (8-12 Hz)
                alpha_mask = (f >= 8) & (f <= 12)
                alpha_power = np.mean(psd[alpha_mask])
                alpha_powers.append(alpha_power)
        
        if alpha_powers:
            # Normalize to 0-1 range (adjust based on your baseline)
            avg_alpha = np.mean(alpha_powers)
            normalized = min(1.0, max(0.0, (avg_alpha - 1.0) / 10.0))  # Adjust scaling
            return normalized
        else:
            return 0.0
    
    def stop_monitoring(self):
        """Stop the alpha monitoring."""
        self.running = False
        if self.streamer:
            self.streamer.cleanup()

class RealAlphaMindControlGUI(QMainWindow):
    """Real Alpha Wave Mind Control with EEG Integration."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 REAL ALPHA WAVE MIND CONTROL - YouTube Ready!")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: #1a1a1a; color: #ffffff;")
        
        self.alpha_controller = None
        self.current_alpha = 0.0
        self.light_state = False
        self.music_state = False
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("🧠 REAL ALPHA WAVE MIND CONTROL")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 32px; font-weight: bold; color: #00ff00; margin: 20px;")
        layout.addWidget(title)
        
        # Status
        self.status_label = QLabel("Ready to connect to your brain...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 18px; color: #ffffff; margin: 10px;")
        layout.addWidget(self.status_label)
        
        # Alpha power display
        self.alpha_display = QLabel("Alpha Power: 0%")
        self.alpha_display.setAlignment(Qt.AlignCenter)
        self.alpha_display.setStyleSheet("font-size: 24px; font-weight: bold; color: #00ff00; margin: 10px; padding: 20px; background-color: #333333; border-radius: 10px;")
        layout.addWidget(self.alpha_display)
        
        # Mind control targets
        targets_frame = QFrame()
        targets_frame.setStyleSheet("background-color: #2a2a2a; border: 2px solid #00ffff; border-radius: 10px; margin: 10px; padding: 20px;")
        targets_layout = QVBoxLayout(targets_frame)
        
        targets_title = QLabel("🎯 MIND CONTROL TARGETS")
        targets_title.setAlignment(Qt.AlignCenter)
        targets_title.setStyleSheet("font-size: 20px; font-weight: bold; color: #00ffff; margin: 10px;")
        targets_layout.addWidget(targets_title)
        
        # Target grid
        grid = QGridLayout()
        
        # Light control
        self.light_widget = self.create_target_widget("💡", "Smart Light", False)
        grid.addWidget(self.light_widget, 0, 0)
        
        # Music control
        self.music_widget = self.create_target_widget("🎵", "Music Player", False)
        grid.addWidget(self.music_widget, 0, 1)
        
        targets_layout.addLayout(grid)
        layout.addWidget(targets_frame)
        
        # Instructions
        instructions = QLabel("""
🎬 REAL MIND CONTROL INSTRUCTIONS:
1. 😌 CLOSE YOUR EYES to increase alpha waves
2. 👁️ OPEN YOUR EYES to decrease alpha waves  
3. 🎯 Alpha > 60% = Light control activated!
4. 🎯 Alpha > 80% = Music control activated!
5. 📹 Perfect for YouTube content!
        """)
        instructions.setStyleSheet("font-size: 14px; color: #ffff00; margin: 10px; padding: 10px; background-color: #333333; border-radius: 5px;")
        layout.addWidget(instructions)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.connect_button = QPushButton("🔌 CONNECT TO BRAIN")
        self.connect_button.setStyleSheet("background-color: #00ff00; color: black; font-size: 16px; font-weight: bold; padding: 15px; border-radius: 5px;")
        self.connect_button.clicked.connect(self.connect_brain)
        button_layout.addWidget(self.connect_button)
        
        self.stop_button = QPushButton("⏹️ STOP")
        self.stop_button.setStyleSheet("background-color: #ff0000; color: white; font-size: 16px; font-weight: bold; padding: 15px; border-radius: 5px;")
        self.stop_button.clicked.connect(self.stop_monitoring)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        layout.addLayout(button_layout)
    
    def create_target_widget(self, emoji, name, active):
        """Create a visual widget for a mind control target."""
        widget = QFrame()
        widget.setMinimumSize(250, 120)
        widget.setStyleSheet(f"""
            background-color: {'#004400' if active else '#333333'};
            border: 3px solid {'#00ff00' if active else '#666666'};
            border-radius: 15px;
            margin: 10px;
        """)
        
        layout = QVBoxLayout(widget)
        
        # Emoji
        emoji_label = QLabel(emoji)
        emoji_label.setAlignment(Qt.AlignCenter)
        emoji_label.setStyleSheet("font-size: 50px; margin: 5px;")
        layout.addWidget(emoji_label)
        
        # Name
        name_label = QLabel(name)
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #ffffff;")
        layout.addWidget(name_label)
        
        return widget
    
    def connect_brain(self):
        """Connect to the EEG device and start monitoring."""
        self.status_label.setText("🔌 Connecting to your brain...")
        self.connect_button.setEnabled(False)
        
        # Start alpha monitoring thread
        self.alpha_controller = RealAlphaController()
        self.alpha_controller.alpha_updated.connect(self.update_alpha_power)
        self.alpha_controller.status_updated.connect(self.update_status)
        self.alpha_controller.start()
        
        self.stop_button.setEnabled(True)
    
    def update_alpha_power(self, alpha_power):
        """Update the alpha power display and control targets."""
        self.current_alpha = alpha_power
        
        # Update display
        self.alpha_display.setText(f"Alpha Power: {alpha_power:.1%}")
        
        # Color coding
        if alpha_power > 0.8:
            color = "#ff0000"  # Red for very high
            self.status_label.setText("🔥 INTENSE ALPHA WAVES - FULL MIND CONTROL!")
        elif alpha_power > 0.6:
            color = "#ffff00"  # Yellow for high
            self.status_label.setText("⚡ HIGH ALPHA WAVES - MIND CONTROL ACTIVE!")
        elif alpha_power > 0.4:
            color = "#00ff00"  # Green for medium
            self.status_label.setText("📈 MODERATE ALPHA WAVES - Getting stronger!")
        else:
            color = "#666666"  # Gray for low
            self.status_label.setText("😐 LOW ALPHA WAVES - Try closing your eyes!")
        
        self.alpha_display.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {color}; margin: 10px; padding: 20px; background-color: #333333; border-radius: 10px;")
        
        # Control targets
        self.control_targets(alpha_power)
    
    def control_targets(self, alpha_power):
        """Control targets based on alpha power."""
        # Light control (60% threshold)
        if alpha_power > 0.6:
            if not self.light_state:
                self.light_state = True
                self.update_target_widget(self.light_widget, True)
                print("💡 LIGHT ACTIVATED BY MIND CONTROL!")
        else:
            if self.light_state:
                self.light_state = False
                self.update_target_widget(self.light_widget, False)
                print("💡 Light deactivated")
        
        # Music control (80% threshold)
        if alpha_power > 0.8:
            if not self.music_state:
                self.music_state = True
                self.update_target_widget(self.music_widget, True)
                print("🎵 MUSIC ACTIVATED BY MIND CONTROL!")
        else:
            if self.music_state:
                self.music_state = False
                self.update_target_widget(self.music_widget, False)
                print("🎵 Music deactivated")
    
    def update_target_widget(self, widget, active):
        """Update the visual state of a target widget."""
        widget.setStyleSheet(f"""
            background-color: {'#004400' if active else '#333333'};
            border: 3px solid {'#00ff00' if active else '#666666'};
            border-radius: 15px;
            margin: 10px;
        """)
    
    def update_status(self, status):
        """Update the status label."""
        self.status_label.setText(status)
    
    def stop_monitoring(self):
        """Stop alpha wave monitoring."""
        if self.alpha_controller:
            self.alpha_controller.stop_monitoring()
            self.alpha_controller.wait()
        
        self.status_label.setText("⏹️ Monitoring stopped")
        self.connect_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        # Reset targets
        self.light_state = False
        self.music_state = False
        self.update_target_widget(self.light_widget, False)
        self.update_target_widget(self.music_widget, False)
    
    def closeEvent(self, event):
        """Handle window close event."""
        self.stop_monitoring()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RealAlphaMindControlGUI()
    window.show()
    sys.exit(app.exec_()) 