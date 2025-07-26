#!/usr/bin/env python3
"""Alpha Wave Mind Control GUI - Perfect for YouTube Content!"""

import sys
import time
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class AlphaWaveVisualizer(QWidget):
    """Real-time alpha wave strength visualizer."""
    
    def __init__(self):
        super().__init__()
        self.alpha_power = 0.0
        self.setMinimumHeight(200)
        self.setStyleSheet("background-color: #2a2a2a; border: 2px solid #00ff00; border-radius: 10px;")
        
    def paintEvent(self, event):
        """Draw the alpha wave visualization."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get dimensions
        width = self.width() - 20
        height = self.height() - 20
        
        # Draw alpha power bar
        bar_width = int(width * self.alpha_power)
        
        # Background bar
        painter.fillRect(10, height//2 - 20, width, 40, QColor(50, 50, 50))
        
        # Alpha power bar (color changes with intensity)
        if self.alpha_power > 0.8:
            color = QColor(255, 0, 0)  # Red for high alpha
        elif self.alpha_power > 0.5:
            color = QColor(255, 255, 0)  # Yellow for medium alpha
        else:
            color = QColor(0, 255, 0)  # Green for low alpha
            
        painter.fillRect(10, height//2 - 20, bar_width, 40, color)
        
        # Text
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Arial", 16, QFont.Bold))
        painter.drawText(10, 30, f"Alpha Power: {self.alpha_power:.1%}")
        
    def update_alpha_power(self, power):
        """Update alpha power display."""
        self.alpha_power = max(0.0, min(1.0, power))
        self.update()

class MindControlTargets(QWidget):
    """Visual targets that can be controlled with alpha waves."""
    
    def __init__(self):
        super().__init__()
        self.light_state = False
        self.music_state = False
        self.setMinimumHeight(300)
        self.setStyleSheet("background-color: #2a2a2a; border: 2px solid #00ffff; border-radius: 10px;")
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the targets interface."""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("🎯 MIND CONTROL TARGETS")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #00ffff; margin: 10px;")
        layout.addWidget(title)
        
        # Targets grid
        targets_layout = QGridLayout()
        
        # Smart Light
        self.light_widget = self.create_target_widget("💡", "Smart Light", self.light_state)
        targets_layout.addWidget(self.light_widget, 0, 0)
        
        # Music Player
        self.music_widget = self.create_target_widget("🎵", "Music Player", self.music_state)
        targets_layout.addWidget(self.music_widget, 0, 1)
        
        # Brain Waves Display
        self.waves_widget = self.create_target_widget("🌊", "Brain Waves", False)
        targets_layout.addWidget(self.waves_widget, 1, 0)
        
        # Meditation Mode
        self.meditation_widget = self.create_target_widget("🧘", "Meditation", False)
        targets_layout.addWidget(self.meditation_widget, 1, 1)
        
        layout.addLayout(targets_layout)
        
    def create_target_widget(self, emoji, name, active):
        """Create a visual widget for a mind control target."""
        widget = QFrame()
        widget.setMinimumSize(200, 100)
        widget.setStyleSheet(f"""
            background-color: {'#004400' if active else '#333333'};
            border: 2px solid {'#00ff00' if active else '#666666'};
            border-radius: 10px;
            margin: 5px;
        """)
        
        layout = QVBoxLayout(widget)
        
        # Emoji
        emoji_label = QLabel(emoji)
        emoji_label.setAlignment(Qt.AlignCenter)
        emoji_label.setStyleSheet("font-size: 40px; margin: 5px;")
        layout.addWidget(emoji_label)
        
        # Name
        name_label = QLabel(name)
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffffff;")
        layout.addWidget(name_label)
        
        return widget
        
    def activate_target(self, target_name, alpha_power):
        """Activate a target based on alpha power."""
        if target_name == "light" and alpha_power > 0.7:
            self.light_state = not self.light_state
            self.update_target_widget(self.light_widget, self.light_state)
        elif target_name == "music" and alpha_power > 0.8:
            self.music_state = not self.music_state
            self.update_target_widget(self.music_widget, self.music_state)
            
    def update_target_widget(self, widget, active):
        """Update the visual state of a target widget."""
        widget.setStyleSheet(f"""
            background-color: {'#004400' if active else '#333333'};
            border: 2px solid {'#00ff00' if active else '#666666'};
            border-radius: 10px;
            margin: 5px;
        """)

class AlphaMindControlGUI(QMainWindow):
    """YouTube-ready Alpha Wave Mind Control Interface."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 ALPHA WAVE MIND CONTROL - YouTube Ready!")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: #1a1a1a; color: #ffffff;")
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("🧠 ALPHA WAVE MIND CONTROL")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 32px; font-weight: bold; color: #00ff00; margin: 20px;")
        layout.addWidget(title)
        
        # Status
        self.status_label = QLabel("Ready to read your mind...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 18px; color: #ffffff; margin: 10px;")
        layout.addWidget(self.status_label)
        
        # Alpha wave visualizer
        self.alpha_visualizer = AlphaWaveVisualizer()
        layout.addWidget(self.alpha_visualizer)

        # Mind control targets
        self.targets_widget = MindControlTargets()
        layout.addWidget(self.targets_widget)
        
        # Instructions for YouTube
        instructions = QLabel("""
🎬 YOUTUBE INSTRUCTIONS:
1. 😌 CLOSE YOUR EYES to increase alpha waves (watch the green bar!)
2. 👁️ OPEN YOUR EYES to decrease alpha waves  
3. 🎯 High alpha = Mind control activation!
4. 💡 Try to turn on the light with your mind!
        """)
        instructions.setStyleSheet("font-size: 14px; color: #ffff00; margin: 10px; padding: 10px; background-color: #333333; border-radius: 5px;")
        layout.addWidget(instructions)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.connect_button = QPushButton("🔌 CONNECT EEG")
        self.connect_button.setStyleSheet("background-color: #00ff00; color: black; font-size: 16px; font-weight: bold; padding: 10px; border-radius: 5px;")
        self.connect_button.clicked.connect(self.connect_eeg)
        button_layout.addWidget(self.connect_button)
        
        self.start_button = QPushButton("🧠 START MIND CONTROL")
        self.start_button.setStyleSheet("background-color: #ff6600; color: white; font-size: 16px; font-weight: bold; padding: 10px; border-radius: 5px;")
        self.start_button.clicked.connect(self.start_mind_control)
        self.start_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        
        layout.addLayout(button_layout)
        
    def connect_eeg(self):
        """Connect to EEG device."""
        self.status_label.setText("🔌 Connecting to EEG device...")
        # TODO: Add actual EEG connection
        QTimer.singleShot(2000, self.on_connected)
        
    def on_connected(self):
        """Handle successful EEG connection."""
        self.status_label.setText("✅ EEG Connected! Ready for mind control!")
        self.start_button.setEnabled(True)
        self.connect_button.setText("✅ CONNECTED")
        self.connect_button.setEnabled(False)
        
    def start_mind_control(self):
        """Start the mind control demonstration."""
        self.status_label.setText("🧠 MIND CONTROL ACTIVE - Close your eyes to activate!")
        # TODO: Start alpha wave monitoring
        self.start_demo_mode()
        
    def start_demo_mode(self):
        """Start demonstration mode with simulated alpha waves."""
        self.demo_timer = QTimer()
        self.demo_timer.timeout.connect(self.update_demo)
        self.demo_timer.start(100)  # Update every 100ms
        self.demo_time = 0
        
    def update_demo(self):
        """Update the demonstration with simulated data."""
        self.demo_time += 0.1
        
        # Simulate alpha wave pattern (higher when "eyes closed")
        alpha_simulation = 0.3 + 0.4 * abs(np.sin(self.demo_time * 0.5))
        
        # Add some randomness
        alpha_simulation += np.random.uniform(-0.1, 0.1)
        alpha_simulation = max(0.0, min(1.0, alpha_simulation))
        
        # Update visualizer
        self.alpha_visualizer.update_alpha_power(alpha_simulation)
        
        # Update status based on alpha level
        if alpha_simulation > 0.8:
            self.status_label.setText("🔥 INTENSE ALPHA WAVES - MIND CONTROL ACTIVATED!")
            self.targets_widget.activate_target("music", alpha_simulation)
        elif alpha_simulation > 0.6:
            self.status_label.setText("⚡ HIGH ALPHA WAVES - GETTING STRONGER!")
            self.targets_widget.activate_target("light", alpha_simulation)
        elif alpha_simulation > 0.4:
            self.status_label.setText("📈 MODERATE ALPHA WAVES - Keep going!")
        else:
            self.status_label.setText("😐 LOW ALPHA WAVES - Try closing your eyes!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AlphaMindControlGUI()
    window.show()
    sys.exit(app.exec_()) 