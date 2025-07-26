#!/usr/bin/env python3
"""
OpenBCI Cyton EEG Streaming Script - VIRAL EDITION! 🧠🚀

This script connects to an OpenBCI Cyton board, streams EEG data in real-time,
applies signal processing, computes band power features, provides live visualization,
AND includes viral features like brain state classification, 3D brain visualization,
brain-controlled music, smart home integration, and MIND READING NEURAL NETWORK!

Author: Senior Python Engineer - BrainPower Project
"""

import argparse
import csv
import json
import logging
import signal
import sys
import time
import threading
from collections import deque
from typing import Optional, Tuple, List, Dict
import os
import random
import colorsys
import pickle
from datetime import datetime

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets, QtGui
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
from scipy import signal as scipy_signal
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sounddevice as sd
import librosa
import requests

# Optional imports (graceful degradation if not available)
try:
    import mido
    import rtmidi
    MIDI_AVAILABLE = True
except ImportError:
    MIDI_AVAILABLE = False
    print("MIDI features disabled - install python-rtmidi for brain-controlled music")

try:
    from phue import Bridge
    HUE_AVAILABLE = True
except ImportError:
    HUE_AVAILABLE = False
    print("Philips Hue features disabled - install phue for smart lighting")

try:
    import dash
    from dash import dcc, html, Input, Output
    import dash_bootstrap_components as dbc
    from flask import Flask
    from flask_socketio import SocketIO
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False
    print("Web dashboard disabled - install dash and flask-socketio for web features")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for graceful shutdown
board_shim: Optional[BoardShim] = None
app: Optional[QtWidgets.QApplication] = None

# Brain state emojis for viral visualization
BRAIN_STATE_EMOJIS = {
    'focused': '🎯',
    'relaxed': '😌', 
    'stressed': '😰',
    'meditative': '🧘',
    'excited': '🤩',
    'drowsy': '😴',
    'neutral': '😐'
}

# Mind reading emojis for neural network predictions
MIND_READING_EMOJIS = {
    'left_hand': '👈',
    'right_hand': '👉',
    'rest': '😐',
    'math': '🧮',
    'music': '🎵',
    'face': '😊',
    'word': '📝'
}

# Color schemes for brain states
BRAIN_STATE_COLORS = {
    'focused': '#FF6B35',      # Orange-red
    'relaxed': '#4ECDC4',      # Teal
    'stressed': '#FF3366',     # Red
    'meditative': '#9B59B6',   # Purple
    'excited': '#F39C12',      # Yellow
    'drowsy': '#3498DB',       # Blue
    'neutral': '#95A5A6'       # Gray
}

# Color schemes for mind reading
MIND_READING_COLORS = {
    'left_hand': '#E74C3C',    # Red
    'right_hand': '#3498DB',   # Blue
    'rest': '#95A5A6',         # Gray
    'math': '#F39C12',         # Orange
    'music': '#9B59B6',        # Purple
    'face': '#E67E22',         # Orange
    'word': '#27AE60'          # Green
}


def signal_handler(signum, frame):
    """Handle Ctrl-C gracefully by stopping the stream and cleaning up."""
    logger.info("Received interrupt signal. Shutting down gracefully...")
    cleanup()
    sys.exit(0)


def cleanup():
    """Clean up resources before shutdown."""
    global board_shim, app
    
    if board_shim:
        try:
            if board_shim.is_prepared():
                board_shim.stop_stream()
                board_shim.release_session()
            logger.info("BrainFlow session released successfully")
        except Exception as e:
            logger.error(f"Error during BrainFlow cleanup: {e}")
    
    if app:
        try:
            app.quit()
        except Exception as e:
            logger.error(f"Error during Qt app cleanup: {e}")


def apply_filters(data: np.ndarray, sampling_rate: int) -> np.ndarray:
    """
    Apply bandpass (1-40 Hz) and notch (60 Hz) filters to EEG data.
    
    Args:
        data: EEG data array (channels x samples)
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Filtered EEG data
    """
    try:
        filtered_data = data.copy()
        
        # Handle both 1D and 2D arrays
        if len(filtered_data.shape) == 1:
            # Single channel data
            # Apply bandpass filter (1-40 Hz)
            DataFilter.perform_bandpass(
                filtered_data, sampling_rate, 1.0, 40.0, 2,
                FilterTypes.BUTTERWORTH.value, 0
            )
            
            # Apply notch filter (60 Hz)
            DataFilter.perform_bandstop(
                filtered_data, sampling_rate, 59.0, 61.0, 2,
                FilterTypes.BUTTERWORTH.value, 0
            )
        else:
            # Multi-channel data - filter each channel separately
            for ch in range(filtered_data.shape[0]):
                channel_data = filtered_data[ch, :].copy()
                
                # Apply bandpass filter (1-40 Hz)
                DataFilter.perform_bandpass(
                    channel_data, sampling_rate, 1.0, 40.0, 2,
                    FilterTypes.BUTTERWORTH.value, 0
                )
                
                # Apply notch filter (60 Hz)
                DataFilter.perform_bandstop(
                    channel_data, sampling_rate, 59.0, 61.0, 2,
                    FilterTypes.BUTTERWORTH.value, 0
                )
                
                filtered_data[ch, :] = channel_data
        
        return filtered_data
        
    except Exception as e:
        logger.error(f"Error applying filters: {e}")
        return data


def compute_band_powers(data: np.ndarray, sampling_rate: int) -> np.ndarray:
    """
    Compute band power features for each channel using Welch's method.
    
    Args:
        data: Filtered EEG data (channels x samples)
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Band powers array (channels x 5 bands)
        Bands: delta (1-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), 
               beta (13-30 Hz), gamma (30-40 Hz)
    """
    try:
        num_channels = data.shape[0]
        band_powers = np.zeros((num_channels, 5))
        
        # Define frequency bands
        bands = {
            'delta': (1, 4),
            'theta': (4, 8), 
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 40)
        }
        
        for ch in range(num_channels):
            # Compute power spectral density using Welch's method
            freqs, psd = scipy_signal.welch(
                data[ch], fs=sampling_rate, nperseg=min(256, len(data[ch])), 
                noverlap=None, scaling='density'
            )
            
            # Calculate power in each band
            for i, (band_name, (low_freq, high_freq)) in enumerate(bands.items()):
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                if np.any(band_mask):
                    band_powers[ch, i] = np.trapz(psd[band_mask], freqs[band_mask])
                else:
                    band_powers[ch, i] = 0.0
                    
        return band_powers
        
    except Exception as e:
        logger.error(f"Error computing band powers: {e}")
        return np.zeros((data.shape[0], 5))


class EEGVisualizer:
    """Enhanced real-time EEG visualization with viral features."""
    
    def __init__(self, sampling_rate: int, window_size: int = 1000):
        """
        Initialize the enhanced EEG visualizer.
        
        Args:
            sampling_rate: Sampling rate in Hz
            window_size: Number of samples to display in raw signal plot
        """
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        
        # Initialize Qt application
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication([])
        
        # Create main window with enhanced layout
        self.win = pg.GraphicsLayoutWidget(title="🧠 OpenBCI EEG Stream - VIRAL EDITION! 🚀")
        self.win.resize(1600, 1200)
        self.win.show()
        
        # Set dark theme for better visual appeal
        self.win.setBackground('black')
        
        # Brain state display (top row)
        self.state_label = pg.LabelItem(justify='center')
        self.state_label.setText("🧠 Initializing Brain State Detection...", size='24pt', color='white')
        self.win.addItem(self.state_label, row=0, col=0, colspan=2)
        
        # 🧠 VIRAL FEATURE: Mind Reading Display
        self.win.nextRow()
        self.mind_reading_label = pg.LabelItem(justify='center')
        self.mind_reading_label.setText("🎯 Mind Reader Neural Network: Initializing...", size='20pt', color='cyan')
        self.win.addItem(self.mind_reading_label, row=1, col=0, colspan=2)
        
        # Raw signal plot (enhanced)
        self.win.nextRow()
        self.raw_plot = self.win.addPlot(title="🎯 Raw EEG Signal (Channel 1)", row=2, col=0)
        self.raw_plot.setLabel('left', 'Amplitude', units='µV', color='white')
        self.raw_plot.setLabel('bottom', 'Time', units='s', color='white')
        self.raw_plot.showGrid(x=True, y=True, alpha=0.3)
        self.raw_curve = self.raw_plot.plot(pen=pg.mkPen(color='cyan', width=2))
        
        # Real-time spectrogram
        self.spectrogram_plot = self.win.addPlot(title="🌈 Real-Time Spectrogram", row=2, col=1)
        self.spectrogram_plot.setLabel('left', 'Frequency', units='Hz', color='white')
        self.spectrogram_plot.setLabel('bottom', 'Time', units='s', color='white')
        
        # Band power plots (enhanced with colors)
        self.win.nextRow()
        self.band_plot = self.win.addPlot(title="📊 Band Power Features", row=3, col=0, colspan=2)
        self.band_plot.setLabel('left', 'Power', units='µV²', color='white')
        self.band_plot.setLabel('bottom', 'Time', units='s', color='white')
        self.band_plot.showGrid(x=True, y=True, alpha=0.3)
        
        # Enhanced band power curves with gradients
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        band_names = ['Delta (1-4Hz)', 'Theta (4-8Hz)', 'Alpha (8-13Hz)', 'Beta (13-30Hz)', 'Gamma (30-40Hz)']
        self.band_curves = []
        
        for i, (color, name) in enumerate(zip(colors, band_names)):
            pen = pg.mkPen(color=color, width=3)
            curve = self.band_plot.plot(pen=pen, name=name)
            self.band_curves.append(curve)
        
        # Add legend with enhanced styling
        legend = self.band_plot.addLegend()
        legend.setParentItem(self.band_plot.graphicsItem())
        
        # Brain state probability bars
        self.win.nextRow()
        self.prob_plot = self.win.addPlot(title="🎲 Brain State Probabilities", row=4, col=0, colspan=2)
        self.prob_plot.setLabel('left', 'Probability', color='white')
        self.prob_plot.setLabel('bottom', 'Brain States', color='white')
        self.prob_plot.showGrid(x=False, y=True, alpha=0.3)
        
        # Data buffers for plotting
        self.raw_buffer = deque(maxlen=self.window_size)
        self.band_buffers = [deque(maxlen=300) for _ in range(5)]  # 300 points = 15s at 20Hz
        self.time_buffer = deque(maxlen=self.window_size)
        self.band_time_buffer = deque(maxlen=300)
        
        # Spectrogram data
        self.spectrogram_data = deque(maxlen=100)  # 5 seconds of spectrogram data
        self.spectrogram_freqs = np.linspace(1, 40, 50)  # 1-40 Hz
        
        # Brain state tracking
        self.current_state = 'neutral'
        self.current_confidence = 0.0
        self.state_probabilities = {}
        
        # Mind reading tracking
        self.current_thought = 'rest'
        self.thought_confidence = 0.0
        self.thought_probabilities = {}
        self.is_training_mode = False
        
        # Initialize buffers with zeros
        self.raw_buffer.extend([0] * self.window_size)
        self.time_buffer.extend(np.linspace(0, self.window_size/self.sampling_rate, self.window_size))
        
        for band_buffer in self.band_buffers:
            band_buffer.extend([0] * 300)
        self.band_time_buffer.extend(np.linspace(0, 15, 300))
        
        # Animation timer for smooth updates
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self._update_animations)
        self.update_timer.start(50)  # 20 FPS
        
        logger.info("Enhanced EEG visualizer initialized with viral features")
        
    def update_brain_state(self, state: str, confidence: float, probabilities: Dict[str, float]):
        """Update brain state display with enhanced visuals."""
        try:
            self.current_state = state
            self.current_confidence = confidence
            self.state_probabilities = probabilities
            
            # Get emoji and color for state
            emoji = BRAIN_STATE_EMOJIS.get(state, '🧠')
            color = BRAIN_STATE_COLORS.get(state, '#95A5A6')
            
            # Create animated state display
            confidence_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
            state_text = f"{emoji} {state.upper()} {emoji} | Confidence: {confidence:.1%} [{confidence_bar}]"
            
            # Update state label with color
            self.state_label.setText(state_text, size='20pt', color=color)
            
            # Update probability bars
            self._update_probability_bars(probabilities)
            
        except Exception as e:
            logger.error(f"Error updating brain state display: {e}")
    
    def _update_probability_bars(self, probabilities: Dict[str, float]):
        """Update the brain state probability bar chart."""
        try:
            self.prob_plot.clear()
            
            if not probabilities:
                return
                
            states = list(probabilities.keys())
            probs = list(probabilities.values())
            colors = [BRAIN_STATE_COLORS.get(state, '#95A5A6') for state in states]
            
            # Create bar chart
            x = np.arange(len(states))
            bar_width = 0.8
            
            for i, (state, prob, color) in enumerate(zip(states, probs, colors)):
                brush = pg.mkBrush(color=color)
                bar = pg.BarGraphItem(x=[i], height=[prob], width=bar_width, brush=brush)
                self.prob_plot.addItem(bar)
                
                # Add state labels
                text = pg.TextItem(text=state, anchor=(0.5, 0))
                text.setPos(i, -0.05)
                self.prob_plot.addItem(text)
            
            self.prob_plot.setXRange(-0.5, len(states) - 0.5)
            self.prob_plot.setYRange(0, 1.0)
            
        except Exception as e:
            logger.error(f"Error updating probability bars: {e}")
    
    def update_raw_signal(self, new_data: np.ndarray, timestamps: np.ndarray):
        """Update the raw signal plot with enhanced visuals."""
        try:
            # Add new data to buffer
            self.raw_buffer.extend(new_data)
            
            # Update time buffer
            if len(timestamps) > 0:
                time_offset = timestamps[-1] - (len(self.raw_buffer) - 1) / self.sampling_rate
                new_times = time_offset + np.arange(len(self.raw_buffer)) / self.sampling_rate
                self.time_buffer.clear()
                self.time_buffer.extend(new_times)
            
            # Update plot with dynamic color based on brain state
            color = BRAIN_STATE_COLORS.get(self.current_state, '#00FFFF')
            pen = pg.mkPen(color=color, width=2)
            self.raw_curve.setPen(pen)
            self.raw_curve.setData(list(self.time_buffer), list(self.raw_buffer))
            
            # Update spectrogram
            self._update_spectrogram(new_data)
            
        except Exception as e:
            logger.error(f"Error updating raw signal plot: {e}")
    
    def _update_spectrogram(self, new_data: np.ndarray):
        """Update the real-time spectrogram."""
        try:
            if len(new_data) < 64:  # Need minimum samples for FFT
                return
                
            # Compute power spectral density
            freqs, psd = scipy_signal.welch(new_data, fs=self.sampling_rate, 
                                          nperseg=min(128, len(new_data)), 
                                          noverlap=None, scaling='density')
            
            # Filter to 1-40 Hz range
            freq_mask = (freqs >= 1) & (freqs <= 40)
            filtered_freqs = freqs[freq_mask]
            filtered_psd = psd[freq_mask]
            
            # Interpolate to fixed frequency grid
            interp_psd = np.interp(self.spectrogram_freqs, filtered_freqs, filtered_psd)
            
            # Add to spectrogram buffer
            self.spectrogram_data.append(interp_psd)
            
            # Update spectrogram plot
            if len(self.spectrogram_data) > 10:
                spectrogram_array = np.array(list(self.spectrogram_data)).T
                
                # Create image item
                img = pg.ImageItem(spectrogram_array)
                img.setLookupTable(pg.colormap.get('viridis').getLookupTable())
                
                # Clear and add new image
                self.spectrogram_plot.clear()
                self.spectrogram_plot.addItem(img)
                
                # Set proper scaling
                img.setRect(QtCore.QRectF(0, 1, len(self.spectrogram_data), 39))
                
        except Exception as e:
            logger.error(f"Error updating spectrogram: {e}")
    
    def update_band_powers(self, band_powers: np.ndarray, timestamp: float):
        """Update band power plots with enhanced animations."""
        try:
            # Add new band power data
            for i, power in enumerate(band_powers):
                self.band_buffers[i].append(power)
            
            # Update time buffer
            self.band_time_buffer.append(timestamp)
            
            # Update plots with enhanced styling
            times = list(self.band_time_buffer)
            for i, curve in enumerate(self.band_curves):
                powers = list(self.band_buffers[i])
                
                # Add glow effect for current dominant band
                if i == np.argmax(band_powers):
                    pen = pg.mkPen(color=curve.opts['pen'].color(), width=4)
                    curve.setPen(pen)
                else:
                    pen = pg.mkPen(color=curve.opts['pen'].color(), width=2)
                    curve.setPen(pen)
                
                curve.setData(times, powers)
                
        except Exception as e:
            logger.error(f"Error updating band power plots: {e}")
    
    def _update_animations(self):
        """Update animations and visual effects."""
        try:
            # Pulse effect for high confidence states
            if self.current_confidence > 0.8:
                alpha = 0.5 + 0.5 * np.sin(time.time() * 5)  # Pulse at 5 Hz
                # Could add pulsing effects here
                
        except Exception as e:
            logger.error(f"Error in animation update: {e}")
    
    def create_3d_brain_plot(self):
        """Create a 3D brain visualization (placeholder for future implementation)."""
        try:
            # This would create a 3D brain model using plotly
            # For now, we'll add this as a future enhancement
            logger.info("3D brain visualization feature planned for future release")
            
        except Exception as e:
            logger.error(f"Error creating 3D brain plot: {e}")
    
    def process_events(self):
        """Process Qt events to keep the GUI responsive."""
        try:
            self.app.processEvents()
        except Exception as e:
            logger.error(f"Error processing Qt events: {e}")
    
    def close(self):
        """Close the visualization window."""
        try:
            if hasattr(self, 'update_timer'):
                self.update_timer.stop()
            self.win.close()
        except Exception as e:
            logger.error(f"Error closing visualization: {e}")
    
    def update_mind_reading(self, thought: str, confidence: float, probabilities: Dict[str, float], 
                           is_training: bool = False, training_progress: Dict = None):
        """Update mind reading display with neural network predictions."""
        try:
            self.current_thought = thought
            self.thought_confidence = confidence
            self.thought_probabilities = probabilities
            self.is_training_mode = is_training
            
            # Get emoji and color for thought
            emoji = MIND_READING_EMOJIS.get(thought, '🧠')
            color = MIND_READING_COLORS.get(thought, '#95A5A6')
            
            if is_training and training_progress:
                # Training mode display
                current_class = training_progress.get('current_class', 'unknown')
                samples_collected = training_progress.get('samples_collected', 0)
                samples_needed = training_progress.get('samples_needed', 50)
                progress_percent = training_progress.get('progress_percent', 0)
                
                training_emoji = MIND_READING_EMOJIS.get(current_class, '🧠')
                progress_bar = "█" * int(progress_percent / 10) + "░" * (10 - int(progress_percent / 10))
                
                mind_text = f"🎬 TRAINING: {training_emoji} {current_class.upper()} | {samples_collected}/{samples_needed} [{progress_bar}] {progress_percent:.0f}%"
                color = '#FF6B35'  # Orange for training mode
                
            elif not probabilities:
                # No model trained yet
                mind_text = f"🎯 Mind Reader: Not trained yet - Ready for viral content!"
                color = '#95A5A6'  # Gray
                
            else:
                # Prediction mode display
                confidence_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
                mind_text = f"🧠 READING MIND: {emoji} {thought.upper()} {emoji} | Confidence: {confidence:.1%} [{confidence_bar}]"
            
            # Update mind reading label with color
            self.mind_reading_label.setText(mind_text, size='18pt', color=color)
            
        except Exception as e:
            logger.error(f"Error updating mind reading display: {e}")


class OpenBCIStreamer:
    """Enhanced OpenBCI EEG streaming with viral features."""
    
    def __init__(self, board_id: int, serial_port: str, csv_path: str, 
                 enable_music: bool = False, enable_smart_home: bool = False, 
                 hue_bridge_ip: Optional[str] = None):
        """
        Initialize the enhanced OpenBCI streamer.
        
        Args:
            board_id: BrainFlow board ID (0 for Cyton)
            serial_port: Serial port for the board
            csv_path: Path to save CSV logs
            enable_music: Enable brain-controlled music generation
            enable_smart_home: Enable smart home integration
            hue_bridge_ip: IP address of Philips Hue bridge
        """
        self.board_id = board_id
        self.serial_port = serial_port
        self.csv_path = csv_path
        
        # BrainFlow setup
        self.board_shim = None
        self.sampling_rate = None
        self.eeg_channels = None
        self.timestamp_channel = None
        
        # Data processing
        self.ring_buffer = None
        self.ring_buffer_size = None
        
        # Enhanced features
        self.visualizer = None
        self.brain_classifier = None
        self.music_generator = None
        self.smart_home = None
        
        # Feature flags
        self.enable_music = enable_music
        self.enable_smart_home = enable_smart_home
        
        # CSV logging
        self.csv_file = None
        self.csv_writer = None
        
        # Performance tracking
        self.stats = {
            'total_samples': 0,
            'state_changes': 0,
            'music_notes_played': 0,
            'start_time': None
        }
        
        # Initialize enhanced features
        self._initialize_enhanced_features(hue_bridge_ip)
        
    def _initialize_enhanced_features(self, hue_bridge_ip: Optional[str]):
        """Initialize all the viral features."""
        try:
            # Brain state classifier
            logger.info("Initializing AI brain state classifier...")
            self.brain_classifier = BrainStateClassifier()
            
            # 🧠 VIRAL FEATURE: Mind Reading Neural Network
            logger.info("Initializing Mind Reading Neural Network...")
            self.mind_reader = MindReaderNN()
            
            # Music generator
            if self.enable_music:
                logger.info("Initializing brain-controlled music generator...")
                self.music_generator = BrainMusicGenerator()
            
            # Smart home controller
            if self.enable_smart_home and hue_bridge_ip:
                logger.info("Initializing smart home controller...")
                self.smart_home = SmartHomeController(hue_bridge_ip)
                
        except Exception as e:
            logger.error(f"Error initializing enhanced features: {e}")
    
    def setup_board(self) -> bool:
        """
        Set up the BrainFlow board connection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Configure BrainFlow parameters
            params = BrainFlowInputParams()
            params.serial_port = self.serial_port
            
            # Create board instance
            self.board_shim = BoardShim(self.board_id, params)
            
            # Get board information
            self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
            self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
            self.timestamp_channel = BoardShim.get_timestamp_channel(self.board_id)
            
            logger.info(f"🧠 Board ID: {self.board_id}")
            logger.info(f"📊 Sampling rate: {self.sampling_rate} Hz")
            logger.info(f"🔌 EEG channels: {self.eeg_channels}")
            
            # Set up ring buffer for 10 seconds of data
            self.ring_buffer_size = int(self.sampling_rate * 10)
            
            # Prepare session
            self.board_shim.prepare_session()
            logger.info("✅ BrainFlow session prepared successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error setting up board: {e}")
            return False
    
    def setup_csv_logging(self) -> bool:
        """
        Set up enhanced CSV logging for raw data, band powers, and brain states.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.csv_file = open(self.csv_path, 'w', newline='')
            
            # Create enhanced CSV headers
            eeg_headers = [f'ch_{i+1}' for i in range(len(self.eeg_channels))]
            band_headers = []
            for ch in range(len(self.eeg_channels)):
                for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
                    band_headers.append(f'ch_{ch+1}_{band}')
            
            # Add brain state headers
            state_headers = ['brain_state', 'confidence', 'focused_prob', 'relaxed_prob', 
                           'stressed_prob', 'meditative_prob', 'excited_prob', 'drowsy_prob', 'neutral_prob']
            
            headers = ['timestamp'] + eeg_headers + band_headers + state_headers
            
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(headers)
            
            logger.info(f"📝 Enhanced CSV logging initialized: {self.csv_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error setting up CSV logging: {e}")
            return False
    
    def setup_visualization(self) -> bool:
        """
        Set up enhanced real-time visualization.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.visualizer = EEGVisualizer(self.sampling_rate)
            logger.info("🎨 Enhanced visualization initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error setting up visualization: {e}")
            return False
    
    def start_stream(self):
        """Start the EEG data stream."""
        try:
            self.board_shim.start_stream()
            self.stats['start_time'] = time.time()
            logger.info("🚀 EEG stream started with viral features enabled!")
        except Exception as e:
            logger.error(f"❌ Error starting stream: {e}")
            raise
    
    def stop_stream(self):
        """Stop the EEG data stream."""
        try:
            if self.board_shim and self.board_shim.is_prepared():
                self.board_shim.stop_stream()
                logger.info("⏹️ EEG stream stopped")
        except Exception as e:
            logger.error(f"❌ Error stopping stream: {e}")
    
    def get_new_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get new data from the board buffer.
        
        Returns:
            Tuple of (eeg_data, timestamps)
        """
        try:
            # Get new data from board
            data = self.board_shim.get_board_data()
            
            if data.shape[1] == 0:
                return np.array([]).reshape(len(self.eeg_channels), 0), np.array([])
            
            # Extract EEG channels and timestamps
            eeg_data = data[self.eeg_channels, :]
            timestamps = data[self.timestamp_channel, :]
            
            # Update stats
            self.stats['total_samples'] += data.shape[1]
            
            return eeg_data, timestamps
            
        except Exception as e:
            logger.error(f"❌ Error getting new data: {e}")
            return np.array([]).reshape(len(self.eeg_channels), 0), np.array([])
    
    def process_brain_state(self, band_powers: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Process brain state classification and trigger viral features.
        
        Args:
            band_powers: Band power features
            
        Returns:
            Tuple of (state, confidence, probabilities)
        """
        try:
            # Classify brain state
            state, confidence, probabilities = self.brain_classifier.predict_state(band_powers)
            
            # Update visualization
            if self.visualizer:
                self.visualizer.update_brain_state(state, confidence, probabilities)
            
            # 🧠 VIRAL FEATURE: Mind Reading Neural Network
            if hasattr(self, 'mind_reader') and self.mind_reader:
                # Check if we're in training mode
                if self.mind_reader.is_training_mode:
                    # Collect training sample
                    self.mind_reader.collect_training_sample(band_powers)
                    
                    # Update visualization with training progress
                    if self.visualizer:
                        training_progress = self.mind_reader.get_training_progress()
                        self.visualizer.update_mind_reading(
                            self.mind_reader.current_training_class or 'unknown',
                            0.0, {}, is_training=True, training_progress=training_progress
                        )
                else:
                    # Make mind reading prediction
                    thought, thought_confidence, thought_probabilities = self.mind_reader.predict_thought(band_powers)
                    
                    # Update visualization with prediction
                    if self.visualizer:
                        self.visualizer.update_mind_reading(thought, thought_confidence, thought_probabilities)
                    
                    # Log significant thought changes
                    if thought != self.mind_reader.last_prediction and thought_confidence > 0.7:
                        emoji = MIND_READING_EMOJIS.get(thought, '🧠')
                        logger.info(f"🧠 MIND READING: {emoji} {thought.upper()} (confidence: {thought_confidence:.1%})")
            
            # Trigger music generation
            if self.enable_music and self.music_generator and confidence > 0.6:
                self.music_generator.play_brain_music(state, band_powers)
                self.stats['music_notes_played'] += 1
            
            # Control smart home devices
            if self.enable_smart_home and self.smart_home and confidence > 0.7:
                self.smart_home.set_lights_for_brain_state(state, confidence)
            
            return state, confidence, probabilities
            
        except Exception as e:
            logger.error(f"❌ Error processing brain state: {e}")
            return 'neutral', 0.0, {}
    
    def log_to_csv(self, eeg_data: np.ndarray, band_powers: np.ndarray, 
                   timestamp: float, state: str, confidence: float, 
                   probabilities: Dict[str, float]):
        """
        Log enhanced data to CSV file including brain states.
        
        Args:
            eeg_data: Raw EEG data for current sample
            band_powers: Band power features
            timestamp: Current timestamp
            state: Predicted brain state
            confidence: Prediction confidence
            probabilities: All state probabilities
        """
        try:
            # Flatten band powers (channels x bands -> single array)
            band_powers_flat = band_powers.flatten()
            
            # Extract individual probabilities
            prob_values = [
                probabilities.get('focused', 0.0),
                probabilities.get('relaxed', 0.0),
                probabilities.get('stressed', 0.0),
                probabilities.get('meditative', 0.0),
                probabilities.get('excited', 0.0),
                probabilities.get('drowsy', 0.0),
                probabilities.get('neutral', 0.0)
            ]
            
            # Create enhanced row
            row = ([timestamp] + list(eeg_data) + list(band_powers_flat) + 
                   [state, confidence] + prob_values)
            
            self.csv_writer.writerow(row)
            self.csv_file.flush()  # Ensure data is written immediately
            
        except Exception as e:
            logger.error(f"❌ Error logging to CSV: {e}")
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics for the session."""
        try:
            elapsed_time = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
            return {
                'session_duration': elapsed_time,
                'total_samples': self.stats['total_samples'],
                'samples_per_second': self.stats['total_samples'] / elapsed_time if elapsed_time > 0 else 0,
                'state_changes': self.stats['state_changes'],
                'music_notes_played': self.stats['music_notes_played']
            }
        except Exception as e:
            logger.error(f"❌ Error getting performance stats: {e}")
            return {}
    
    def cleanup(self):
        """Clean up all resources."""
        try:
            # Stop stream
            self.stop_stream()
            
            # Release BrainFlow session
            if self.board_shim:
                self.board_shim.release_session()
                logger.info("🧠 BrainFlow session released")
            
            # Close CSV file
            if self.csv_file:
                self.csv_file.close()
                logger.info("📝 CSV file closed")
            
            # Close visualization
            if self.visualizer:
                self.visualizer.close()
                logger.info("🎨 Visualization closed")
            
            # Print final stats
            stats = self.get_performance_stats()
            logger.info("📊 Session Statistics:")
            logger.info(f"   Duration: {stats.get('session_duration', 0):.1f}s")
            logger.info(f"   Samples processed: {stats.get('total_samples', 0)}")
            logger.info(f"   Music notes played: {stats.get('music_notes_played', 0)}")
                
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")


def run_streaming_loop(streamer: OpenBCIStreamer, duration: Optional[int] = None):
    """
    Enhanced real-time processing loop with viral features running at 20 Hz.
    
    Args:
        streamer: OpenBCIStreamer instance
        duration: Optional duration in seconds to run (None for indefinite)
    """
    try:
        # Setup global board reference for signal handler
        global board_shim
        board_shim = streamer.board_shim
        
        # Processing parameters
        process_interval = 1.0 / 20  # 20 Hz processing rate
        window_duration = 1.0  # 1 second window for analysis
        window_samples = int(streamer.sampling_rate * window_duration)
        
        # Initialize ring buffer for 10 seconds of data
        ring_buffer = deque(maxlen=streamer.ring_buffer_size)
        
        # Timing variables
        start_time = time.time()
        last_process_time = start_time
        last_music_time = start_time
        iteration_count = 0
        
        # Brain state tracking
        last_state = 'neutral'
        state_change_count = 0
        
        logger.info("🚀 Starting enhanced real-time processing loop (20 Hz)")
        logger.info(f"📊 Window size: {window_samples} samples ({window_duration}s)")
        logger.info("🎵 Brain-controlled music: " + ("ENABLED" if streamer.enable_music else "DISABLED"))
        logger.info("🏠 Smart home control: " + ("ENABLED" if streamer.enable_smart_home else "DISABLED"))
        
        while True:
            current_time = time.time()
            
            # Check duration limit
            if duration and (current_time - start_time) >= duration:
                logger.info(f"⏰ Reached duration limit of {duration} seconds")
                break
            
            # Process data at 20 Hz
            if current_time - last_process_time >= process_interval:
                try:
                    # Get new data from board
                    new_eeg_data, timestamps = streamer.get_new_data()
                    
                    if new_eeg_data.shape[1] > 0:
                        # Add new samples to ring buffer
                        for i in range(new_eeg_data.shape[1]):
                            sample_data = {
                                'eeg': new_eeg_data[:, i],
                                'timestamp': timestamps[i] if len(timestamps) > i else current_time
                            }
                            ring_buffer.append(sample_data)
                    
                    # Process if we have enough data (1 second window)
                    if len(ring_buffer) >= window_samples:
                        # Extract recent 1-second window
                        recent_samples = list(ring_buffer)[-window_samples:]
                        window_eeg = np.array([sample['eeg'] for sample in recent_samples]).T
                        window_timestamps = np.array([sample['timestamp'] for sample in recent_samples])
                        
                        # Apply filters
                        filtered_eeg = apply_filters(window_eeg, streamer.sampling_rate)
                        
                        # Compute band powers
                        band_powers = compute_band_powers(filtered_eeg, streamer.sampling_rate)
                        
                        # 🧠 VIRAL FEATURE: Brain state classification
                        state, confidence, probabilities = streamer.process_brain_state(band_powers)
                        
                        # Track state changes for stats
                        if state != last_state and confidence > 0.6:
                            state_change_count += 1
                            last_state = state
                            emoji = BRAIN_STATE_EMOJIS.get(state, '🧠')
                            logger.info(f"🧠 Brain state changed: {emoji} {state.upper()} (confidence: {confidence:.1%})")
                        
                        # Update visualization with enhanced features
                        if streamer.visualizer:
                            streamer.visualizer.update_raw_signal(
                                filtered_eeg[0, -50:],  # Last 50 samples for display
                                window_timestamps[-50:]
                            )
                            streamer.visualizer.update_band_powers(
                                band_powers[0, :],  # First channel band powers
                                current_time
                            )
                            streamer.visualizer.process_events()
                        
                        # 📝 Enhanced CSV logging with brain states
                        if len(recent_samples) > 0:
                            latest_sample = recent_samples[-1]
                            streamer.log_to_csv(
                                latest_sample['eeg'],
                                band_powers,
                                latest_sample['timestamp'],
                                state,
                                confidence,
                                probabilities
                            )
                    
                    last_process_time = current_time
                    iteration_count += 1
                    
                    # Enhanced progress logging every 100 iterations (~5 seconds)
                    if iteration_count % 100 == 0:
                        elapsed = current_time - start_time
                        stats = streamer.get_performance_stats()
                        logger.info(f"📊 Processed {iteration_count} iterations in {elapsed:.1f}s")
                        logger.info(f"   Buffer size: {len(ring_buffer)} | State changes: {state_change_count}")
                        logger.info(f"   Music notes: {stats.get('music_notes_played', 0)} | Current state: {last_state}")
                
                except Exception as e:
                    logger.error(f"❌ Error in processing loop iteration: {e}")
                    time.sleep(0.1)  # Brief pause before continuing
            
            else:
                # Sleep briefly to avoid busy waiting
                time.sleep(0.001)
                
                # Still process GUI events even when not processing data
                if streamer.visualizer:
                    streamer.visualizer.process_events()
        
        # Final statistics
        final_stats = streamer.get_performance_stats()
        logger.info("🎉 Processing loop completed successfully!")
        logger.info(f"📊 Final Statistics:")
        logger.info(f"   Total iterations: {iteration_count}")
        logger.info(f"   Session duration: {final_stats.get('session_duration', 0):.1f}s")
        logger.info(f"   Samples processed: {final_stats.get('total_samples', 0)}")
        logger.info(f"   Brain state changes: {state_change_count}")
        logger.info(f"   Music notes played: {final_stats.get('music_notes_played', 0)}")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Processing loop interrupted by user")
    except Exception as e:
        logger.error(f"❌ Error in processing loop: {e}")
        raise


def parse_arguments():
    """Parse enhanced command line arguments."""
    parser = argparse.ArgumentParser(
        description='🧠 OpenBCI Cyton EEG Streaming Script - VIRAL EDITION! 🚀',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--port', 
        type=str, 
        required=True,
        help='Serial port for OpenBCI board (e.g., COM3 on Windows, /dev/ttyUSB0 on Linux)'
    )
    
    parser.add_argument(
        '--board-id', 
        type=int, 
        default=0,
        help='BrainFlow board ID (0 for OpenBCI Cyton)'
    )
    
    parser.add_argument(
        '--duration', 
        type=int, 
        default=None,
        help='Duration to run in seconds (default: run indefinitely)'
    )
    
    parser.add_argument(
        '--csv-path', 
        type=str, 
        default='./eeg_log.csv',
        help='Path to save enhanced CSV log file'
    )
    
    # Viral feature flags
    parser.add_argument(
        '--disable-music',
        action='store_true',
        help='Disable brain-controlled music generation'
    )
    
    parser.add_argument(
        '--enable-music',
        action='store_true',
        help='Enable brain-controlled music generation'
    )
    
    parser.add_argument(
        '--enable-smart-home',
        action='store_true',
        help='Enable smart home integration (requires Philips Hue bridge)'
    )
    
    parser.add_argument(
        '--hue-bridge-ip',
        type=str,
        default=None,
        help='IP address of Philips Hue bridge for smart lighting'
    )
    
    return parser.parse_args()


def main():
    """Enhanced main entry point with viral features."""
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse command line arguments
    args = parse_arguments()
    
    logger.info("=" * 80)
    logger.info("🧠 OpenBCI Cyton EEG Streaming Script - VIRAL EDITION! 🚀")
    logger.info("=" * 80)
    logger.info(f"🔌 Serial port: {args.port}")
    logger.info(f"📊 Board ID: {args.board_id}")
    logger.info(f"⏰ Duration: {args.duration if args.duration else 'indefinite'}")
    logger.info(f"📝 CSV path: {args.csv_path}")
    logger.info(f"🎵 Music generation: {'DISABLED' if args.disable_music else 'ENABLED'}")
    logger.info(f"🏠 Smart home: {'ENABLED' if args.enable_smart_home else 'DISABLED'}")
    if args.hue_bridge_ip:
        logger.info(f"💡 Hue bridge IP: {args.hue_bridge_ip}")
    logger.info("=" * 80)
    
    # Create enhanced streamer instance
    streamer = OpenBCIStreamer(
        args.board_id, 
        args.port, 
        args.csv_path,
        enable_music=args.enable_music and not args.disable_music,  # Enable only if explicitly requested and not disabled
        enable_smart_home=args.enable_smart_home,
        hue_bridge_ip=args.hue_bridge_ip
    )
    
    try:
        # Setup board connection
        logger.info("🔧 Setting up BrainFlow board connection...")
        if not streamer.setup_board():
            logger.error("❌ Failed to setup board connection")
            return 1
        
        # Setup enhanced CSV logging
        logger.info("📝 Setting up enhanced CSV logging...")
        if not streamer.setup_csv_logging():
            logger.error("❌ Failed to setup CSV logging")
            return 1
        
        # Setup enhanced visualization
        logger.info("🎨 Setting up enhanced real-time visualization...")
        if not streamer.setup_visualization():
            logger.error("❌ Failed to setup visualization")
            return 1
        
        # Start streaming
        logger.info("🚀 Starting EEG data stream with viral features...")
        streamer.start_stream()
        
        # Allow initial buffer to fill
        logger.info("⏳ Allowing initial buffer to fill (2 seconds)...")
        time.sleep(2)
        
        # Run enhanced processing loop
        run_streaming_loop(streamer, args.duration)
        
        logger.info("✅ Streaming completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        return 1
        
    finally:
        # Cleanup
        logger.info("🧹 Cleaning up resources...")
        streamer.cleanup()
        logger.info("✅ Cleanup completed")


class BrainStateClassifier:
    """AI-powered real-time brain state classification system."""
    
    def __init__(self):
        """Initialize the brain state classifier."""
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        self.training_data = []
        self.training_labels = []
        self.state_history = deque(maxlen=10)  # For smoothing predictions
        self.expected_features = None  # Will be set dynamically
        self.is_personal_model = False
        
        # Try to load personal model first
        self._load_personal_model()
    
    def _load_personal_model(self):
        """Try to load a personal brain state model."""
        try:
            import pickle
            with open('personal_brain_classifier.pkl', 'rb') as f:
                model_data = pickle.load(f)
            
            self.scaler = model_data['scaler']
            self.classifier = model_data['classifier']
            self.expected_features = len(model_data['training_data'][0])
            self.is_trained = True
            self.is_personal_model = True
            
            logger.info("🎯 Loaded personal brain state classifier!")
            logger.info(f"📊 Trained on {len(model_data['training_data'])} personal samples")
            
        except FileNotFoundError:
            logger.info("💡 No personal model found - using synthetic training data")
            logger.info("💡 Run 'python calibrate_brain_states.py --port COM3 --board-id 2' to create one!")
        except Exception as e:
            logger.warning(f"⚠️  Error loading personal model: {e}")
            logger.info("🔄 Falling back to synthetic training data")
    
    def _initialize_synthetic_model(self, n_channels: int, n_bands: int):
        """Initialize with synthetic training data for immediate functionality."""
        # DISABLED: No synthetic data generation in open source version
        # try:
        #     if self.is_personal_model:
        #         logger.info("✅ Personal model already loaded, skipping synthetic initialization")
        #         return
        #         
        #     # Generate synthetic training data based on EEG research patterns
        #     n_samples = 1000
        #     
        #     # Synthetic data for different brain states
        #     states = ['focused', 'relaxed', 'stressed', 'meditative', 'excited', 'drowsy', 'neutral']
        #     
        #     training_features = []
        #     training_labels = []
        #     
        #     for state in states:
        #         for _ in range(n_samples // len(states)):
        #             # Generate synthetic band power features
        #             features = self._generate_synthetic_features(state, n_channels, n_bands)
        #             training_features.append(features.flatten())
        #             training_labels.append(state)
        #     
        #     # Train the model
        #     X = np.array(training_features)
        #     y = np.array(training_labels)
        #     
        #     self.scaler.fit(X)
        #     X_scaled = self.scaler.transform(X)
        #     self.classifier.fit(X_scaled, y)
        #     self.is_trained = True
        #     self.expected_features = n_channels * n_bands
        #     
        #     logger.info(f"🤖 Synthetic brain state classifier initialized ({n_channels} channels, {n_bands} bands)")
        #     
        # except Exception as e:
        #     logger.error(f"Error initializing brain state classifier: {e}")
        
        logger.info("❌ Synthetic model initialization disabled - requires real EEG training data")
    
    def _generate_synthetic_features(self, state: str, n_channels: int, n_bands: int) -> np.ndarray:
        """Generate synthetic band power features for a given brain state."""
        # DISABLED: No synthetic data generation in open source version
        # features = np.random.random((n_channels, n_bands))
        # 
        # # Apply state-specific patterns based on EEG research
        # if state == 'focused':
        #     # Higher beta waves, moderate alpha
        #     features[:, min(3, n_bands-1)] *= 2.0  # Beta (if available)
        #     if n_bands > 2:
        #         features[:, 2] *= 1.2  # Alpha
        # elif state == 'relaxed':
        #     # Higher alpha waves, lower beta
        #     if n_bands > 2:
        #         features[:, 2] *= 3.0  # Alpha
        #     if n_bands > 3:
        #         features[:, 3] *= 0.5  # Beta
        # elif state == 'meditative':
        #     # Higher theta and alpha, lower beta
        #     if n_bands > 1:
        #         features[:, 1] *= 2.5  # Theta
        #     if n_bands > 2:
        #         features[:, 2] *= 2.0  # Alpha
        #     if n_bands > 3:
        #         features[:, 3] *= 0.3  # Beta
        # elif state == 'stressed':
        #     # Higher beta and gamma, irregular patterns
        #     if n_bands > 3:
        #         features[:, 3] *= 2.5  # Beta
        #     if n_bands > 4:
        #         features[:, 4] *= 1.8  # Gamma
        #     features += np.random.random(features.shape) * 0.5  # Add noise
        # elif state == 'excited':
        #     # Higher gamma and beta
        #     if n_bands > 4:
        #         features[:, 4] *= 2.2  # Gamma
        #     if n_bands > 3:
        #         features[:, 3] *= 1.8  # Beta
        # elif state == 'drowsy':
        #     # Higher delta and theta, lower beta and gamma
        #     features[:, 0] *= 2.5  # Delta
        #     if n_bands > 1:
        #         features[:, 1] *= 1.8  # Theta
        #     if n_bands > 3:
        #         features[:, 3] *= 0.4  # Beta
        #     if n_bands > 4:
        #         features[:, 4] *= 0.3  # Gamma
        # 
        # return features
        
        import numpy as np
        return np.zeros((n_channels, n_bands))  # Return empty array instead of synthetic data
    
    def predict_state(self, band_powers: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict brain state from band power features.
        
        Args:
            band_powers: Band power array (channels x bands)
            
        Returns:
            Tuple of (predicted_state, confidence, all_probabilities)
        """
        try:
            # Initialize model if not done yet
            if not self.is_trained:
                n_channels, n_bands = band_powers.shape
                self._initialize_synthetic_model(n_channels, n_bands)
            
            if not self.is_trained:
                return 'neutral', 0.0, {}
            
            # Prepare features
            features = band_powers.flatten().reshape(1, -1)
            
            # Check if feature dimensions match
            if features.shape[1] != self.expected_features:
                logger.warning(f"Feature dimension mismatch: expected {self.expected_features}, got {features.shape[1]}")
                return 'neutral', 0.0, {}
            
            features_scaled = self.scaler.transform(features)
            
            # Get prediction probabilities
            probabilities = self.classifier.predict_proba(features_scaled)[0]
            classes = self.classifier.classes_
            
            # Create probability dictionary
            prob_dict = {classes[i]: probabilities[i] for i in range(len(classes))}
            
            # Get most likely state
            predicted_state = classes[np.argmax(probabilities)]
            confidence = np.max(probabilities)
            
            # Add to history for smoothing
            self.state_history.append((predicted_state, confidence))
            
            # Smooth prediction using recent history
            if len(self.state_history) >= 3:
                smoothed_state = self._smooth_prediction()
                return smoothed_state, confidence, prob_dict
            
            return predicted_state, confidence, prob_dict
            
        except Exception as e:
            logger.error(f"Error predicting brain state: {e}")
            return 'neutral', 0.0, {}
    
    def _smooth_prediction(self) -> str:
        """Smooth predictions using recent history to reduce jitter."""
        try:
            # Count occurrences of each state in recent history
            state_counts = {}
            for state, _ in self.state_history:
                state_counts[state] = state_counts.get(state, 0) + 1
            
            # Return most common state
            return max(state_counts, key=state_counts.get)
            
        except Exception as e:
            logger.error(f"Error smoothing prediction: {e}")
            return 'neutral'
    
    def get_state_emoji(self, state: str) -> str:
        """Get emoji for brain state."""
        return BRAIN_STATE_EMOJIS.get(state, '🧠')
    
    def get_state_color(self, state: str) -> str:
        """Get color for brain state."""
        return BRAIN_STATE_COLORS.get(state, '#95A5A6')


class BrainMusicGenerator:
    """Generate music based on brain wave patterns."""
    
    def __init__(self):
        """Initialize the brain music generator."""
        self.sample_rate = 44100
        self.duration = 0.5  # 500ms notes
        self.current_notes = []
        self.is_playing = False
        
        # Musical scales for different brain states
        self.state_scales = {
            'focused': [60, 62, 64, 65, 67, 69, 71, 72],  # C major
            'relaxed': [60, 62, 63, 65, 67, 68, 70, 72],  # C minor pentatonic
            'meditative': [60, 65, 67, 72, 77, 79, 84],   # Perfect fifths
            'stressed': [60, 61, 64, 66, 67, 70, 71],     # Chromatic/dissonant
            'excited': [60, 64, 67, 72, 76, 79, 84],      # Major triads
            'drowsy': [48, 50, 52, 55, 57, 60, 62],       # Low, slow notes
            'neutral': [60, 64, 67, 71, 74, 77, 81]       # Suspended chords
        }
    
    def generate_tone(self, frequency: float, duration: float = 0.5, volume: float = 0.3) -> np.ndarray:
        """Generate a sine wave tone."""
        try:
            t = np.linspace(0, duration, int(self.sample_rate * duration))
            # Add some harmonics for richer sound
            tone = volume * (
                np.sin(2 * np.pi * frequency * t) +
                0.3 * np.sin(4 * np.pi * frequency * t) +
                0.1 * np.sin(6 * np.pi * frequency * t)
            )
            # Apply envelope to avoid clicks
            envelope = np.exp(-t * 3)
            return tone * envelope
        except Exception as e:
            logger.error(f"Error generating tone: {e}")
            return np.zeros(int(self.sample_rate * duration))
    
    def midi_to_frequency(self, midi_note: int) -> float:
        """Convert MIDI note number to frequency."""
        return 440.0 * (2 ** ((midi_note - 69) / 12.0))
    
    def brain_state_to_music(self, state: str, band_powers: np.ndarray) -> np.ndarray:
        """Convert brain state and band powers to musical notes."""
        try:
            scale = self.state_scales.get(state, self.state_scales['neutral'])
            
            # Use band powers to influence musical parameters
            alpha_power = np.mean(band_powers[:, 2])  # Alpha
            beta_power = np.mean(band_powers[:, 3])   # Beta
            
            # Select notes based on brain activity
            note_count = max(1, int(beta_power * 3))  # More beta = more notes
            selected_notes = random.sample(scale, min(note_count, len(scale)))
            
            # Generate audio
            audio = np.zeros(int(self.sample_rate * self.duration))
            for note in selected_notes:
                frequency = self.midi_to_frequency(note)
                volume = min(0.5, alpha_power * 0.8)  # Alpha influences volume
                tone = self.generate_tone(frequency, self.duration, volume)
                audio += tone
            
            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.7
                
            return audio
            
        except Exception as e:
            logger.error(f"Error generating brain music: {e}")
            return np.zeros(int(self.sample_rate * self.duration))
    
    def play_brain_music(self, state: str, band_powers: np.ndarray):
        """Play music based on brain state (non-blocking)."""
        try:
            if not self.is_playing:
                audio = self.brain_state_to_music(state, band_powers)
                threading.Thread(target=self._play_audio, args=(audio,), daemon=True).start()
        except Exception as e:
            logger.error(f"Error playing brain music: {e}")
    
    def _play_audio(self, audio: np.ndarray):
        """Play audio in separate thread."""
        try:
            self.is_playing = True
            sd.play(audio, self.sample_rate)
            sd.wait()
            self.is_playing = False
        except Exception as e:
            logger.error(f"Error in audio playback: {e}")
            self.is_playing = False


class SmartHomeController:
    """Control smart home devices based on brain states."""
    
    def __init__(self, hue_bridge_ip: Optional[str] = None):
        """Initialize smart home controller."""
        self.hue_bridge = None
        self.lights_available = False
        
        if HUE_AVAILABLE and hue_bridge_ip:
            try:
                self.hue_bridge = Bridge(hue_bridge_ip)
                self.hue_bridge.connect()
                self.lights_available = True
                logger.info("Connected to Philips Hue bridge")
            except Exception as e:
                logger.warning(f"Could not connect to Hue bridge: {e}")
    
    def set_lights_for_brain_state(self, state: str, confidence: float):
        """Set smart lights based on brain state."""
        try:
            if not self.lights_available:
                return
                
            # Map brain states to light colors
            state_colors = {
                'focused': (1.0, 0.4, 0.2),      # Orange
                'relaxed': (0.2, 0.8, 0.7),      # Teal
                'stressed': (1.0, 0.2, 0.4),     # Red
                'meditative': (0.6, 0.3, 0.9),   # Purple
                'excited': (1.0, 0.8, 0.1),      # Yellow
                'drowsy': (0.2, 0.5, 1.0),       # Blue
                'neutral': (0.5, 0.5, 0.5)       # Gray
            }
            
            rgb = state_colors.get(state, (0.5, 0.5, 0.5))
            brightness = int(confidence * 254)  # Confidence affects brightness
            
            # Convert RGB to XY color space for Hue
            xy = self._rgb_to_xy(rgb)
            
            # Set all lights
            lights = self.hue_bridge.lights
            for light in lights:
                light.brightness = brightness
                light.xy = xy
                
        except Exception as e:
            logger.error(f"Error controlling smart lights: {e}")
    
    def _rgb_to_xy(self, rgb: Tuple[float, float, float]) -> Tuple[float, float]:
        """Convert RGB to XY color space for Philips Hue."""
        try:
            r, g, b = rgb
            
            # Apply gamma correction
            r = pow((r + 0.055) / 1.055, 2.4) if r > 0.04045 else r / 12.92
            g = pow((g + 0.055) / 1.055, 2.4) if g > 0.04045 else g / 12.92
            b = pow((b + 0.055) / 1.055, 2.4) if b > 0.04045 else b / 12.92
            
            # Convert to XYZ
            X = r * 0.664511 + g * 0.154324 + b * 0.162028
            Y = r * 0.283881 + g * 0.668433 + b * 0.047685
            Z = r * 0.000088 + g * 0.072310 + b * 0.986039
            
            # Convert to xy
            x = X / (X + Y + Z) if (X + Y + Z) > 0 else 0
            y = Y / (X + Y + Z) if (X + Y + Z) > 0 else 0
            
            return (x, y)
            
        except Exception as e:
            logger.error(f"Error converting RGB to XY: {e}")
            return (0.3, 0.3)


class MindReaderNN:
    """🧠 VIRAL FEATURE: Neural Network Mind Reader! 🚀
    
    Train a neural network to recognize specific thoughts like:
    - Motor imagery (left hand vs right hand movement)
    - Mental math vs rest
    - Imagining faces vs words
    - Any custom thoughts you want to train!
    
    Perfect for viral YouTube content - "I TRAINED AN AI TO READ MY MIND!"
    """
    
    def __init__(self):
        """Initialize the mind reading neural network."""
        self.scaler = StandardScaler()
        self.neural_network = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),  # 3 hidden layers
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='adaptive',
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20
        )
        
        # Training data storage
        self.training_data = []
        self.training_labels = []
        self.is_trained = False
        self.is_training_mode = False
        self.current_training_class = None
        self.training_samples_collected = 0
        self.samples_per_class = 50  # Collect 50 samples per thought
        
        # Prediction smoothing
        self.prediction_history = deque(maxlen=10)
        self.confidence_threshold = 0.6
        
        # Available thought classes for training
        self.thought_classes = [
            'left_hand',    # Imagine moving left hand
            'right_hand',   # Imagine moving right hand  
            'rest',         # Rest/neutral state
            'math',         # Mental math (count backwards from 100)
            'music',        # Imagine your favorite song
            'face',         # Imagine a familiar face
            'word'          # Think of specific words
        ]
        
        # Training instructions for each class
        self.training_instructions = {
            'left_hand': "🤚 Imagine clenching and unclenching your LEFT HAND repeatedly",
            'right_hand': "🤚 Imagine clenching and unclenching your RIGHT HAND repeatedly", 
            'rest': "😌 Just relax and don't think of anything specific",
            'math': "🧮 Count backwards from 100 by 7s (100, 93, 86, 79...)",
            'music': "🎵 Imagine your favorite song playing in your head",
            'face': "😊 Visualize a familiar person's face in detail",
            'word': "📝 Think of the word 'ELEPHANT' and spell it mentally"
        }
        
        # Performance tracking
        self.training_accuracy = 0.0
        self.last_prediction = 'rest'
        self.last_confidence = 0.0
        
        # Try to load existing model
        self._load_trained_model()
        
        logger.info("🧠 Mind Reader Neural Network initialized!")
        logger.info(f"🎯 Available thought classes: {', '.join(self.thought_classes)}")
    
    def _load_trained_model(self):
        """Try to load a previously trained mind reading model."""
        try:
            with open('mind_reader_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            
            self.scaler = model_data['scaler']
            self.neural_network = model_data['neural_network']
            self.training_accuracy = model_data.get('accuracy', 0.0)
            self.is_trained = True
            
            logger.info("🎯 Loaded existing mind reader model!")
            logger.info(f"📊 Training accuracy: {self.training_accuracy:.1%}")
            
        except FileNotFoundError:
            logger.info("💡 No existing mind reader model found")
            logger.info("🎬 Ready to train a new model for viral content!")
        except Exception as e:
            logger.warning(f"⚠️  Error loading mind reader model: {e}")
    
    def save_trained_model(self):
        """Save the trained mind reading model."""
        try:
            model_data = {
                'scaler': self.scaler,
                'neural_network': self.neural_network,
                'accuracy': self.training_accuracy,
                'training_data': self.training_data,
                'training_labels': self.training_labels,
                'thought_classes': self.thought_classes,
                'timestamp': datetime.now().isoformat()
            }
            
            with open('mind_reader_model.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info("💾 Mind reader model saved successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error saving mind reader model: {e}")
    
    def start_training_mode(self, thought_class: str):
        """Start collecting training data for a specific thought."""
        if thought_class not in self.thought_classes:
            logger.error(f"❌ Unknown thought class: {thought_class}")
            return False
        
        self.is_training_mode = True
        self.current_training_class = thought_class
        self.training_samples_collected = 0
        
        emoji = MIND_READING_EMOJIS.get(thought_class, '🧠')
        instruction = self.training_instructions.get(thought_class, "Think about this concept")
        
        logger.info("=" * 80)
        logger.info(f"🎬 TRAINING MODE: {emoji} {thought_class.upper()} {emoji}")
        logger.info("=" * 80)
        logger.info(f"📋 INSTRUCTION: {instruction}")
        logger.info(f"🎯 Target samples: {self.samples_per_class}")
        logger.info("⏰ Starting in 3 seconds... Get ready!")
        logger.info("=" * 80)
        
        return True
    
    def stop_training_mode(self):
        """Stop training mode."""
        self.is_training_mode = False
        self.current_training_class = None
        logger.info("⏹️  Training mode stopped")
    
    def collect_training_sample(self, band_powers: np.ndarray) -> bool:
        """
        Collect a training sample during training mode.
        
        Args:
            band_powers: Current band power features
            
        Returns:
            True if sample collected, False if training complete
        """
        if not self.is_training_mode or self.current_training_class is None:
            return False
        
        # Flatten band powers to create feature vector
        features = band_powers.flatten()
        
        # Add to training data
        self.training_data.append(features)
        self.training_labels.append(self.current_training_class)
        self.training_samples_collected += 1
        
        # Progress update
        progress = self.training_samples_collected / self.samples_per_class
        emoji = MIND_READING_EMOJIS.get(self.current_training_class, '🧠')
        
        if self.training_samples_collected % 10 == 0:  # Update every 10 samples
            logger.info(f"📊 {emoji} {self.current_training_class}: {self.training_samples_collected}/{self.samples_per_class} samples ({progress:.1%})")
        
        # Check if we've collected enough samples
        if self.training_samples_collected >= self.samples_per_class:
            logger.info(f"✅ Completed training for {self.current_training_class}!")
            self.stop_training_mode()
            return False
        
        return True
    
    def train_neural_network(self):
        """Train the neural network on collected data."""
        if len(self.training_data) < 10:
            logger.error("❌ Not enough training data! Need at least 10 samples.")
            return False
        
        try:
            logger.info("🧠 Training neural network mind reader...")
            
            # Prepare data
            X = np.array(self.training_data)
            y = np.array(self.training_labels)
            
            logger.info(f"📊 Training data shape: {X.shape}")
            logger.info(f"🎯 Classes: {np.unique(y)}")
            logger.info(f"📈 Samples per class: {[np.sum(y == cls) for cls in np.unique(y)]}")
            
            # Split data for validation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train neural network
            logger.info("🚀 Training neural network...")
            self.neural_network.fit(X_train_scaled, y_train)
            
            # Evaluate performance
            train_accuracy = self.neural_network.score(X_train_scaled, y_train)
            test_accuracy = self.neural_network.score(X_test_scaled, y_test)
            
            # Make predictions for detailed metrics
            y_pred = self.neural_network.predict(X_test_scaled)
            
            self.training_accuracy = test_accuracy
            self.is_trained = True
            
            logger.info("🎉 Neural network training completed!")
            logger.info(f"📊 Training accuracy: {train_accuracy:.1%}")
            logger.info(f"🎯 Test accuracy: {test_accuracy:.1%}")
            logger.info("📋 Classification Report:")
            print(classification_report(y_test, y_pred))
            
            # Save the trained model
            self.save_trained_model()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error training neural network: {e}")
            return False
    
    def predict_thought(self, band_powers: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict what the person is thinking based on brain activity.
        
        Args:
            band_powers: Current band power features
            
        Returns:
            Tuple of (predicted_thought, confidence, all_probabilities)
        """
        if not self.is_trained:
            return 'rest', 0.0, {}
        
        try:
            # Prepare features
            features = band_powers.flatten().reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            
            # Get prediction probabilities
            probabilities = self.neural_network.predict_proba(features_scaled)[0]
            classes = self.neural_network.classes_
            
            # Create probability dictionary
            prob_dict = {classes[i]: probabilities[i] for i in range(len(classes))}
            
            # Get most likely thought
            predicted_thought = classes[np.argmax(probabilities)]
            confidence = np.max(probabilities)
            
            # Add to history for smoothing
            self.prediction_history.append((predicted_thought, confidence))
            
            # Smooth prediction if we have enough history
            if len(self.prediction_history) >= 5:
                smoothed_thought = self._smooth_thought_prediction()
                self.last_prediction = smoothed_thought
                self.last_confidence = confidence
                return smoothed_thought, confidence, prob_dict
            
            self.last_prediction = predicted_thought
            self.last_confidence = confidence
            return predicted_thought, confidence, prob_dict
            
        except Exception as e:
            logger.error(f"❌ Error predicting thought: {e}")
            return 'rest', 0.0, {}
    
    def _smooth_thought_prediction(self) -> str:
        """Smooth thought predictions to reduce jitter."""
        try:
            # Count occurrences of each thought in recent history
            thought_counts = {}
            for thought, confidence in self.prediction_history:
                if confidence > self.confidence_threshold:  # Only count confident predictions
                    thought_counts[thought] = thought_counts.get(thought, 0) + 1
            
            if thought_counts:
                return max(thought_counts, key=thought_counts.get)
            else:
                return 'rest'  # Default to rest if no confident predictions
                
        except Exception as e:
            logger.error(f"❌ Error smoothing thought prediction: {e}")
            return 'rest'
    
    def get_training_progress(self) -> Dict:
        """Get current training progress information."""
        return {
            'is_training': self.is_training_mode,
            'current_class': self.current_training_class,
            'samples_collected': self.training_samples_collected,
            'samples_needed': self.samples_per_class,
            'progress_percent': (self.training_samples_collected / self.samples_per_class * 100) if self.is_training_mode else 0,
            'total_training_samples': len(self.training_data),
            'classes_trained': list(set(self.training_labels)) if self.training_labels else [],
            'is_trained': self.is_trained,
            'training_accuracy': self.training_accuracy
        }
    
    def get_thought_emoji(self, thought: str) -> str:
        """Get emoji for thought class."""
        return MIND_READING_EMOJIS.get(thought, '🧠')
    
    def get_thought_color(self, thought: str) -> str:
        """Get color for thought class."""
        return MIND_READING_COLORS.get(thought, '#95A5A6')
    
    def reset_training_data(self):
        """Reset all training data (use with caution!)."""
        self.training_data = []
        self.training_labels = []
        self.is_trained = False
        self.training_accuracy = 0.0
        logger.info("🔄 Training data reset")


if __name__ == "__main__":
    sys.exit(main()) 