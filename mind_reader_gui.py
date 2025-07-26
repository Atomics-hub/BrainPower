#!/usr/bin/env python3
"""
🧠 MIND READER GUI - VIRAL NEURAL NETWORK VISUALIZATION! 🚀

Real-time GUI showing:
- Neural network architecture and learning
- Live training progress with animations
- Real-time mind reading predictions
- "AI thinking" visual effects

Perfect for viral YouTube content!
"""

import sys
import time
import threading
from typing import Optional, Dict, List
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import QTimer, pyqtSignal, QThread, QPointF
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import logging
logger = logging.getLogger(__name__)

class TrainingThread(QThread):
    """Thread to run neural network training without blocking the GUI."""
    
    # Signals for updating GUI
    progress_updated = pyqtSignal(str, int, int)  # thought_class, samples_collected, total_samples
    training_complete = pyqtSignal(float)  # accuracy
    status_updated = pyqtSignal(str)  # status message
    prediction_updated = pyqtSignal(str, float, dict)  # thought, confidence, probabilities
    band_power_updated = pyqtSignal(np.ndarray)  # band_powers
    neural_network_updated = pyqtSignal(str, float, np.ndarray, str)  # mode, progress/confidence, band_powers, thought
    eeg_data_updated = pyqtSignal(np.ndarray)  # NEW: raw EEG data for live channels display
    
    def __init__(self, port: str, board_id: int = 2):
        super().__init__()
        self.port = port
        self.board_id = board_id
        self.is_training = False
        self.is_testing = False
        self.is_quick_training = False  # Add quick training flag
        self.streamer = None
        self.data_buffer = deque(maxlen=1000)  # Buffer for accumulating data
    
    def run_training(self):
        """Start training mode."""
        self.is_training = True
        self.is_testing = False
    
    def run_testing(self):
        """Start testing mode."""
        self.is_training = False
        self.is_testing = True
    
    def run_quick_training(self):
        """Start quick training mode for testing."""
        self.is_training = False
        self.is_testing = False
        self.is_quick_training = True
    
    def run(self):
        """Main thread execution."""
        try:
            # DEBUG: Print thread state
            print(f"🔍 THREAD DEBUG: is_testing={self.is_testing}, board_id={self.board_id}, port={self.port}")
            
            # Check if we're doing synthetic testing (bypass BrainFlow)
            if self.is_testing and (self.board_id == -1 or self.port == "COM999"):
                print("🎮 TAKING SYNTHETIC TESTING PATH!")
                self.status_updated.emit("🎮 Running synthetic mind reading test...")
                self._run_testing_sequence()  # This will handle synthetic testing
                return
            
            print("🔧 TAKING REAL BRAINFLOW PATH")
            
            # Regular BrainFlow path for real data
            # Import here to avoid circular imports
            from openbci_stream import OpenBCIStreamer, apply_filters, compute_band_powers
            
            # Create streamer
            self.status_updated.emit("🔧 Connecting to OpenBCI...")
            self.streamer = OpenBCIStreamer(self.board_id, self.port, './gui_training.csv')
            
            # Setup board
            if not self.streamer.setup_board():
                self.status_updated.emit("❌ Failed to connect to board")
                return
            
            # Start streaming
            self.streamer.start_stream()
            self.status_updated.emit("🚀 EEG stream started!")
            
            # Allow buffer to fill
            time.sleep(3)
            
            if self.is_training:
                self._run_training_sequence()
            elif self.is_testing:
                self._run_testing_sequence()
            elif self.is_quick_training:
                self._run_training_sequence(quick_mode=True)
                
        except Exception as e:
            self.status_updated.emit(f"❌ Error: {e}")
        finally:
            if hasattr(self, 'streamer') and self.streamer:
                self.streamer.cleanup()
    
    def _accumulate_data(self, timeout_seconds=5):
        """Accumulate data until we have enough for processing."""
        print(f"🔄 _accumulate_data() called with timeout={timeout_seconds}s")
        
        # Check if we're in synthetic mode (no real streamer)
        if not hasattr(self, 'streamer') or self.streamer is None or self.board_id == -1:
            print("🎮 SYNTHETIC MODE: Generating synthetic EEG data...")
            # Generate synthetic EEG data for training
            synthetic_data = self._generate_synthetic_eeg_data()
            # Emit for live display
            self.eeg_data_updated.emit(synthetic_data)
            return synthetic_data
        
        start_time = time.time()
        target_samples = int(self.streamer.sampling_rate * 0.8)  # 0.8 seconds of data
        
        print(f"🎯 Target: {target_samples} samples ({self.streamer.sampling_rate} Hz)")
        print(f"📊 Current buffer size: {len(self.data_buffer)}")
        
        while time.time() - start_time < timeout_seconds:
            try:
                new_eeg_data, timestamps = self.streamer.get_new_data()
                
                if new_eeg_data is not None and new_eeg_data.shape[1] > 0:
                    print(f"📥 Got {new_eeg_data.shape[1]} new samples from streamer")
                    
                    # Emit raw EEG data for live display
                    self.eeg_data_updated.emit(new_eeg_data)
                    
                    # Add each sample to buffer
                    for i in range(new_eeg_data.shape[1]):
                        sample_data = {
                            'eeg': new_eeg_data[:, i],
                            'timestamp': timestamps[i] if len(timestamps) > i else time.time()
                        }
                        self.data_buffer.append(sample_data)
                    
                    print(f"📊 Buffer now has {len(self.data_buffer)} samples")
                    
                    # Check if we have enough data
                    if len(self.data_buffer) >= target_samples:
                        # Extract recent data
                        recent_samples = list(self.data_buffer)[-target_samples:]
                        window_eeg = np.array([sample['eeg'] for sample in recent_samples]).T
                        print(f"✅ SUCCESS: Returning {window_eeg.shape[1]} samples")
                        return window_eeg
                    
                    status_msg = f"📊 Accumulating data: {len(self.data_buffer)}/{target_samples} samples"
                    print(status_msg)
                    self.status_updated.emit(status_msg)
                else:
                    print("⚠️ No new data from streamer")
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                error_msg = f"❌ Error accumulating data: {e}"
                print(error_msg)
                self.status_updated.emit(error_msg)
                time.sleep(0.1)
        
        # Return whatever we have if timeout reached, or synthetic data as fallback
        if len(self.data_buffer) > 50:  # At least 50 samples (0.2 seconds)
            recent_samples = list(self.data_buffer)
            window_eeg = np.array([sample['eeg'] for sample in recent_samples]).T
            print(f"⏰ TIMEOUT: Returning {window_eeg.shape[1]} samples (not ideal)")
            # Emit for live display
            self.eeg_data_updated.emit(window_eeg)
            return window_eeg
        
        print(f"❌ FAILED: Only {len(self.data_buffer)} samples after timeout, generating synthetic data...")
        synthetic_data = self._generate_synthetic_eeg_data()
        # Emit for live display
        self.eeg_data_updated.emit(synthetic_data)
        return synthetic_data
    
    def _generate_synthetic_eeg_data(self):
        """Generate synthetic EEG data for training when real data isn't available."""
        # Create realistic synthetic EEG data (16 channels, ~100 samples)
        num_channels = 16
        num_samples = 100  # About 0.8 seconds at 125 Hz
        
        # Generate realistic EEG patterns
        synthetic_eeg = np.random.normal(0, 50, (num_channels, num_samples))  # Baseline noise
        
        # Add some realistic frequency components
        time_vec = np.linspace(0, 0.8, num_samples)
        
        for ch in range(num_channels):
            # Add alpha waves (8-12 Hz)
            alpha_freq = 10 + np.random.uniform(-2, 2)
            alpha_amplitude = np.random.uniform(20, 80)
            synthetic_eeg[ch, :] += alpha_amplitude * np.sin(2 * np.pi * alpha_freq * time_vec)
            
            # Add beta waves (13-30 Hz) 
            beta_freq = 20 + np.random.uniform(-5, 5)
            beta_amplitude = np.random.uniform(10, 40)
            synthetic_eeg[ch, :] += beta_amplitude * np.sin(2 * np.pi * beta_freq * time_vec)
            
            # Add some random spikes
            for spike in range(np.random.randint(0, 3)):
                spike_pos = np.random.randint(0, num_samples)
                spike_width = np.random.randint(3, 8)
                spike_amp = np.random.uniform(50, 150)
                start_idx = max(0, spike_pos - spike_width)
                end_idx = min(num_samples, spike_pos + spike_width)
                synthetic_eeg[ch, start_idx:end_idx] += spike_amp
        
        print(f"🎮 Generated synthetic EEG: {synthetic_eeg.shape}")
        return synthetic_eeg
    
    def _run_training_sequence(self, quick_mode=False):
        """Run the complete training sequence."""
        try:
            print("\n" + "="*60)
            if quick_mode:
                print("🚀 STARTING QUICK TRAINING SEQUENCE")
                # Quick training: fewer samples but still reasonable
                focused_classes = ['left_hand', 'right_hand', 'rest']
                samples_per_class = 15  # Increased from 3 to 15
                collection_time = 8     # Reduced from 15 to 8 seconds
            else:
                print("🧠 STARTING FULL TRAINING SEQUENCE")
                focused_classes = ['left_hand', 'right_hand', 'rest']
                samples_per_class = 50
                collection_time = 15
            
            print(f"🎯 Classes: {focused_classes}")
            print(f"📊 Samples per class: {samples_per_class}")
            print(f"⏱️  Collection time per class: {collection_time}s")
            print("="*60)
            
            # Get mind_reader from streamer
            mind_reader = self.streamer.mind_reader if self.streamer else None
            if not mind_reader:
                # If no streamer (synthetic mode), create a mind reader
                from openbci_stream import MindReaderNN
                mind_reader = MindReaderNN()
            
            # Reset training samples storage
            mind_reader.training_samples = {}
            
            # Training phase
            for i, thought_class in enumerate(focused_classes):
                self.status_updated.emit(f"🎯 Phase {i+1}/{len(focused_classes)}: Training {thought_class}")
                
                instructions = {
                    'left_hand': "🤚 Imagine clenching and unclenching your LEFT HAND repeatedly",
                    'right_hand': "🤚 Imagine clenching and unclenching your RIGHT HAND repeatedly", 
                    'rest': "😌 Just relax and don't think of anything specific"
                }
                
                instruction = instructions.get(thought_class, f"Think about {thought_class}")
                
                # Show instruction
                self.status_updated.emit(f"📋 INSTRUCTION: {instruction}")
                
                # Countdown
                for j in range(3, 0, -1):
                    self.status_updated.emit(f"⏰ Starting in {j}...")
                    time.sleep(1)
                
                # Data collection
                self.status_updated.emit(f"🔴 COLLECTING {thought_class.upper()} DATA...")
                
                samples_collected = 0
                collection_start = time.time()
                mind_reader.training_samples[thought_class] = []
                
                # Calculate timing: aim for one sample every 0.5 seconds
                sample_interval = collection_time / samples_per_class
                next_sample_time = collection_start
                
                print(f"🎯 Collection plan: {samples_per_class} samples over {collection_time}s")
                print(f"⏱️  Sample interval: {sample_interval:.1f}s per sample")
                
                while (time.time() - collection_start < collection_time and 
                       samples_collected < samples_per_class):
                    
                    current_time = time.time()
                    
                    # Wait until it's time for the next sample
                    if current_time >= next_sample_time:
                        # Get data (synthetic or real)
                        window_data = self._accumulate_data(timeout_seconds=2)
                        
                        if window_data is not None:
                            # Process the data
                            from openbci_stream import apply_filters, compute_band_powers
                            filtered_data = apply_filters(window_data, 125)  # Use standard 125 Hz
                            band_powers = compute_band_powers(filtered_data, 125)
                            
                            # Store sample (flatten to create feature vector)
                            features = band_powers.flatten()
                            mind_reader.training_samples[thought_class].append(features)
                            samples_collected += 1
                            
                            # Update progress
                            progress = (samples_collected / samples_per_class) * 100
                            self.progress_updated.emit(thought_class, samples_collected, samples_per_class)
                            self.neural_network_updated.emit("training", progress, band_powers, thought_class)
                            
                            self.status_updated.emit(f"📊 {thought_class}: {samples_collected}/{samples_per_class} samples ({progress:.0f}%)")
                            print(f"✅ Sample {samples_collected}: {thought_class} collected")
                            
                            # Schedule next sample
                            next_sample_time = current_time + sample_interval
                        else:
                            print("⚠️ Failed to get data, retrying...")
                            next_sample_time = current_time + 1  # Retry in 1 second
                    
                    # Small sleep to prevent busy waiting
                    time.sleep(0.1)
                    
                    # Update status every few seconds
                    elapsed = time.time() - collection_start
                    if int(elapsed) % 2 == 0 and elapsed > 0:  # Every 2 seconds
                        remaining = max(0, collection_time - elapsed)
                        self.status_updated.emit(f"⏱️  {thought_class}: {samples_collected}/{samples_per_class} samples - {remaining:.0f}s remaining")
                
                final_count = len(mind_reader.training_samples[thought_class])
                self.status_updated.emit(f"✅ {thought_class}: {final_count} samples collected")
                print(f"✅ Collected {final_count} samples for {thought_class}")
                
                # Brief pause between classes
                if i < len(focused_classes) - 1:
                    for countdown in range(10, 0, -1):
                        self.status_updated.emit(f"⏸️  Break: {countdown} seconds until next class...")
                        time.sleep(1)
            
            # Training phase
            self.status_updated.emit("🧠 Training neural network...")
            
            try:
                # Prepare training data
                X_train = []
                y_train = []
                
                for class_idx, thought_class in enumerate(focused_classes):
                    if thought_class in mind_reader.training_samples:
                        samples = mind_reader.training_samples[thought_class]
                        print(f"📊 {thought_class}: {len(samples)} samples")
                        
                        for sample in samples:
                            X_train.append(sample)
                            y_train.append(thought_class)  # Use string labels, not indices
                
                if len(X_train) > 0:
                    X_train = np.array(X_train)
                    y_train = np.array(y_train)
                    
                    print(f"🔧 Training data shape: {X_train.shape}")
                    self.status_updated.emit(f"🔧 Training data shape: {X_train.shape}")
                    
                    print("🧠 Starting neural network training...")
                    
                    # IMPROVED MODEL PARAMETERS for small dataset
                    from sklearn.neural_network import MLPClassifier
                    from sklearn.preprocessing import StandardScaler
                    
                    # Create scaler
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    
                    # Better model for small datasets
                    improved_model = MLPClassifier(
                        hidden_layer_sizes=(32, 16),    # Smaller network for small data
                        activation='relu',
                        alpha=0.1,                      # Strong regularization to prevent overfitting
                        early_stopping=True,
                        validation_fraction=0.3,        # More validation data
                        max_iter=2000,                  # More iterations
                        learning_rate_init=0.001,       # Lower learning rate
                        random_state=42,
                        solver='adam'
                    )
                    
                    # Train the model
                    print("🚀 Training improved model...")
                    improved_model.fit(X_train_scaled, y_train)
                    
                    # Update the mind_reader with BOTH the model AND scaler
                    mind_reader.neural_network = improved_model
                    mind_reader.scaler = scaler  
                    mind_reader.is_trained = True
                    
                    print("✅ Neural network training completed!")
                    
                    # Calculate accuracy using cross-validation for better estimate
                    from sklearn.model_selection import cross_val_score
                    if len(X_train_scaled) >= 5:  # Only if we have enough samples
                        cv_scores = cross_val_score(improved_model, X_train_scaled, y_train, cv=min(3, len(X_train_scaled)//2))
                        accuracy = cv_scores.mean()
                        accuracy_std = cv_scores.std()
                        print(f"🎯 Cross-validation Accuracy: {accuracy:.1%} ± {accuracy_std:.1%}")
                    else:
                        # Fallback to training accuracy for very small datasets
                        y_pred = improved_model.predict(X_train_scaled)
                        accuracy = np.mean(y_pred == y_train)
                        print(f"🎯 Training Accuracy: {accuracy:.1%} (Warning: Small dataset)")
                    
                    mind_reader.training_accuracy = accuracy
                    
                    # Convert training_samples to the format expected by save_trained_model()
                    print("🔄 Converting training data format for saving...")
                    
                    # Convert training_samples dict to flat lists
                    mind_reader.training_data = []
                    mind_reader.training_labels = []
                    
                    for thought_class in focused_classes:
                        if thought_class in mind_reader.training_samples:
                            samples = mind_reader.training_samples[thought_class]
                            print(f"📊 Converting {len(samples)} samples for {thought_class}")
                            
                            for sample in samples:
                                features = sample.flatten() if hasattr(sample, 'flatten') else sample
                                mind_reader.training_data.append(features)
                                mind_reader.training_labels.append(thought_class)  
                    
                    print(f"✅ Converted to {len(mind_reader.training_data)} total samples")
                    print(f"📊 Classes: {set(mind_reader.training_labels)}")
                    
                    # Save the model
                    print("💾 Saving trained model...")
                    mind_reader.save_trained_model()
                    
                    self.training_complete.emit(accuracy)
                    self.status_updated.emit(f"🎉 Training completed! Accuracy: {accuracy:.1%}")
                    
                    # Give guidance based on accuracy and dataset size
                    total_samples = len(mind_reader.training_data)
                    if total_samples < 30:
                        guidance = "⚠️ SMALL DATASET - Collect more data for better accuracy"
                    elif accuracy >= 0.8:
                        guidance = "🏆 EXCELLENT! Ready for viral content!"
                    elif accuracy >= 0.6:
                        guidance = "✅ GOOD! Ready for testing!"
                    elif accuracy >= 0.4:
                        guidance = "⚠️ OK - Try retraining with more focus"
                    else:
                        guidance = "🔄 LOW - Need more training data"
                    
                    print(guidance)
                    self.status_updated.emit(guidance)
                        
                else:
                    error_msg = "❌ No training data collected!"
                    print(error_msg)
                    self.status_updated.emit(error_msg)
                    
            except Exception as e:
                error_msg = f"❌ Training failed: {e}"
                print(error_msg)
                self.status_updated.emit(error_msg)
                
        except Exception as e:
            error_msg = f"❌ Training sequence error: {e}"
            print(error_msg)
            self.status_updated.emit(error_msg)
            
        print("\n" + "="*60)
        print("🎉 TRAINING SEQUENCE COMPLETED")
        print("="*60)
    
    def _run_testing_sequence(self):
        """Run testing with live predictions."""
        mind_reader = self.streamer.mind_reader if self.streamer else None
        
        # Check if we're using synthetic testing
        is_synthetic = (self.board_id == -1 or self.port == "COM999")
        
        # For synthetic testing, we don't need the actual streamer
        if is_synthetic:
            # Load the trained model directly without streamer dependency
            from openbci_stream import MindReaderNN
            if not mind_reader:
                mind_reader = MindReaderNN()
            
            if not mind_reader.is_trained:
                mind_reader._load_trained_model()
                
            if not mind_reader.is_trained:
                self.status_updated.emit("❌ No trained model found! Please train a model first.")
                return
                
            self.status_updated.emit("🎮 Starting SYNTHETIC mind reading test...")
            
            # Run synthetic testing loop
            self._run_synthetic_testing_loop(mind_reader)
            return
        
        # Original real data testing path
        if not mind_reader.is_trained:
            mind_reader._load_trained_model()
        
        if not mind_reader.is_trained:
            self.status_updated.emit("❌ No trained model found! Please train a model first.")
            return
        
        self.status_updated.emit("🧠 Starting mind reading test...")
        
        test_duration = 60  # 60 seconds
        start_time = time.time()
        
        while time.time() - start_time < test_duration:
            # Accumulate real data for prediction
            window_data = self._accumulate_data(timeout_seconds=3)
            
            if window_data is not None:
                try:
                    # Process data
                    from openbci_stream import apply_filters, compute_band_powers
                    filtered_data = apply_filters(window_data, self.streamer.sampling_rate)
                    band_powers = compute_band_powers(filtered_data, self.streamer.sampling_rate)
                    
                    # Update live band power display
                    self.band_power_updated.emit(band_powers)
                    
                    # DEBUG: Show actual brain patterns
                    print(f"🧠 REAL BRAIN PATTERNS:")
                    print(f"   Motor channels (C3,Cz,C4): {band_powers[8:11, :].mean(axis=0)}")
                    print(f"   Alpha power: {band_powers[:, 2].mean():.2f}")
                    print(f"   Beta power: {band_powers[:, 3].mean():.2f}")
                    print(f"   Overall pattern: {band_powers.flatten()[:10]}")  # First 10 features
                    
                    # Make prediction
                    thought, confidence, probabilities = mind_reader.predict_thought(band_powers)
                    
                    # FIX: Map class indices back to class names
                    focused_classes = ['left_hand', 'right_hand', 'rest']
                    if isinstance(thought, (int, np.integer)):
                        # Convert index to class name
                        if 0 <= thought < len(focused_classes):
                            thought_name = focused_classes[thought]
                        else:
                            thought_name = 'rest'  # Default fallback
                    else:
                        thought_name = str(thought)
                    
                    # Fix probabilities dictionary to use class names
                    fixed_probabilities = {}
                    for i, prob in probabilities.items():
                        if isinstance(i, (int, np.integer)) and 0 <= i < len(focused_classes):
                            fixed_probabilities[focused_classes[i]] = prob
                        else:
                            fixed_probabilities[str(i)] = prob
                    
                    # DEBUG: Print prediction info
                    print(f"🧠 PREDICTION: {thought_name} (confidence: {confidence:.2f})")
                    print(f"📊 Probabilities: {fixed_probabilities}")
                    
                    # Update neural network visualization with prediction
                    self.neural_network_updated.emit("prediction", confidence, band_powers, thought_name)
                    
                    # Update GUI
                    self.prediction_updated.emit(thought_name, confidence, fixed_probabilities)
                    
                    # Update status with current prediction
                    self.status_updated.emit(f"🧠 (real) Predicting: {thought_name} ({confidence:.1%} confidence)")
                    
                except Exception as e:
                    self.status_updated.emit(f"❌ Error making prediction: {e}")
            
            time.sleep(1)  # Update every second
        
        self.status_updated.emit("✅ Testing completed!")
    
    def _run_synthetic_testing_loop(self, mind_reader):
        """Run synthetic testing loop with varied data patterns."""
        test_duration = 60  # 60 seconds
        start_time = time.time()
        prediction_count = 0
        
        print("🎮 STARTING SYNTHETIC TESTING LOOP")
        print("🎮 This will cycle through different brain patterns...")
        
        while time.time() - start_time < test_duration:
            try:
                # Generate realistic VARIED synthetic data for proper testing
                band_powers = self._generate_varied_synthetic_data(prediction_count)
                
                # Emit synthetic band powers for visualization
                self.band_power_updated.emit(band_powers)
                
                # Make prediction with synthetic data
                thought, confidence, probabilities = mind_reader.predict_thought(band_powers)
                
                # FIX: Map class indices back to class names
                focused_classes = ['left_hand', 'right_hand', 'rest']
                if isinstance(thought, (int, np.integer)):
                    # Convert index to class name
                    if 0 <= thought < len(focused_classes):
                        thought_name = focused_classes[thought]
                    else:
                        thought_name = 'rest'  # Default fallback
                else:
                    thought_name = str(thought)
                
                # Fix probabilities dictionary to use class names
                fixed_probabilities = {}
                for i, prob in probabilities.items():
                    if isinstance(i, (int, np.integer)) and 0 <= i < len(focused_classes):
                        fixed_probabilities[focused_classes[i]] = prob
                    else:
                        fixed_probabilities[str(i)] = prob
                
                # DEBUG: Print prediction info
                print(f"🎮 SYNTHETIC PREDICTION: {thought_name} (confidence: {confidence:.2f})")
                print(f"🎮 SYNTHETIC Probabilities: {fixed_probabilities}")
                
                # Update neural network visualization with prediction
                self.neural_network_updated.emit("prediction", confidence, band_powers, thought_name)
                
                # Update GUI
                self.prediction_updated.emit(thought_name, confidence, fixed_probabilities)
                
                # Update status with current prediction
                self.status_updated.emit(f"🎮 (synthetic) Predicting: {thought_name} ({confidence:.1%} confidence)")
                
                prediction_count += 1
                
            except Exception as e:
                error_msg = f"❌ Error making synthetic prediction: {e}"
                print(error_msg)
                self.status_updated.emit(error_msg)
            
            time.sleep(2)  # Update every 2 seconds for better visualization
        
        self.status_updated.emit("✅ Synthetic testing completed!")
    
    def _generate_varied_synthetic_data(self, prediction_count: int) -> np.ndarray:
        """Generate varied synthetic EEG band power data for realistic testing."""
        np.random.seed(int(time.time() * 1000) % 10000)  # Time-based seed for variation
        
        # Create 16 channels × 5 bands array
        band_powers = np.zeros((16, 5))
        
        # Generate different patterns based on time/count to simulate different mental states
        pattern_type = prediction_count % 3  # Cycle through 3 different patterns
        
        if pattern_type == 0:  # "left_hand" pattern
            # Higher beta activity in motor cortex areas (C3, C4, Cz)
            motor_channels = [8, 9, 10]  # C3, Cz, C4 indices
            for ch in motor_channels:
                band_powers[ch, 3] = np.random.uniform(15, 25)  # High beta
                band_powers[ch, 2] = np.random.uniform(5, 10)   # Low alpha (ERD)
            
            # Background activity for other channels
            for ch in range(16):
                if ch not in motor_channels:
                    band_powers[ch, :] = np.random.uniform(2, 8, 5)
                    
        elif pattern_type == 1:  # "right_hand" pattern  
            # Similar to left_hand but slightly different pattern
            motor_channels = [8, 9, 10]  # C3, Cz, C4
            for ch in motor_channels:
                band_powers[ch, 3] = np.random.uniform(12, 22)  # High beta
                band_powers[ch, 2] = np.random.uniform(3, 8)    # Low alpha
                band_powers[ch, 4] = np.random.uniform(3, 6)    # Some gamma
            
            # Background activity
            for ch in range(16):
                if ch not in motor_channels:
                    band_powers[ch, :] = np.random.uniform(1, 7, 5)
                    
        else:  # "rest" pattern
            # More relaxed pattern with higher alpha
            for ch in range(16):
                band_powers[ch, 0] = np.random.uniform(3, 8)   # Delta
                band_powers[ch, 1] = np.random.uniform(4, 10)  # Theta  
                band_powers[ch, 2] = np.random.uniform(10, 20) # High alpha (relaxed)
                band_powers[ch, 3] = np.random.uniform(2, 6)   # Low beta
                band_powers[ch, 4] = np.random.uniform(1, 4)   # Low gamma
        
        # Add some random noise to make it more realistic
        noise = np.random.uniform(-1, 1, band_powers.shape)
        band_powers += noise
        
        # Ensure positive values
        band_powers = np.maximum(band_powers, 0.1)
        
        print(f"🎮 Generated synthetic pattern {pattern_type} - variation {prediction_count}")
        
        return band_powers

class NeuralNetworkWidget(QWidget):
    """Widget to visualize the neural network structure and LIVE activity."""
    
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(300)
        self.setStyleSheet("background-color: #2a2a2a; border: 2px solid #00ffff; border-radius: 10px;")
        
        # Network structure (matches actual MLPClassifier for 16 channels)
        self.input_nodes = 80  # Band power features (16 channels × 5 bands)
        self.hidden_nodes = [64, 32]  # Hidden layers 
        self.output_nodes = 3  # Thought classes (left_hand, right_hand, rest)
        
        # Animation and activity variables
        self.neuron_activations = {}
        self.connection_weights = {}
        self.is_thinking = False
        self.is_training = False
        self.current_prediction = None
        self.training_progress = 0.0
        
        # Animation state
        self.animation_frame = 0
        self.data_flow_positions = []
        self.pulse_intensities = {}
        
        # Initialize activation values
        self.reset_activations()
        
        # Timer for smooth animations (reduced frequency to prevent conflicts)
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(50)  # 20 FPS instead of 60 FPS to reduce conflicts
    
    def reset_activations(self):
        """Reset all neuron activations to baseline."""
        layers = [self.input_nodes] + self.hidden_nodes + [self.output_nodes]
        
        for layer_idx, layer_size in enumerate(layers):
            layer_name = f"layer_{layer_idx}"
            self.neuron_activations[layer_name] = np.random.uniform(0.1, 0.3, layer_size)
            self.pulse_intensities[layer_name] = np.zeros(layer_size)
    
    def paintEvent(self, event):
        """Draw the live neural network."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        try:
            # Draw title with activity indicator
            title_color = QColor("#00ffff") if not self.is_thinking else QColor("#ff00ff")
            painter.setPen(QPen(title_color, 2))
            painter.setFont(QFont("Arial", 14, QFont.Bold))
            
            status = ""
            if self.is_training:
                status = f" - TRAINING {self.training_progress:.0f}%"
            elif self.is_thinking:
                status = " - ACTIVE"
            elif self.current_prediction:
                status = f" - PREDICTING: {self.current_prediction}"
            
            painter.drawText(10, 25, f"🧠 LIVE NEURAL NETWORK{status}")
            
            # Draw the network
            self.draw_live_network(painter)
            
        except Exception as e:
            print(f"Error in paintEvent: {e}")
        finally:
            # Ensure painter is properly ended
            painter.end()
    
    def draw_live_network(self, painter):
        """Draw the neural network with live animations."""
        width = self.width() - 40
        height = self.height() - 60
        
        # Layer configuration
        layers = [self.input_nodes, self.hidden_nodes[0], self.hidden_nodes[1], self.output_nodes]
        layer_names = ["INPUT\n(EEG)", "HIDDEN 1\n(64)", "HIDDEN 2\n(32)", "OUTPUT\n(3)"]
        layer_positions = []
        
        # Calculate layer positions
        for i in range(len(layers)):
            x = 40 + (i * (width - 80) // (len(layers) - 1))
            layer_positions.append(x)
        
        # Draw connections first (behind neurons)
        self.draw_connections(painter, layers, layer_positions, height)
        
        # Draw neurons
        self.draw_neurons(painter, layers, layer_names, layer_positions, height)
        
        # Draw data flow animation
        self.draw_data_flow(painter, layer_positions, height)
    
    def draw_neurons(self, painter, layers, layer_names, layer_positions, height):
        """Draw animated neurons with activity levels."""
        for layer_idx, (layer_size, layer_name) in enumerate(zip(layers, layer_names)):
            x = layer_positions[layer_idx]
            display_size = min(layer_size, 8)  # Limit display neurons
            
            for i in range(display_size):
                y = 60 + (i * (height - 20) // display_size)
                
                # Get neuron activation
                activation = self.get_neuron_activation(layer_idx, i)
                pulse = self.get_neuron_pulse(layer_idx, i)
                
                # Neuron size based on activation (ensure integers)
                radius = int(8 + activation * 12 + pulse * 5)
                
                # Color based on activation and layer
                if layer_idx == 0:  # Input layer
                    base_color = QColor(0, 255, 0)  # Green
                elif layer_idx == len(layers) - 1:  # Output layer
                    base_color = QColor(255, 0, 255)  # Magenta
                else:  # Hidden layers
                    base_color = QColor(0, 255, 255)  # Cyan
                
                # Intensity based on activation
                intensity = int(100 + activation * 155)
                color = QColor(
                    int(base_color.red() * intensity / 255),
                    int(base_color.green() * intensity / 255),
                    int(base_color.blue() * intensity / 255),
                    200
                )
                
                # Draw neuron with glow effect (convert to integers)
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(QColor(255, 255, 255, 150), 1))
                painter.drawEllipse(int(x - radius), int(y - radius), 
                                  int(radius * 2), int(radius * 2))
                
                # Draw pulse effect for highly active neurons
                if pulse > 0.3:
                    pulse_color = QColor(255, 255, 255, int(pulse * 100))
                    painter.setBrush(QBrush(pulse_color))
                    painter.setPen(QPen(pulse_color, 1))
                    pulse_radius = int(radius + pulse * 10)
                    painter.drawEllipse(int(x - pulse_radius), int(y - pulse_radius), 
                                      int(pulse_radius * 2), int(pulse_radius * 2))
            
            # Draw layer label (convert coordinates to integers)
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.setFont(QFont("Arial", 9))
            painter.drawText(int(x - 25), int(height + 40), layer_name)
            
            # Show layer info
            if layer_size > 8:
                painter.setFont(QFont("Arial", 7))
                painter.drawText(int(x - 15), int(height + 55), f"({layer_size} total)")
    
    def draw_connections(self, painter, layers, layer_positions, height):
        """Draw animated connections between neurons."""
        for layer_idx in range(len(layers) - 1):
            current_layer_size = min(layers[layer_idx], 8)  # Limit display
            next_layer_size = min(layers[layer_idx + 1], 8)
            
            current_x = layer_positions[layer_idx]
            next_x = layer_positions[layer_idx + 1]
            
            for i in range(current_layer_size):
                current_y = 60 + (i * (height - 20) // current_layer_size)
                
                for j in range(next_layer_size):
                    next_y = 60 + (j * (height - 20) // next_layer_size)
                    
                    # Get connection strength (simulated)
                    strength = self.get_connection_strength(layer_idx, i, j)
                    
                    # Color based on strength and activity
                    if strength > 0.5:
                        color = QColor(int(255 * strength), int(255 * strength), 0, 100)  # Yellow
                    else:
                        color = QColor(0, int(255 * strength), int(255 * strength), 60)   # Cyan
                    
                    # Draw line with integer coordinates
                    painter.setPen(QPen(color, max(1, int(strength * 3))))
                    painter.drawLine(int(current_x + 15), int(current_y), 
                                   int(next_x - 15), int(next_y))
    
    def draw_data_flow(self, painter, layer_positions, height):
        """Draw animated data flowing through the network."""
        if not self.is_thinking and not self.is_training:
            return
        
        # Create flowing particles
        for flow in self.data_flow_positions:
            x, y, progress, layer = flow
            
            if layer < len(layer_positions) - 1:
                start_x = layer_positions[layer]
                end_x = layer_positions[layer + 1]
                
                current_x = start_x + (end_x - start_x) * progress
                
                # Particle color
                color = QColor(255, 255, 0, int(255 * (1 - progress)))
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(color, 1))
                # Draw particle with integer coordinates
                painter.drawEllipse(int(current_x - 3), int(y - 3), 6, 6)
    
    def get_neuron_activation(self, layer_idx, neuron_idx):
        """Get current activation level for a neuron."""
        layer_name = f"layer_{layer_idx}"
        if layer_name in self.neuron_activations and neuron_idx < len(self.neuron_activations[layer_name]):
            return self.neuron_activations[layer_name][neuron_idx]
        return 0.1
    
    def get_neuron_pulse(self, layer_idx, neuron_idx):
        """Get current pulse intensity for a neuron."""
        layer_name = f"layer_{layer_idx}"
        if layer_name in self.pulse_intensities and neuron_idx < len(self.pulse_intensities[layer_name]):
            return self.pulse_intensities[layer_name][neuron_idx]
        return 0.0
    
    def get_connection_strength(self, layer_idx, from_neuron, to_neuron):
        """Get connection strength between neurons (simulated)."""
        # Simulate connection weights based on training progress and activity
        base_strength = 0.3 + (self.training_progress / 100) * 0.4
        
        # Add some variation
        variation = np.sin(self.animation_frame * 0.1 + from_neuron + to_neuron) * 0.2
        
        return max(0.1, min(1.0, base_strength + variation))
    
    def update_animation(self):
        """Update animation frame."""
        try:
            self.animation_frame += 1
            
            # Update pulse intensities (decay over time)
            for layer_name in self.pulse_intensities:
                self.pulse_intensities[layer_name] *= 0.95  # Decay
            
            # Update data flow positions
            for i, flow in enumerate(self.data_flow_positions):
                x, y, progress, layer = flow
                progress += 0.05  # Speed of data flow
                
                if progress >= 1.0:
                    # Move to next layer or remove
                    if layer < 3:  # Move to next layer
                        self.data_flow_positions[i] = (x, y, 0.0, layer + 1)
                    else:
                        # Remove completed flow
                        self.data_flow_positions.pop(i)
                        break
                else:
                    self.data_flow_positions[i] = (x, y, progress, layer)
            
            # Update random baseline activity (less frequently)
            if self.animation_frame % 60 == 0:  # Every ~1 second instead of 0.5
                self.add_random_activity()
            
            # Update display (less frequently to reduce conflicts)
            if self.animation_frame % 3 == 0:  # Update every 3 frames instead of every frame
                self.update()
                
        except Exception as e:
            print(f"Error in update_animation: {e}")
    
    def add_random_activity(self):
        """Add random baseline neural activity."""
        for layer_name in self.neuron_activations:
            activations = self.neuron_activations[layer_name]
            
            # Add small random variations
            noise = np.random.uniform(-0.1, 0.1, len(activations))
            activations += noise
            
            # Keep in valid range
            activations = np.clip(activations, 0.1, 1.0)
            
            self.neuron_activations[layer_name] = activations
    
    def set_thinking_mode(self, thinking: bool):
        """Set whether the network is actively thinking."""
        self.is_thinking = thinking
        
        if thinking:
            # Start data flow animation
            self.start_data_flow()
            # Increase overall activity
            self.boost_activity()
    
    def set_training_mode(self, training: bool, progress: float = 0.0):
        """Set training mode and progress."""
        self.is_training = training
        self.training_progress = progress
        
        if training:
            # Add training-specific activity
            self.add_training_activity()
    
    def update_live_prediction(self, prediction: str, confidence: float, band_powers: np.ndarray = None):
        """Update network with live prediction data."""
        self.current_prediction = prediction
        
        # Update input layer with real band power data
        if band_powers is not None:
            flattened = band_powers.flatten()[:self.input_nodes]
            # Normalize to 0-1 range for visualization
            normalized = (flattened - np.min(flattened)) / (np.max(flattened) - np.min(flattened) + 1e-8)
            
            # Update input activations
            self.neuron_activations["layer_0"][:len(normalized)] = normalized
            
            # Pulse input neurons
            self.pulse_intensities["layer_0"][:len(normalized)] = normalized
        
        # Update output layer based on prediction
        output_activations = np.array([0.2, 0.2, 0.2])  # Baseline
        
        if prediction == "left_hand":
            output_activations[0] = confidence
        elif prediction == "right_hand":
            output_activations[1] = confidence  
        elif prediction == "rest":
            output_activations[2] = confidence
        
        self.neuron_activations["layer_3"] = output_activations
        self.pulse_intensities["layer_3"] = output_activations * 0.8
        
        # Trigger data flow
        self.start_data_flow()
    
    def start_data_flow(self):
        """Start animated data flow through network."""
        # Add multiple data particles
        height = self.height() - 60
        
        for i in range(3):  # 3 data streams
            y = 60 + (i * height // 3) + np.random.randint(-20, 20)
            self.data_flow_positions.append((0, y, 0.0, 0))
    
    def boost_activity(self):
        """Boost neural activity levels."""
        for layer_name in self.neuron_activations:
            self.neuron_activations[layer_name] *= 1.5
            self.neuron_activations[layer_name] = np.clip(self.neuron_activations[layer_name], 0.1, 1.0)
    
    def add_training_activity(self):
        """Add training-specific neural activity."""
        # Pulse all layers during training
        for layer_name in self.pulse_intensities:
            self.pulse_intensities[layer_name] += 0.3
            self.pulse_intensities[layer_name] = np.clip(self.pulse_intensities[layer_name], 0.0, 1.0)

class TrainingProgressWidget(QWidget):
    """Widget to show neural network training progress."""
    
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(200)
        self.setStyleSheet("background-color: #2a2a2a; border: 2px solid #ff6b35; border-radius: 10px;")
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("🎬 TRAINING PROGRESS")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #ff6b35; margin: 10px;")
        layout.addWidget(title)
        
        # Current class being trained
        self.current_class_label = QLabel("Ready to train...")
        self.current_class_label.setAlignment(QtCore.Qt.AlignCenter)
        self.current_class_label.setStyleSheet("font-size: 14px; color: #ffffff; margin: 5px;")
        layout.addWidget(self.current_class_label)
        
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
            }
            QProgressBar::chunk {
                background-color: #ff6b35;
                border-radius: 8px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Samples collected
        self.samples_label = QLabel("0 / 50 samples collected")
        self.samples_label.setAlignment(QtCore.Qt.AlignCenter)
        self.samples_label.setStyleSheet("font-size: 12px; color: #cccccc; margin: 5px;")
        layout.addWidget(self.samples_label)
        
        # Training accuracy (after training)
        self.accuracy_label = QLabel("")
        self.accuracy_label.setAlignment(QtCore.Qt.AlignCenter)
        self.accuracy_label.setStyleSheet("font-size: 14px; color: #00ff00; margin: 5px;")
        layout.addWidget(self.accuracy_label)
    
    def update_progress(self, thought_class: str, samples_collected: int, total_samples: int):
        """Update training progress."""
        from openbci_stream import MIND_READING_EMOJIS
        
        emoji = MIND_READING_EMOJIS.get(thought_class, '🧠')
        self.current_class_label.setText(f"Training: {emoji} {thought_class.upper()}")
        
        progress = int((samples_collected / total_samples) * 100)
        self.progress_bar.setValue(progress)
        
        self.samples_label.setText(f"{samples_collected} / {total_samples} samples collected")
    
    def show_training_complete(self, accuracy: float):
        """Show training completion with accuracy."""
        self.current_class_label.setText("🎉 TRAINING COMPLETED!")
        self.progress_bar.setValue(100)
        self.accuracy_label.setText(f"🎯 Accuracy: {accuracy:.1%}")

class PredictionsWidget(QWidget):
    """Widget to show real-time mind reading predictions."""
    
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(250)
        self.setStyleSheet("background-color: #2a2a2a; border: 2px solid #00ff00; border-radius: 10px;")
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("🧠 LIVE MIND READING")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #00ff00; margin: 10px;")
        layout.addWidget(title)
        
        # Current prediction
        self.prediction_label = QLabel("🤔 Waiting for thoughts...")
        self.prediction_label.setAlignment(QtCore.Qt.AlignCenter)
        self.prediction_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #ffffff; margin: 10px;")
        layout.addWidget(self.prediction_label)
        
        # Confidence bar
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Confidence:"))
        
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #666666;
                border-radius: 10px;
                text-align: center;
                background-color: #333333;
                color: white;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #00ff00;
                border-radius: 8px;
            }
        """)
        confidence_layout.addWidget(self.confidence_bar)
        layout.addLayout(confidence_layout)
        
        # Probability bars for all thoughts
        self.probability_bars = {}
        self.create_probability_bars(layout)
    
    def create_probability_bars(self, layout):
        """Create probability bars for each thought class."""
        from openbci_stream import MIND_READING_EMOJIS, MIND_READING_COLORS
        
        # UPDATED: Only show the 3 classes we're actually training on
        thought_classes = ['left_hand', 'right_hand', 'rest']
        
        prob_frame = QFrame()
        prob_frame.setStyleSheet("border: 1px solid #444444; border-radius: 5px; margin: 5px;")
        prob_layout = QVBoxLayout(prob_frame)
        
        prob_title = QLabel("🎲 Thought Probabilities (3 Classes)")
        prob_title.setAlignment(QtCore.Qt.AlignCenter)
        prob_title.setStyleSheet("font-size: 12px; color: #cccccc; margin: 5px;")
        prob_layout.addWidget(prob_title)
        
        for thought in thought_classes:
            row_layout = QHBoxLayout()
            
            # Emoji and label
            emoji = MIND_READING_EMOJIS.get(thought, '🧠')
            label = QLabel(f"{emoji} {thought}")
            label.setFixedWidth(100)
            label.setStyleSheet("font-size: 10px; color: #ffffff;")
            row_layout.addWidget(label)
            
            # Probability bar
            bar = QProgressBar()
            bar.setMaximum(100)
            bar.setStyleSheet(f"""
                QProgressBar {{
                    border: 1px solid #666666;
                    border-radius: 3px;
                    text-align: center;
                    background-color: #333333;
                    color: white;
                    font-size: 8px;
                }}
                QProgressBar::chunk {{
                    background-color: {MIND_READING_COLORS.get(thought, '#666666')};
                    border-radius: 2px;
                }}
            """)
            row_layout.addWidget(bar)
            
            prob_layout.addLayout(row_layout)
            self.probability_bars[thought] = bar
        
        layout.addWidget(prob_frame)
    
    def update_prediction(self, thought: str, confidence: float, probabilities: Dict[str, float]):
        """Update the current prediction display."""
        from openbci_stream import MIND_READING_EMOJIS
        
        # DEBUG: Print received prediction
        print(f"🎯 GUI UPDATE: Received prediction - {thought} ({confidence:.2f})")
        print(f"🎯 GUI UPDATE: Probabilities - {probabilities}")
        
        emoji = MIND_READING_EMOJIS.get(thought, '🧠')
        self.prediction_label.setText(f"{emoji} {thought.upper()} {emoji}")
        
        # Update confidence
        self.confidence_bar.setValue(int(confidence * 100))
        
        # Update probability bars
        for thought_class, probability in probabilities.items():
            if thought_class in self.probability_bars:
                self.probability_bars[thought_class].setValue(int(probability * 100))

class LiveBandPowerWidget(QWidget):
    """Widget to show live EEG band powers for all channels and frequency bands."""
    
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(200)
        self.setStyleSheet("background-color: #2a2a2a; border: 2px solid #00ffff; border-radius: 10px;")
        
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("📊 LIVE EEG BAND POWERS - 16 CHANNELS")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #00ffff; margin: 5px;")
        layout.addWidget(title)
        
        # Data buffers for each channel and band (UPDATED FOR 16 CHANNELS)
        self.band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        # Standard 16-channel OpenBCI electrode names following 10-20 system
        self.channel_names = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7',
            'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4'
        ]
        self.band_data = np.zeros((len(self.channel_names), len(self.band_names)))
        
        # Colors for each band
        self.band_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(12, 3), facecolor='#2a2a2a')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: #2a2a2a;")
        layout.addWidget(self.canvas)
        
        # Initialize plots (AFTER setting attributes)
        self.setup_plots()
    
    def setup_plots(self):
        """Setup the matplotlib plots."""
        self.figure.clear()
        
        # Create subplot for band powers
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#2a2a2a')
        
        # Initialize empty bars for ALL 16 channels
        x_pos = np.arange(len(self.channel_names))
        self.bars = []
        
        # Create stacked bars for each frequency band
        bottom = np.zeros(len(self.channel_names))
        for i, color in enumerate(self.band_colors):
            bars = self.ax.bar(x_pos, np.zeros(len(self.channel_names)), 
                              bottom=bottom, color=color, alpha=0.8, 
                              label=self.band_names[i])
            self.bars.append(bars)
        
        # Styling
        self.ax.set_xlabel('EEG Channels (16 Total)', color='white', fontsize=10)
        self.ax.set_ylabel('Power (μV²)', color='white', fontsize=10)
        self.ax.set_title('Real-time EEG Band Powers - All 16 Channels', color='#00ffff', fontsize=12, fontweight='bold')
        
        # Set channel names as x-tick labels with rotation for better fit
        self.ax.set_xticks(x_pos)
        self.ax.set_xticklabels(self.channel_names, 
                               color='white', fontsize=8, rotation=45)
        
        # Style the axes
        self.ax.tick_params(colors='white', labelsize=8)
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_color('white')
        self.ax.spines['right'].set_color('white')
        self.ax.spines['left'].set_color('white')
        
        # Legend
        self.ax.legend(loc='upper right', framealpha=0.8, facecolor='#2a2a2a', 
                      edgecolor='white', labelcolor='white', fontsize=8)
        
        # Set reasonable y-limits
        self.ax.set_ylim(0, 100)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def update_band_powers(self, band_powers):
        """Update the live band power display for 16 channels."""
        if band_powers is None or band_powers.shape[0] == 0:
            return
            
        try:
            # Ensure we have the right shape (channels x bands)
            if len(band_powers.shape) == 1:
                # Single channel - reshape
                band_powers = band_powers.reshape(1, -1)
            
            # IMPORTANT: Handle 16 channels properly
            expected_channels = len(self.channel_names)  # Should be 16
            num_channels = min(band_powers.shape[0], expected_channels)
            num_bands = min(band_powers.shape[1], len(self.band_names))
            
            print(f"🔍 Band powers shape: {band_powers.shape}, expecting ({expected_channels}, {len(self.band_names)})")
            
            # If we have fewer channels in data than expected, pad with zeros
            if band_powers.shape[0] < expected_channels:
                print(f"⚠️ Only {band_powers.shape[0]} channels in data, padding to {expected_channels}")
                padded_powers = np.zeros((expected_channels, band_powers.shape[1]))
                padded_powers[:band_powers.shape[0], :] = band_powers
                band_powers = padded_powers
                num_channels = expected_channels
            
            # Normalize band powers (log scale for better visualization)
            normalized_powers = np.log10(band_powers[:num_channels, :num_bands] + 1) * 10
            
            # Clear previous bars
            for band_bars in self.bars:
                for bar in band_bars:
                    bar.set_height(0)
            
            # Update bars for all channels
            bottom = np.zeros(num_channels)
            for band_idx in range(num_bands):
                if band_idx < len(self.bars):
                    band_bars = self.bars[band_idx]
                    for ch_idx in range(num_channels):
                        if ch_idx < len(band_bars):
                            height = normalized_powers[ch_idx, band_idx]
                            band_bars[ch_idx].set_height(height)
                            band_bars[ch_idx].set_y(bottom[ch_idx])
                    bottom[:num_channels] += normalized_powers[:num_channels, band_idx]
            
            # Adjust y-limits dynamically
            max_power = np.max(np.sum(normalized_powers, axis=1)) if normalized_powers.size > 0 else 100
            self.ax.set_ylim(0, max_power * 1.1)
            
            # Update the canvas
            self.canvas.draw_idle()
            
            print(f"✅ Updated band powers for {num_channels} channels")
            
        except Exception as e:
            print(f"❌ Error updating band powers visualization: {e}")
            import traceback
            traceback.print_exc()

class LiveEEGChannelsWidget(QWidget):
    """Live display of all 16 EEG channels for viral YouTube content."""
    
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(400)
        self.setStyleSheet("background-color: #1a1a1a; border: 2px solid #ff6600; border-radius: 10px;")
        
        # Channel data storage
        self.channel_data = [deque(maxlen=250) for _ in range(16)]  # 2 seconds at 125 Hz
        self.channel_colors = [
            '#ff0000', '#00ff00', '#0000ff', '#ffff00',  # Red, Green, Blue, Yellow
            '#ff00ff', '#00ffff', '#ff8000', '#8000ff',  # Magenta, Cyan, Orange, Purple
            '#80ff00', '#0080ff', '#ff0080', '#80ff80',  # Lime, Blue, Pink, Light Green
            '#ff8080', '#8080ff', '#ffff80', '#80ffff'   # Light Red, Light Blue, Light Yellow, Light Cyan
        ]
        
        # Animation variables
        self.scroll_position = 0
        self.highlight_channels = set()  # Channels to highlight when active
        
        # Setup timer for animation
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_display)
        self.animation_timer.start(50)  # 20 FPS
        
        # Initialize with some baseline data
        for ch in range(16):
            for _ in range(250):
                self.channel_data[ch].append(np.random.normal(0, 10))
    
    def paintEvent(self, event):
        """Draw the live EEG channels."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get dimensions
        width = self.width() - 40
        height = self.height() - 40
        
        # Background
        painter.fillRect(20, 20, width, height, QColor(26, 26, 26))
        
        # Title
        painter.setPen(QColor(255, 102, 0))
        painter.setFont(QFont("Arial", 14, QFont.Bold))
        painter.drawText(30, 40, "🧠 LIVE EEG CHANNELS (16 Brain Sensors)")
        
        # Channel layout (4x4 grid)
        channels_per_row = 4
        channel_width = width // channels_per_row - 10
        channel_height = (height - 60) // 4 - 10
        
        for ch in range(16):
            row = ch // channels_per_row
            col = ch % channels_per_row
            
            x = 30 + col * (channel_width + 10)
            y = 60 + row * (channel_height + 10)
            
            self.draw_channel(painter, ch, x, y, channel_width, channel_height)
    
    def draw_channel(self, painter, channel_num, x, y, width, height):
        """Draw a single EEG channel waveform."""
        # Channel background
        is_highlighted = channel_num in self.highlight_channels
        bg_color = QColor(0, 50, 0) if is_highlighted else QColor(40, 40, 40)
        painter.fillRect(x, y, width, height, bg_color)
        
        # Channel border
        border_color = QColor(0, 255, 0) if is_highlighted else QColor(100, 100, 100)
        painter.setPen(QPen(border_color, 2))
        painter.drawRect(x, y, width, height)
        
        # Channel label
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Arial", 10, QFont.Bold))
        painter.drawText(x + 5, y + 15, f"Ch{channel_num + 1}")
        
        # Get channel data
        data = list(self.channel_data[channel_num])
        if len(data) < 2:
            return
        
        # Draw waveform using simple lines
        painter.setPen(QPen(QColor(self.channel_colors[channel_num]), 2))
        
        # Draw the waveform with simple line segments
        recent_data = data[-100:]  # Last 100 samples (0.8 seconds)
        if len(recent_data) > 1:
            for i in range(len(recent_data) - 1):
                # Scale and position current point
                x1 = x + 5 + (i / 100) * (width - 10)
                y1 = y + height//2 - (recent_data[i] / 100) * (height//3)
                y1 = max(y + 5, min(y + height - 5, y1))  # Clamp to bounds
                
                # Scale and position next point
                x2 = x + 5 + ((i + 1) / 100) * (width - 10)
                y2 = y + height//2 - (recent_data[i + 1] / 100) * (height//3)
                y2 = max(y + 5, min(y + height - 5, y2))  # Clamp to bounds
                
                # Draw line segment
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        
        # Activity indicator (RMS of recent samples)
        if len(data) > 10:
            rms = np.sqrt(np.mean(np.array(data[-10:]) ** 2))
            activity_width = min(width - 10, max(5, int((rms / 50) * (width - 10))))
            
            # Activity bar
            activity_color = QColor(255, 255 - int(rms * 2), 0) if rms > 50 else QColor(0, 255, 0)
            painter.fillRect(x + 5, y + height - 15, activity_width, 8, activity_color)
    
    def update_eeg_data(self, new_eeg_data):
        """Update with new EEG data from the streamer."""
        if new_eeg_data is not None and new_eeg_data.shape[0] >= 16:
            # Add the latest sample from each channel
            latest_sample = new_eeg_data[:, -1] if new_eeg_data.shape[1] > 0 else new_eeg_data[:, 0]
            
            for ch in range(16):
                if ch < len(latest_sample):
                    self.channel_data[ch].append(latest_sample[ch])
                    
            # Highlight motor channels during high activity
            motor_channels = [8, 9, 10]  # C3, Cz, C4
            self.highlight_channels.clear()
            
            for ch in motor_channels:
                if ch < len(latest_sample):
                    recent_data = list(self.channel_data[ch])[-10:]
                    if len(recent_data) > 5:
                        rms = np.sqrt(np.mean(np.array(recent_data) ** 2))
                        if rms > 75:  # Threshold for highlighting
                            self.highlight_channels.add(ch)
    
    def update_display(self):
        """Update the display animation."""
        self.scroll_position += 1
        if self.scroll_position > 250:
            self.scroll_position = 0
        
        # Add some random activity if no real data
        if np.random.random() < 0.1:  # 10% chance each frame
            ch = np.random.randint(16)
            self.channel_data[ch].append(np.random.normal(0, 20))
        
        self.update()

class MindReaderGUI(QMainWindow):
    """Main GUI window for viral mind reading visualization."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 MIND READER AI - VIRAL EDITION! 🚀")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet("background-color: #1a1a1a; color: white;")
        
        # Training thread
        self.training_thread = None
        
        # Initialize UI
        self.init_ui()
        
        # Connect to default port (can be changed via dialog)
        self.port = "COM3"
        self.board_id = 2
    
    def init_ui(self):
        """Initialize the comprehensive mind reading interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Neural Network and Controls
        left_panel = QVBoxLayout()
        
        # Title
        title = QLabel("🧠 VIRAL MIND READER - NEURAL NETWORK AI")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: #00ff00; margin: 10px;")
        left_panel.addWidget(title)
        
        # Status label
        self.status_label = QLabel("Ready to read minds...")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 16px; color: #ffff00; margin: 5px;")
        left_panel.addWidget(self.status_label)
        
        # Neural network visualization
        self.network_widget = NeuralNetworkWidget()
        left_panel.addWidget(self.network_widget)
        
        # Training progress
        self.progress_widget = TrainingProgressWidget()
        left_panel.addWidget(self.progress_widget)
        
        # Control buttons
        self.create_control_buttons(left_panel)
        
        main_layout.addLayout(left_panel, 1)  # 1/3 of width
        
        # Right panel - Live Data and Predictions
        right_panel = QVBoxLayout()
        
        # Live EEG Channels (NEW!)
        self.eeg_channels_widget = LiveEEGChannelsWidget()
        right_panel.addWidget(self.eeg_channels_widget)
        
        # Predictions
        self.predictions_widget = PredictionsWidget()
        right_panel.addWidget(self.predictions_widget)
        
        # Band powers
        self.band_power_widget = LiveBandPowerWidget()
        right_panel.addWidget(self.band_power_widget)
        
        main_layout.addLayout(right_panel, 2)  # 2/3 of width
    
    def create_control_buttons(self, layout):
        """Create control buttons for training and testing."""
        button_layout = QHBoxLayout()
        
        # Start Training button
        self.train_button = QPushButton("🎬 START TRAINING")
        self.train_button.setStyleSheet("""
            QPushButton {
                background-color: #ff6b35;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ff8555;
            }
        """)
        self.train_button.clicked.connect(self.start_training)
        button_layout.addWidget(self.train_button)
        
        # Quick Training button
        self.quick_train_button = QPushButton("⚡ QUICK TRAIN")
        self.quick_train_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #f56c6c;
            }
        """)
        self.quick_train_button.clicked.connect(self.start_quick_training)
        button_layout.addWidget(self.quick_train_button)
        
        # Test Model button  
        self.test_button = QPushButton("🧠 TEST MODEL")
        self.test_button.setStyleSheet("""
            QPushButton {
                background-color: #00ff00;
                color: black;
                border: none;
                border-radius: 10px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #33ff33;
            }
        """)
        self.test_button.clicked.connect(self.start_testing)
        button_layout.addWidget(self.test_button)
        
        # Demo Mode button
        self.demo_button = QPushButton("🎬 VIRAL DEMO")
        self.demo_button.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #b769d6;
            }
        """)
        self.demo_button.clicked.connect(self.start_demo)
        button_layout.addWidget(self.demo_button)
        
        layout.addLayout(button_layout)
    
    def start_training(self):
        """Start neural network training."""
        if self.training_thread and self.training_thread.isRunning():
            return  # Already training
        
        # Ask user for port
        port, ok = QInputDialog.getText(self, 'OpenBCI Port', 'Enter OpenBCI port (e.g., COM3):', text=self.port)
        if not ok:
            return
        
        self.port = port
        self.status_label.setText("🎬 Starting viral training mode...")
        self.network_widget.set_thinking_mode(True)
        
        # Create and start training thread
        self.training_thread = TrainingThread(self.port, self.board_id)
        self.training_thread.progress_updated.connect(self.progress_widget.update_progress)
        self.training_thread.training_complete.connect(self.progress_widget.show_training_complete)
        self.training_thread.status_updated.connect(self.status_label.setText)
        self.training_thread.band_power_updated.connect(self.band_power_widget.update_band_powers)
        self.training_thread.neural_network_updated.connect(self.update_neural_network)
        self.training_thread.eeg_data_updated.connect(self.eeg_channels_widget.update_eeg_data)
        
        # Set training mode and start the thread
        self.training_thread.run_training()
        self.training_thread.start()  # This actually starts the thread
    
    def start_testing(self):
        """Start model testing."""
        if self.training_thread and self.training_thread.isRunning():
            return  # Already running
        
        # Ask user for port and offer synthetic option
        port, ok = QInputDialog.getText(self, 'OpenBCI Port', 
            'Enter OpenBCI port (e.g., COM3) or type "SYNTHETIC" for demo:', text=self.port)
        if not ok:
            return
        
        self.port = port
        
        # Check if user wants synthetic testing
        if port.upper() == "SYNTHETIC":
            self.port = "COM999"  # Dummy port for synthetic
            self.board_id = -1    # Synthetic board ID
            self.status_label.setText("🎮 Testing with synthetic EEG data...")
            
            # Skip BrainFlow entirely for synthetic testing
            self.network_widget.set_thinking_mode(True)
            
            # Create minimal testing thread for synthetic only
            self.training_thread = TrainingThread(self.port, self.board_id)
            self.training_thread.prediction_updated.connect(self.predictions_widget.update_prediction)
            self.training_thread.status_updated.connect(self.status_label.setText)
            self.training_thread.band_power_updated.connect(self.band_power_widget.update_band_powers)
            self.training_thread.neural_network_updated.connect(self.update_neural_network)
            self.training_thread.eeg_data_updated.connect(self.eeg_channels_widget.update_eeg_data)
            
            # Set testing mode and start synthetic testing
            self.training_thread.run_testing()
            self.training_thread.start()
            return
        else:
            self.board_id = 2     # Real 16-channel board
            self.status_label.setText("🧠 Testing mind reading model...")
        
        self.network_widget.set_thinking_mode(True)
        
        # Create and start testing thread for real data
        self.training_thread = TrainingThread(self.port, self.board_id)
        self.training_thread.prediction_updated.connect(self.predictions_widget.update_prediction)
        self.training_thread.status_updated.connect(self.status_label.setText)
        self.training_thread.band_power_updated.connect(self.band_power_widget.update_band_powers)
        self.training_thread.neural_network_updated.connect(self.update_neural_network)
        self.training_thread.eeg_data_updated.connect(self.eeg_channels_widget.update_eeg_data)
        
        # Set testing mode and start the thread
        self.training_thread.run_testing()
        self.training_thread.start()  # This actually starts the thread
    
    def start_demo(self):
        """Start viral demo mode."""
        self.status_label.setText("🎬 Running viral demo for YouTube...")
        self.network_widget.set_thinking_mode(True)
        
        # Start demo with synthetic predictions
        self.demo_timer = QTimer()
        self.demo_timer.timeout.connect(self._update_demo)
        self.demo_timer.start(2000)  # Update every 2 seconds
    
    def _update_demo(self):
        """Update demo with fake predictions for content creation."""
        import random
        from openbci_stream import MIND_READING_EMOJIS
        
        # FIXED: Only use the 3 classes we actually trained on
        thoughts = ['left_hand', 'right_hand', 'rest']
        thought = random.choice(thoughts)
        confidence = random.uniform(0.4, 0.9)
        
        # Create fake probabilities for ONLY the 3 trained classes
        probabilities = {}
        for t in thoughts:
            if t == thought:
                probabilities[t] = confidence
            else:
                probabilities[t] = random.uniform(0.0, 1.0 - confidence) / len(thoughts)
        
        # UPDATE PREDICTIONS
        self.predictions_widget.update_prediction(thought, confidence, probabilities)
        
        # GENERATE SYNTHETIC EEG DATA FOR VIRAL DEMO
        # Generate realistic EEG band powers (16 channels × 5 bands)
        synthetic_band_powers = self._generate_demo_band_powers(thought, confidence)
        
        # Generate raw EEG data for live channels display
        synthetic_eeg_data = self._generate_demo_eeg_data(thought)
        
        # UPDATE ALL VISUALIZATIONS
        self.band_power_widget.update_band_powers(synthetic_band_powers)
        self.eeg_channels_widget.update_eeg_data(synthetic_eeg_data)
        self.network_widget.update_live_prediction(thought, confidence, synthetic_band_powers)
        
        # Update status to show what's happening
        self.status_label.setText(f"🎬 VIRAL DEMO: Predicting {thought} ({confidence:.1%} confidence)")
    
    def _generate_demo_band_powers(self, thought: str, confidence: float) -> np.ndarray:
        """Generate realistic synthetic EEG band powers for viral demo."""
        import numpy as np
        
        # Create 16 channels × 5 bands array
        band_powers = np.zeros((16, 5))
        
        # Generate different patterns based on the predicted thought
        if thought == 'left_hand':
            # Higher beta activity in motor cortex areas (simulate left hand movement)
            motor_channels = [8, 9, 10]  # C3, Cz, C4 indices
            for ch in motor_channels:
                band_powers[ch, 3] = np.random.uniform(15, 25) * confidence  # High beta
                band_powers[ch, 2] = np.random.uniform(3, 8)   # Low alpha (ERD)
                band_powers[ch, 4] = np.random.uniform(2, 5)   # Some gamma
            
            # Background activity for other channels
            for ch in range(16):
                if ch not in motor_channels:
                    band_powers[ch, :] = np.random.uniform(2, 8, 5)
                    
        elif thought == 'right_hand':
            # Similar to left_hand but with slightly different pattern
            motor_channels = [8, 9, 10]  # C3, Cz, C4
            for ch in motor_channels:
                band_powers[ch, 3] = np.random.uniform(12, 22) * confidence  # High beta
                band_powers[ch, 2] = np.random.uniform(2, 7)    # Low alpha
                band_powers[ch, 4] = np.random.uniform(3, 7)    # More gamma than left
            
            # Background activity
            for ch in range(16):
                if ch not in motor_channels:
                    band_powers[ch, :] = np.random.uniform(1, 7, 5)
                    
        else:  # 'rest' pattern
            # More relaxed pattern with higher alpha across all channels
            for ch in range(16):
                band_powers[ch, 0] = np.random.uniform(3, 8)   # Delta
                band_powers[ch, 1] = np.random.uniform(4, 10)  # Theta  
                band_powers[ch, 2] = np.random.uniform(12, 22) # High alpha (relaxed)
                band_powers[ch, 3] = np.random.uniform(2, 6)   # Low beta
                band_powers[ch, 4] = np.random.uniform(1, 4)   # Low gamma
        
        # Add some realistic noise and ensure positive values
        noise = np.random.uniform(-1, 1, band_powers.shape)
        band_powers += noise
        band_powers = np.maximum(band_powers, 0.1)
        
        return band_powers
    
    def _generate_demo_eeg_data(self, thought: str) -> np.ndarray:
        """Generate synthetic raw EEG data for live channels display."""
        import numpy as np
        
        # Create realistic synthetic EEG data (16 channels, ~100 samples)
        num_channels = 16
        num_samples = 100  # About 0.8 seconds at 125 Hz
        
        # Generate realistic EEG patterns
        synthetic_eeg = np.random.normal(0, 30, (num_channels, num_samples))  # Baseline noise
        
        # Add realistic frequency components based on the thought
        time_vec = np.linspace(0, 0.8, num_samples)
        
        for ch in range(num_channels):
            # Motor channels get special treatment for hand movements
            if ch in [8, 9, 10] and thought != 'rest':  # C3, Cz, C4 for hand movements
                # Add stronger beta oscillations for motor imagery
                beta_freq = 20 + np.random.uniform(-3, 3)
                beta_amplitude = np.random.uniform(40, 80)  # Stronger for motor
                synthetic_eeg[ch, :] += beta_amplitude * np.sin(2 * np.pi * beta_freq * time_vec)
                
                # Add motor-related gamma bursts
                gamma_freq = 35 + np.random.uniform(-5, 5)
                gamma_amplitude = np.random.uniform(15, 30)
                synthetic_eeg[ch, :] += gamma_amplitude * np.sin(2 * np.pi * gamma_freq * time_vec)
                
            else:
                # Regular alpha waves for all channels
                alpha_freq = 10 + np.random.uniform(-2, 2)
                alpha_amplitude = np.random.uniform(20, 60)
                if thought == 'rest':
                    alpha_amplitude *= 1.5  # Stronger alpha during rest
                synthetic_eeg[ch, :] += alpha_amplitude * np.sin(2 * np.pi * alpha_freq * time_vec)
            
            # Add some random spikes for realism
            for spike in range(np.random.randint(0, 2)):
                spike_pos = np.random.randint(0, num_samples)
                spike_width = np.random.randint(2, 5)
                spike_amp = np.random.uniform(30, 100)
                start_idx = max(0, spike_pos - spike_width)
                end_idx = min(num_samples, spike_pos + spike_width)
                synthetic_eeg[ch, start_idx:end_idx] += spike_amp
        
        return synthetic_eeg

    def update_neural_network(self, mode: str, progress: float, band_powers: np.ndarray, thought: str):
        """Update neural network visualization with training or prediction activity."""
        if mode == "training":
            self.network_widget.set_training_mode(True, progress)
            # Also update with current training data
            if band_powers is not None:
                self.network_widget.update_live_prediction(thought, progress/100, band_powers)
        elif mode == "prediction":
            self.network_widget.set_thinking_mode(True)
            self.network_widget.update_live_prediction(thought, progress, band_powers)

    def start_quick_training(self):
        """Start quick neural network training for testing the save fix."""
        if self.training_thread and self.training_thread.isRunning():
            return  # Already training
        
        # Ask user for port
        port, ok = QInputDialog.getText(self, 'OpenBCI Port', 'Enter OpenBCI port (e.g., COM3) for QUICK test:', text=self.port)
        if not ok:
            return
        
        self.port = port
        self.status_label.setText("⚡ Starting QUICK training to test save fix...")
        self.network_widget.set_thinking_mode(True)
        
        # Create and start quick training thread
        self.training_thread = TrainingThread(self.port, self.board_id)
        self.training_thread.progress_updated.connect(self.progress_widget.update_progress)
        self.training_thread.training_complete.connect(self.progress_widget.show_training_complete)
        self.training_thread.status_updated.connect(self.status_label.setText)
        self.training_thread.band_power_updated.connect(self.band_power_widget.update_band_powers)
        self.training_thread.neural_network_updated.connect(self.update_neural_network)
        self.training_thread.eeg_data_updated.connect(self.eeg_channels_widget.update_eeg_data)
        
        # Set quick training mode and start the thread
        self.training_thread.run_quick_training()
        self.training_thread.start()  # This actually starts the thread

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = MindReaderGUI()
    gui.show()
    sys.exit(app.exec_()) 