#!/usr/bin/env python3
"""
🔍 Brain Pattern Inspector - Lightweight EEG Pattern Viewer

This tool shows your real-time brain patterns in a simple format
so we can verify your brain data is varied enough before retraining.

Usage: python brain_pattern_inspector.py
"""

import time
import numpy as np
from collections import deque
from openbci_stream import OpenBCIStreamer, apply_filters, compute_band_powers

class BrainPatternInspector:
    def __init__(self, port="COM3", duration=60):
        self.port = port
        self.duration = duration
        self.patterns = []
        
    def run_inspection(self):
        """Run brain pattern inspection."""
        print("🔍 BRAIN PATTERN INSPECTOR")
        print("=" * 50)
        print("🧠 This will show your real-time brain patterns")
        print("🎯 Try different mental states and see if patterns change")
        print("=" * 50)
        
        # Create streamer
        try:
            streamer = OpenBCIStreamer(2, self.port, './pattern_inspection.csv')
            
            if not streamer.setup_board():
                print("❌ Failed to connect to board")
                return
                
            streamer.start_stream()
            print("🚀 EEG stream started!")
            
            # Allow buffer to fill
            time.sleep(3)
            
            print("\\n📊 LIVE BRAIN PATTERNS (every 2 seconds):")
            print("Try these mental states:")
            print("   1️⃣ Complete relaxation")  
            print("   2️⃣ Intense left hand imagery")
            print("   3️⃣ Intense right hand imagery")
            print("\\n" + "-" * 50)
            
            data_buffer = deque(maxlen=1000)
            start_time = time.time()
            pattern_count = 0
            
            while time.time() - start_time < self.duration:
                # Collect data for 1 second
                collection_start = time.time()
                while time.time() - collection_start < 1.0:
                    try:
                        new_eeg_data, timestamps = streamer.get_new_data()
                        if new_eeg_data is not None and new_eeg_data.shape[1] > 0:
                            for i in range(new_eeg_data.shape[1]):
                                sample_data = {
                                    'eeg': new_eeg_data[:, i],
                                    'timestamp': timestamps[i] if len(timestamps) > i else time.time()
                                }
                                data_buffer.append(sample_data)
                        time.sleep(0.01)
                    except Exception as e:
                        print(f"❌ Error: {e}")
                
                # Process collected data
                if len(data_buffer) >= 100:  # Need at least 100 samples
                    recent_samples = list(data_buffer)[-100:]
                    window_eeg = np.array([sample['eeg'] for sample in recent_samples]).T
                    
                    try:
                        # Process the brain data
                        filtered_data = apply_filters(window_eeg, streamer.sampling_rate)
                        band_powers = compute_band_powers(filtered_data, streamer.sampling_rate)
                        
                        # Calculate key metrics
                        alpha_avg = band_powers[:, 2].mean()  # Alpha band
                        beta_avg = band_powers[:, 3].mean()   # Beta band
                        motor_activity = band_powers[8:11, 3].mean()  # Beta in motor areas
                        overall_power = band_powers.sum()
                        
                        # Store pattern
                        pattern = {
                            'time': time.time() - start_time,
                            'alpha': alpha_avg,
                            'beta': beta_avg, 
                            'motor': motor_activity,
                            'total': overall_power,
                            'features': band_powers.flatten()[:10]  # First 10 features
                        }
                        self.patterns.append(pattern)
                        pattern_count += 1
                        
                        # Display current pattern
                        print(f"Pattern {pattern_count:2d} | "
                              f"Alpha: {alpha_avg:6.1f} | "
                              f"Beta: {beta_avg:6.1f} | "
                              f"Motor: {motor_activity:6.1f} | "
                              f"Total: {overall_power:8.0f}")
                        
                        # Show pattern changes
                        if len(self.patterns) >= 2:
                            prev = self.patterns[-2]
                            alpha_change = abs(pattern['alpha'] - prev['alpha'])
                            beta_change = abs(pattern['beta'] - prev['beta'])
                            
                            if alpha_change > 100 or beta_change > 50:
                                print(f"         ⚡ PATTERN CHANGE DETECTED! α:{alpha_change:+.1f} β:{beta_change:+.1f}")
                        
                    except Exception as e:
                        print(f"❌ Processing error: {e}")
                
                time.sleep(1)  # Wait 1 second between readings
            
            print("\\n" + "=" * 50)
            print("🎯 PATTERN ANALYSIS SUMMARY")
            print("=" * 50)
            
            if len(self.patterns) >= 5:
                alphas = [p['alpha'] for p in self.patterns]
                betas = [p['beta'] for p in self.patterns]
                motors = [p['motor'] for p in self.patterns]
                
                alpha_range = max(alphas) - min(alphas)
                beta_range = max(betas) - min(betas)
                motor_range = max(motors) - min(motors)
                
                print(f"📊 Alpha variation: {alpha_range:.1f} (need >200 for good classification)")
                print(f"📊 Beta variation: {beta_range:.1f} (need >100 for good classification)")
                print(f"📊 Motor variation: {motor_range:.1f} (need >50 for motor imagery)")
                
                print("\\n🎯 RECOMMENDATIONS:")
                if alpha_range < 200:
                    print("   ⚠️ Alpha variation too low - try more relaxation vs focus")
                if beta_range < 100:
                    print("   ⚠️ Beta variation too low - try more intense mental imagery")
                if motor_range < 50:
                    print("   ⚠️ Motor activity too stable - try stronger left/right imagery")
                    
                if alpha_range > 200 and beta_range > 100:
                    print("   ✅ Good pattern variation - model should work!")
                else:
                    print("   🔄 Need more distinct mental states for better model")
            
            streamer.cleanup()
            
        except Exception as e:
            print(f"❌ Error: {e}")
            
    def save_patterns(self, filename="brain_patterns.txt"):
        """Save patterns for analysis."""
        with open(filename, 'w') as f:
            f.write("Brain Pattern Analysis\\n")
            f.write("=" * 30 + "\\n")
            for i, pattern in enumerate(self.patterns):
                f.write(f"Pattern {i+1}: Alpha={pattern['alpha']:.1f}, Beta={pattern['beta']:.1f}, Motor={pattern['motor']:.1f}\\n")
        print(f"📄 Patterns saved to {filename}")

if __name__ == "__main__":
    inspector = BrainPatternInspector()
    inspector.run_inspection()
    inspector.save_patterns() 