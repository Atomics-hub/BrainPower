#!/usr/bin/env python3
"""Simple alpha wave test using eyes open/closed paradigm."""

import numpy as np
import time
from openbci_stream import OpenBCIStreamer
from scipy.signal import welch

def run_alpha_test():
    """Test alpha wave modulation with eyes open/closed."""
    
    print("🧠 ALPHA WAVE TEST")
    print("="*50)
    print("This will test if your brain shows clear alpha wave differences")
    print("between eyes open and eyes closed states.")
    print("We'll use your working motor channels (C3, Cz, C4)")
    print()
    
    # Setup streamer
    streamer = OpenBCIStreamer(2, 'COM3', './alpha_test.csv')
    
    if not streamer.setup_board():
        print("❌ Failed to connect to OpenBCI board")
        return
    
    streamer.start_stream()
    print("✅ Connected! Starting alpha test...")
    time.sleep(3)  # Let buffer fill
    
    def collect_data_for_duration(duration_seconds, description):
        """Collect EEG data for specified duration."""
        print(f"📊 Collecting {description} for {duration_seconds} seconds...")
        
        data_buffer = []
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            new_eeg_data, timestamps = streamer.get_new_data()
            if new_eeg_data.shape[1] > 0:
                data_buffer.append(new_eeg_data)
            time.sleep(0.1)  # Small delay to prevent excessive CPU usage
        
        if data_buffer:
            # Concatenate all collected data
            full_data = np.concatenate(data_buffer, axis=1)
            print(f"✅ Collected {full_data.shape[1]} samples")
            return full_data
        else:
            print("❌ No data collected")
            return None
    
    # EYES CLOSED TEST
    print("\n👁️ PHASE 1: EYES CLOSED TEST")
    print("Close your eyes and keep them closed...")
    print("Starting in: 3..."); time.sleep(1)
    print("2..."); time.sleep(1) 
    print("1..."); time.sleep(1)
    
    eyes_closed_data = collect_data_for_duration(10, "EYES CLOSED data")
    
    # EYES OPEN TEST  
    print("\n👁️ PHASE 2: EYES OPEN TEST")
    print("Now OPEN your eyes and focus on this text...")
    print("Keep reading this text and focus on the screen...")
    time.sleep(3)
    
    eyes_open_data = collect_data_for_duration(10, "EYES OPEN data")
    
    streamer.cleanup()
    print("✅ Data collection complete!")
    
    # ANALYSIS
    if eyes_closed_data is not None and eyes_open_data is not None:
        print("\n📊 ALPHA ANALYSIS:")
        print("-" * 30)
        
        motor_channels = [8, 9, 10]  # C3, Cz, C4 (0-indexed)
        channel_names = ['C3', 'Cz', 'C4']
        
        alpha_ratios = []
        
        for i, ch in enumerate(motor_channels):
            if ch < eyes_closed_data.shape[0] and ch < eyes_open_data.shape[0]:
                # Power spectral density
                f_closed, psd_closed = welch(eyes_closed_data[ch], fs=125, nperseg=256)
                f_open, psd_open = welch(eyes_open_data[ch], fs=125, nperseg=256)
                
                # Alpha band (8-12 Hz)
                alpha_mask = (f_closed >= 8) & (f_closed <= 12)
                alpha_closed = np.mean(psd_closed[alpha_mask])
                alpha_open = np.mean(psd_open[alpha_mask])
                
                ratio = alpha_closed / alpha_open if alpha_open > 0 else 0
                alpha_ratios.append(ratio)
                
                print(f"{channel_names[i]:2} | Closed: {alpha_closed:6.1f} | Open: {alpha_open:6.1f} | Ratio: {ratio:.2f}")
            else:
                print(f"{channel_names[i]:2} | ERROR: Channel not available")
        
        if alpha_ratios:
            # Overall assessment
            max_ratio = max(alpha_ratios)
            avg_ratio = np.mean(alpha_ratios)
            
            print("\n🎯 ALPHA TEST RESULTS:")
            print("=" * 40)
            print(f"Best channel ratio: {max_ratio:.2f}")
            print(f"Average ratio: {avg_ratio:.2f}")
            
            if max_ratio > 3.0:
                print("🎉 EXCELLENT alpha modulation!")
                print("✅ Eyes open/closed paradigm will work great!")
                print("✅ Your brain shows very clear alpha patterns!")
                print("\n🚀 RECOMMENDATION: Train with eyes open/closed instead of motor imagery!")
            elif max_ratio > 2.0:
                print("😊 GOOD alpha modulation!")
                print("✅ Eyes open/closed paradigm should work well!")
                print("\n🚀 RECOMMENDATION: Try eyes open/closed training!")
            elif max_ratio > 1.5:
                print("😐 MODERATE alpha modulation")
                print("⚠️ Eyes open/closed might work with more training data")
            else:
                print("😞 WEAK alpha modulation")
                print("⚠️ May need better electrode contact")
                
            print(f"\n💡 INTERPRETATION:")
            print(f"   Ratio > 2.0 = Good for BCI training")
            print(f"   Ratio > 3.0 = Excellent for BCI training") 
            print(f"   Your best ratio: {max_ratio:.2f}")
            
            if max_ratio > 2.0:
                print(f"\n✅ CONCLUSION: Alpha-based BCI should work better than motor imagery!")
                print(f"   Your motor imagery showed almost no variation")
                print(f"   But alpha shows {max_ratio:.1f}x difference between states")
            else:
                print(f"\n⚠️ CONCLUSION: Both paradigms show weak patterns")
                print(f"   May need to improve electrode contact further")
            
            return max_ratio > 2.0
        else:
            print("❌ No valid channel data")
            return False
    else:
        print("❌ Failed to collect data")
        return False

if __name__ == "__main__":
    run_alpha_test() 