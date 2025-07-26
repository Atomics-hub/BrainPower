#!/usr/bin/env python3
"""Guide for better real brain signal training with realistic expectations."""

import numpy as np
import time

def print_real_brain_training_guide():
    """Print comprehensive guide for training with real brain signals."""
    
    print("🧠 REAL BRAIN SIGNAL TRAINING GUIDE")
    print("="*60)
    
    print("\n1. 🔧 HARDWARE SETUP (CRITICAL):")
    print("   ✅ All electrodes should show <50kΩ impedance")
    print("   ✅ NO 'Railed 100%' channels (fix with more gel)")
    print("   ✅ Focus on motor cortex: C3, Cz, C4 (channels 9, 10, 11)")
    print("   ✅ Reference electrode on earlobe or mastoid")
    print("   ✅ Quiet environment (no electrical interference)")
    
    print("\n2. 🎯 REALISTIC EXPECTATIONS:")
    print("   • Motor imagery EEG is HARD (even for experts)")
    print("   • Typical accuracy: 60-80% (not 90%+)")
    print("   • Need 100-200 samples per class minimum")
    print("   • Takes multiple training sessions")
    print("   • Individual differences are huge")
    
    print("\n3. 🧘 MENTAL STRATEGY:")
    print("   LEFT HAND:")
    print("   • Imagine SQUEEZING a stress ball with left hand")
    print("   • Feel the muscles in your left arm tensing")
    print("   • Keep rhythm: squeeze for 2s, relax 1s, repeat")
    print("   • DON'T actually move - just imagine")
    
    print("\n   RIGHT HAND:")
    print("   • Imagine LIFTING a heavy weight with right arm")
    print("   • Different sensation than left hand")
    print("   • Focus on shoulder and bicep muscles")
    print("   • Same rhythm but different muscle group")
    
    print("\n   REST:")
    print("   • Complete mental relaxation")
    print("   • Focus on breathing")
    print("   • NO specific motor thoughts")
    print("   • Let your mind wander naturally")
    
    print("\n4. 📊 IMPROVED TRAINING PROCEDURE:")
    print("   • Collect 50-100 samples per class")
    print("   • 5-10 second trials per sample")
    print("   • 10-15 second breaks between trials")
    print("   • Train when you're alert (not tired)")
    print("   • Multiple short sessions > one long session")
    
    print("\n5. 🔍 DEBUGGING CHECKLIST:")
    print("   ❓ Are motor cortex channels (C3, Cz, C4) working?")
    print("   ❓ Can you see alpha waves (8-12 Hz) when eyes closed?")
    print("   ❓ Are you consistent with mental imagery?")
    print("   ❓ Is the training data actually different between classes?")
    
    print("\n6. 🚀 ADVANCED TIPS:")
    print("   • Practice mental imagery WITHOUT recording first")
    print("   • Use biofeedback: watch alpha/beta bands during imagery")
    print("   • Try different mental strategies if first doesn't work")
    print("   • Consider spatial filtering (Common Spatial Patterns)")
    print("   • Multiple training sessions over several days")

def create_improved_training_config():
    """Create configuration for better real brain training."""
    config = {
        'samples_per_class': 80,  # More samples
        'trial_duration': 6,      # Longer trials
        'rest_between_trials': 12, # Longer rest
        'rest_between_classes': 60, # Much longer break
        'motor_channels': [8, 9, 10],  # C3, Cz, C4 (0-indexed)
        'feature_bands': {
            'mu_rhythm': (8, 12),      # Sensorimotor rhythm
            'beta': (13, 30),          # Motor beta
            'high_beta': (20, 35),     # High beta
        },
        'mental_strategies': {
            'left_hand': 'Squeeze stress ball with LEFT hand - feel the forearm muscles',
            'right_hand': 'Lift heavy weight with RIGHT arm - feel the bicep and shoulder',
            'rest': 'Complete relaxation - breathe naturally, no motor thoughts'
        }
    }
    return config

def analyze_signal_quality(channel_data):
    """Analyze if EEG signal quality is good enough for training."""
    print("\n🔍 SIGNAL QUALITY ANALYSIS:")
    
    # Check for flat lines (railed channels)
    for i, channel in enumerate(channel_data):
        if np.std(channel) < 1.0:
            print(f"   ⚠️ Channel {i+1}: FLAT LINE - check electrode contact")
        elif np.std(channel) > 200:
            print(f"   ⚠️ Channel {i+1}: TOO NOISY - check for artifacts")
        else:
            print(f"   ✅ Channel {i+1}: Good signal quality")
    
    # Check for alpha rhythm (eyes closed test)
    print("\n📊 ALPHA RHYTHM TEST:")
    print("   Close your eyes for 10 seconds...")
    print("   Look for 8-12 Hz peaks in posterior channels")
    print("   This confirms brain signal detection")

if __name__ == "__main__":
    print_real_brain_training_guide()
    
    print("\n" + "="*60)
    print("💡 KEY INSIGHT: The issue isn't your brain or the software.")
    print("💡 Motor imagery EEG is genuinely difficult!")
    print("💡 Focus on electrode contact quality FIRST.")
    print("💡 Then train with realistic expectations.")
    print("="*60) 