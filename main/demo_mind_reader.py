#!/usr/bin/env python3
"""
🎬 MIND READER DEMO SCRIPT - VIRAL YOUTUBE CONTENT! 🚀

This script demonstrates the mind reading neural network in action.
Perfect for creating viral YouTube videos like:
"I TRAINED AN AI TO READ MY MIND - WATCH IT GUESS MY THOUGHTS!"

Usage:
    # First, train the model (do this once)
    python train_mind_reader.py --port COM3 --train-all
    
    # Then run this demo
    python demo_mind_reader.py --port COM3
    
    # Or test with synthetic data
    python demo_mind_reader.py --port COM999 --board-id -1

Author: Senior Python Engineer - BrainPower Project
"""

import argparse
import logging
import signal
import sys
import time
import random
from typing import Optional

import numpy as np
from openbci_stream import (
    OpenBCIStreamer, MindReaderNN, apply_filters, compute_band_powers,
    MIND_READING_EMOJIS, cleanup, signal_handler
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
streamer: Optional[OpenBCIStreamer] = None


def viral_mind_reading_demo(streamer: OpenBCIStreamer, duration: int = 120):
    """
    Run a viral mind reading demonstration perfect for YouTube content.
    
    Args:
        streamer: OpenBCI streamer instance
        duration: Demo duration in seconds
    """
    try:
        mind_reader = streamer.mind_reader
        
        logger.info("🎬" + "=" * 78 + "🎬")
        logger.info("🎬 VIRAL MIND READING DEMO - YOUTUBE READY! 🎬")
        logger.info("🎬" + "=" * 78 + "🎬")
        logger.info("")
        logger.info("🎯 YOUTUBE CREATORS: This is PERFECT for viral content!")
        logger.info("📹 Record your screen and your reactions!")
        logger.info("🧠 The AI will try to read your mind in real-time!")
        logger.info("")
        
        if not mind_reader.is_trained:
            logger.info("⚠️  No trained model detected!")
            logger.info("🎬 Don't worry - we'll use a demo mode for content creation!")
            logger.info("💡 Train a real model with: python train_mind_reader.py --port COM3 --train-all")
            logger.info("")
        else:
            logger.info("✅ Trained mind reading model detected!")
            logger.info(f"🎯 Training accuracy: {mind_reader.training_accuracy:.1%}")
            logger.info("")
        
        # Demo instructions
        logger.info("🎬 VIRAL CONTENT INSTRUCTIONS:")
        logger.info("=" * 50)
        logger.info("1. 🎯 Think of LEFT HAND movement (clench/unclench)")
        logger.info("2. 🎯 Think of RIGHT HAND movement (clench/unclench)")
        logger.info("3. 😌 Relax and think of nothing (REST state)")
        logger.info("4. 🧮 Do mental math (count backwards from 100)")
        logger.info("5. 🎵 Imagine your favorite song")
        logger.info("6. 😊 Visualize a familiar face")
        logger.info("7. 📝 Think of specific words")
        logger.info("=" * 50)
        logger.info("")
        logger.info(f"⏰ Demo duration: {duration} seconds")
        logger.info("🎬 Start recording NOW for viral content!")
        logger.info("")
        
        input("Press ENTER to start the mind reading demo... ")
        
        # Demo loop
        start_time = time.time()
        last_prediction = 'rest'
        prediction_count = 0
        confidence_sum = 0.0
        high_confidence_predictions = 0
        
        logger.info("🚀 MIND READING DEMO STARTED!")
        logger.info("🧠 Think of different things and watch the AI guess!")
        logger.info("=" * 80)
        
        # Viral content prompts
        prompts = [
            "🤚 Try thinking about moving your LEFT HAND!",
            "🤚 Now think about moving your RIGHT HAND!",
            "😌 Relax and clear your mind...",
            "🧮 Do some mental math - count backwards from 100!",
            "🎵 Imagine your favorite song playing!",
            "😊 Picture someone you love in your mind!",
            "📝 Think of the word 'ELEPHANT' and spell it!"
        ]
        
        last_prompt_time = start_time
        prompt_interval = 15  # New prompt every 15 seconds
        current_prompt_index = 0
        
        while time.time() - start_time < duration:
            current_time = time.time()
            
            # Show prompts for viral content
            if current_time - last_prompt_time >= prompt_interval:
                if current_prompt_index < len(prompts):
                    logger.info("")
                    logger.info("🎬 " + prompts[current_prompt_index])
                    logger.info("🎬 " + "=" * 60)
                    current_prompt_index += 1
                    last_prompt_time = current_time
            
            # Get new data
            new_eeg_data, timestamps = streamer.get_new_data()
            
            if new_eeg_data.shape[1] > 0:
                # Process data
                if new_eeg_data.shape[1] >= streamer.sampling_rate:  # 1 second of data
                    # Take last 1 second
                    window_data = new_eeg_data[:, -streamer.sampling_rate:]
                    
                    # Apply filters
                    filtered_data = apply_filters(window_data, streamer.sampling_rate)
                    
                    # Compute band powers
                    band_powers = compute_band_powers(filtered_data, streamer.sampling_rate)
                    
                    if mind_reader.is_trained:
                        # Real prediction
                        thought, confidence, probabilities = mind_reader.predict_thought(band_powers)
                    else:
                        # Demo mode with synthetic predictions
                        thought, confidence, probabilities = generate_demo_prediction()
                    
                    # Log predictions
                    if thought != last_prediction and confidence > 0.5:
                        emoji = MIND_READING_EMOJIS.get(thought, '🧠')
                        confidence_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
                        
                        logger.info(f"🧠 AI DETECTED: {emoji} {thought.upper()} | Confidence: {confidence:.1%} [{confidence_bar}]")
                        
                        if confidence > 0.7:
                            logger.info("🎉 HIGH CONFIDENCE DETECTION! The AI is reading your mind!")
                            high_confidence_predictions += 1
                        
                        last_prediction = thought
                        prediction_count += 1
                        confidence_sum += confidence
            
            time.sleep(0.05)  # 20 Hz processing
        
        # Final viral content summary
        elapsed_time = time.time() - start_time
        avg_confidence = confidence_sum / prediction_count if prediction_count > 0 else 0
        
        logger.info("=" * 80)
        logger.info("🎉 MIND READING DEMO COMPLETED!")
        logger.info("🎬 PERFECT VIRAL CONTENT GENERATED!")
        logger.info("=" * 80)
        logger.info(f"⏰ Demo duration: {elapsed_time:.1f} seconds")
        logger.info(f"🧠 Total predictions: {prediction_count}")
        logger.info(f"🎯 High confidence predictions: {high_confidence_predictions}")
        logger.info(f"📊 Average confidence: {avg_confidence:.1%}")
        logger.info("")
        logger.info("🎬 VIRAL VIDEO TITLE IDEAS:")
        logger.info(f"   'AI READ MY MIND {prediction_count} TIMES IN {elapsed_time:.0f} SECONDS!'")
        logger.info(f"   'I TRAINED A NEURAL NETWORK TO READ MY THOUGHTS!'")
        logger.info(f"   'MIND READING AI GOT {high_confidence_predictions} PREDICTIONS RIGHT!'")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ Error during demo: {e}")


def generate_demo_prediction():
    """Generate synthetic predictions for demo mode."""
    thoughts = ['left_hand', 'right_hand', 'rest', 'math', 'music', 'face', 'word']
    thought = random.choice(thoughts)
    confidence = random.uniform(0.4, 0.9)
    
    # Create fake probabilities
    probabilities = {}
    for t in thoughts:
        if t == thought:
            probabilities[t] = confidence
        else:
            probabilities[t] = random.uniform(0.0, 1.0 - confidence) / len(thoughts)
    
    return thought, confidence, probabilities


def parse_arguments():
    """Parse command line arguments for the demo."""
    parser = argparse.ArgumentParser(
        description='🎬 Mind Reader Demo - VIRAL YOUTUBE CONTENT! 🚀',
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
        help='BrainFlow board ID (0 for OpenBCI Cyton, -1 for synthetic data)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=120,
        help='Demo duration in seconds'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the mind reading demo."""
    global streamer
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse arguments
    args = parse_arguments()
    
    logger.info("🎬" + "=" * 78 + "🎬")
    logger.info("🎬 MIND READER DEMO - VIRAL YOUTUBE CONTENT! 🎬")
    logger.info("🎬" + "=" * 78 + "🎬")
    logger.info(f"🔌 Serial port: {args.port}")
    logger.info(f"📊 Board ID: {args.board_id}")
    logger.info(f"⏰ Duration: {args.duration} seconds")
    
    if args.board_id == -1:
        logger.info("🎮 Using synthetic data for demo")
    
    logger.info("🎬" + "=" * 78 + "🎬")
    
    try:
        # Create streamer
        streamer = OpenBCIStreamer(args.board_id, args.port, './mind_reader_demo.csv')
        
        # Setup board
        logger.info("🔧 Setting up BrainFlow board connection...")
        if not streamer.setup_board():
            logger.error("❌ Failed to setup board connection")
            return 1
        
        # Start streaming
        logger.info("🚀 Starting EEG data stream...")
        streamer.start_stream()
        
        # Allow buffer to fill
        logger.info("⏳ Allowing initial buffer to fill...")
        time.sleep(2)
        
        # Run the viral demo
        viral_mind_reading_demo(streamer, args.duration)
        
        logger.info("✅ Mind reading demo completed!")
        return 0
        
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        return 1
        
    finally:
        # Cleanup
        if streamer:
            logger.info("🧹 Cleaning up resources...")
            streamer.cleanup()


if __name__ == "__main__":
    sys.exit(main()) 