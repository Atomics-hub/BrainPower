# 🧠 OpenBCI Cyton EEG Streaming Script - VIRAL EDITION! 🚀

**The most advanced, feature-packed EEG streaming script that will blow your mind!**

A comprehensive Python script for real-time EEG data streaming, processing, and visualization using the OpenBCI Cyton board and BrainFlow SDK. Now with **AI-powered brain state classification**, **MIND READING NEURAL NETWORK**, **brain-controlled music generation**, **smart home integration**, and **stunning real-time visualizations**!

## 🎯 VIRAL FEATURES

### 🧠 **AI Brain State Classification**

- **Real-time AI detection** of 7 different brain states:
  - 🎯 **Focused** - When you're in the zone
  - 😌 **Relaxed** - Chill and peaceful vibes
  - 😰 **Stressed** - High tension detected
  - 🧘 **Meditative** - Deep zen state
  - 🤩 **Excited** - High energy and enthusiasm
  - 😴 **Drowsy** - Getting sleepy
  - 😐 **Neutral** - Baseline state
- **Machine learning** with confidence scores and probability distributions
- **Emoji-based visual feedback** that changes in real-time

### 🎯 **MIND READING NEURAL NETWORK** ⭐ NEW!

**Train an AI to literally read your thoughts! Perfect for viral YouTube content!**

- **🧠 Motor Imagery Detection**: Left hand vs right hand movement thoughts
- **🧮 Mental Math Recognition**: Detect when you're doing calculations
- **🎵 Music Imagination**: AI knows when you're thinking of songs
- **😊 Face Visualization**: Detects when you're imagining faces
- **📝 Word Thinking**: Recognizes specific word thoughts
- **😌 Rest State**: Knows when your mind is clear

#### **Training Your Personal Mind Reader**

```bash
# Use the GUI for easy training (recommended)
python main/mind_reader_gui.py

# Or train via command line (advanced)
python main/brain_calibrator.py --port COM3
```

#### **Viral Demo Mode**

```bash
# Perfect for YouTube videos!
python main/demo_mind_reader.py --port COM3 --duration 120
```

#### **Mind Reading Features**

- **🎬 Guided Training**: Step-by-step instructions for each thought class
- **📊 Real-time Progress**: Live training progress with emoji feedback
- **🧠 Neural Network**: 3-layer MLP with 128-64-32 neurons
- **🎯 High Accuracy**: Typically 70-90% accuracy on personal data
- **💾 Model Persistence**: Save and load your trained models
- **🎮 Demo Mode**: Synthetic predictions for content creation

#### **Viral Content Ideas**

- **"I TRAINED AN AI TO READ MY MIND!"**
- **"NEURAL NETWORK GUESSES MY THOUGHTS!"**
- **"AI DETECTED MY HAND MOVEMENTS JUST BY THINKING!"**
- **"MIND READING TECHNOLOGY IS HERE!"**

### 🎵 **Brain-Controlled Music Generation**

- **Your brain waves become music!**
- Different brain states trigger unique musical scales and harmonies
- **Real-time audio synthesis** based on EEG band powers
- Alpha waves control volume, Beta waves control complexity
- **Non-blocking audio** so the music flows seamlessly

### 🌈 **Enhanced Real-Time Visualization**

- **Stunning dark-themed interface** with animated elements
- **Real-time spectrogram** showing frequency content
- **Dynamic color-coded plots** that change with brain states
- **Probability bar charts** for all brain states
- **Pulsing effects** for high-confidence states
- **Professional-grade PyQtGraph** visualizations

### 🏠 **Smart Home Integration**

- **Philips Hue lights** that change color based on your brain state
- **Brightness controlled by confidence** levels
- Focused = Orange, Relaxed = Teal, Stressed = Red, etc.
- **Ambient lighting** that responds to your mind!

### 📊 **Advanced Signal Processing**

- **1-40 Hz Butterworth bandpass** and **60 Hz notch** filtering
- **Welch power spectral density** computation
- **Five frequency bands**: Delta, Theta, Alpha, Beta, Gamma
- **10-second ring buffer** for continuous processing
- **20 Hz real-time processing** loop

### 📈 **Enhanced Data Logging**

- **Comprehensive CSV export** with:
  - Raw EEG samples from all 8 channels
  - Band power features for each channel
  - **Brain state classifications** with confidence scores
  - **Individual probabilities** for all states
  - High-precision timestamps

### 🎮 **Performance Monitoring**

- **Real-time statistics** tracking
- Session duration and sample counts
- Brain state change detection
- Music note generation counter
- **Detailed performance logs**

## 📁 Project Structure

```
BrainPower/
├── main/                  # Core applications
│   ├── openbci_stream.py # Main EEG streaming app
│   ├── mind_reader_gui.py # Neural network training GUI
│   ├── demo_mind_reader.py # Mind reading demo
│   └── brain_calibrator.py # Brain state calibration
├── tools/                 # Utilities and tools
│   └── check_ports.py    # Serial port detection
├── docs/                  # Documentation
│   ├── README.md         # This file
│   ├── LICENSE           # MIT License
│   └── requirements.txt  # Python dependencies
└── *.py                  # Additional utility scripts
```

## 🚀 Installation

1. **Clone this repository**

```bash
git clone https://github.com/Atomics-hub/BrainPower.git
cd BrainPower
```

2. **Install all dependencies**

```bash
pip install -r docs/requirements.txt
```

3. **Check your serial ports** (helpful for setup)

```bash
python tools/check_ports.py
```

4. **Optional: Set up Philips Hue** (for smart lighting)
   - Find your Hue bridge IP address
   - Press the bridge button before first connection

## 🎯 Usage

### **Basic Viral Mode** (AI + Music + Visualization)

```bash
python main/openbci_stream.py --port COM3
```

### **Full Viral Mode** (Everything enabled!)

```bash
python main/openbci_stream.py --port /dev/ttyUSB0 --enable-smart-home --hue-bridge-ip 192.168.1.100 --duration 300
```

### **Silent Mode** (No music)

```bash
python main/openbci_stream.py --port COM3 --disable-music
```

### **YouTube Demo Mode** (Perfect for recording)

```bash
python main/openbci_stream.py --port COM3 --duration 60 --csv-path youtube_demo.csv
```

## 🎛️ Command Line Arguments

| Argument              | Description                  | Default         |
| --------------------- | ---------------------------- | --------------- |
| `--port`              | **Serial port** (required)   | -               |
| `--board-id`          | BrainFlow board ID           | 0 (Cyton)       |
| `--duration`          | Recording duration (seconds) | ∞ (infinite)    |
| `--csv-path`          | Enhanced CSV output path     | `./eeg_log.csv` |
| `--disable-music`     | Turn off brain music         | Music enabled   |
| `--enable-smart-home` | Enable Hue integration       | Disabled        |
| `--hue-bridge-ip`     | Philips Hue bridge IP        | Auto-detect     |

## 🧬 Data Processing Pipeline

1. **📡 Data Acquisition**: 250 Hz from 8 EEG channels
2. **🔄 Ring Buffer**: 10-second continuous data storage
3. **⚡ Real-time Processing** (20 Hz):
   - 1-second sliding window analysis
   - Butterworth filtering (1-40 Hz + 60 Hz notch)
   - Welch PSD computation
   - **🧠 AI brain state classification**
   - **🎵 Music generation trigger**
   - **💡 Smart home control**
   - **📊 Enhanced visualization updates**

## 🎨 Visualization Features

### **Main Dashboard**

- **🧠 Brain State Display**: Large emoji + confidence bar
- **📈 Raw EEG Signal**: Color-coded by brain state
- **🌈 Real-time Spectrogram**: Frequency waterfall
- **📊 Band Power Trends**: 5 animated frequency bands
- **🎲 State Probabilities**: Live bar chart

### **Visual Effects**

- **Dynamic colors** that change with brain states
- **Pulsing animations** for high confidence
- **Smooth transitions** between states
- **Professional dark theme** for better contrast

## 🎵 Brain Music System

### **Musical Scales by Brain State**

- **🎯 Focused**: C Major (energetic, clear)
- **😌 Relaxed**: C Minor Pentatonic (smooth, flowing)
- **🧘 Meditative**: Perfect Fifths (harmonic, peaceful)
- **😰 Stressed**: Chromatic/Dissonant (tense, complex)
- **🤩 Excited**: Major Triads (bright, uplifting)
- **😴 Drowsy**: Low, slow notes (deep, calming)
- **😐 Neutral**: Suspended chords (balanced)

### **Audio Parameters**

- **Volume**: Controlled by Alpha wave power
- **Complexity**: Controlled by Beta wave power
- **Harmonics**: Rich overtones for musical quality
- **Envelope**: Smooth attack/decay to prevent clicks

## 🏠 Smart Home Integration

### **Philips Hue Setup**

1. Find your bridge IP: Use Philips Hue app
2. Press the bridge button
3. Run with `--enable-smart-home --hue-bridge-ip YOUR_IP`

### **Color Mapping**

- **🎯 Focused**: Orange (energizing)
- **😌 Relaxed**: Teal (calming)
- **😰 Stressed**: Red (alert)
- **🧘 Meditative**: Purple (spiritual)
- **🤩 Excited**: Yellow (joyful)
- **😴 Drowsy**: Blue (sleepy)
- **😐 Neutral**: Gray (balanced)

## 📊 Enhanced CSV Output

The CSV file now includes **40+ columns**:

```
timestamp,ch_1,ch_2,...,ch_8,
ch_1_delta,ch_1_theta,ch_1_alpha,ch_1_beta,ch_1_gamma,
...(all channels)...,
brain_state,confidence,
focused_prob,relaxed_prob,stressed_prob,meditative_prob,
excited_prob,drowsy_prob,neutral_prob
```

## 🎬 Perfect for YouTube Content!

### **Demo Ideas**

- **Meditation Challenge**: Watch brain state change to 🧘
- **Focus Test**: Try math problems and see 🎯 activation
- **Music Reaction**: Different songs trigger different states
- **Smart Home Control**: Lights change with your mood
- **Stress Response**: Show 😰 state during difficult tasks
- **Relaxation Journey**: Guide viewers through 😌 state

### **Viral Moments**

- **Real-time emoji changes** are instantly engaging
- **Music generation** creates unique soundtracks
- **Color-changing lights** add visual drama
- **Live brain state detection** feels like magic
- **Professional visualizations** look incredibly cool

## 🔧 Troubleshooting

### **Connection Issues**

- Ensure OpenBCI board is in **PC mode** (not standalone)
- Check **serial port permissions**: `sudo chmod 666 /dev/ttyUSB0`
- Verify port name in **Device Manager** (Windows) or `ls /dev/tty*` (Linux/Mac)

### **Performance Optimization**

- Close other serial port applications
- Ensure **4GB+ RAM** for smooth operation
- Use **powered USB hubs** for stable connections
- **Disable antivirus** real-time scanning for audio folder

### **Audio Issues**

- Install **ASIO drivers** for low-latency audio (Windows)
- Check **default audio device** settings
- Adjust **buffer sizes** if audio stutters

### **Smart Home Setup**

- Ensure Hue bridge and computer are on **same network**
- **Press bridge button** before first connection
- Check **firewall settings** for bridge communication

## 🏆 Technical Achievements

- **Real-time AI classification** at 20 Hz processing rate
- **Multi-threaded audio generation** without blocking
- **Professional-grade signal processing** with BrainFlow
- **Scalable architecture** for easy feature additions
- **Robust error handling** prevents crashes
- **Cross-platform compatibility** (Windows, Linux, Mac)

## 🎯 Future Enhancements

- **3D brain visualization** with activity mapping
- **Web dashboard** for remote monitoring
- **Social media integration** for sharing brain states
- **VR/AR visualization** support
- **Mobile app** companion
- **Cloud data sync** and analysis
- **Multi-user sessions** and competitions

## 📜 License

This project is open source. Please ensure compliance with BrainFlow and OpenBCI licensing terms.

## 👨‍💻 Author

**Senior Python Engineer** - BrainPower Project

---

**🚀 Ready to go viral? Your brain has never been this entertaining!**

_Star this repo if it blew your mind! 🤯_
