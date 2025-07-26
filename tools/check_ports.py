#!/usr/bin/env python3
"""
🔍 OpenBCI Port Detection Utility

Helps users find available serial ports for their OpenBCI board.
Useful for troubleshooting connection issues.

Usage:
    python tools/check_ports.py

Author: BrainPower Project
"""

import sys
import platform
import serial
import serial.tools.list_ports


def check_available_ports():
    """Check and display available serial ports."""
    print("🔍 Scanning for available serial ports...")
    print("=" * 50)
    
    # Get list of available ports
    ports = serial.tools.list_ports.comports()
    
    if not ports:
        print("❌ No serial ports found!")
        print("\n🔧 Troubleshooting tips:")
        print("1. Make sure your OpenBCI dongle is plugged in")
        print("2. Check Device Manager (Windows) or dmesg (Linux)")
        print("3. Try different USB ports")
        return []
    
    available_ports = []
    
    print(f"✅ Found {len(ports)} serial port(s):")
    print()
    
    for i, port in enumerate(ports, 1):
        print(f"{i}. Port: {port.device}")
        print(f"   Description: {port.description}")
        print(f"   Hardware ID: {port.hwid}")
        
        # Check if it might be an OpenBCI device
        description_lower = port.description.lower()
        if any(keyword in description_lower for keyword in ['ftdi', 'usb', 'serial', 'rfcomm']):
            print("   🎯 Likely candidate for OpenBCI board!")
            available_ports.append(port.device)
        
        print()
    
    return available_ports


def test_port_connection(port_name):
    """Test if we can open a connection to the specified port."""
    print(f"🔌 Testing connection to {port_name}...")
    
    try:
        # Try to open the port
        ser = serial.Serial(port_name, 115200, timeout=2)
        print(f"✅ Successfully opened {port_name}")
        ser.close()
        return True
        
    except serial.SerialException as e:
        print(f"❌ Failed to open {port_name}: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error with {port_name}: {e}")
        return False


def get_platform_specific_tips():
    """Provide platform-specific troubleshooting tips."""
    system = platform.system().lower()
    
    print("🛠️ Platform-specific tips:")
    print("=" * 30)
    
    if system == "windows":
        print("Windows:")
        print("• Check Device Manager for COM ports")
        print("• OpenBCI typically appears as 'USB Serial Port (COM#)'") 
        print("• Try COM3, COM4, COM5, COM6 commonly")
        print("• Run as Administrator if you get permission errors")
        
    elif system == "linux":
        print("Linux:")
        print("• Ports typically appear as /dev/ttyUSB0, /dev/ttyUSB1, etc.")
        print("• You may need to add yourself to the dialout group:")
        print("  sudo usermod -a -G dialout $USER")
        print("• Then log out and back in")
        print("• Check permissions: ls -l /dev/ttyUSB*")
        
    elif system == "darwin":  # macOS
        print("macOS:")
        print("• Ports typically appear as /dev/cu.usbserial-* or /dev/tty.usbserial-*")
        print("• Use /dev/cu.* (callout) rather than /dev/tty.* (dialin)")
        print("• No special permissions usually needed")
    
    print()


def main():
    """Main function."""
    print("🧠 OpenBCI Port Detection Utility")
    print("=" * 40)
    print()
    
    # Check available ports
    available_ports = check_available_ports()
    
    # Test connections to likely candidates
    if available_ports:
        print("🧪 Testing connections to likely candidates...")
        print("-" * 45)
        
        working_ports = []
        for port in available_ports:
            if test_port_connection(port):
                working_ports.append(port)
        
        print()
        if working_ports:
            print("🎉 Working ports found:")
            for port in working_ports:
                print(f"✅ {port}")
            
            print()
            print("🚀 Try running BrainPower with:")
            for port in working_ports:
                print(f"   python src/openbci_stream.py --port {port}")
        else:
            print("⚠️ No working ports found")
    
    print()
    get_platform_specific_tips()
    
    print("💡 For synthetic testing (no hardware needed):")
    print("   python src/openbci_stream.py --port COM999 --board-id -1")


if __name__ == "__main__":
    main() 