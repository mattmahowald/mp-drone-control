#!/usr/bin/env python3
"""
Simple script to make the drone hover in place.
Usage:
    poetry run python scripts/simple_hover.py
"""

import time
import cflib.crtp
from cflib.crazyflie import Crazyflie

# --- Crazyflie drone control ---
DRONE_URI = "radio://0/80/2M/E7E7E7E7E7"

def main():
    # Initialize the drivers
    cflib.crtp.init_drivers()
    
    # Create and connect to the drone
    cf = Crazyflie()
    print("Connecting to drone...")
    cf.open_link(DRONE_URI)
    print("Connected!")
    
    try:
        # Hover parameters
        roll = 0
        pitch = 0
        yaw = 0
        thrust = 50000  # Increased thrust for better hover
        
        print("Starting hover...")
        print("Press Ctrl+C to stop")
        
        # Keep sending hover commands
        while True:
            cf.commander.send_setpoint(roll, pitch, yaw, thrust)
            time.sleep(0.1)  # Send command every 100ms
            
    except KeyboardInterrupt:
        print("\nStopping hover...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Emergency stop
        cf.commander.send_setpoint(0, 0, 0, 0)
        cf.close_link()
        print("Drone connection closed.")

if __name__ == "__main__":
    main() 