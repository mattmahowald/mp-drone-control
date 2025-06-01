#!/usr/bin/env python3
import cflib.crtp
from cflib.crazyflie import Crazyflie

# (Replace with your Crazyflie's URI if needed)
DRONE_URI = "radio://0/80/2M/E7E7E7E7E7"

# --- ping (connect–disconnect) ---
cflib.crtp.init_drivers()
cf = Crazyflie()
try:
    cf.open_link(DRONE_URI)
    print("Ping OK – Connected to Crazyflie drone (URI: {})".format(DRONE_URI))
except Exception as e:
    print("Ping failed – Could not connect to Crazyflie drone (URI: {}): {}".format(DRONE_URI, e))
finally:
    cf.close_link() 