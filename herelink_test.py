import time
from pymavlink import mavutil

print("Connecting to drone...")
time.sleep(2)
master = mavutil.mavlink_connection("udpout:192.168.144.10:14552")

target_lon, target_lat = None, None

# Send a ping to verify connection
master.mav.ping_send(
    int(time.time() * 1e6),  # Unix time in microseconds
    0,  # Ping number
    0,  # Request ping of all systems
    0)   # Request ping of all components
    
# Wait for the first heartbeat to confirm connection
master.wait_heartbeat()
# print("Heartbeat received! Drone is online.")

msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True)       
if msg:
    print("Heartbeat received! Drone is online.")
    target_lat = msg.lat / 1e7
    target_lon = msg.lon / 1e7
    target_alt = msg.relative_alt / 1000.0
    print(target_lat, target_lon)

print(target_lat, target_lon)