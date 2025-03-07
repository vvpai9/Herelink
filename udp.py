from dronekit import connect, LocationGlobalRelative

vehicle = None

# Custom mode handler to bypass DroneKit errors
def custom_mode_handler(_, name, msg):
    try:
        if vehicle is None:
            print("Vehicle not initialized yet, skipping mode handling.")
            return
        
        if not hasattr(vehicle, "_mode_mapping_bynumber") or msg.custom_mode not in vehicle._mode_mapping_bynumber:
            print(f"Unknown mode {msg.custom_mode}, ignoring...")
            return
        
        vehicle._flightmode = vehicle._mode_mapping_bynumber[msg.custom_mode]
        print(f"Vehicle mode set to: {vehicle.mode}")

    except Exception as e:
        print(f"Mode handling error: {e}")

try:
    print("Connecting to vehicle...")
    vehicle = connect('udpout:192.168.144.10:14552', wait_ready=True, timeout=60)  # ✅ FIXED `udp:`
    
    if vehicle:
        print("Vehicle connected successfully")
        print("Vehicle Location: ", vehicle.location.global_frame)
        
        # Manually replace the mode listener to prevent crashes
        vehicle.add_message_listener('HEARTBEAT', custom_mode_handler)

except KeyboardInterrupt:
    print("Aborted by user")
except Exception as e: 
    print(f"Error connecting to vehicle: {e}")
    if vehicle:
        vehicle.close()
