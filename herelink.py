import logging
import sys
import time
from datetime import datetime
from pymavlink import mavutil
from geopy.distance import geodesic
import threading
import argparse
import numpy as np
import importlib.util
import os
import cv2

# Configure logging
log_filename = "mission_debug.log"
logging.basicConfig(
    level=logging.DEBUG,  # Capture all levels of logs (DEBUG, INFO, ERROR)
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, mode='a'),  # Write logs to file
        logging.StreamHandler(sys.stdout)  # Still print to console
    ]
)

# Redirect stdout and stderr to logger
class LoggerWriter:
    def __init__(self, level):
        self.level = level
        self.buffer = ''

    def write(self, message):
        if message != '\n':
            self.buffer += message
        if '\n' in message:
            self.flush()

    def flush(self):
        if self.buffer:
            self.level(self.buffer)
            self.buffer = ''

sys.stdout = LoggerWriter(logging.info)
sys.stderr = LoggerWriter(logging.error)

logging.info("-----------------------------------------------") # Separator for new runs

# Establish connection to the drone
logging.info("Connecting to drone...")
time.sleep(2)
master = mavutil.mavlink_connection("udpout:192.168.144.10:14552")

# Send a ping to verify connection
master.mav.ping_send(
    int(time.time() * 1e6),  # Unix time in microseconds
    0,  # Ping number
    0,  # Request ping of all systems
    0)   # Request ping of all components
    
# Wait for the first heartbeat to confirm connection
master.wait_heartbeat()
logging.info("Heartbeat received! Drone is online.")

msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True)       
if msg:
    target_lat = msg.lat / 1e7
    target_lon = msg.lon / 1e7
    target_alt = msg.relative_alt / 1000.0

# ArduPilot modes mapped to their IDs
MODES = {
    "STABILIZE": 0,
    "ACRO": 1,
    "ALT_HOLD": 2,
    "AUTO": 3,
    "GUIDED": 4,
    "LOITER": 5,
    "RTL": 6,
    "CIRCLE": 7,
    "LAND": 9,
    "DRIFT": 11,
    "SPORT": 13,
    "FLIP": 14,
    "AUTOTUNE": 15,
    "POSHOLD": 16,
    "BRAKE": 17,
    "THROW": 18,
    "AVOID_ADSB": 19,
    "GUIDED_NO_GPS": 20,
    "SMART_RTL": 21,
    "FLOWHOLD": 22,
    "FOLLOW": 23,
    "ZIGZAG": 24,
    "SYSTEMID": 25,
    "AUTOROTATE": 26,
    "AUTO_RTL": 27
}

stop_detection = False
interruption_flag = threading.Event()
stop_rtsp = False
targetDetectFlag = True
cap = None
latest_frame = None
frame_lock = threading.Lock()

# Define geofence
geofence = [
    (15.367808276128258, 75.12540728300411),  # Bottom-Left
    (15.367466640407201, 75.12536764581941),  # Bottom-Right
    (15.36743396482936, 75.12575188156819),  # Top-Right
    (15.367774321214313, 75.1258001664985)   # Top-Left
] # Change values as required

x_divisions = 5
y_divisions = 4
altitude = 10

def rtsp_stream():
    """ Continuously captures frames from RTSP stream and feeds them into a queue for detection. """
    global cap, stop_rtsp, latest_frame
    try:
        rtsp_url = "rtsp://192.168.43.1:8554/fpv_stream"
        cap = cv2.VideoCapture(rtsp_url)

        if not cap.isOpened():
            logging.error("Error: Could not open RTSP stream")
            return

        while not stop_rtsp:
            ret, frame = cap.read()
            if not ret:
                print("Error in RTSP")
                logging.error("Error: Could not read frame")
                continue

            with frame_lock:
                latest_frame = frame.copy()  # Always store the latest frame

            # Display video feed (optional)
            cv2.imshow("Herelink RTSP Stream", frame)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        logging.error("RTSP stream interrupted by user.")
    except Exception as e:
        logging.error(e)
    finally:
        stop_rtsp = True
        if cap:
            cap.release()
        cv2.destroyAllWindows()

def detectionClassify():
    global targetDetectFlag, stop_detection, latest_frame, interruption_flag
    print("Detection thread started")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"image_{timestamp}.mp4"
    frameCount = 0
    targetCount = 0
    start_time = time.time()


    # Define and parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                        required=False,default='targets')
    parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                        default='detect.tflite')
    parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                        default='labelmap.txt')
    parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                        default=0.8)
    parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                        default='640x480')
    parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                        action='store_true')

    args = parser.parse_args()

    MODEL_NAME = args.modeldir
    GRAPH_NAME = args.graph
    LABELMAP_NAME = args.labels
    min_conf_threshold = float(args.threshold)
    resW, resH = args.resolution.split('x')
    imW, imH = int(resW), int(resH)
    use_TPU = args.edgetpu

    # Import TensorFlow libraries
    # If using Coral Edge TPU, import the load_delegate library
    pkg = importlib.util.find_spec('tflite_runtime')
    if pkg:
        from tflite_runtime.interpreter import Interpreter
        if use_TPU:
            from tflite_runtime.interpreter import load_delegate
    else:
        from tensorflow.lite.python.interpreter import Interpreter
        if use_TPU:
            from tensorflow.lite.python.interpreter import load_delegate

    if use_TPU:
        # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
        if (GRAPH_NAME == 'detect.tflite'):
            GRAPH_NAME = 'edgetpu.tflite'       

    # Get path to current working directory
    CWD_PATH = os.getcwd()

    # Path to .tflite file, which contains the model that is used for object detection
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

    # Load the label map
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # First label is '???', which has to be removed.
    if labels[0] == '???':
        del(labels[0])

    # Load the Tensorflow Lite model.
    if use_TPU:
        interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
        print(PATH_TO_CKPT)
    else:
        interpreter = Interpreter(model_path=PATH_TO_CKPT)

    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # because outputs are ordered differently for TF2 and TF1 models
    outname = output_details[0]['name']

    if ('StatefulPartitionedCall' in outname): # This is a TF2 model
        boxes_idx, classes_idx, scores_idx = 1, 3, 0
    else: # This is a TF1 model
        boxes_idx, classes_idx, scores_idx = 0, 1, 2

    # Initialize video stream
    frame_width = int(640)
    frame_height = int(480)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4
    # fourcc = cv2.VideoWriter_fourcc(*'X264')  # Use 'X264' for .mkv
    out = cv2.VideoWriter(filename, fourcc, 20.0, (frame_width, frame_height))

    #for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
    try:
        while not stop_detection:
            # Acquire frame and resize to expected shape [1xHxWx3]
            frame = latest_frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame, (width, height))
            input_data = np.expand_dims(frame_resized, axis=0)

            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            if floating_model:
                input_data = (np.float32(input_data) - input_mean) / input_std

            # Perform the actual detection by running the model with the image as input
            interpreter.set_tensor(input_details[0]['index'],input_data)
            interpreter.invoke()

            # Retrieve detection results
            boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
            classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
            scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

            # Loop over all detections and draw detection box if confidence is above minimum threshold
            for i in range(len(scores)):
                if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))
                    
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 4)
                    cx = (xmax+xmin)//2
                    cy = (ymax+ymin)//2
                    #print("The coordinates:",cx,cy)

                    # Draw label
                    object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

                    if object_name == "target" :
                       with open("YelloWcoordinates.txt", "a") as f:
                               f.write(f"{cx}, {cy} \n")
                            #    print("target The coordinates:",cx,cy,object_name)

                    if (object_name == "target") and targetDetectFlag:
                        with open("pixel.txt", "a") as f:
                                f.write(f"{cx}, {cy} \n")
                        print("The coordinates:",cx,cy,object_name)     
                        targetCount +=1
                    elif object_name == "hotspot":
                        # if abs(cx-320)==35 and abs(cy-240)==35:
                        #     cv2.imwrite(f'{frameCount}.jpg',frame)
                        #     frameCount +=1
                        frameCount +=1
                        cv2.imwrite(f'{frameCount}.jpg',frame)

            if targetDetectFlag and targetCount >= 4:
                interruption_flag.set()
                time.sleep(1)
            current_time = time.time()            
            if current_time - start_time >= 2 and targetDetectFlag:
                targetCount = 0 
                start_time = current_time

            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow('Detect', frame)
            frame_output = cv2.resize(frame, (int(imW), int(imH)))
            out.write(frame_output)
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Recording stopped by user")
                break

    except KeyboardInterrupt:
        print("interrupted by user")
    finally:
        # Release everything when done
        out.release()
        cv2.destroyAllWindows()

    # Clean up
    out.release()
    cv2.destroyAllWindows()

def take_picture(detection_queue):
    global targetDetectFlag,target_lat, target_lon, target_alt  # Add this line to modify the global variable
    logging.info("taking picture")
    for i in range(4):
        frame = detection_queue.pop(0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #filename = f"image_{timestamp}_lat{location.lat}_lon{location.lon}_alt{location.alt}.jpg"
        filename = f"image_{timestamp}{i}.jpg"
        # time.sleep()
        cv2.imwrite(filename, frame)
        time.sleep(0.3)
        logging.info(f"Saved image: {filename}")
    # time.sleep(0.1)
    msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True)       
    if msg:
        target_lat = msg.lat / 1e7
        target_lon = msg.lon / 1e7
        target_alt = msg.relative_alt / 1000.0
    time.sleep(0.3)
    targetDetectFlag = False

def generate_grid(geofence, x_divisions, y_divisions):
    """Generate a search grid based on geofence coordinates."""
    logging.info("Generating grid...")
    
    bottom_left, bottom_right, top_right, top_left = geofence
    lat_start, lon_start = bottom_left
    lat_end, lon_end = top_right

    lat_step = (lat_end - lat_start) / y_divisions
    lon_step = (lon_end - lon_start) / x_divisions

    grid = []
    for i in range(y_divisions + 1):
        row = []
        for j in range(x_divisions + 1):
            lat = lat_start + i * lat_step
            lon = lon_start + j * lon_step
            row.append((lat, lon))
        grid.append(row)
    
    return grid

def serpentine_path(grid):
    """Follow a serpentine path over the grid."""
    for i, row in enumerate(grid):
        path = row if i % 2 == 0 else reversed(row)
        for point in path:
            lat, lon = point
            go_to_location(lat, lon, altitude)
            time.sleep(0.5)

# Function to change mode and confirm it
def set_mode(mode):
    # Convert mode name to uppercase just in case user sends 'guided' or 'rtl'
    mode = mode.upper()

    # Validate the mode
    if mode not in MODES:
        raise ValueError(f"Mode '{mode}' not recognized. Available modes: {list(MODES.keys())}")

    mode_id = MODES[mode]
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id
    )
    logging.info(f"Mode change command sent for mode ID {mode}")
    # Confirm mode change with timeout
    start_time = time.time()
    timeout = 10  # seconds
    while time.time() - start_time < timeout:
        msg = master.recv_match(type='HEARTBEAT', blocking=True)
        if msg and msg.custom_mode == mode_id:
            logging.info(f"Mode successfully changed to ID {mode}")
            return
    logging.error(f"Mode change to '{mode}' failed or timed out.")

def move_ned_frame(target_lat, target_lon):
    """
    Moves the drone first along the X-axis (North) and then Y-axis (East) sequentially.
    Sends 1 m/s velocity until the distance is covered.
    """
    msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
    current_lat = msg.lat / 1e7
    current_lon = msg.lon / 1e7

    # Calculate distances separately for latitude and longitude
    distance_x = geodesic((current_lat, current_lon), (target_lat, current_lon)).meters  # North-South
    distance_y = geodesic((target_lat, current_lon), (target_lat, target_lon)).meters  # East-West

    # Adjust signs based on target position
    if target_lat < current_lat:
        distance_x = -distance_x  # Move South
    if target_lon < current_lon:
        distance_y = -distance_y  # Move West

    logging.info(f"Distance X (North/South): {distance_x:.2f} meters")
    logging.info(f"Distance Y (East/West): {distance_y:.2f} meters")

    # Move along the X-axis (Right of Drone)
    if abs(distance_x) > 0:
        direction_x = 1 if distance_x > 0 else -1  # Determine if moving North or South
        duration_x = int(abs(distance_x))  # Time to travel at 1 m/s
        logging.info(f"Moving along X for {duration_x} seconds.")
        send_ned_velocity(0, direction_x, 0, duration=duration_x)

    # Move along the Y-axis (Front of Drone)
    if abs(distance_y) > 0:
        direction_y = 1 if distance_y > 0 else -1  # Determine if moving East or West
        duration_y = int(abs(distance_y))  # Time to travel at 1 m/s
        logging.info(f"Moving along Y for {duration_y} seconds.")
        send_ned_velocity(direction_y, 0, 0, duration=duration_y)

    logging.info("Reached target location.")

# Function to arm the drone and takeoff to altitude
def arm_and_takeoff(altitude):
    logging.info("GUIDED Mode")
    set_mode('STABILIZE')
    time.sleep(2)
    
    logging.info("Arming motors...")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1,  # 1 to arm, 0 to disarm
        0, 0, 0, 0, 0, 0
    )

    time.sleep(2)
    '''logging.info(f"Initiating takeoff to {altitude} meters...")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0, 0, 0, 0, 0, 0, altitude
    )
    logging.info("Takeoff command sent.")
    while True:
        msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
        if msg:
            current_altitude = msg.relative_alt / 1000.0  # Convert from mm to meters
            logging.info(f"Altitude: {current_altitude:.2f} meters")
            if current_altitude >= altitude * 0.95:
                logging.info("Reached target altitude.")
                break'''

# Function to send velocity command in NED frame
def send_ned_velocity(velocity_x, velocity_y, velocity_z, duration=1):
    for _ in range(duration):
        master.mav.set_position_target_local_ned_send(
            0,  # time_boot_ms (not used)
            master.target_system, # target system
            master.target_component, # target component
            mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED, # frame
            0b10111000111,  # type_mask (only speeds enabled)
            0, 0, 0, # x, y, z positions (not used)
            velocity_x, velocity_y, velocity_z,  # x, y, z velocity in m/s
            0, 0, 0, # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
            0, 0) # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)
        time.sleep(1)

# Function to drop payload using servo
def drop_payload(PWM):
    logging.info("Dropping payload...")
    master.mav.command_long_send(
        master.target_system, # target_system
        master.target_component, # target_component
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO, # command
        0, # confirmation
        9,  # Servo number
        PWM, # servo position between 1000 and 2000
        0, 0, 0, 0, 0) # param 3 ~ 7 not used
    logging.info("Payload Dropped.")

def go_to_location(latitude, longitude, altitude, groundspeed=3):
    """
    Sends a command to go to the specified GPS location with adjustable groundspeed.
    
    Parameters:
    - latitude: Target latitude in degrees
    - longitude: Target longitude in degrees
    - altitude: Target altitude in meters (relative to home position)
    - groundspeed: Desired speed in m/s
    """
    # Set the desired groundspeed
    master.mav.command_long_send(
        0, 0,  # Target system and component ID
        mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,  # Command to change speed
        0,  # Confirmation
        0,  # Speed type (0 = airspeed, 1 = groundspeed)
        groundspeed,  # Desired speed in m/s
        -1, 0, 0, 0, 0  # Unused parameters
    )
    logging.info(f"Groundspeed set to {groundspeed} m/s.")

    # Sending command using MAV_CMD_NAV_WAYPOINT
    master.mav.mission_item_send(
        0, 0,  # Target component ID
        0,  # Sequence number (position in mission)
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,  # Coordinate frame
        mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,  # Command type (fly to waypoint)
        2,  # Current (set to 1 if this is the current mission item)
        0,  # Auto-continue (move to next waypoint automatically)
        0, 0, 0, 0,  # Params 1-4 (unused here)
        latitude,  # Latitude of waypoint
        longitude,  # Longitude of waypoint
        altitude  # Altitude (in meters)
    )

    logging.info(f"Command sent to go to Latitude: {latitude}, Longitude: {longitude}, Altitude: {altitude}m")

    # Monitor distance to target
    while True:
        msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
        if msg:
            current_lat = msg.lat / 1e7
            current_lon = msg.lon / 1e7
            current_alt = msg.relative_alt / 1000.0
            distance = geodesic((latitude, longitude), (current_lat, current_lon)).meters
            logging.info(f"Distance to target: {distance:.2f} meters | Current Altitude: {current_alt:.2f} m")
            if distance <= 1.0 and abs(current_alt - altitude) <= 0.5:
                logging.info("Reached target location.")
                break

def mission():
    try:
        # Mission sequence
        global stop_rtsp, stop_detection
        rtsp_thread = threading.Thread(target=rtsp_stream)
        start_time = time.time()
        grid = generate_grid(geofence, x_divisions, y_divisions)
        logging.info("Mission Begins")
        rtsp_thread.start()
        detection_thread = threading.Thread(target=detectionClassify)
    
        # Arm and takeoff to 15 meters
        arm_and_takeoff(altitude)
        detection_thread.start()
        serpentine_path(grid)
        time.sleep(1)

        logging.info("RTL Mode")
        set_mode('RTL')
        time.sleep(2)

        while True:
            msg = master.recv_match(type='HEARTBEAT', blocking=True)
            if msg:
                # system_status: 4 means "Standby" (disarmed on ground)
                # base_mode bit 7 (128) = armed
                disarmed = (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
                if disarmed:
                    logging.info("Drone disarmed. Mission complete.")
                    break
                time.sleep(1)
    except KeyboardInterrupt:
        logging.error("Mission interrupted by user.")
        stop_detection = True
        stop_rtsp = True
    except Exception as e:
        logging.error(e)
    finally:
        stop_detection = True
        stop_rtsp = True
        end_time = time.time()
        detection_thread.join()
        rtsp_thread.join()
        cv2.destroyAllWindows()
        mission_time = end_time - start_time
        logging.info(f"Mission Time: {int(mission_time // 60)} minutes {int(mission_time % 60)} seconds")

if __name__ == "__main__":
    mission()
