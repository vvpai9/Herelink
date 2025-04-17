# Herelink
This repository guides the reader to interface Herelink with PixHawk and run autonomous missions through Python scripts.

Make connections as shown in the figure:

![image](https://github.com/user-attachments/assets/78ef98ca-e857-46f3-9b7b-a7125d76a78a)

Refer to this link for detailed documentation:
<br /> https://docs.cubepilot.org/user-guides/herelink/herelink-user-guides/assembly-and-connection

1. Power the Herelink Air Unit with a recommended power supply with minimum 6V supply (Recommended: 7V - 12V; 4W).
2. Turn on the Herelink Transmitter and connect to the drone via QGC.
3. Connect the laptop to the Transmitter via ```herelink``` hotspot. Connect Mission Planner via laptop through herelink via UDP.
4. The drone will now be publishing telemetry on ```udpout:192.168.144.10:14552``` by default if the herelink is not connected to the internet. If connected to the internet, the IP address and port may vary. Go to  ```Settings -> About Phone -> Status -> IP Address```.
5. The default ```baud_rate``` is  ```115200```. This can be changed based on the baudrate supported by the flight controller.
6. These scripts are tested on Python 3.11.9

Requirements:
1. Python 3.9 or higher. (Python 3.13 or higher versions currently do not support all the functionalities)
2. Libraries: ```pymavlink```, ```geopy```, ```logging```

In ```terminal ``` if using ```Linux``` or ```Command Prompt``` if using ```Windows```, run the following command:
```
python3 <script_name>.py
```

Replace ```<script_name>``` by the name of the script you want to run.
1. ```herelink_basic.py``` allows you to run basic functionalities which can be later combined for performing autnomous missions.
2. ```herelink_test.py``` is a test script. It attempts to connect to the drone using MAVLink and tries to fetch its home location and print it. It is highly recommended to run this script after appropriate connections.
3. ```herelink.py``` executes an autonomous survey and payload drop to a target site using a TensorFlow Lite custom trained model for surveillance and target detection using RTSP stream of the camera.

Refer https://chatgpt.com/share/67c2c947-93bc-8002-8072-b4ed4c4a010f for common issues
