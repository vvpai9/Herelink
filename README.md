# Herelink
This repository guides the reader to interface Herelink with PixHawk and run autonomous missions through Python scripts.

Make connections as shown in the figure:

![image](https://github.com/user-attachments/assets/78ef98ca-e857-46f3-9b7b-a7125d76a78a)

Refer to this link for detailed documentation:
\n https://docs.cubepilot.org/user-guides/herelink/herelink-user-guides/assembly-and-connection

1. Power the Herelink Air Unit with a recommended power supply with minimum 6V supply (Recommended: 7V - 12V; 4W).
2. Turn on the Herelink Transmitter and connect to the drone via QGC.
3. Connect the laptop to the Transmitter via ```herelink``` hotspot. Connect Mission Planner via laptop through herelink via UDP.
4. The drone will now be publishing telemetry on ```udpout:192.168.144.10:14552``` by default if the herelink is not connected to the internet. If connected to the internet, the IP address and port may vary. Go to  ```Settings -> About Phone -> Status -> IP Address```. The default ```baud_rate``` is  ```115200```.

Requirements:
1. Python 3.9 or higher
2. Libraries: ```pymavlink```, ```geopy```, ```logging```

In ```terminal ``` if using ```Linux``` or ```Command Prompt``` if using ```Windows```, run the following command:
```
python3 herelink.py
```
