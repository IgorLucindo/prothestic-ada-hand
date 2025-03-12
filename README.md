# Prothestic Ada Hand
Budget prothestic hand that grasps objecs automatically.

# Setup
First, to install all the libraries needed, just run the command bellow:
```
$ pip install -r requirements.txt
```

Now for installing the ros packages, follow the instructions in this link http://wiki.ros.org/ROS/Installation.

The next step is to upload the arduino code to your board, for that, open the file ``ada_hand.ino`` inside the arduino folder, select the board ``Arduino Mega or Mega 2560`` and upload the code.

After that, run the following commands to start and setup the ros workspace:
```
$ cd ada_visual_ws
$ catkin_make
$ source devel/setup.bash
```

# Execution
Now, every time you want to start the application, just run:
```
$ roslaunch ada_visual_control ada_visual_init.launch
```
