# Prothestic ada hand
prothestic hand that grasps objecs automatically

# Setup
First, for uploading the arduino code to your board, open the file ``ada_hand.ino`` inside the arduino folder, select the board ``Arduino Mega or Mega 2560`` and upload.

Now, to install all the libraries needed, just run the command bellow:
```
$ pip install -r requirements.txt
```

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
