# Prothestic ada hand
prothestic hand that grasps objecs automatically

# Setup
First, run the command bellow to install all the libraries needed to run the application:
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
