execute_process(COMMAND "/home/bioinlab/Desktop/carlosIgor/prothestic_ada_hand/ada_visual_ws/build/ada_visual_control/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/bioinlab/Desktop/carlosIgor/prothestic_ada_hand/ada_visual_ws/build/ada_visual_control/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
