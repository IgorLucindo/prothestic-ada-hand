#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/bioinlab/Desktop/carlosIgor/prothestic_ada_hand/ada_visual_ws/src/ada_visual_control"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/bioinlab/Desktop/carlosIgor/prothestic_ada_hand/ada_visual_ws/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/bioinlab/Desktop/carlosIgor/prothestic_ada_hand/ada_visual_ws/install/lib/python3/dist-packages:/home/bioinlab/Desktop/carlosIgor/prothestic_ada_hand/ada_visual_ws/build/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/bioinlab/Desktop/carlosIgor/prothestic_ada_hand/ada_visual_ws/build" \
    "/home/bioinlab/anaconda3/bin/python3" \
    "/home/bioinlab/Desktop/carlosIgor/prothestic_ada_hand/ada_visual_ws/src/ada_visual_control/setup.py" \
     \
    build --build-base "/home/bioinlab/Desktop/carlosIgor/prothestic_ada_hand/ada_visual_ws/build/ada_visual_control" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/bioinlab/Desktop/carlosIgor/prothestic_ada_hand/ada_visual_ws/install" --install-scripts="/home/bioinlab/Desktop/carlosIgor/prothestic_ada_hand/ada_visual_ws/install/bin"
