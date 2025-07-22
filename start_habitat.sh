#!/usr/bin/env bash
# file: start_habitat_stack.sh
# usage:  bash start_habitat_stack.sh

# 1. core – background
roscore >/tmp/roscore.log 2>&1 &
ROSCORE_PID=$!
echo "✓ roscore started  (pid $ROSCORE_PID)"

# 2. Habitat roamer – background
python src/scripts/roam_with_joy.py \
  --hab-env-config-path  ~/catkin_ws/src/ros_x_habitat/configs/roam_configs/pointnav_rgbd_roam_mp3d_test_scenes.yaml \
  --episode-id -1 \
  --scene-id data/scene_datasets/mp3d/1LXtFkjw3qL/1LXtFkjw3qL.glb \
  --video-frame-period 10 \
  >/tmp/roam_with_joy.log 2>&1 &
JOY_PID=$!
echo "✓ roam_with_joy.py started  (pid $JOY_PID)"

# 3. dummy subscriber – background
rostopic echo /pointgoal_with_gps_compass \
  >/tmp/pointgoal.log 2>&1 &
ECHO_PID=$!
echo "✓ rostopic echo started  (pid $ECHO_PID)"

# Give the background nodes a moment to come up
sleep 3

# 4. mapping / visual SLAM – foreground
roslaunch launch/rtabmap_mapping_021.launch

# ───────────────────── cleanup on Ctrl-C ────────────────────────────────
trap "echo 'Stopping…'; kill $ECHO_PID $JOY_PID $ROSCORE_PID" EXIT
