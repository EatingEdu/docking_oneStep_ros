# run by sudo bash UAV1.bash

# sudo chmod 777 /dev/ttyACM0

# cd /home/jetson/airdocking/src/transmitacr/launch

# roslaunch UAV1mavros.launch

# roslaunch sample.launch

# bash ./UAV1.bash


#1. roslaunch /home/jetson/airdocking/src/transmitacr/launch/UAV1mavros.launch

#2.  roslaunch /home/jetson/airdocking/src/transmitacr/launch/sample.launch

#3.  bash /home/jetson/airdocking/src/transmitacr/launch/UAV1.bash

#4. python /home/jetson/airdocking/src/velocity_control/scripts/control_velocity_pid_vrpn.py
rosrun mavros mavcmd -n /child1/mavros long 511 32 450 0 0 0 0 0
rosrun mavros mavcmd -n /child1/mavros long 511 31 450 0 0 0 0 0
rosrun mavros mavcmd -n /child1/mavros long 511 65 900 0 0 0 0 0
rosrun topic_tools relay /vrpn_client_node2/uav1/pose /child1/mavros/vision_pose/pose

# rosrun topic_tools relay /vrpn_client_node1/UAV1/pose /child1/mavros/vision_pose/pose

# python /home/jetson/airdocking/src/velocity_control/scripts/control_velocity_pid_vrpn.py


# rostopic echo /child1/mavros/local_position/pose
# rostopic echo /child2/mavros/local_position/pose
# rostopic echo /child2/nominal_pos_enu 
# rostopic echo /nominal_pid_pos1



# rostopic echo /mavros/local_position/pose
# rostopic echo /vrpn_client_node1/UAV1/pose 
# rostopic echo /vrpn_client_node2/UAV2/pose 
# rostopic echo /vrpn_client_node/UAV1/pose 
# 


# 只能在一架启动

# rostopic echo /child1/mavros/state
