#!/usr/bin/env python3
import rospy
from uav_state import UAVState
from dual_uav_controller import DualUAVController

rospy.init_node("dual_uav_rl_controller", anonymous=True)

uav1 = UAVState("/child1")
uav2 = UAVState("/child2")

controller = DualUAVController(uav1, uav2)

rospy.spin()
