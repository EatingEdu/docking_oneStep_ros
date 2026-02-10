#!/usr/bin/env python3
import rospy
from uav_state import UAVState
from dual_uav_controller import DualUAVController

rospy.init_node("dual_uav_rl_controller", anonymous=True)

uav1 = UAVState("/child1", mass = 1.5)
uav2 = UAVState("/child2", mass = 2.1)

controller = DualUAVController(uav1, uav2)

rospy.spin()
