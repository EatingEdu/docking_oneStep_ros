from std_msgs.msg import Float32, Float64MultiArray, Bool, Int32
from geometry_msgs.msg import Vector3, Point, PoseStamped, TwistStamped
from mavros_msgs.msg import State, AttitudeTarget, RCIn
# import pdb
# pdb.set_trace()
import numpy as np


sw = 1099
# global  sw, start, first, randinit_pos, err_pos, rot
# r_now=Vector3()
r_now=np.zeros(9)
err_pos = np.zeros(3)
r_d = np.zeros(9)
# rot = np.zeros(9)
pos = np.zeros(3)
vel = np.zeros(3)
quat = np.zeros(4)
nominal_pos = np.zeros(3)
#randinit_pos = np.zeros(3)



local_pos = PoseStamped()
px4_command = AttitudeTarget()
px4_command.type_mask = 128

mode = TwistStamped()
nominal_pos_enu = Point()
# nominal_pos_enu
randinit_pos = PoseStamped()
mav_vel_receive = TwistStamped()
control_name = "rl"


arm = False
start = 0 #0
mode = State()
offboard_start = Int32()
first =0
timev = 0