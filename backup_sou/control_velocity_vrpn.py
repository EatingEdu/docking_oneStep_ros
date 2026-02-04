from params import *
import numpy as np
from model_predict import *
#from model_predict_ppo import *
#from model_predict_pid import *
import rospy 
from std_msgs.msg import Float32, Float64MultiArray, Bool, Int32
from geometry_msgs.msg import Vector3, Point, PoseStamped, TwistStamped
from mavros_msgs.msg import State, AttitudeTarget, RCIn
from copy import deepcopy
import pdb
# pdb.set_trace()

def state_pos_cb(data):
    global state_pos
    state_pos = data
    print(f"state_pos is {state_pos}")

def nominal_position_cb(data):
    global nominal_pos
    nominal_pos = np.array([data.x,data.y,data.z])
    print(f"nominal_pos is {nominal_pos}")

def nominal_euler_angles_cb(data):
    global nominal_euler
    nominal_euler = data
    print(f"nominal_euler is {nominal_euler}")

def arm_cb(data):
    global arm
    arm = data
    print(f"arm is {arm}")


def get_rc_channel_cb(data):
    global sw, start2, first2
    get_rc_channel = data
    sw = get_rc_channel.channels[4]
    if sw > 19000:
        start2 = 1
    else:
        start2 = 0
        first2 = 0
    #print(f"mode is {mode}")
    #ropy.lognfo("------------------------------------------mode is {mode}")
    #print(f"sw is {sw}, start is {start}")

def state_callback(msg):
    # 在这里处理接收到的消息
    print('Received state: %s' % msg)
    
def mode_cb(data):
    global mode, start, first
    #pdb.set_trace()
    mode = data
    if mode.mode == 'OFFBOARD':
        start = 1
    else:
        start = 0
        first = 0
    #rospy.lognfo("------------------------------------------mode is {mode}")
    #print(start)
    #print(f"------------------------------------------mode is {mode}")

def local_pos_cb(data):
    global pos, quat, first ,randinit_pos, sw
    pos = np.array([data.pose.position.y, -data.pose.position.x, data.pose.position.z])
    #local_pos = PoseStamped()
    #print(f"pos is {pos}")
    
    #w,x,y,z
    quat = np.array([data.pose.orientation.w, 
                     data.pose.orientation.x, 
                     data.pose.orientation.y, 
                     data.pose.orientation.z])
    if data.pose.orientation.w < 0:
        quat = -np.array([data.pose.orientation.w, 
                     data.pose.orientation.x, 
                     data.pose.orientation.y, 
                     data.pose.orientation.z])
    if first == 0 and start == 1:
        randinit_pos = deepcopy(data)
        randinit_pos.pose.position.x = pos[0]
        randinit_pos.pose.position.y = pos[1]
        randinit_pos.pose.position.z = pos[2]
        first = 1
    elif first == 0:
        nominal_pos[0] = pos[0]
        nominal_pos[1] = pos[1]
        nominal_pos[2] = pos[2]
    # print(f"randinit_pos is ")
    # print(randinit_pos)
    # print(f" data is ")
    # print(data)
    # print(f"first is {first}, start is {start},sw is {sw}")

def mav_vel_receive_cb(data):
    global mav_vel_receive
    mav_vel_receive = data
    global vel
    vel = np.array([data.twist.linear.x,
                    data.twist.linear.y,
                    data.twist.linear.z])

def quat2rot_change(quat):
    global r_now
    w = quat[0]
    x = quat[1]
    y = quat[2] 
    z = quat[3]
    r_now[0] = 2*x*y + 2*z*w
    r_now[1] = 1.0 - 2*x*x - 2*z*z
    r_now[2] = 2*y*z - 2*x*w
    r_now[3] = 2*y*y + 2*z*z -1.0
    r_now[4] = 2*z*w - 2*x*y
    r_now[5] = -2*x*z - 2*y*w
    r_now[6] = 2*x*z - 2*y*w
    r_now[7] = 2*y*z + 2*x*w
    r_now[8] = 1.0 - 2*x*x -2*y*y

def body2worldVel():
    global r_now, vel
    local_vel = np.zeros(3)
    x = vel[0]
    y = vel[1]
    z = vel[2]
    local_vel[0] = r_now[0] * x + r_now[1] * y + r_now[2] * z
    local_vel[1] = r_now[3] * x + r_now[4] * y + r_now[5] * z
    local_vel[2] = r_now[6] * x + r_now[7] * y + r_now[8] * z
    return local_vel

def errorRot():
    global r_now, r_d
    rot = np.zeros(9)
    rot[0] = r_d[0] * r_now[0] + r_d[3] * r_now[3] + r_d[6] * r_now[6]
    rot[1] = r_d[0] * r_now[1] + r_d[3] * r_now[4] + r_d[6] * r_now[7]
    rot[2] = r_d[0] * r_now[2] + r_d[3] * r_now[5] + r_d[6] * r_now[8]
    rot[3] = r_d[1] * r_now[0] + r_d[4] * r_now[3] + r_d[7] * r_now[6]
    rot[4] = r_d[1] * r_now[1] + r_d[4] * r_now[4] + r_d[7] * r_now[7]
    rot[5] = r_d[1] * r_now[2] + r_d[4] * r_now[5] + r_d[7] * r_now[8]
    rot[6] = r_d[2] * r_now[0] + r_d[5] * r_now[3] + r_d[8] * r_now[6]
    rot[7] = r_d[2] * r_now[1] + r_d[5] * r_now[4] + r_d[8] * r_now[7]
    rot[8] = r_d[2] * r_now[2] + r_d[5] * r_now[5] + r_d[8] * r_now[8]
    return rot

def timecallback(event):
    global pos, quat, r_now, vel, nominal_pos, nominal_pos_enu, randinit_pos, r_d, err_pos, rot
    global timev, sw, first
    global px4_command
    global mav_vel_receive
    global control_name
    global mode, start

    print(f"mode is {mode}")
    #pdb.set_trace()
    if 1: #start == 1:
        quat2rot_change(quat)
        #pdb.set_trace()
        # nominal_pos[0] = -0.71
        # nominal_pos[1] = 0.82 #-1.2 #
        # nominal_pos[2] = 1.6
        print(f"first is {first}")
        if first == 1:
            nominal_pos[0] = randinit_pos.pose.position.x
            nominal_pos[1] = randinit_pos.pose.position.y
            nominal_pos[2] = randinit_pos.pose.position.z
            first = 2
        nominal_pos_enu.x = - nominal_pos[1]
        nominal_pos_enu.y =   nominal_pos[0]
        nominal_pos_enu.z =   nominal_pos[2]
        #print(control_name)
        nominal_pos_enu_pub.publish(nominal_pos_enu)
        
        first_pos_pub.publish(randinit_pos)
        timev = timev + 0.1

        err_pos[0] = pos[0] - nominal_pos[0]
        err_pos[1] = pos[1] - nominal_pos[1]
        err_pos[2] = pos[2] - nominal_pos[2]
        err_vel = body2worldVel()
        r_d[0] = 1
        r_d[1] = 0
        r_d[2] = 0
        r_d[3] = 0
        r_d[4] = 1
        r_d[5] = 0
        r_d[6] = 0
        r_d[7] = 0
        r_d[8] = 1
    
        rot_err = errorRot()
        state_error = np.array([err_pos[0], err_pos[1], err_pos[2],
                                err_vel[0], err_vel[1], err_vel[2],
                                rot_err[0], rot_err[1], rot_err[2],
                                rot_err[3], rot_err[4], rot_err[5],
                                rot_err[6], rot_err[7], rot_err[8],
                                #mav_vel_receive.twist.angular.x, mav_vel_receive.twist.angular.y, mav_vel_receive.twist.angular.z])
                                0,0,0])
        
        state_error_pub.publish(data = state_error)
        #pdb.set_trace()
        if control_name == "rl":
            action = modelPredict(state_error)
            #print(f"state_error is {state_error}")
        ## pid_control 
        if control_name == "pid":
            omega = np.array([mav_vel_receive.twist.angular.x, mav_vel_receive.twist.angular.y, mav_vel_receive.twist.angular.z])
            #goal = np.array([])
            action = modelPredict(pos,vel, r_now.reshape(3,3), omega, nominal_pos)
        
        #print(f"action is {action}")
        px4_command.body_rate.x = action[1]
        px4_command.body_rate.y = action[2] 
        px4_command.body_rate.z = action[3]
        px4_command.thrust = (action[0] + 1) / 1.93 * 2 * 1.0 * 0.31 #0.50 #0.50
        px4_command_pub_.publish(px4_command)
        
    # else:
    #     print(f"mode.system_status : {mode.system_status}" )
    #     print("No RL")
    print(f"start is {start}")
    offboard_start.data = start
    offboard_start_pub.publish(offboard_start)

rospy.init_node('control_velocity_child1', anonymous=True)
px4_command_pub_ = rospy.Publisher("/rl_cmd1", AttitudeTarget, queue_size=1)
nominal_pos_enu_pub = rospy.Publisher("/child1/nominal_pos_enu", Point, queue_size=1)
first_pos_pub = rospy.Publisher("/child1/randinit_pos", PoseStamped, queue_size=1)
offboard_start_pub = rospy.Publisher("/child1/offboard_start", Int32, queue_size=1)
state_error_pub = rospy.Publisher("/child1/state_error", Float64MultiArray, queue_size=1)
# mode_sub_


state_pos_sub = rospy.Subscriber("/child1/state_pos", Point, state_pos_cb, queue_size=1)
nominal_position_sub = rospy.Subscriber("/child1/nominal_position", Point, nominal_position_cb, queue_size=1)
#nominal_eular_angles_sub = rospy.Subscriber("/nominal_euler_angles", Point, nominal_euler_angles_cb, queue_size=1)
arm_sub = rospy.Subscriber("/child1/start_pub_att", Bool, arm_cb, queue_size=1)
#get_rc_channel_sub = rospy.Subscriber("/mavros/rc/in", RCIn, get_rc_channel_cb, queue_size=1)

#rospy.Subscriber("/mavros/state", State, state_callback)
local_pos_sub = rospy.Subscriber("/child1/mavros/local_position/pose", PoseStamped, local_pos_cb, queue_size=1)
mav_vel_receive_sub = rospy.Subscriber("/child1/mavros/local_position/velocity_body", TwistStamped, mav_vel_receive_cb, queue_size=1)
mode_sub = rospy.Subscriber("/child1/mavros/state", State, mode_cb, queue_size=1)




rospy.Timer(rospy.Duration(0.01), timecallback)
rospy.spin()