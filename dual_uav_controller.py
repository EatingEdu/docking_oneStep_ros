import rospy
import numpy as np
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import Float64MultiArray, Int32
from mavros_msgs.msg import AttitudeTarget

import sys 
sys.path.append("../")
from model_predict_forArm import modelPredict
from model_predict_pid import *
from uav_state import UAVState
from utils.math_util import *

import pdb


class DualUAVController:
    def __init__(self, uav1: UAVState, uav2: UAVState):
        self.uav1 = uav1
        self.uav2 = uav2

        self.control_name = "rl"  # or "pid"

        # ---- rotation matrices ----
        self.r_now1 = np.zeros(9)
        self.r_now2 = np.zeros(9)
        self.r_d = np.eye(3).reshape(-1)
        
        self.rt = self._init_estimate_force_torque(self.uav1)
        self.estimate_force_torque = np.zeros(6)
        self.estimate_force_torque_bias = np.zeros(6)
        self.force_torque_cmd = np.zeros(4)

        # ---- publishers ----
        # self.cmd_pub1 = rospy.Publisher("/rl_cmd1", AttitudeTarget, queue_size=1) # /mavros/setpoint_raw/attitude
        # self.cmd_pub2 = rospy.Publisher("/rl_cmd2", AttitudeTarget, queue_size=1)
        
        
        # ---- 实飞控制节点输出 ----
        self.cmd_pub1 = rospy.Publisher("/child1/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=1)
        self.cmd_pub2 = rospy.Publisher("/child2/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=1)
        
        

        self.nominal_pub1 = rospy.Publisher("/child1/nominal_pos_enu", Point, queue_size=1)
        self.nominal_pub2 = rospy.Publisher("/child2/nominal_pos_enu", Point, queue_size=1)

        self.randinit_pos_pub1 = rospy.Publisher("/child1/randinit_pos", PoseStamped, queue_size=1)
        self.randinit_pos_pub2 = rospy.Publisher("/child2/randinit_pos", PoseStamped, queue_size=1)

        self.offboard_pub1 = rospy.Publisher("/child1/offboard_start", Int32, queue_size=1)
        self.offboard_pub2 = rospy.Publisher("/child2/offboard_start", Int32, queue_size=1)

        self.state_error_pub = rospy.Publisher("/dual/state_error", Float64MultiArray, queue_size=1)
        self.estimate_force_torque_pub = rospy.Publisher("/estimate_force_torque", Float64MultiArray, queue_size=1)
        self.estimate_force_torque_bias_pub = rospy.Publisher("/estimate_force_torque_bias", Float64MultiArray, queue_size=1)
        
        self.force_torque_cmd_pub = rospy.Publisher("/force_torque_cmd", Float64MultiArray, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(0.01), self.control_loop)

    # ================= control loop ================= #

    def control_loop(self, event):
        if not (self.uav1.ready() and self.uav2.ready()):
            return

        # ---- rotation ----
        #pdb.set_trace()
        self.r_now1 = eulerAnglesToRotationMatrix(quan2angle(self.uav1.quat)).reshape(-1)
        self.r_now2 = eulerAnglesToRotationMatrix(quan2angle(self.uav2.quat)).reshape(-1)
        
        # self.r_now1 = quat2rot_change(self.uav1.quat)
        # self.r_now2 = quat2rot_change(self.uav2.quat)

        # ---- nominal position init ----
        self._handle_first(self.uav1, self.nominal_pub1, self.randinit_pos_pub1)
        self._handle_first(self.uav2, self.nominal_pub2, self.randinit_pos_pub2)

        # ---- state error ----
        # state_error1 = [-2.42855889e-03, -8.27417672e-02, -2.42711231e-02, 
        #                     -1.34465797e-02, -1.06117360e-01,  2.56579280e-01,  
        #                     9.99968410e-01, -5.39710606e-03, -5.83150331e-03,  
        #                     5.47043514e-03,  9.99905229e-01,  1.26327295e-02,
        #                 5.76277077e-03, -1.26642315e-02,  9.99903202e-01,
        #                 -6.94541559e-02,5.43837212e-02, -6.90532802e-03, 
        #                 -1.17029937e-03, -4.12333943e-02, -4.74429736e-03,  
        #                 5.27418451e-03,  7.28572858e-03,  1.52184721e-03,
        #                 0.00000000e+00,  -0.15,  0.00000000e+00 , 
        #                 8.53046961e-03, -1.31720593e-02,  6.15611486e-03,  
        #                 9.99682307e-01,  2.36368924e-02, 8.75284709e-03, 
        #                 -2.34367475e-02,  9.99476552e-01, -2.23034639e-02,
        #                 -9.27545037e-03,  2.20912397e-02,  9.99712944e-01 ,
        #                 -4.28698817e-03, -8.48978641e-04, -5.55298466e-04]
        s1 = self._compute_state_error(self.uav1, self.r_now1)
        #s1 = state_error1[:18]
         
        s2 = self._compute_state_error(self.uav2, self.r_now2)  #这里需要注意，现在是都只把子机当前的位置作为了悬停位置，不做进一步的对接操作，需要注意
        #force_torque = np.zeros(6)\
        force_torque_input =  self.estimate_force_torque - self.estimate_force_torque_bias
        force_torque_input = np.zeros(6)
        joint_state = np.concatenate([s1,force_torque_input , s2]) #这里还需要加入力估计器的值
        self.state_error_pub.publish(data=joint_state.tolist())
        #pdb.set_trace()
        if self.control_name == "rl":
            #pdb.set_trace()
            # ---- RL inference (8D action) ----
            action, self.estimate_force_torque, self.force_torque_cmd = modelPredict(self.uav1, joint_state, self.r_now1.reshape(3,3), self.rt)   # shape (8,)
        else: #control_name == "pid" 做稳定悬停时使用
            action, self.estimate_force_torque, self.force_torque_cmd = modelPredict_pid(self.uav1, self.uav2, self.r_now1.reshape(3,3), self.r_now2.reshape(3,3), self.rt , self.uav1.first, self.uav2.first)
            

        a1 = action[:4]
        a2 = action[4:]
        #pdb.set_trace()
        #print(joint_state)
        print(action)

        # ---- publish commands ----
        self._publish_cmd(self.cmd_pub1, a1)
        self._publish_cmd(self.cmd_pub2, a2)

        self._publish_offboard(self.uav1, self.offboard_pub1)
        self._publish_offboard(self.uav2, self.offboard_pub2)
        
        self._publish_estimate_force_torque(self.estimate_force_torque-self.estimate_force_torque_bias, self.estimate_force_torque_pub) #以uav1为参照
        self._publish_estimate_force_torque(self.estimate_force_torque_bias, self.estimate_force_torque_bias_pub) #以uav1为参照
        self._publish_estimate_force_torque(self.force_torque_cmd, self.force_torque_cmd_pub) #以uav1为参照

    # ================= helper funcs ================= #

    def _handle_first(self, uav, nominal_pub, first_pub):
        if uav.first == 0 : 
            #pdb.set_trace()
            """
            进入oddboard，记录当前位置为目标位置，
            当前这一部分两架无人机都是为了完成悬停
            """
            uav.nominal_pos[0] = uav.randinit_pos.pose.position.x
            uav.nominal_pos[1] = uav.randinit_pos.pose.position.y
            uav.nominal_pos[2] = uav.randinit_pos.pose.position.z
        elif uav.first == 1:
            uav.first = 2
            if uav.ns == "/child1":
                self.estimate_force_torque_bias = self.estimate_force_torque

        p = Point()  #这里北西天又转回了东北天，为了和之前的mavros消息对齐
        p.x = uav.nominal_pos[0]
        p.y = uav.nominal_pos[1]
        p.z = uav.nominal_pos[2]
        nominal_pub.publish(p) 
        # pdb.set_trace()
        # print(uav.randinit_pos)
        if uav.randinit_pos is not None:
            #print(uav.randinit_pos)
            first_pub.publish(uav.randinit_pos)

    # 这里统一check一下，这里的nominal_pos与pos是否是同一个坐标系下的
    def _compute_state_error(self, uav, r_now):
        err_pos = uav.pos - uav.nominal_pos 
        err_vel = body2worldVel(r_now, uav.vel)
        rot_err = errorRot(r_now, self.r_d)
        err_omega = errOmega(r_now, uav.omega)
        return np.concatenate([
            err_pos,
            err_vel,
            rot_err,
            err_omega   # 这里的角速度值可以适当做出调整
        ])

    def _publish_cmd(self, pub, action):
        msg = AttitudeTarget()
        msg.body_rate.x = 0. #action[1] #* 0.7
        msg.body_rate.y = 0. #action[2] #* 0.7
        msg.body_rate.z = 0. #action[3] #* 0.7
        
        msg.thrust = (action[0] + 1) / 1.93 * 2 * 1.0 * 0.3  #这个值可以根据各子机进行调控 uav1 0.3
        pub.publish(msg)

    
    def _init_estimate_force_torque(self, uav):
        w = 7
        epsilon  = 1
        inertia = np.array([0.0685339,0.08342368,0.1501083])
        K1,K1_K2 = getK(w,epsilon)
        I_b = inertia
        v = uav.vel
        w_vel = uav.omega
        dt = 0.01
        rt = RT(v, w_vel , I_b = inertia * np.eye(3), u = np.array([0,0,0]), 
                        R_b = np.eye(3), torq_b = np.zeros(3).reshape(-1,1), K1 = K1, K1_K2 = K1_K2, dt = dt, m=uav.mass)

        return rt
    
    def _publish_offboard(self, uav, pub):
        m = Int32()
        m.data = uav.start
        pub.publish(m)
        
    def _publish_estimate_force_torque(self, estimate_force_torque, pub):
        msg = Float64MultiArray()
        msg.data = estimate_force_torque
        pub.publish(msg)
        
