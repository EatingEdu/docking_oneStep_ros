import rospy
import numpy as np
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import Float64MultiArray, Int32
from mavros_msgs.msg import AttitudeTarget

from model_predict_forArm import modelPredictforArm
from uav_state import UAVState
from math_util import *


class DualUAVController:
    def __init__(self, uav1: UAVState, uav2: UAVState):
        self.uav1 = uav1
        self.uav2 = uav2

        self.control_name = "rl"  # or "pid"

        # ---- rotation matrices ----
        self.r_now1 = np.zeros(9)
        self.r_now2 = np.zeros(9)
        self.r_d = np.eye(3).reshape(-1)

        # ---- publishers ----
        self.cmd_pub1 = rospy.Publisher("/rl_cmd1", AttitudeTarget, queue_size=1)
        self.cmd_pub2 = rospy.Publisher("/rl_cmd2", AttitudeTarget, queue_size=1)

        self.nominal_pub1 = rospy.Publisher("/child1/nominal_pos_enu", Point, queue_size=1)
        self.nominal_pub2 = rospy.Publisher("/child2/nominal_pos_enu", Point, queue_size=1)

        self.first_pub1 = rospy.Publisher("/child1/randinit_pos", PoseStamped, queue_size=1)
        self.first_pub2 = rospy.Publisher("/child2/randinit_pos", PoseStamped, queue_size=1)

        self.offboard_pub1 = rospy.Publisher("/child1/offboard_start", Int32, queue_size=1)
        self.offboard_pub2 = rospy.Publisher("/child2/offboard_start", Int32, queue_size=1)

        self.state_error_pub = rospy.Publisher("/dual/state_error", Float64MultiArray, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration(0.01), self.control_loop)

    # ================= control loop ================= #

    def control_loop(self, event):
        if not (self.uav1.ready() and self.uav2.ready()):
            return

        # ---- rotation ----
        self.r_now1 = quat2rot_change(self.r_now1, self.uav1.quat)
        self.r_now2 = quat2rot_change(self.r_now2, self.uav2.quat)

        # ---- nominal position init ----
        self._handle_first(self.uav1, self.nominal_pub1, self.first_pub1)
        self._handle_first(self.uav2, self.nominal_pub2, self.first_pub2)

        # ---- state error ----
        s1 = self._compute_state_error(self.uav1, self.r_now1)
        s2 = self._compute_state_error(self.uav2, self.r_now2)

        joint_state = np.concatenate([s1, s2]) #这里还需要加入力估计器的值
        self.state_error_pub.publish(data=joint_state.tolist())

        # ---- RL inference (8D action) ----
        action = modelPredictforArm(joint_state)   # shape (8,)

        a1 = action[:4]
        a2 = action[4:]

        # ---- publish commands ----
        self._publish_cmd(self.cmd_pub1, a1)
        self._publish_cmd(self.cmd_pub2, a2)

        self._publish_offboard(self.uav1, self.offboard_pub1)
        self._publish_offboard(self.uav2, self.offboard_pub2)

    # ================= helper funcs ================= #

    def _handle_first(self, uav, nominal_pub, first_pub):
        if uav.first == 1: 
            """
            进入oddboard，记录当前位置为目标位置，
            当前这一部分两架无人机都是为了完成悬停
            """
            uav.nominal_pos[0] = uav.randinit_pos.pose.position.x
            uav.nominal_pos[1] = uav.randinit_pos.pose.position.y
            uav.nominal_pos[2] = uav.randinit_pos.pose.position.z
            uav.first = 2

        p = Point()  #这里北西天又转回了东北天，为了和之前的mavros消息对齐
        p.x = -uav.nominal_pos[1]
        p.y = uav.nominal_pos[0]
        p.z = uav.nominal_pos[2]
        nominal_pub.publish(p) 

        if uav.randinit_pos is not None:
            first_pub.publish(uav.randinit_pos)

    # check_end 20260204 2140
    def _compute_state_error(self, uav, r_now):
        err_pos = uav.pos - uav.nominal_pos 
        err_vel = body2worldVel(r_now, uav.vel)
        rot_err = errorRot(r_now, self.r_d)

        return np.concatenate([
            err_pos,
            err_vel,
            rot_err,
            np.zeros(3)   # 这里的角速度值可以适当做出调整
        ])

    def _publish_cmd(self, pub, action):
        msg = AttitudeTarget()
        msg.body_rate.x = action[1]
        msg.body_rate.y = action[2]
        msg.body_rate.z = action[3]
        msg.thrust = (action[0] + 1) / 1.93 * 2 * 1.0 * 0.31  #这个值可以根据各子机进行调控
        pub.publish(msg)

    def _publish_offboard(self, uav, pub):
        m = Int32()
        m.data = uav.start
        pub.publish(m)
