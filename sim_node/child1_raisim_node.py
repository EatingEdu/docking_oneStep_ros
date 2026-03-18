#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Raisim 仿真节点（MAVROS接口对齐版）

功能：
- 使用 Raisim 仿真 UAV1
- 接收 HITL 控制输入（/thrust_omega_command）
- 接收 UAV2 状态（/uav2/state_error）
- 发布 MAVROS 标准话题（可被 UAVState 直接读取）

"""
import sys
sys.path.append("/home/fyt/project/SPC_MIQL/AlignIQL/")
sys.path.append("/home/fyt/project/SPC_MIQL/AlignIQL/Airdocking")
sys.path.append("../")
import rospy
import numpy as np
import gym
import Airdocking
import pdb

from std_msgs.msg import Float64MultiArray, Int32, Bool
from mavros_msgs.msg import AttitudeTarget, State, RCIn
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3Stamped

from jaxrl5 import wrappers
from raisim_airdocking.env_air_sb3.env_params import rew_coeff_sou
from miql import Airdocking_env
from raisim_airdocking.env_air_sb3.AMP_sample_HITL_uav1 import * 
from utils.quad_utils import *

# ===================== 主节点 ===================== #

class RaisimMavrosBridge:

    def __init__(self, ns="/child2",): #ns为/child1 or /child2

        # ---------- 参数 ----------
        self.env_name = "AirDocking-v6"
        self.dt = 0.005  # 200Hz

        # ---------- 状态 ----------
        self.action = np.zeros(4)
        self.state2 = np.zeros(18)
        self.state2[6:15] = np.eye(3).reshape(-1)

        self.init_flag = 0
        self.done = False
        self.step_count = 0

        #pdb.set_trace()
        # ---------- 创建环境 ----------
        self.env = self.make_env(self.env_name)

        # ---------- MAVROS Publishers ----------
        self.pose_pub = rospy.Publisher(
            f"{ns}/mavros/local_position/pose",
            PoseStamped,
            queue_size=1
        )

        self.vel_pub = rospy.Publisher(
            f"{ns}/mavros/local_position/velocity_body",
            TwistStamped,
            queue_size=1
        )

        self.state_pub = rospy.Publisher(
            f"{ns}/mavros/state",
            State,
            queue_size=1
        )

        self.rc_pub = rospy.Publisher(
            f"{ns}/mavros/rc/in",
            RCIn,
            queue_size=1)

        self.euler_angles = rospy.Publisher(
            f"{ns}/nominal_euler_angles",
            Vector3Stamped,
            queue_size=1
        )
        
        self.arm_pub = rospy.Publisher(
            f"{ns}/arm",
            Bool, 
            queue_size=1)
        
        
        # ---------- 其他（保留调试用） ----------
        self.contact_force_pub = rospy.Publisher(
            "/contact_force",
            Float64MultiArray,
            queue_size=1
        )

        # ---------- Subscribers ----------
        rospy.Subscriber(
            f"{ns}/mavros/setpoint_raw/attitude",
            AttitudeTarget,
            self.command_cb,
            queue_size=1
        )
        
        # 这里订阅的是另一个无人机的位姿
        rospy.Subscriber(
            "/uav_airsim/state_error",
            Float64MultiArray,
            self.state2_cb,
            queue_size=1
        )

        # ---------- 定时器 ----------
        rospy.Timer(rospy.Duration(self.dt), self.step)

    # ===================== Env ===================== #

    def make_env(self, env_name):
        gym.envs.register(
            id='AirDocking-v6',
            entry_point='env_air_sb3.AMP_sample_HITL_uav1:AMP',  # 指向您的环境类
        )
        env = gym.make(
            env_name,
            rew_coeff=rew_coeff_sou,
            sense_noise="default", # self_define_hard_05 #self_define_hard_0301
            control_name="forceThrustOmega",
            max_step=50000
        )

        env = wrappers.EpisodeMonitor(env)
        env = wrappers.SinglePrecision(env)

        return env

    # ===================== 回调 ===================== #

    def command_cb(self, data):
        self.action[0] = data.thrust /0.3 /1.0/2*1.93-1
        self.action[1] = data.body_rate.x
        self.action[2] = data.body_rate.y
        self.action[3] = data.body_rate.z

    def state2_cb(self, data):
        self.state2 = np.array(data.data)

    # ===================== 主循环 ===================== #

    def step(self, event):

        # ---------- 初始化 ----------
        if not self.init_flag:
            self.obs = self.env.reset()
            self.done = False
            self.step_count = 0
            self.init_flag = 1
            return

        # ---------- step ----------
        if not self.done:

            self.step_count += 1

            # 注入 UAV2 状态
            self.env.env.env.env.state2 = self.state2

            # 执行动作
            self.obs, _, done, _ = self.env.step(
                np.concatenate([self.action, np.zeros(4)])
            )

            # 接触力（调试）
            self.contact_force_pub.publish(
                Float64MultiArray(data=self.obs[18:24])
            )

            self.done = done

            # 发布 MAVROS
            self.publish_mavros()

            # 防止锁死（保持运行）
            self.done = False

        else:
            rospy.loginfo("Reset env")
            self.obs = self.env.reset()
            self.done = False
            self.step_count = 0

    # ===================== MAVROS发布 ===================== #

    def publish_mavros(self):

        env = self.env.env.env.env.env

        pos = env.position_w
        omega = env.angVel_b
        quat = rot_nwu_to_quat_enu(env.rot_b)
        vel = env.rot_b.T @ env.lineVel_w

        # ===== Pose =====
        pose_msg = PoseStamped()

        # ⚠️ NWU  → ENU 转换（与你 UAVState 一致）
        pose_msg.pose.position.x = -pos[1]
        pose_msg.pose.position.y = pos[0]
        pose_msg.pose.position.z = pos[2]

        pose_msg.pose.orientation.w = quat[0]
        pose_msg.pose.orientation.x = quat[1]
        pose_msg.pose.orientation.y = quat[2]
        pose_msg.pose.orientation.z = quat[3]

        self.pose_pub.publish(pose_msg)

        # ===== Velocity =====
        vel_msg = TwistStamped()

        vel_msg.twist.linear.x = vel[0]
        vel_msg.twist.linear.y = vel[1]
        vel_msg.twist.linear.z = vel[2]

        vel_msg.twist.angular.x = omega[0]
        vel_msg.twist.angular.y = omega[1]
        vel_msg.twist.angular.z = omega[2]

        self.vel_pub.publish(vel_msg)

        # ===== State =====
        state_msg = State()
        state_msg.mode = "OFFBOARD"
        self.state_pub.publish(state_msg)


# ===================== main ===================== #

if __name__ == "__main__":

    rospy.init_node("uav1_raisim_mavros_bridge")

    node = RaisimMavrosBridge()

    rospy.spin()