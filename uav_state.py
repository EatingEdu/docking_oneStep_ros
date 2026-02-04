import rospy
import numpy as np
from copy import deepcopy
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3Stamped
from mavros_msgs.msg import State, RCIn
from std_msgs.msg import Bool


class UAVState:
    def __init__(self, ns="/child1"):
        self.ns = ns

        # ---------- 原始变量 ----------
        self.state_pos = None
        self.nominal_pos = np.zeros(3)
        self.nominal_euler = None
        self.arm = None

        self.sw = None
        self.start = 0
        self.first = 0

        self.mode = None

        # ---------- 新增变量 ----------
        self.pos = None
        self.quat = None
        #self.r_now = None
        self.vel = None
        self.mav_vel_receive = None
        self.randinit_pos = None

        # ---------- Subscribers ----------
        rospy.Subscriber(f"{ns}/mavros/local_position/pose",
                         PoseStamped, self.local_pos_cb, queue_size=1)

        rospy.Subscriber(f"{ns}/mavros/local_position/velocity_body",
                         TwistStamped, self.mav_vel_receive_cb, queue_size=1)

        rospy.Subscriber(f"{ns}/mavros/state",
                         State, self.mode_cb, queue_size=1)

        rospy.Subscriber(f"{ns}/mavros/rc/in",
                         RCIn, self.get_rc_channel_cb, queue_size=1)

        rospy.Subscriber(f"{ns}/nominal_position",
                         Vector3Stamped, self.nominal_position_cb, queue_size=1)

        rospy.Subscriber(f"{ns}/nominal_euler_angles",
                         Vector3Stamped, self.nominal_euler_angles_cb, queue_size=1)

        rospy.Subscriber(f"{ns}/arm",
                         Bool, self.arm_cb, queue_size=1)

    # ===================== 原始 callbacks ===================== #

    def state_pos_cb(self, data):
        self.state_pos = data
        # print(f"[{self.ns}] state_pos is {self.state_pos}")

    # 这里的nominal_position 还需要考察是什么
    def nominal_position_cb(self, data):
        self.nominal_pos = np.array([data.x, data.y, data.z])
        # print(f"[{self.ns}] nominal_pos is {self.nominal_pos}")

    def nominal_euler_angles_cb(self, data):
        self.nominal_euler = data
        # print(f"[{self.ns}] nominal_euler is {self.nominal_euler}")

    def arm_cb(self, data):
        self.arm = data
        # print(f"[{self.ns}] arm is {self.arm}")

    def get_rc_channel_cb(self, data):
        self.sw = data.channels[4]  #offoard的判定项，需注意
        if self.sw > 1900:
            self.start = 1
        else:
            self.start = 0
            self.first = 0

    def mode_cb(self, data):
        self.mode = data
        if data.mode == 'OFFBOARD':
            self.start = 1
        else:
            self.start = 0
            self.first = 0

    # ===================== 新增 callbacks ===================== #

    def local_pos_cb(self, data):  #这里是之前的代码，东北天->北西天，建立坐标系的时候需要注意一下
        self.pos = np.array([
            data.pose.position.y,
            -data.pose.position.x,
            data.pose.position.z
        ])

        quat = np.array([
            data.pose.orientation.w,
            data.pose.orientation.x,
            data.pose.orientation.y,
            data.pose.orientation.z
        ])
        if quat[0] < 0:
            quat = -quat
        self.quat = quat

        if self.first == 0 and self.start == 1:
            self.randinit_pos = deepcopy(data)
            self.randinit_pos.pose.position.x = self.pos[0]
            self.randinit_pos.pose.position.y = self.pos[1]
            self.randinit_pos.pose.position.z = self.pos[2]
            self.first = 1
        elif self.first == 0:
            self.nominal_pos[:] = self.pos

    def mav_vel_receive_cb(self, data):
        self.mav_vel_receive = data
        self.vel = np.array([
            data.twist.linear.x,
            data.twist.linear.y,
            data.twist.linear.z
        ])

    # ===================== Utilities ===================== #

    def ready(self):
        return self.pos is not None and self.vel is not None and self.mode is not None

# uav1 = UAVState("/child1")
# uav2 = UAVState("/child2")

# rate = rospy.Rate(100)
# while not rospy.is_shutdown():
#     if uav1.ready() and uav2.ready():
#         print("uav1 pos:", uav1.pos, "vel:", uav1.vel)
#         print("uav2 pos:", uav2.pos, "vel:", uav2.vel)
#     rate.sleep()
