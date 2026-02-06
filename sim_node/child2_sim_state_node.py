#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped, Point
from mavros_msgs.msg import State
from std_msgs.msg import Bool


def main():
    rospy.init_node("child2_fake_state_node")

    pose_pub = rospy.Publisher("/child2/mavros/local_position/pose", PoseStamped, queue_size=1)
    vel_pub = rospy.Publisher("/child2/mavros/local_position/velocity_body", TwistStamped, queue_size=1)
    state_pub = rospy.Publisher("/child2/mavros/state", State, queue_size=1)
    state_pos_pub = rospy.Publisher("/child2/state_pos", Point, queue_size=1)
    nominal_pos_pub = rospy.Publisher("/child2/nominal_position", Point, queue_size=1)
    arm_pub = rospy.Publisher("/child2/start_pub_att", Bool, queue_size=1)

    rate = rospy.Rate(50)
    t = 0.0

    while not rospy.is_shutdown():
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = -np.sin(t)
        pose.pose.position.y = -np.cos(t)
        pose.pose.position.z = 1.5
        pose.pose.orientation.w = 1.0
        pose_pub.publish(pose)

        vel = TwistStamped()
        vel.twist.linear.x = -0.1*np.cos(t)
        vel.twist.linear.y = 0.1*np.sin(t)
        vel.twist.linear.z = 0.0
        vel_pub.publish(vel)

        s = State()
        s.mode = "OFFBOARD"
        state_pub.publish(s)

        p = Point()
        p.x, p.y, p.z = pose.pose.position.x, pose.pose.position.y, pose.pose.position.z
        state_pos_pub.publish(p)
        nominal_pos_pub.publish(p)

        arm_pub.publish(Bool(data=True))

        t += 0.02
        rate.sleep()


if __name__ == "__main__":
    main()
