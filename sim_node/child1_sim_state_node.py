#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped, Point
from mavros_msgs.msg import State
from std_msgs.msg import Bool


def main():
    rospy.init_node("child1_fake_state_node")

    pose_pub = rospy.Publisher("/child1/mavros/local_position/pose", PoseStamped, queue_size=1)
    vel_pub = rospy.Publisher("/child1/mavros/local_position/velocity_body", TwistStamped, queue_size=1)
    state_pub = rospy.Publisher("/child1/mavros/state", State, queue_size=1)
    state_pos_pub = rospy.Publisher("/child1/state_pos", Point, queue_size=1)
    nominal_pos_pub = rospy.Publisher("/child1/nominal_position", Point, queue_size=1)
    arm_pub = rospy.Publisher("/child1/start_pub_att", Bool, queue_size=1)

    rate = rospy.Rate(20)
    t = 0.0

    while not rospy.is_shutdown():
        # ---- pose ----
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        if t < 11:
            pose.pose.position.x = 0.1 #np.sin(t)
            pose.pose.position.y = 0.2 #np.cos(t)
            pose.pose.position.z = 1.5
        else:
            pose.pose.position.x = 0.2 #np.sin(t)
            pose.pose.position.y = 0.4 #np.cos(t)
            pose.pose.position.z = 1.2
            
        pose.pose.orientation.w = -0.691
        pose.pose.orientation.x = -0.001
        pose.pose.orientation.y = -0.006
        pose.pose.orientation.z = -0.723
        
        pose_pub.publish(pose)

        # ---- velocity ----
        vel = TwistStamped()
        vel.twist.linear.x = 0.0001 #*np.cos(t)
        vel.twist.linear.y = -0.001#*np.sin(t)
        vel.twist.linear.z = 0.002
        
        vel.twist.angular.x = 0.1
        vel.twist.angular.y = 0.1
        vel.twist.angular.z = 0.1
        
        vel_pub.publish(vel)

        # ---- state ----
        s = State()
        if t > 10.:
            s.mode = "OFFBOARD"
            
        else:
            s.mode = "POS"
        state_pub.publish(s)

        # ---- state_pos ----
        p = Point()
        p.x, p.y, p.z = pose.pose.position.x, pose.pose.position.y, pose.pose.position.z
        state_pos_pub.publish(p)

        # ---- nominal_position ----
        nominal_pos_pub.publish(p)

        # ---- arm ----
        arm_pub.publish(Bool(data=True))

        t += 0.02
        rate.sleep()


if __name__ == "__main__":
    main()
