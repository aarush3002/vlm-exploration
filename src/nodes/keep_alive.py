#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist

last_cmd_time = rospy.Time(0)
pub           = rospy.Publisher("cmd_vel", Twist, queue_size=10)

def relay(_):
    global last_cmd_time
    last_cmd_time = rospy.Time.now()            # update on any cmd_vel

rospy.init_node("cmd_vel_keep_alive")
rospy.Subscriber("cmd_vel", Twist, relay, queue_size=10)

rate = rospy.Rate(40)     # 4 Hz watchdog
zero = Twist()
while not rospy.is_shutdown():
    if (rospy.Time.now() - last_cmd_time).to_sec() > 1/4:
        pub.publish(zero)                      # only if silent >0.4 s
    rate.sleep()
