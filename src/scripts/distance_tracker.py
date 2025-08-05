#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
import math
import os

experiment_name = "PX4nDJXEHrG"

class DistanceTracker:
    def __init__(self):
        # Initialize the node
        rospy.init_node('distance_tracker', anonymous=True)

        # A place to store the last recorded position
        self.last_position = None

        # The running total of distance traveled
        self.total_distance = 0.0

        # How often to print the total distance (in seconds)
        self.print_interval = 2.0
        self.last_print_time = rospy.Time.now()

        self.output_file_path = f"/home/aarush/final_gemini_results/{experiment_name}/total_distance.txt"

        # Ensure output directory exists and the file is created
        output_dir = os.path.dirname(self.output_file_path)
        try:
            os.makedirs(output_dir, exist_ok=True)  # mkdir -p
        except OSError as e:
            rospy.logerr(f"Could not create output directory '{output_dir}': {e}")

        # Touch the file so later writes won't fail
        try:
            if not os.path.exists(self.output_file_path):
                with open(self.output_file_path, 'w') as f:
                    f.write("0.00\n")
        except IOError as e:
            rospy.logerr(f"Could not create output file '{self.output_file_path}': {e}")

        rospy.loginfo(f"Will write total distance to {self.output_file_path}")

        # Subscribe to the /odom topic
        rospy.Subscriber('/odom', Odometry, self.odom_callback)

        rospy.loginfo("Distance tracker node started.")

    def odom_callback(self, msg):
        # Get the current position
        current_position = msg.pose.pose.position

        if self.last_position is not None:
            # Calculate the distance moved since the last message
            dx = current_position.x - self.last_position.x
            dy = current_position.y - self.last_position.y
            distance_increment = math.sqrt(dx**2 + dy**2)

            # Add it to the total
            self.total_distance += distance_increment

        # Update the last position for the next calculation
        self.last_position = current_position

        # Check if it's time to print the total distance
        if (rospy.Time.now() - self.last_print_time).to_sec() > self.print_interval:
            rospy.loginfo(f"Total distance traveled: {self.total_distance:.2f} meters")
            self.last_print_time = rospy.Time.now()

            try:
                with open(self.output_file_path, 'w') as f:
                    f.write(f"{self.total_distance:.2f}\n")
            except IOError as e:
                rospy.logerr(f"Could not write to distance file: {e}")

    def run(self):
        # Keep the node alive
        rospy.spin()

if __name__ == '__main__':
    try:
        tracker = DistanceTracker()
        tracker.run()
    except rospy.ROSInterruptException:
        pass