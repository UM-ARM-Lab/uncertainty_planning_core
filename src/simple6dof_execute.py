#!/usr/bin/python

#####################################################
#                                                   #
#   Copyright (c) 2015, Calder Phillips-Grafflin    #
#                                                   #
#   Planner parameter testing                       #
#                                                   #
#####################################################

import math
import rospy
import std_srvs.srv
import geometry_msgs.msg
import nomdp_planning.srv
import nav_msgs.msg
import threading
import thruster_robot_examples.srv
from sensitivity_learning.transformation_helper import *


def angle_between_quaternions(q1, q2):
    nq1 = normalize_quaternion(q1)
    nq2 = normalize_quaternion(q2)
    dot_product = abs(nq1[0] * nq2[0] + nq1[1] * nq2[1] + nq1[2] * nq2[2] + nq1[3] * nq2[3])
    if dot_product < 0.9999:
        return math.acos(2.0 * (dot_product ** 2) - 1.0)
    else:
        return 0.0


def normalize_quaternion(q_raw):
    magnitude = math.sqrt(q_raw[0]**2 + q_raw[1]**2 + q_raw[2]**2 + q_raw[3]**2)
    x = q_raw[0] / magnitude
    y = q_raw[1] / magnitude
    z = q_raw[2] / magnitude
    w = q_raw[3] / magnitude
    return [x, y, z, w]


class ExecuteServer(object):

    def __init__(self, service_path, command_topic, abort_service, teleport_service, feedback_topic, max_execution_time, error_threshold, exec_time_limit):
        self.max_execution_time = max_execution_time
        self.error_threshold = error_threshold
        self.exec_time_limit = exec_time_limit
        self.trajectory_storage = []
        self.storage_lock = threading.Lock()
        self.execution_lock = threading.Lock()
        self.command_publisher = rospy.Publisher(command_topic, geometry_msgs.msg.PoseStamped, queue_size=1)
        self.abort_client = rospy.ServiceProxy(abort_service, std_srvs.srv.Empty)
        self.teleport_client = rospy.ServiceProxy(teleport_service, thruster_robot_examples.srv.Teleport)
        self.server = rospy.Service(service_path, nomdp_planning.srv.Simple6dofRobotMove, self.service_handler)
        self.feedback_sub = rospy.Subscriber(feedback_topic, nav_msgs.msg.Odometry, self.feedback_cb, queue_size=1)
        spin_rate = rospy.Rate(10.0)
        while not rospy.is_shutdown():
            spin_rate.sleep()
        rospy.loginfo("Shutting down")

    def service_handler(self, request):
        rospy.loginfo("Received execution service call")
        with self.execution_lock:
            if request.reset:
                rospy.loginfo("Resetting to " + str(request.target))
                self.command_stop()
                rospy.sleep(2.5)
                self.command_teleport(request.target)
            else:
                rospy.loginfo("Executing to " + str(request.target))
                robot_target = request.target
                self.command_to_target(robot_target)
                start_time = rospy.Time.now()
                rospy.sleep(5.0)
                cur_time = rospy.Time.now()
                elapsed_time = (cur_time - start_time).to_sec()
                done = False
                while not done and elapsed_time <= self.exec_time_limit:
                    with self.storage_lock:
                        error = self.compute_error(self.trajectory_storage[-1].pose, request.target.pose)
                        if error <= self.error_threshold:
                            done = True
                    rospy.sleep(0.1)
                    cur_time = rospy.Time.now()
                    elapsed_time = (cur_time - start_time).to_sec()
                rospy.loginfo("Commanding stop")
                #self.command_stop()
                #rospy.sleep(2.5)
            # Make the response
            with self.storage_lock:
                rospy.loginfo("Assembling response")
                response = nomdp_planning.srv.Simple6dofRobotMoveResponse()
                response.trajectory = self.trajectory_storage
                self.trajectory_storage = []
                rospy.loginfo("Response with " + str(len(response.trajectory)) + " states")
                return response

    def compute_error(self, current_pose, target_pose):
        error_vector = (target_pose.position.x - current_pose.position.x, target_pose.position.y - current_pose.position.y, target_pose.position.z - current_pose.position.z)
        error_vector_magnitude = math.sqrt(error_vector[0] ** 2 + error_vector[1] ** 2 + error_vector[2] ** 2)
        q1 = [current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z, current_pose.orientation.w]
        q2 = [target_pose.orientation.x, target_pose.orientation.y, target_pose.orientation.z, target_pose.orientation.w]
        error_angle = angle_between_quaternions(q1, q2)
        error = error_vector_magnitude + error_angle
        return error

    def feedback_cb(self, msg):
        feedback_pose = geometry_msgs.msg.PoseStamped()
        feedback_pose.pose = msg.pose.pose
        feedback_pose.header = msg.header
        with self.storage_lock:
            self.trajectory_storage.append(feedback_pose)

    def command_teleport(self, target_pose):
        req = thruster_robot_examples.srv.TeleportRequest()
        req.target_pose = target_pose
        self.teleport_client.call(req)

    def command_to_target(self, target_pose):
        self.command_publisher.publish(target_pose)

    def command_stop(self):
        req = std_srvs.srv.EmptyRequest()
        self.abort_client.call(req)

if __name__ == '__main__':
    rospy.init_node("simple6dof_execute_server")
    rospy.loginfo("Starting...")
    ExecuteServer("simple_6dof_robot_move", "vehicle_bus/target_pose", "vehicle_bus/target_pose/abort", "vehicle_bus/bus/teleport", "vehicle_bus/pose", 10.0, 0.01, 20.0)