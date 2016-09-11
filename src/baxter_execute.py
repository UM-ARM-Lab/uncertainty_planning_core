#!/usr/bin/python

#####################################################
#                                                   #
#   Copyright (c) 2015, Calder Phillips-Grafflin    #
#                                                   #
#   Baxter Execute Shim                             #
#                                                   #
#####################################################

import rospy
import std_srvs.srv
import uncertainty_planning_core.srv
import actionlib
import control_msgs.msg
import gazebo_msgs.srv
import baxter_robot_interface.msg
import baxter_robot_interface.srv


class ExecuteServer(object):

    def __init__(self, service_path, command_action, abort_service, teleport_service):
        self.command_action_client = actionlib.SimpleActionClient(command_action, baxter_robot_interface.msg.GoToJointTargetAction)
        self.command_action_client.wait_for_server()
        rospy.loginfo("Connected to action server")
        self.abort_client = rospy.ServiceProxy(abort_service, std_srvs.srv.Empty)
        self.abort_client.wait_for_service()
        rospy.loginfo("Connected to abort server")
        if teleport_service != "":
            self.teleport_client = rospy.ServiceProxy(teleport_service, gazebo_msgs.srv.SetModelConfiguration)
            self.teleport_client.wait_for_service()
            rospy.loginfo("Connected to teleport server")
        else:
            self.teleport_client = None
            rospy.logwarn("Teleport service disabled")
        self.server = rospy.Service(service_path, uncertainty_planning_core.srv.SimpleLinkedRobotMove, self.service_handler)
        rospy.loginfo("...ready")
        spin_rate = rospy.Rate(10.0)
        while not rospy.is_shutdown():
            spin_rate.sleep()
        rospy.loginfo("Shutting down")

    def service_handler(self, request):
        rospy.loginfo("Received execution service call")
        if request.mode == uncertainty_planning_core.srv.Simple6dofRobotMoveRequest.RESET:
            rospy.loginfo("Resetting to " + str(zip(request.joint_name, request.start_position)))
            self.command_stop()
            rospy.sleep(2.5)
            self.command_teleport(request.joint_name, request.start_position)
            result = control_msgs.msg.JointTrajectoryControllerState()
            result.joint_names = request.joint_name
            result.actual.positions = request.start_position
            trajectory = [result]
        elif request.mode == uncertainty_planning_core.srv.Simple6dofRobotMoveRequest.EXECUTE:
            rospy.loginfo("Executing to " + str(zip(request.joint_name, request.target_position)))
            trajectory = self.command_to_target(request.joint_name, request.target_position, request.expected_result_position, request.max_execution_time)
        elif request.mode == uncertainty_planning_core.srv.Simple6dofRobotMoveRequest.EXECUTE_FROM_START:
            rospy.loginfo("First, resetting to " + str(zip(request.joint_name, request.start_position)))
            self.command_stop()
            rospy.sleep(2.5)
            self.command_teleport(request.joint_name, request.start_position)
            rospy.loginfo("Executing to " + str(zip(request.joint_name, request.target_position)))
            trajectory = self.command_to_target(request.joint_name, request.target_position, request.expected_result_position, request.max_execution_time)
        else:
            rospy.logerr("Invalid mode command")
            trajectory = []
        rospy.loginfo("Assembling response")
        response = uncertainty_planning_core.srv.SimpleLinkedRobotMoveResponse()
        response.trajectory = trajectory
        rospy.loginfo("Response with " + str(len(response.trajectory)) + " states")
        reached_state = response.trajectory[-1]
        rospy.loginfo("Reached " + str(zip(reached_state.joint_names, reached_state.actual.positions)))
        return response

    def command_teleport(self, joint_names, target_positions):
        if self.teleport_client is not None:
            req = gazebo_msgs.srv.SetModelConfigurationRequest()
            req.model_name = "baxter"
            req.urdf_param_name = "robot_description"
            req.joint_names = joint_names
            req.joint_positions = target_positions
            self.teleport_client.call(req)
        else:
            rospy.loginfo("Teleport ignored")

    def command_to_target(self, joint_names, target_positions, expected_result_positions, time_limit):
        goal = baxter_robot_interface.msg.GoToJointTargetGoal()
        goal.max_execution_time = time_limit
        goal.target.actual.positions = expected_result_positions
        goal.target.desired.positions = target_positions
        goal.target.joint_names = joint_names
        self.command_action_client.send_goal(goal)
        self.command_action_client.wait_for_result()
        result = self.command_action_client.get_result()
        return result.trajectory

    def command_stop(self):
        req = std_srvs.srv.EmptyRequest()
        self.abort_client.call(req)

if __name__ == '__main__':
    rospy.init_node("simplelinked_execute_server")
    rospy.loginfo("Starting...")
    can_teleport = rospy.get_param("~can_teleport", True)
    if can_teleport:
        ExecuteServer("simple_linked_robot_move", "baxter_robot_position_controller/go_to_joint_target", "baxter_robot_position_controller/abort", "gazebo/set_model_configuration")
    else:
        ExecuteServer("simple_linked_robot_move", "baxter_robot_position_controller/go_to_joint_target", "baxter_robot_position_controller/abort", "")