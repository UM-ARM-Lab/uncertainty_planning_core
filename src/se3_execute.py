#!/usr/bin/python

#####################################################
#                                                   #
#   Copyright (c) 2015, Calder Phillips-Grafflin    #
#                                                   #
#   Planner parameter testing                       #
#                                                   #
#####################################################

import rospy
import std_srvs.srv
import nomdp_planning.srv
import actionlib
import thruster_robot_controllers.msg
import thruster_robot_examples.srv


class ExecuteServer(object):

    def __init__(self, service_path, command_action, abort_service, teleport_service):
        self.command_action_client = actionlib.SimpleActionClient(command_action, thruster_robot_controllers.msg.GoToPoseTargetAction)
        self.command_action_client.wait_for_server()
        rospy.loginfo("Connected to action server")
        self.abort_client = rospy.ServiceProxy(abort_service, std_srvs.srv.Empty)
        self.abort_client.wait_for_service()
        rospy.loginfo("Connected to abort server")
        self.teleport_client = rospy.ServiceProxy(teleport_service, thruster_robot_examples.srv.Teleport)
        self.teleport_client.wait_for_service()
        rospy.loginfo("Connected to teleport server")
        self.server = rospy.Service(service_path, nomdp_planning.srv.Simple6dofRobotMove, self.service_handler)
        rospy.loginfo("...ready")
        spin_rate = rospy.Rate(10.0)
        while not rospy.is_shutdown():
            spin_rate.sleep()
        rospy.loginfo("Shutting down")

    def service_handler(self, request):
        rospy.loginfo("Received execution service call")
        if request.reset:
            rospy.loginfo("Resetting to " + str(request.target))
            self.command_stop()
            rospy.sleep(2.5)
            self.command_teleport(request.target)
            trajectory = [request.target]
        else:
            rospy.loginfo("Executing to " + str(request.target))
            robot_target = request.target
            trajectory = self.command_to_target(robot_target, request.time_limit)
        rospy.loginfo("Assembling response")
        response = nomdp_planning.srv.Simple6dofRobotMoveResponse()
        response.trajectory = trajectory
        rospy.loginfo("Response with " + str(len(response.trajectory)) + " states")
        return response

    def command_teleport(self, target_pose):
        req = thruster_robot_examples.srv.TeleportRequest()
        req.target_pose = target_pose
        self.teleport_client.call(req)

    def command_to_target(self, target_pose, time_limit):
        goal = thruster_robot_controllers.msg.GoToPoseTargetGoal()
        goal.max_execution_time = time_limit
        goal.target_pose = target_pose
        self.command_action_client.send_goal(goal)
        self.command_action_client.wait_for_result()
        result = self.command_action_client.get_result()
        return result.trajectory

    def command_stop(self):
        req = std_srvs.srv.EmptyRequest()
        self.abort_client.call(req)

if __name__ == '__main__':
    rospy.init_node("simple6dof_execute_server")
    rospy.loginfo("Starting...")
    ExecuteServer("simple_6dof_robot_move", "vehicle_bus/go_to_target_pose", "vehicle_bus/target_pose/abort", "vehicle_bus/bus/teleport")