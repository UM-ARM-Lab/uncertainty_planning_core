#!/usr/bin/python

#####################################################
#                                                   #
#   Copyright (c) 2015, Calder Phillips-Grafflin    #
#                                                   #
#   Planner parameter testing                       #
#                                                   #
#####################################################

import os
import sys
import subprocess
import string


def test_icc_serial(num_repeats, num_particles, num_executions, goal_probability_threshold, goal_bias, signature_threshold, feasibility_alpha, variance_alpha, max_attempt_count, use_spur_actions):
    filename = "./logs/icc_nomdp_contact_planning_node_no_ros_" + str(num_repeats) + "_" + str(num_particles) + "_" + str(num_executions) + "_" + str(goal_probability_threshold) + "_" + str(goal_bias) + "_" + str(signature_threshold) + "_" + str(feasibility_alpha) + "_" + str(variance_alpha) + "_" + str(max_attempt_count) + "_" + str(int(use_spur_actions)) + ".log"
    command_str = "./icc_nomdp_contact_planning_node_no_ros " + str(num_repeats) + " \"" + filename + "\" " + str(num_particles) + " " + str(num_executions) + " " + str(goal_probability_threshold) + " " + str(goal_bias) + " " + str(signature_threshold) + " " + str(feasibility_alpha) + " " + str(variance_alpha) + " " + str(max_attempt_count) + " " + str(int(use_spur_actions))
    subprocess.call(command_str, shell=True)


def test_icc_parallel(num_repeats, num_particles, num_executions, goal_probability_threshold, goal_bias, signature_threshold, feasibility_alpha, variance_alpha, max_attempt_count, use_spur_actions):
    filename = "./logs/icc_nomdp_contact_planning_node_no_ros_parallel_" + str(num_repeats) + "_" + str(num_particles) + "_" + str(num_executions) + "_" + str(goal_probability_threshold) + "_" + str(goal_bias) + "_" + str(signature_threshold) + "_" + str(feasibility_alpha) + "_" + str(variance_alpha) + "_" + str(max_attempt_count) + "_" + str(int(use_spur_actions)) + ".log"
    command_str = "./icc_nomdp_contact_planning_node_no_ros_parallel " + str(num_repeats) + " \"" + filename + "\" " + str(num_particles) + " " + str(num_executions) + " " + str(goal_probability_threshold) + " " + str(goal_bias) + " " + str(signature_threshold) + " " + str(feasibility_alpha) + " " + str(variance_alpha) + " " + str(max_attempt_count) + " " + str(int(use_spur_actions))
    subprocess.call(command_str, shell=True)


def test_gcc_serial(num_repeats, num_particles, num_executions, goal_probability_threshold, goal_bias, signature_threshold, feasibility_alpha, variance_alpha, max_attempt_count, use_spur_actions):
    filename = "./logs/gcc_nomdp_contact_planning_node_no_ros_" + str(num_repeats) + "_" + str(num_particles) + "_" + str(num_executions) + "_" + str(goal_probability_threshold) + "_" + str(goal_bias) + "_" + str(signature_threshold) + "_" + str(feasibility_alpha) + "_" + str(variance_alpha) + "_" + str(max_attempt_count) + "_" + str(int(use_spur_actions)) + ".log"
    command_str = "./gcc_nomdp_contact_planning_node_no_ros " + str(num_repeats) + " \"" + filename + "\" " + str(num_particles) + " " + str(num_executions) + " " + str(goal_probability_threshold) + " " + str(goal_bias) + " " + str(signature_threshold) + " " + str(feasibility_alpha) + " " + str(variance_alpha) + " " + str(max_attempt_count) + " " + str(int(use_spur_actions))
    subprocess.call(command_str, shell=True)


def test_gcc_parallel(num_repeats, num_particles, num_executions, goal_probability_threshold, goal_bias, signature_threshold, feasibility_alpha, variance_alpha, max_attempt_count, use_spur_actions):
    filename = "./logs/gcc_nomdp_contact_planning_node_no_ros_parallel_" + str(num_repeats) + "_" + str(num_particles) + "_" + str(num_executions) + "_" + str(goal_probability_threshold) + "_" + str(goal_bias) + "_" + str(signature_threshold) + "_" + str(feasibility_alpha) + "_" + str(variance_alpha) + "_" + str(max_attempt_count) + "_" + str(int(use_spur_actions)) + ".log"
    command_str = "./gcc_nomdp_contact_planning_node_no_ros_parallel " + str(num_repeats) + " \"" + filename + "\" " + str(num_particles) + " " + str(num_executions) + " " + str(goal_probability_threshold) + " " + str(goal_bias) + " " + str(signature_threshold) + " " + str(feasibility_alpha) + " " + str(variance_alpha) + " " + str(max_attempt_count) + " " + str(int(use_spur_actions))
    subprocess.call(command_str, shell=True)


def test_parameters():
    print("Testing planners")
    particle_counts = [25, 50, 100]
    max_attempt_counts = [5, 10, 20]
    use_spur_actions = [False, True]
    for particle_count in particle_counts:
        for max_attempt_count in max_attempt_counts:
            for use_spur_action in use_spur_actions:
                test_gcc_serial(5, particle_count, 100, 0.51, 0.1, 0.125, 0.75, 0.75, max_attempt_count, use_spur_action)
                test_gcc_parallel(5, particle_count, 100, 0.51, 0.1, 0.125, 0.75, 0.75, max_attempt_count, use_spur_action)
                test_icc_serial(5, particle_count, 100, 0.51, 0.1, 0.125, 0.75, 0.75, max_attempt_count, use_spur_action)
                test_icc_parallel(5, particle_count, 100, 0.51, 0.1, 0.125, 0.75, 0.75, max_attempt_count, use_spur_action)
    print("Finished testing")

if __name__ == '__main__':
    test_parameters()
