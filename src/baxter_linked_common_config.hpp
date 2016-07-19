#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <functional>
#include <random>
#include <Eigen/Geometry>
#include <visualization_msgs/Marker.h>
#include "arc_utilities/eigen_helpers.hpp"
#include "arc_utilities/eigen_helpers_conversions.hpp"
#include "arc_utilities/pretty_print.hpp"
#include "arc_utilities/voxel_grid.hpp"
#include "arc_utilities/simple_rrt_planner.hpp"
#include "uncertainty_planning_core/simple_pid_controller.hpp"
#include "uncertainty_planning_core/simple_uncertainty_models.hpp"
#include "uncertainty_planning_core/uncertainty_contact_planning.hpp"
#include "uncertainty_planning_core/simplelinked_robot_helpers.hpp"
#include "common_config.hpp"
#include "baxter_joint_actuator_model.hpp"

#ifndef BAXTER_LINKED_COMMON_CONFIG_HPP
#define BAXTER_LINKED_COMMON_CONFIG_HPP

namespace linked_common_config
{
    typedef simplelinked_robot_helpers::SimpleJointModel SJM;
    typedef simplelinked_robot_helpers::SimpleLinkedConfiguration SLC;

    inline common_config::OPTIONS GetDefaultOptions()
    {
        common_config::OPTIONS options;
        options.clustering_type = uncertainty_contact_planning::CONVEX_REGION_SIGNATURE;
        options.environment_resolution = 0.025;
        options.planner_time_limit = 120.0;
        options.goal_bias = 0.1;
        options.step_size = M_PI;
        options.goal_probability_threshold = 0.51;
        options.goal_distance_threshold = M_PI_4;
        options.signature_matching_threshold = 0.99;
        options.distance_clustering_threshold = M_PI_2;
        options.feasibility_alpha = 0.75;
        options.variance_alpha = 0.75;
        options.actuator_error = M_PI * 0.02;
        options.sensor_error = 0.0;
        options.edge_attempt_count = 50u;
        options.num_particles = 24u;
        options.use_contact = true;
        options.use_reverse = true;
        options.use_spur_actions = true;
        options.max_exec_actions = 1000u;
        options.max_policy_exec_time = 0.0;
        options.num_policy_simulations = 1u;
        options.num_policy_executions = 1u;
        options.policy_action_attempt_count = 100u;
        options.enable_contact_manifold_target_adjustment = false;
        options.planner_log_file = "/tmp/linked_planner_log.txt";
        options.policy_log_file = "/tmp/linked_policy_log.txt";
        options.planned_policy_file = "/tmp/linked_planned_policy.policy";
        options.executed_policy_file = "/tmp/linked_executed_policy.policy";
        return options;
    }

#ifdef USE_ROS
    inline common_config::OPTIONS GetOptions(const common_config::OPTIONS::TYPE& type)
    {
        return common_config::GetOptions(GetDefaultOptions(), type);
    }
#else
    inline common_config::OPTIONS GetOptions(int argc, char** argv, const common_config::OPTIONS::TYPE& type)
    {
        return common_config::GetOptions(GetDefaultOptions(), argc, argv, type);
    }
#endif

    inline simplelinked_robot_helpers::ROBOT_CONFIG GetDefaultRobotConfig(const common_config::OPTIONS& options)
    {
        const double env_resolution = options.environment_resolution;
        const double kp = 0.1;
        const double ki = 0.0;
        const double kd = 0.01;
        const double i_clamp = 0.0;
        const double velocity_limit = env_resolution * 2.0;
        const double max_sensor_noise = options.sensor_error;
        const double max_actuator_noise = options.actuator_error;
        const simplelinked_robot_helpers::ROBOT_CONFIG robot_config(kp, ki, kd, i_clamp, velocity_limit, max_sensor_noise, max_actuator_noise);
        return robot_config;
    }

    inline Eigen::Affine3d GetBaseTransform()
    {
        const Eigen::Affine3d base_transform = Eigen::Translation3d(0.0, 0.0, 0.0) * Eigen::Quaterniond(Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitZ()));
        return base_transform;
    }

    inline SLC MakeBaxterRightArmConfiguration(const std::vector<double>& joint_values)
    {
        assert(joint_values.size() == 7);
        SLC right_arm_configuration(7);
        const double right_s0 = joint_values[0];
        right_arm_configuration[0] = SJM(std::pair<double, double>(-1.70167993878, 1.70167993878), right_s0, SJM::REVOLUTE); // right_s0
        const double right_s1 = joint_values[1];
        right_arm_configuration[1] = SJM(std::pair<double, double>(-2.147, 1.047), right_s1, SJM::REVOLUTE); // right_s1
        const double right_e0 = joint_values[2];
        right_arm_configuration[2] = SJM(std::pair<double, double>(-3.05417993878, 3.05417993878), right_e0, SJM::REVOLUTE); // right_e0
        const double right_e1 = joint_values[3];
        right_arm_configuration[3] = SJM(std::pair<double, double>(-0.05, 2.618), right_e1, SJM::REVOLUTE); // right_e1
        const double right_w0 = joint_values[4];
        right_arm_configuration[4] = SJM(std::pair<double, double>(-3.059, 3.059), right_w0, SJM::REVOLUTE); // right_w0
        const double right_w1 = joint_values[5];
        right_arm_configuration[5] = SJM(std::pair<double, double>(-1.57079632679, 2.094), right_w1, SJM::REVOLUTE); // right_w1
        const double right_w2 = joint_values[6];
        right_arm_configuration[6] = SJM(std::pair<double, double>(-3.059, 3.059), right_w2, SJM::REVOLUTE); // right_w2
        return right_arm_configuration;
    }

    inline std::pair<SLC, SLC> GetStartAndGoal()
    {
        // Define the goals of the plan
        const SLC goal = MakeBaxterRightArmConfiguration(std::vector<double>{0.84, 0.24, 0.0, 0.01, 0.22, 0.24, 0.26});
        const SLC start = MakeBaxterRightArmConfiguration(std::vector<double>{-0.84, 0.24, 0.0, 0.01, 0.22, 0.24, 0.26});
        //const SLC goal = MakeBaxterRightArmConfiguration(std::vector<double>{0.0, 0.24, 0.0, 0.01, 0.22, 0.24, 0.26});
        return std::make_pair(start, goal);
    }

    inline SLC GetReferenceConfiguration()
    {
        const SLC reference_configuration = MakeBaxterRightArmConfiguration(std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        return reference_configuration;
    }

    inline Eigen::Quaterniond QuaternionFromRPY(const double R, const double P, const double Y)
    {
        const Eigen::AngleAxisd roll(R, Eigen::Vector3d::UnitX());
        const Eigen::AngleAxisd pitch(P, Eigen::Vector3d::UnitY());
        const Eigen::AngleAxisd yaw(Y, Eigen::Vector3d::UnitZ());
        const Eigen::Quaterniond quat(roll * pitch * yaw);
        return quat;
    }

    /* OSRF CAN GO FUCK THEMSELVES - URDF RPY IS ACTUALLY APPLIED Y*P*R */
    inline Eigen::Quaterniond QuaternionFromUrdfRPY(const double R, const double P, const double Y)
    {
        const Eigen::AngleAxisd roll(R, Eigen::Vector3d::UnitX());
        const Eigen::AngleAxisd pitch(P, Eigen::Vector3d::UnitY());
        const Eigen::AngleAxisd yaw(Y, Eigen::Vector3d::UnitZ());
        const Eigen::Quaterniond quat(yaw * pitch * roll);
        return quat;
    }

    class SimpleLinkedDimDistancer
    {
    public:

        Eigen::VectorXd operator()(const simplelinked_robot_helpers::SimpleLinkedConfiguration& q1, const simplelinked_robot_helpers::SimpleLinkedConfiguration& q2) const
        {
            return Distance(q1, q2);
        }

        static Eigen::VectorXd Distance(const simplelinked_robot_helpers::SimpleLinkedConfiguration& q1, const simplelinked_robot_helpers::SimpleLinkedConfiguration& q2)
        {
            return RawDistance(q1, q2).cwiseAbs();
        }

        static double RawJointDistance(const simplelinked_robot_helpers::SimpleJointModel& j1, const simplelinked_robot_helpers::SimpleJointModel& j2)
        {
            assert(j1.IsContinuous() == j2.IsContinuous());
            return j1.SignedDistance(j1.GetValue(), j2.GetValue());
        }

        static Eigen::VectorXd RawDistance(const simplelinked_robot_helpers::SimpleLinkedConfiguration& q1, const simplelinked_robot_helpers::SimpleLinkedConfiguration& q2)
        {
            const std::vector<double> distance_weights = {2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0};
            assert(q1.size() == q2.size());
            assert(q1.size() == 7);
            Eigen::VectorXd distances = Eigen::VectorXd::Zero(q1.size());
            for (size_t idx = 0; idx < q1.size(); idx++)
            {
                const simplelinked_robot_helpers::SimpleJointModel& j1 = q1[idx];
                const simplelinked_robot_helpers::SimpleJointModel& j2 = q2[idx];
                const double joint_weight = distance_weights[idx];
                distances((int64_t)idx) = RawJointDistance(j1, j2) * joint_weight;
            }
            return distances;
        }

        static inline std::string TypeName()
        {
            return std::string("BaxterSimpleLinkedDimDistancer");
        }
    };

    class SimpleLinkedDistancer
    {
    public:

        double operator()(const simplelinked_robot_helpers::SimpleLinkedConfiguration& q1, const simplelinked_robot_helpers::SimpleLinkedConfiguration& q2) const
        {
            return Distance(q1, q2);
        }

        static double Distance(const simplelinked_robot_helpers::SimpleLinkedConfiguration& q1, const simplelinked_robot_helpers::SimpleLinkedConfiguration& q2)
        {
            const Eigen::VectorXd dim_distances = SimpleLinkedDimDistancer::Distance(q1, q2);
            return dim_distances.norm();
        }

        static inline std::string TypeName()
        {
            return std::string("BaxterSimpleLinkedDistancer");
        }
    };

    //typedef baxter_joint_actuator_model::BaxterJointActuatorModel BaxterJointActuatorModel;
    typedef simple_uncertainty_models::SimpleUncertainVelocityActuator BaxterJointActuatorModel;

    inline simplelinked_robot_helpers::SimpleLinkedRobot<BaxterJointActuatorModel> GetRobot(const Eigen::Affine3d& base_transform, const simplelinked_robot_helpers::ROBOT_CONFIG& joint_config)
    {
        const double s0_noise = joint_config.max_actuator_noise;
        const double s1_noise = joint_config.max_actuator_noise;
        const double e0_noise = joint_config.max_actuator_noise;
        const double e1_noise = joint_config.max_actuator_noise;
        const double w0_noise = joint_config.max_actuator_noise;
        const double w1_noise = joint_config.max_actuator_noise;
        const double w2_noise = joint_config.max_actuator_noise;
        // Make the reference configuration
        const SLC reference_configuration = GetReferenceConfiguration();
        // Make the robot model
        simplelinked_robot_helpers::RobotLink torso;
        torso.link_name = "torso";
        torso.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
        torso.link_points.push_back(Eigen::Vector3d(0.025, 0.0, 0.0));
        torso.link_points.push_back(Eigen::Vector3d(0.0, 0.025, 0.0));
        torso.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.025));
        torso.link_points.push_back(Eigen::Vector3d(0.0, -0.025, 0.0));
        torso.link_points.push_back(Eigen::Vector3d(0.0, -0.05, 0.0));
        torso.link_points.push_back(Eigen::Vector3d(0.0, -0.075, 0.0));
        torso.link_points.push_back(Eigen::Vector3d(0.0, -0.1, 0.0));
        torso.link_points.push_back(Eigen::Vector3d(0.0, -0.125, 0.0));
        torso.link_points.push_back(Eigen::Vector3d(0.0, -0.15, 0.0));
        torso.link_points.push_back(Eigen::Vector3d(0.0, -0.175, 0.0));
        torso.link_points.push_back(Eigen::Vector3d(0.0, -0.2, 0.0));
        torso.link_points.push_back(Eigen::Vector3d(0.0, -0.225, 0.0));
        torso.link_points.push_back(Eigen::Vector3d(0.0, -0.225, 0.025));
        torso.link_points.push_back(Eigen::Vector3d(0.0, -0.225, 0.05));
        torso.link_points.push_back(Eigen::Vector3d(0.0, -0.225, 0.075));
        torso.link_points.push_back(Eigen::Vector3d(0.0, -0.225, 0.1));
        torso.link_points.push_back(Eigen::Vector3d(0.025, -0.225, 0.1));
        simplelinked_robot_helpers::RobotLink right_arm_mount;
        right_arm_mount.link_name = "right_arm_mount";
        right_arm_mount.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
        right_arm_mount.link_points.push_back(Eigen::Vector3d(0.025, 0.0, 0.0));
        right_arm_mount.link_points.push_back(Eigen::Vector3d(0.05, 0.0, 0.0));
        simplelinked_robot_helpers::RobotLink right_upper_shoulder;
        right_upper_shoulder.link_name = "right_upper_shoulder";
        right_upper_shoulder.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
        right_upper_shoulder.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.025));
        right_upper_shoulder.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.05));
        right_upper_shoulder.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.075));
        right_upper_shoulder.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.1));
        right_upper_shoulder.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.125));
        right_upper_shoulder.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.15));
        right_upper_shoulder.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.175));
        right_upper_shoulder.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.2));
        right_upper_shoulder.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.225));
        right_upper_shoulder.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.25));
        right_upper_shoulder.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.275));
        right_upper_shoulder.link_points.push_back(Eigen::Vector3d(0.025, 0.0, 0.275));
        right_upper_shoulder.link_points.push_back(Eigen::Vector3d(0.05, 0.0, 0.275));
        simplelinked_robot_helpers::RobotLink right_lower_shoulder;
        right_lower_shoulder.link_name = "right_lower_shoulder";
        right_lower_shoulder.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
        right_lower_shoulder.link_points.push_back(Eigen::Vector3d(0.025, 0.0, 0.0));
        right_lower_shoulder.link_points.push_back(Eigen::Vector3d(0.05, 0.0, 0.0));
        right_lower_shoulder.link_points.push_back(Eigen::Vector3d(0.075, 0.0, 0.0));
        right_lower_shoulder.link_points.push_back(Eigen::Vector3d(0.1, 0.0, 0.0));
        simplelinked_robot_helpers::RobotLink right_upper_elbow;
        right_upper_elbow.link_name = "right_upper_elbow";
        right_upper_elbow.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
        right_upper_elbow.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.025));
        right_upper_elbow.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.05));
        right_upper_elbow.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.075));
        right_upper_elbow.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.1));
        right_upper_elbow.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.125));
        right_upper_elbow.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.15));
        right_upper_elbow.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.175));
        right_upper_elbow.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.2));
        right_upper_elbow.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.225));
        right_upper_elbow.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.25));
        right_upper_elbow.link_points.push_back(Eigen::Vector3d(0.025, 0.0, 0.25));
        right_upper_elbow.link_points.push_back(Eigen::Vector3d(0.05, 0.0, 0.25));
        simplelinked_robot_helpers::RobotLink right_lower_elbow;
        right_lower_elbow.link_name = "right_lower_elbow";
        right_lower_elbow.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
        right_lower_elbow.link_points.push_back(Eigen::Vector3d(0.025, 0.0, 0.0));
        right_lower_elbow.link_points.push_back(Eigen::Vector3d(0.05, 0.0, 0.0));
        right_lower_elbow.link_points.push_back(Eigen::Vector3d(0.075, 0.0, 0.0));
        right_lower_elbow.link_points.push_back(Eigen::Vector3d(0.1, 0.0, 0.0));
        simplelinked_robot_helpers::RobotLink right_upper_forearm;
        right_upper_forearm.link_name = "right_upper_forearm";
        right_upper_forearm.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
        right_upper_forearm.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.025));
        right_upper_forearm.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.05));
        right_upper_forearm.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.075));
        right_upper_forearm.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.1));
        right_upper_forearm.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.125));
        right_upper_forearm.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.15));
        right_upper_forearm.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.175));
        right_upper_forearm.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.2));
        right_upper_forearm.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.225));
        right_upper_forearm.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.25));
        right_upper_forearm.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.275));
        simplelinked_robot_helpers::RobotLink right_lower_forearm;
        right_lower_forearm.link_name = "right_lower_forearm";
        right_lower_forearm.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
        right_lower_forearm.link_points.push_back(Eigen::Vector3d(0.025, 0.0, 0.0));
        right_lower_forearm.link_points.push_back(Eigen::Vector3d(0.05, 0.0, 0.0));
        right_lower_forearm.link_points.push_back(Eigen::Vector3d(0.075, 0.0, 0.0));
        right_lower_forearm.link_points.push_back(Eigen::Vector3d(0.1, 0.0, 0.0));
        right_lower_forearm.link_points.push_back(Eigen::Vector3d(0.125, 0.0, 0.0));
        simplelinked_robot_helpers::RobotLink right_wrist;
        right_wrist.link_name = "right_wrist";
        right_wrist.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
        right_wrist.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.025));
        right_wrist.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.05));
        right_wrist.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.075));
        right_wrist.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.1));
        right_wrist.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.125));
        right_wrist.link_points.push_back(Eigen::Vector3d(0.0, 0.0, 0.15));
        std::vector<simplelinked_robot_helpers::RobotLink> links = {torso, right_arm_mount, right_upper_shoulder, right_lower_shoulder, right_upper_elbow, right_lower_elbow, right_upper_forearm, right_lower_forearm, right_wrist};
        std::vector<std::pair<size_t, size_t>> allowed_self_collisions = {std::pair<size_t, size_t>(0, 1), std::pair<size_t, size_t>(1, 2), std::pair<size_t, size_t>(2, 3), std::pair<size_t, size_t>(3, 4), std::pair<size_t, size_t>(4, 5), std::pair<size_t, size_t>(5, 6), std::pair<size_t, size_t>(6, 7), std::pair<size_t, size_t>(7, 8)};
        // right_s0
        simplelinked_robot_helpers::RobotJoint<BaxterJointActuatorModel> right_arm_mount_joint;
        right_arm_mount_joint.parent_link_index = 0;
        right_arm_mount_joint.child_link_index = 1;
        right_arm_mount_joint.joint_axis = Eigen::Vector3d::UnitZ();
        right_arm_mount_joint.joint_transform = Eigen::Translation3d(0.024645, -0.219645, 0.118588) * QuaternionFromUrdfRPY(0.0, 0.0, -0.7854);
        right_arm_mount_joint.joint_model = SJM(std::make_pair(0.0, 0.0), 0.0, SJM::FIXED);
        // We don't need an uncertainty model for a fixed joint
        right_arm_mount_joint.joint_controller = simplelinked_robot_helpers::JointControllerGroup<BaxterJointActuatorModel>(joint_config);
        // right_s0
        simplelinked_robot_helpers::RobotJoint<BaxterJointActuatorModel> right_s0;
        right_s0.parent_link_index = 1;
        right_s0.child_link_index = 2;
        right_s0.joint_axis = Eigen::Vector3d::UnitZ();
        right_s0.joint_transform = Eigen::Translation3d(0.055695, 0.0, 0.011038) * QuaternionFromUrdfRPY(0.0, 0.0, 0.0);
        right_s0.joint_model = reference_configuration[0];
        //std::shared_ptr<baxter_joint_actuator_model::JointUncertaintySampleModel> right_s0_model_samples(baxter_joint_actuator_model::LoadModel("/home/calderpg/Dropbox/ROS_workspace/src/Research/baxter_uncertainty_models/right_s0.csv", 0.5, 50.0, 50.0));
        //const BaxterJointActuatorModel right_s0_joint_model(right_s0_model_samples, 0.5);
        const BaxterJointActuatorModel right_s0_joint_model(-fabs(s0_noise), fabs(s0_noise), 0.5);
        simplelinked_robot_helpers::ROBOT_CONFIG s0_config = joint_config;
        s0_config.velocity_limit = 0.5;
        right_s0.joint_controller = simplelinked_robot_helpers::JointControllerGroup<BaxterJointActuatorModel>(s0_config, right_s0_joint_model);
        // Base pitch
        simplelinked_robot_helpers::RobotJoint<BaxterJointActuatorModel> right_s1;
        right_s1.parent_link_index = 2;
        right_s1.child_link_index = 3;
        right_s1.joint_axis = Eigen::Vector3d::UnitZ();
        right_s1.joint_transform = Eigen::Translation3d(0.069, 0.0, 0.27035) * QuaternionFromUrdfRPY(-1.57079632679, 0.0, 0.0);
        right_s1.joint_model = reference_configuration[1];
        //std::shared_ptr<baxter_joint_actuator_model::JointUncertaintySampleModel> right_s1_model_samples(baxter_joint_actuator_model::LoadModel("/home/calderpg/Dropbox/ROS_workspace/src/Research/baxter_uncertainty_models/right_s1.csv", 0.5, 50.0, 50.0));
        //const BaxterJointActuatorModel right_s1_joint_model(right_s1_model_samples, 0.5);
        const BaxterJointActuatorModel right_s1_joint_model(-fabs(s1_noise), fabs(s1_noise), 0.5);
        simplelinked_robot_helpers::ROBOT_CONFIG s1_config = joint_config;
        s1_config.velocity_limit = 0.5;
        right_s1.joint_controller = simplelinked_robot_helpers::JointControllerGroup<BaxterJointActuatorModel>(s1_config, right_s1_joint_model);
        // Elbow pitch
        simplelinked_robot_helpers::RobotJoint<BaxterJointActuatorModel> right_e0;
        right_e0.parent_link_index = 3;
        right_e0.child_link_index = 4;
        right_e0.joint_axis = Eigen::Vector3d::UnitZ();
        right_e0.joint_transform = Eigen::Translation3d(0.102, 0.0, 0.0) * QuaternionFromUrdfRPY(1.57079632679, 0.0, 1.57079632679);
        right_e0.joint_model = reference_configuration[2];
        //std::shared_ptr<baxter_joint_actuator_model::JointUncertaintySampleModel> right_e0_model_samples(baxter_joint_actuator_model::LoadModel("/home/calderpg/Dropbox/ROS_workspace/src/Research/baxter_uncertainty_models/right_e0.csv", 0.5, 50.0, 50.0));
        //const BaxterJointActuatorModel right_e0_joint_model(right_e0_model_samples, 0.5);
        const BaxterJointActuatorModel right_e0_joint_model(-fabs(e0_noise), fabs(e0_noise), 0.5);
        simplelinked_robot_helpers::ROBOT_CONFIG e0_config = joint_config;
        e0_config.velocity_limit = 0.5;
        right_e0.joint_controller = simplelinked_robot_helpers::JointControllerGroup<BaxterJointActuatorModel>(e0_config, right_e0_joint_model);
        // Elbow roll
        simplelinked_robot_helpers::RobotJoint<BaxterJointActuatorModel> right_e1;
        right_e1.parent_link_index = 4;
        right_e1.child_link_index = 5;
        right_e1.joint_axis = Eigen::Vector3d::UnitZ();
        right_e1.joint_transform = Eigen::Translation3d(0.069, 0.0, 0.26242) * QuaternionFromUrdfRPY(-1.57079632679, -1.57079632679, 0.0);
        right_e1.joint_model = reference_configuration[3];
        //std::shared_ptr<baxter_joint_actuator_model::JointUncertaintySampleModel> right_e1_model_samples(baxter_joint_actuator_model::LoadModel("/home/calderpg/Dropbox/ROS_workspace/src/Research/baxter_uncertainty_models/right_e1.csv", 0.5, 50.0, 50.0));
        //const BaxterJointActuatorModel right_e1_joint_model(right_e1_model_samples, 0.5);
        const BaxterJointActuatorModel right_e1_joint_model(-fabs(e1_noise), fabs(e1_noise), 0.5);
        simplelinked_robot_helpers::ROBOT_CONFIG e1_config = joint_config;
        e1_config.velocity_limit = 0.5;
        right_e1.joint_controller = simplelinked_robot_helpers::JointControllerGroup<BaxterJointActuatorModel>(e1_config, right_e1_joint_model);
        // Wrist pitch
        simplelinked_robot_helpers::RobotJoint<BaxterJointActuatorModel> right_w0;
        right_w0.parent_link_index = 5;
        right_w0.child_link_index = 6;
        right_w0.joint_axis = Eigen::Vector3d::UnitZ();
        right_w0.joint_transform = Eigen::Translation3d(0.10359, 0.0, 0.0) * QuaternionFromUrdfRPY(1.57079632679, 0.0, 1.57079632679);
        right_w0.joint_model = reference_configuration[4];
        //std::shared_ptr<baxter_joint_actuator_model::JointUncertaintySampleModel> right_w0_model_samples(baxter_joint_actuator_model::LoadModel("/home/calderpg/Dropbox/ROS_workspace/src/Research/baxter_uncertainty_models/right_w0.csv", 1.0, 50.0, 50.0));
        //const BaxterJointActuatorModel right_w0_joint_model(right_w0_model_samples, 1.0);
        const BaxterJointActuatorModel right_w0_joint_model(-fabs(w0_noise), fabs(w0_noise), 1.0);
        simplelinked_robot_helpers::ROBOT_CONFIG w0_config = joint_config;
        w0_config.velocity_limit = 1.0;
        right_w0.joint_controller = simplelinked_robot_helpers::JointControllerGroup<BaxterJointActuatorModel>(w0_config, right_w0_joint_model);
        // Wrist roll
        simplelinked_robot_helpers::RobotJoint<BaxterJointActuatorModel> right_w1;
        right_w1.parent_link_index = 6;
        right_w1.child_link_index = 7;
        right_w1.joint_axis = Eigen::Vector3d::UnitZ();
        right_w1.joint_transform = Eigen::Translation3d(0.01, 0.0, 0.2707) * QuaternionFromUrdfRPY(-1.57079632679, -1.57079632679, 0.0);
        right_w1.joint_model = reference_configuration[5];
        //std::shared_ptr<baxter_joint_actuator_model::JointUncertaintySampleModel> right_w1_model_samples(baxter_joint_actuator_model::LoadModel("/home/calderpg/Dropbox/ROS_workspace/src/Research/baxter_uncertainty_models/right_w1.csv", 1.0, 50.0, 50.0));
        //const BaxterJointActuatorModel right_w1_joint_model(right_w1_model_samples, 1.0);
        const BaxterJointActuatorModel right_w1_joint_model(-fabs(w1_noise), fabs(w1_noise), 1.0);
        simplelinked_robot_helpers::ROBOT_CONFIG w1_config = joint_config;
        w1_config.velocity_limit = 1.0;
        right_w1.joint_controller = simplelinked_robot_helpers::JointControllerGroup<BaxterJointActuatorModel>(w1_config, right_w1_joint_model);
        // Wrist roll
        simplelinked_robot_helpers::RobotJoint<BaxterJointActuatorModel> right_w2;
        right_w2.parent_link_index = 7;
        right_w2.child_link_index = 8;
        right_w2.joint_axis = Eigen::Vector3d::UnitZ();
        right_w2.joint_transform = Eigen::Translation3d(0.115975, 0.0, 0.0) * QuaternionFromUrdfRPY(1.57079632679, 0.0, 1.57079632679);
        right_w2.joint_model = reference_configuration[6];
        //std::shared_ptr<baxter_joint_actuator_model::JointUncertaintySampleModel> right_w2_model_samples(baxter_joint_actuator_model::LoadModel("/home/calderpg/Dropbox/ROS_workspace/src/Research/baxter_uncertainty_models/right_w2.csv", 1.0, 50.0, 50.0));
        //const BaxterJointActuatorModel right_w2_joint_model(right_w2_model_samples, 1.0);
        const BaxterJointActuatorModel right_w2_joint_model(-fabs(w2_noise), fabs(w2_noise), 1.0);
        simplelinked_robot_helpers::ROBOT_CONFIG w2_config = joint_config;
        w2_config.velocity_limit = 1.0;
        right_w2.joint_controller = simplelinked_robot_helpers::JointControllerGroup<BaxterJointActuatorModel>(w2_config, right_w2_joint_model);
        std::vector<simplelinked_robot_helpers::RobotJoint<BaxterJointActuatorModel>>joints = {right_arm_mount_joint, right_s0, right_s1, right_e0, right_e1, right_w0, right_w1, right_w2};
        const simplelinked_robot_helpers::SimpleLinkedRobot<BaxterJointActuatorModel> robot(base_transform, links, joints, allowed_self_collisions, reference_configuration);
        return robot;
    }

    inline simplelinked_robot_helpers::SimpleLinkedBaseSampler GetSampler()
    {
        // Make the sampler
        const SLC reference_configuration = GetReferenceConfiguration();
        const simplelinked_robot_helpers::SimpleLinkedBaseSampler sampler(reference_configuration);
        return sampler;
    }
}

#endif // BAXTER_LINKED_COMMON_CONFIG_HPP
