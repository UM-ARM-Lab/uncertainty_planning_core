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
#include "uncertainty_planning_core/uncertainty_planning_core.hpp"

#ifndef UR5_LINKED_COMMON_CONFIG_HPP
#define UR5_LINKED_COMMON_CONFIG_HPP

#ifdef LIMIT_UR5_JOINT_LIMITS
    #define UR5_JOINT_LIMITS M_PI
#else
    #define UR5_JOINT_LIMITS (M_PI * 2.0)
#endif

#ifndef OVERRIDE_FIXED_RESOLUTION
    #define RESOLUTION 0.03125
#endif

namespace ur5_linked_common_config
{
    typedef simplelinked_robot_helpers::SimpleJointModel SJM;
    typedef simplelinked_robot_helpers::SimpleLinkedConfiguration SLC;

    inline uncertainty_planning_core::OPTIONS GetDefaultOptions()
    {
        uncertainty_planning_core::OPTIONS options;
        options.clustering_type = uncertainty_contact_planning::CONVEX_REGION_SIGNATURE;
        options.environment_name = "baxter_env";
#ifdef OVERRIDE_FIXED_RESOLUTION
        options.environment_resolution = 0.025;
#else
        options.environment_resolution = RESOLUTION;
#endif
        options.planner_time_limit = 120.0;
        options.goal_bias = 0.1;
        options.step_size = M_PI;
        options.step_duration = 10.0;
        options.goal_probability_threshold = 0.51;
        options.goal_distance_threshold = M_PI_4;
        options.connect_after_first_solution = 0.0;
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
        options.debug_level = 0;
        options.planner_log_file = "/tmp/ur5_planner_log.txt";
        options.policy_log_file = "/tmp/ur5_policy_log.txt";
        options.planned_policy_file = "/tmp/ur5_planned_policy.policy";
        options.executed_policy_file = "/dev/null";
        return options;
    }

    inline uncertainty_planning_core::OPTIONS GetOptions()
    {
        uncertainty_planning_core::OPTIONS options = uncertainty_planning_core::GetOptions(GetDefaultOptions());
#ifndef OVERRIDE_FIXED_RESOLUTION
        options.environment_resolution = RESOLUTION;
#endif
        return options;
    }

    inline simplelinked_robot_helpers::ROBOT_CONFIG GetDefaultRobotConfig(const uncertainty_planning_core::OPTIONS& options)
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

    inline SLC MakeUR5ArmConfiguration(const std::vector<double>& joint_values)
    {
        assert(joint_values.size() == 6);
        SLC arm_configuration(6);
        const double shoulder_pan_joint = joint_values[0];
        arm_configuration[0] = SJM(std::pair<double, double>(-UR5_JOINT_LIMITS, UR5_JOINT_LIMITS), shoulder_pan_joint, SJM::REVOLUTE); // shoulder_pan_joint
        const double shoulder_lift_joint = joint_values[1];
        arm_configuration[1] = SJM(std::pair<double, double>(-UR5_JOINT_LIMITS, UR5_JOINT_LIMITS), shoulder_lift_joint, SJM::REVOLUTE); // shoulder_lift_joint
        const double elbow_joint = joint_values[2];
        arm_configuration[2] = SJM(std::pair<double, double>(-UR5_JOINT_LIMITS, UR5_JOINT_LIMITS), elbow_joint, SJM::REVOLUTE); // elbow_joint
        const double wrist_1_joint = joint_values[3];
        arm_configuration[3] = SJM(std::pair<double, double>(-UR5_JOINT_LIMITS, UR5_JOINT_LIMITS), wrist_1_joint, SJM::REVOLUTE); // wrist_1_joint
        const double wrist_2_joint = joint_values[4];
        arm_configuration[4] = SJM(std::pair<double, double>(-UR5_JOINT_LIMITS, UR5_JOINT_LIMITS), wrist_2_joint, SJM::REVOLUTE); // wrist_2_joint
        const double wrist_3_joint = joint_values[5];
        arm_configuration[5] = SJM(std::pair<double, double>(-UR5_JOINT_LIMITS, UR5_JOINT_LIMITS), wrist_3_joint, SJM::REVOLUTE); // wrist_3_joint
        return arm_configuration;
    }

    inline std::pair<SLC, SLC> GetStartAndGoal()
    {
        // Define the goals of the plan
        const SLC goal = MakeUR5ArmConfiguration(std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        const SLC start = MakeUR5ArmConfiguration(std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        return std::make_pair(start, goal);
    }

    inline SLC GetReferenceConfiguration()
    {
        const SLC reference_configuration = MakeUR5ArmConfiguration(std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        return reference_configuration;
    }

    inline std::vector<double> GetJointUncertaintyParams(const uncertainty_planning_core::OPTIONS& options)
    {
        std::vector<double> uncertainty_params(6, options.actuator_error);
        return uncertainty_params;
    }

    inline std::vector<double> GetJointDistanceWeights()
    {
        const std::vector<double> max_velocities = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        std::vector<double> distance_weights(max_velocities.size(), 0.0);
        for (size_t idx = 0; idx < max_velocities.size(); idx++)
        {
            distance_weights[idx] = 1.0 / max_velocities[idx];
        }
        return distance_weights;
    }

    inline void MakeCylinderPoints(const double radius, const double length, const double resolution, const Eigen::Vector3d& axis, const std::shared_ptr<EigenHelpers::VectorVector3d>& points)
    {
        UNUSED(radius);
        UNUSED(length);
        UNUSED(axis);
        points->push_back(Eigen::Vector3d(0.0, 0.0, 0.0));
        points->push_back(Eigen::Vector3d(resolution, 0.0, 0.0));
        points->push_back(Eigen::Vector3d(0.0, resolution, 0.0));
        points->push_back(Eigen::Vector3d(0.0, resolution * 2.0, 0.0));
        points->push_back(Eigen::Vector3d(0.0, 0.0, resolution));
        points->push_back(Eigen::Vector3d(0.0, 0.0, resolution * 2.0));
        points->push_back(Eigen::Vector3d(0.0, 0.0, resolution * 3.0));
    }

    inline std::shared_ptr<EigenHelpers::VectorVector3d> GetLinkPoints(const std::string& link_name)
    {
        if (link_name == "base_link")
        {
            std::shared_ptr<EigenHelpers::VectorVector3d> link_points(new EigenHelpers::VectorVector3d());
            MakeCylinderPoints(0.06, 0.05, RESOLUTION, Eigen::Vector3d::UnitZ(), link_points);
            return link_points;
        }
        else if (link_name == "shoulder_link")
        {
            std::shared_ptr<EigenHelpers::VectorVector3d> link_points(new EigenHelpers::VectorVector3d());
            MakeCylinderPoints(0.06, 0.05, RESOLUTION, Eigen::Vector3d::UnitZ(), link_points);
            return link_points;
        }
        else if (link_name == "upper_arm_link")
        {
            std::shared_ptr<EigenHelpers::VectorVector3d> link_points(new EigenHelpers::VectorVector3d());
            MakeCylinderPoints(0.06, 0.05, RESOLUTION, Eigen::Vector3d::UnitZ(), link_points);
            return link_points;
        }
        else if (link_name == "forearm_link")
        {
            std::shared_ptr<EigenHelpers::VectorVector3d> link_points(new EigenHelpers::VectorVector3d());
            MakeCylinderPoints(0.06, 0.05, RESOLUTION, Eigen::Vector3d::UnitZ(), link_points);
            return link_points;
        }
        else if (link_name == "wrist_1_link")
        {
            std::shared_ptr<EigenHelpers::VectorVector3d> link_points(new EigenHelpers::VectorVector3d());
            MakeCylinderPoints(0.6, 0.12, RESOLUTION, Eigen::Vector3d::UnitY(), link_points);
            return link_points;
        }
        else if (link_name == "wrist_2_link")
        {
            std::shared_ptr<EigenHelpers::VectorVector3d> link_points(new EigenHelpers::VectorVector3d());
            MakeCylinderPoints(0.6, 0.12, RESOLUTION, Eigen::Vector3d::UnitZ(), link_points);
            return link_points;
        }
        else if (link_name == "wrist_3_link")
        {
            std::shared_ptr<EigenHelpers::VectorVector3d> link_points(new EigenHelpers::VectorVector3d());
            MakeCylinderPoints(0.05, 0.0823, RESOLUTION, Eigen::Vector3d::UnitY(), link_points);
            return link_points;
        }
        else if (link_name == "ee_link")
        {
            std::shared_ptr<EigenHelpers::VectorVector3d> link_points(new EigenHelpers::VectorVector3d());
            MakeCylinderPoints(RESOLUTION * 0.5, RESOLUTION * 0.5, RESOLUTION, Eigen::Vector3d::UnitX(), link_points);
            return link_points;
        }
        else
        {
            throw std::invalid_argument("Invalid link name");
        }
    }

    typedef uncertainty_planning_core::UR5JointActuatorModel UR5JointActuatorModel;

    inline simplelinked_robot_helpers::SimpleLinkedRobot<UR5JointActuatorModel> GetRobot(const Eigen::Affine3d& base_transform, const simplelinked_robot_helpers::ROBOT_CONFIG& joint_config, const std::vector<double>& joint_uncertainty_params, const std::vector<double>& joint_distance_weights)
    {
        const double shoulder_pan_joint_noise = joint_uncertainty_params[0];
        const double shoulder_lift_joint_noise = joint_uncertainty_params[1];
        const double elbow_joint_noise = joint_uncertainty_params[2];
        const double wrist_1_joint_noise = joint_uncertainty_params[3];
        const double wrist_2_joint_noise = joint_uncertainty_params[4];
        const double wrist_3_joint_noise = joint_uncertainty_params[5];
        // Make the robot model
        simplelinked_robot_helpers::RobotLink base_link(GetLinkPoints("base_link"), "base_link");
        simplelinked_robot_helpers::RobotLink shoulder_link(GetLinkPoints("shoulder_link"), "shoulder_link");
        simplelinked_robot_helpers::RobotLink upper_arm_link(GetLinkPoints("upper_arm_link"), "upper_arm_link");
        simplelinked_robot_helpers::RobotLink forearm_link(GetLinkPoints("forearm_link"), "forearm_link");
        simplelinked_robot_helpers::RobotLink wrist_1_link(GetLinkPoints("wrist_1_link"), "wrist_1_link");
        simplelinked_robot_helpers::RobotLink wrist_2_link(GetLinkPoints("wrist_2_link"), "wrist_2_link");
        simplelinked_robot_helpers::RobotLink wrist_3_link(GetLinkPoints("wrist_3_link"), "wrist_3_link");
        simplelinked_robot_helpers::RobotLink ee_link(GetLinkPoints("ee_link"), "ee_link");
        // Collect the links
        const std::vector<simplelinked_robot_helpers::RobotLink> links = {base_link, shoulder_link, upper_arm_link, forearm_link, wrist_1_link, wrist_2_link, wrist_3_link, ee_link};
        // Set allowed self-collisions (i.e. collisions that actually can't happen, so if they occur, they are numerical issues)
        const std::vector<std::pair<size_t, size_t>> allowed_self_collisions = {std::pair<size_t, size_t>(0, 1), std::pair<size_t, size_t>(1, 2), std::pair<size_t, size_t>(2, 3), std::pair<size_t, size_t>(3, 4), std::pair<size_t, size_t>(4, 5), std::pair<size_t, size_t>(5, 6), std::pair<size_t, size_t>(6, 7)};
        // Make the reference configuration
        const SLC reference_configuration = GetReferenceConfiguration();
        // Make the joints
        // Shoulder pan
        simplelinked_robot_helpers::RobotJoint<UR5JointActuatorModel> shoulder_pan_joint;
        shoulder_pan_joint.name = "shoulder_pan_joint";
        shoulder_pan_joint.parent_link_index = 0;
        shoulder_pan_joint.child_link_index = 1;
        shoulder_pan_joint.joint_axis = Eigen::Vector3d::UnitZ();
        shoulder_pan_joint.joint_transform = Eigen::Translation3d(0.0, 0.0, 0.089159) * EigenHelpers::QuaternionFromUrdfRPY(0.0, 0.0, 0.0);
        shoulder_pan_joint.joint_model = reference_configuration[0];
        simplelinked_robot_helpers::ROBOT_CONFIG shoulder_pan_joint_config = joint_config;
        shoulder_pan_joint_config.velocity_limit = 3.15;
        const UR5JointActuatorModel shoulder_pan_joint_model(std::abs(shoulder_pan_joint_noise), shoulder_pan_joint_config.velocity_limit);
        shoulder_pan_joint.joint_controller = simplelinked_robot_helpers::JointControllerGroup<UR5JointActuatorModel>(shoulder_pan_joint_config, shoulder_pan_joint_model);
        // Shoulder lift
        simplelinked_robot_helpers::RobotJoint<UR5JointActuatorModel> shoulder_lift_joint;
        shoulder_lift_joint.name = "shoulder_lift_joint";
        shoulder_lift_joint.parent_link_index = 1;
        shoulder_lift_joint.child_link_index = 2;
        shoulder_lift_joint.joint_axis = Eigen::Vector3d::UnitY();
        shoulder_lift_joint.joint_transform = Eigen::Translation3d(0.0, 0.13585, 0.0) * EigenHelpers::QuaternionFromUrdfRPY(0.0, M_PI_2, 0.0);
        shoulder_lift_joint.joint_model = reference_configuration[1];
        simplelinked_robot_helpers::ROBOT_CONFIG shoulder_lift_joint_config = joint_config;
        shoulder_lift_joint_config.velocity_limit = 3.15;
        const UR5JointActuatorModel shoulder_lift_joint_model(std::abs(shoulder_lift_joint_noise), shoulder_lift_joint_config.velocity_limit);
        shoulder_lift_joint.joint_controller = simplelinked_robot_helpers::JointControllerGroup<UR5JointActuatorModel>(shoulder_lift_joint_config, shoulder_lift_joint_model);
        // Elbow
        simplelinked_robot_helpers::RobotJoint<UR5JointActuatorModel> elbow_joint;
        elbow_joint.name = "elbow_joint";
        elbow_joint.parent_link_index = 2;
        elbow_joint.child_link_index = 3;
        elbow_joint.joint_axis = Eigen::Vector3d::UnitY();
        elbow_joint.joint_transform = Eigen::Translation3d(0.0, -0.1197, 0.42500) * EigenHelpers::QuaternionFromUrdfRPY(0.0, 0.0, 0.0);
        elbow_joint.joint_model = reference_configuration[2];
        simplelinked_robot_helpers::ROBOT_CONFIG elbow_joint_config = joint_config;
        elbow_joint_config.velocity_limit = 3.15;
        const UR5JointActuatorModel elbow_joint_model(std::abs(elbow_joint_noise), elbow_joint_config.velocity_limit);
        elbow_joint.joint_controller = simplelinked_robot_helpers::JointControllerGroup<UR5JointActuatorModel>(elbow_joint_config, elbow_joint_model);
        // Wrist 1
        simplelinked_robot_helpers::RobotJoint<UR5JointActuatorModel> wrist_1_joint;
        wrist_1_joint.name = "wrist_1_joint";
        wrist_1_joint.parent_link_index = 3;
        wrist_1_joint.child_link_index = 4;
        wrist_1_joint.joint_axis = Eigen::Vector3d::UnitY();
        wrist_1_joint.joint_transform = Eigen::Translation3d(0.0, 0.0, 0.39225) * EigenHelpers::QuaternionFromUrdfRPY(0.0, M_PI_2, 0.0);
        wrist_1_joint.joint_model = reference_configuration[3];
        simplelinked_robot_helpers::ROBOT_CONFIG wrist_1_joint_config = joint_config;
        wrist_1_joint_config.velocity_limit = 3.2;
        const UR5JointActuatorModel wrist_1_joint_model(std::abs(wrist_1_joint_noise), wrist_1_joint_config.velocity_limit);
        wrist_1_joint.joint_controller = simplelinked_robot_helpers::JointControllerGroup<UR5JointActuatorModel>(wrist_1_joint_config, wrist_1_joint_model);
        // Wrist 2
        simplelinked_robot_helpers::RobotJoint<UR5JointActuatorModel> wrist_2_joint;
        wrist_2_joint.name = "wrist_2_joint";
        wrist_2_joint.parent_link_index = 4;
        wrist_2_joint.child_link_index = 5;
        wrist_2_joint.joint_axis = Eigen::Vector3d::UnitZ();
        wrist_2_joint.joint_transform = Eigen::Translation3d(0.0, 0.093, 0.0) * EigenHelpers::QuaternionFromUrdfRPY(0.0, 0.0, 0.0);
        wrist_2_joint.joint_model = reference_configuration[4];
        simplelinked_robot_helpers::ROBOT_CONFIG wrist_2_joint_config = joint_config;
        wrist_2_joint_config.velocity_limit = 3.2;
        const UR5JointActuatorModel wrist_2_joint_model(std::abs(wrist_2_joint_noise), wrist_2_joint_config.velocity_limit);
        wrist_2_joint.joint_controller = simplelinked_robot_helpers::JointControllerGroup<UR5JointActuatorModel>(wrist_2_joint_config, wrist_2_joint_model);
        // Wrist 3
        simplelinked_robot_helpers::RobotJoint<UR5JointActuatorModel> wrist_3_joint;
        wrist_3_joint.name = "wrist_3_joint";
        wrist_3_joint.parent_link_index = 5;
        wrist_3_joint.child_link_index = 6;
        wrist_3_joint.joint_axis = Eigen::Vector3d::UnitY();
        wrist_3_joint.joint_transform = Eigen::Translation3d(0.0, 0.0, 0.09465) * EigenHelpers::QuaternionFromUrdfRPY(0.0, 0.0, 0.0);
        wrist_3_joint.joint_model = reference_configuration[5];
        simplelinked_robot_helpers::ROBOT_CONFIG wrist_3_joint_config = joint_config;
        wrist_3_joint_config.velocity_limit = 3.2;
        const UR5JointActuatorModel wrist_3_joint_model(std::abs(wrist_3_joint_noise), wrist_3_joint_config.velocity_limit);
        wrist_3_joint.joint_controller = simplelinked_robot_helpers::JointControllerGroup<UR5JointActuatorModel>(wrist_3_joint_config, wrist_3_joint_model);
        // Fixed wrist->EE joint
        simplelinked_robot_helpers::RobotJoint<UR5JointActuatorModel> ee_fixed_joint;
        ee_fixed_joint.name = "ee_fixed_joint";
        ee_fixed_joint.parent_link_index = 6;
        ee_fixed_joint.child_link_index = 7;
        ee_fixed_joint.joint_axis = Eigen::Vector3d::UnitZ();
        ee_fixed_joint.joint_transform = Eigen::Translation3d(0.0, 0.0823, 0.0) * EigenHelpers::QuaternionFromUrdfRPY(0.0, 0.0, M_PI_2);
        ee_fixed_joint.joint_model = SJM(std::make_pair(0.0, 0.0), 0.0, SJM::FIXED);
        // We don't need an uncertainty model for a fixed joint
        ee_fixed_joint.joint_controller = simplelinked_robot_helpers::JointControllerGroup<UR5JointActuatorModel>(joint_config);
        // Collect the joints
        const std::vector<simplelinked_robot_helpers::RobotJoint<UR5JointActuatorModel>> joints = {shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint, ee_fixed_joint};
        const simplelinked_robot_helpers::SimpleLinkedRobot<UR5JointActuatorModel> robot(base_transform, links, joints, allowed_self_collisions, reference_configuration, joint_distance_weights);
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

#endif // UR5_LINKED_COMMON_CONFIG_HPP
