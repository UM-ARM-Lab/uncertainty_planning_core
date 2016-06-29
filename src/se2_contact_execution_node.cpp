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
#include "nomdp_planning/simple_pid_controller.hpp"
#include "nomdp_planning/simple_uncertainty_models.hpp"
#include "nomdp_planning/nomdp_contact_planning.hpp"
#include "nomdp_planning/simplese2_robot_helpers.hpp"
#include "se2_common_config.hpp"
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <nomdp_planning/Simple6dofRobotMove.h>

using namespace nomdp_contact_planning;

inline std::vector<Eigen::Matrix<double, 3, 1>, std::allocator<Eigen::Matrix<double, 3, 1>>> move_robot(const Eigen::Matrix<double, 3, 1>& target_config, const bool reset, ros::ServiceClient& robot_control_service)
{
    const Eigen::Affine3d target_transform = Eigen::Translation3d(target_config(0), target_config(1), 0.0) * Eigen::Quaterniond(Eigen::AngleAxisd(target_config(2), Eigen::Vector3d::UnitZ()));
    const geometry_msgs::PoseStamped target_pose = EigenHelpersConversions::EigenAffine3dToGeometryPoseStamped(target_transform, "world");
    // Put together service call
    nomdp_planning::Simple6dofRobotMove::Request req;
    req.target = target_pose;
    if (reset)
    {
        std::cout << "Resetting robot to transform: " << PrettyPrint::PrettyPrint(target_transform) << std::endl;
        req.reset = true;
    }
    else
    {
        std::cout << "Commanding robot to transform: " << PrettyPrint::PrettyPrint(target_transform) << std::endl;
        req.reset = false;
    }
    nomdp_planning::Simple6dofRobotMove::Response res;
    // Call service
    try
    {
        robot_control_service.call(req, res);
    }
    catch (...)
    {
        ROS_ERROR("Move service failed");
    }
    // Unpack result
    const std::vector<geometry_msgs::PoseStamped>& poses = res.trajectory;
    std::vector<Eigen::Matrix<double, 3, 1>, std::allocator<Eigen::Matrix<double, 3, 1>>> configs(poses.size());
    for (size_t idx = 0; idx < poses.size(); idx++)
    {
        const Eigen::Affine3d& transform = EigenHelpersConversions::GeometryPoseToEigenAffine3d(poses[idx].pose);
        const Eigen::Vector3d position = transform.translation();
        const Eigen::AngleAxisd rotation(transform.rotation());
        const Eigen::Vector3d axis = rotation.axis();
        const double abs_dot = fabs(axis.dot(Eigen::Vector3d::UnitZ()));
        const double zr = rotation.angle();
        if (((fabs(zr) < 0.00001) || (abs_dot > 0.99)) == false)
        {
            std::cout << "WARNING - Dot product: " << abs_dot << ", Angle: " << zr << ", Rotation axis: " << PrettyPrint::PrettyPrint(axis) << std::endl;
        }
        const double x = position.x();
        const double y = position.y();
        if (fabs(position.z()) >= 0.001)
        {
            std::cout << "WARNING - Z: " << position.z() << std::endl;
        }
        Eigen::Matrix<double, 3, 1> config;
        config << x, y, zr;
        configs[idx] = config;
    }
    std::cout << "Reached transform: " << PrettyPrint::PrettyPrint(configs.back()) << std::endl;
    return configs;
}

void peg_in_hole_env_se2(ros::Publisher& display_debug_publisher, ros::ServiceClient& robot_control_service)
{
    const common_config::OPTIONS options = se2_common_config::GetOptions(common_config::OPTIONS::EXECUTION);
    std::cout << PrettyPrint::PrettyPrint(options) << std::endl;
    const std::pair<Eigen::Matrix<double, 3, 1>, Eigen::Matrix<double, 3, 1>> start_and_goal = se2_common_config::GetStartAndGoal();
    const simplese2_robot_helpers::SimpleSE2BaseSampler sampler = se2_common_config::GetSampler();
    const simplese2_robot_helpers::ROBOT_CONFIG robot_config = se2_common_config::GetDefaultRobotConfig(options);
    const simplese2_robot_helpers::SimpleSE2Robot robot = se2_common_config::GetRobot(robot_config);
    nomdp_contact_planning::NomdpPlanningSpace<simplese2_robot_helpers::SimpleSE2Robot, simplese2_robot_helpers::SimpleSE2BaseSampler, Eigen::Matrix<double, 3, 1>, simplese2_robot_helpers::EigenMatrixD31Serializer, simplese2_robot_helpers::SimpleSE2Averager, simplese2_robot_helpers::SimpleSE2Distancer, simplese2_robot_helpers::SimpleSE2DimDistancer, simplese2_robot_helpers::SimpleSE2Interpolator, std::allocator<Eigen::Matrix<double, 3, 1>>, std::mt19937_64> planning_space(CONVEX_REGION_SIGNATURE, false, options.num_particles, options.step_size, options.goal_distance_threshold, options.goal_probability_threshold, options.signature_matching_threshold, options.distance_clustering_threshold, options.feasibility_alpha, options.variance_alpha, robot, sampler, "se2_maze", options.environment_resolution);
    // Load the policy
    const auto policy = planning_space.LoadPolicy(options.planned_policy_file);
    std::map<std::string, double> complete_policy_stats;
    std::cout << "Press ENTER to simulate policy..." << std::endl;
    std::cin.get();
    const auto policy_simulation_stats = planning_space.SimulateExectionPolicy(policy, start_and_goal.first, start_and_goal.second, options.num_policy_simulations, options.exec_step_limit, options.enable_contact_manifold_target_adjustment, display_debug_publisher, false, 0.001, false);
    std::cout << "Policy simulation success: " << policy_simulation_stats.first.second << std::endl;
    complete_policy_stats["Policy simulation success"] = policy_simulation_stats.first.second;
    complete_policy_stats["Policy simulation successful resolves"] = policy_simulation_stats.second.first;
    complete_policy_stats["Policy simulation unsuccessful resolves"] = policy_simulation_stats.second.second;
    std::cout << "Press ENTER to execute policy..." << std::endl;
    std::cin.get();
    std::function<std::vector<Eigen::Matrix<double, 3, 1>, std::allocator<Eigen::Matrix<double, 3, 1>>>(const Eigen::Matrix<double, 3, 1>&, const bool)> robot_execution_fn = [&] (const Eigen::Matrix<double, 3, 1>& target_configuration, const bool reset) { return move_robot(target_configuration, reset, robot_control_service); };
    const auto execution_results = planning_space.ExecuteExectionPolicy(policy, start_and_goal.first, start_and_goal.second, robot_execution_fn, options.num_policy_executions, options.exec_step_limit, display_debug_publisher, false, 0.1, false);
    std::cout << "Policy execution success: " << PrettyPrint::PrettyPrint(execution_results.second, true) << std::endl;
    complete_policy_stats["Policy execution success"] = execution_results.second;
    // Save the executed policy
    planning_space.SavePolicy(execution_results.first, options.executed_policy_file);
    // Print out the results & save them to the log file
    const std::string log_results = PrettyPrint::PrettyPrint(complete_policy_stats, false, "\n");
    std::cout << "Policy results:\n" << log_results << std::endl;
    std::ofstream log_file(options.policy_log_file, std::ios_base::out | std::ios_base::app);
    if (!log_file.is_open())
    {
        std::cerr << "\x1b[31;1m Unable to create folder/file to log to: " << options.policy_log_file << "\x1b[37m \n";
        throw std::invalid_argument( "Log filename must be write-openable" );
    }
    log_file << log_results << std::endl;
    log_file.close();
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "se2_contact_execution_node");
    ros::NodeHandle nh;
    ROS_INFO("Starting Nomdp Contact Execution Node...");
    ros::Publisher display_debug_publisher = nh.advertise<visualization_msgs::MarkerArray>("nomdp_debug_display_markers", 1, true);
    ros::ServiceClient robot_control_service = nh.serviceClient<nomdp_planning::Simple6dofRobotMove>("simple_6dof_robot_move");
    peg_in_hole_env_se2(display_debug_publisher, robot_control_service);
    return 0;
}
