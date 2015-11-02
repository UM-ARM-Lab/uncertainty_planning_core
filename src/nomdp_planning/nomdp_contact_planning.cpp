#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <functional>
#include <chrono>
#include <random>
#include <cmath>
#include <mutex>
#include <thread>
#include <atomic>
#include "arc_utilities/eigen_helpers.hpp"
#include "arc_utilities/pretty_print.hpp"
#include "nomdp_planning/simple_pid_controller.hpp"
#include "nomdp_planning/simple_uncertainty_models.hpp"
#include "arc_utilities/simple_rrt_planner.hpp"
#include "nomdp_planning/nomdp_contact_planning.hpp"

#ifdef USE_ROS
#include <ros/ros.h>
#include "arc_utilities/eigen_helpers_conversions.hpp"
#endif

#ifdef ENABLE_PARALLEL
#include <omp.h>
#endif

using namespace nomdp_contact_planning;

/*
 * Discretize a cuboid obstacle to resolution - sized cells
 */
EigenHelpers::VectorVector3d SimpleParticleContactSimulator::DiscretizeObstacle(const OBSTACLE_CONFIG& obstacle, const double resolution) const
{
    EigenHelpers::VectorVector3d cells;
    int32_t x_cells = (int32_t)(obstacle.extents.x() * 2.0 * (1.0 / resolution));
    int32_t y_cells = (int32_t)(obstacle.extents.y() * 2.0 * (1.0 / resolution));
    int32_t z_cells = (int32_t)(obstacle.extents.z() * 2.0 * (1.0 / resolution));
    for (int32_t xidx = 0; xidx < x_cells; xidx++)
    {
        for (int32_t yidx = 0; yidx < y_cells; yidx++)
        {
            for (int32_t zidx = 0; zidx < z_cells; zidx++)
            {
                double x_location = -(obstacle.extents.x() - (resolution * 0.5)) + (resolution * xidx);
                double y_location = -(obstacle.extents.y() - (resolution * 0.5)) + (resolution * yidx);
                double z_location = -(obstacle.extents.z() - (resolution * 0.5)) + (resolution * zidx);
                Eigen::Vector3d cell_location(x_location, y_location, z_location);
                cells.push_back(cell_location);
            }
        }
    }
    return cells;
}

/*
 * Build a new environment from the provided obstacles
 */
sdf_tools::CollisionMapGrid SimpleParticleContactSimulator::BuildEnvironment(const std::vector<OBSTACLE_CONFIG>& obstacles, const double resolution) const
{
    if (obstacles.empty())
    {
        std::cerr << "No obstacles provided, not updating the environment" << std::endl;
        double grid_x_size = 1.0;
        double grid_y_size = 1.0;
        double grid_z_size = 1.0;
        // The grid origin is the minimum point, with identity rotation
        Eigen::Translation3d grid_origin_translation(-(grid_x_size * 0.5), -(grid_y_size * 0.5), -(grid_z_size * 0.5));
        Eigen::Quaterniond grid_origin_rotation = Eigen::Quaterniond::Identity();
        Eigen::Affine3d grid_origin_transform = grid_origin_translation * grid_origin_rotation;
        // Make the grid
        sdf_tools::COLLISION_CELL default_cell(0.0);
        sdf_tools::CollisionMapGrid grid(grid_origin_transform, "nomdp_simulator", resolution, grid_x_size, grid_y_size, grid_z_size, default_cell);
        return grid;
    }
    else
    {
        std::cout << "Rebuilding the environment with " << obstacles.size() << " obstacles" << std::endl;
        // We need to loop through the obstacles, discretize each obstacle, and then find the size of the grid we need to store them
        bool xyz_bounds_initialized = false;
        double x_min = 0.0;
        double y_min = 0.0;
        double z_min = 0.0;
        double x_max = 0.0;
        double y_max = 0.0;
        double z_max = 0.0;
        EigenHelpers::VectorVector3d all_obstacle_cells;
        for (size_t idx = 0; idx < obstacles.size(); idx++)
        {
            const OBSTACLE_CONFIG& obstacle = obstacles[idx];
            EigenHelpers::VectorVector3d obstacle_cells = DiscretizeObstacle(obstacle, resolution);
            for (size_t cidx = 0; cidx < obstacle_cells.size(); cidx++)
            {
                const Eigen::Vector3d& relative_location = obstacle_cells[cidx];
                Eigen::Vector3d real_location = obstacle.pose * relative_location;
                all_obstacle_cells.push_back(real_location);
                // Check against the min/max extents
                if (xyz_bounds_initialized)
                {
                    if (real_location.x() < x_min)
                    {
                        x_min = real_location.x();
                    }
                    else if (real_location.x() > x_max)
                    {
                        x_max = real_location.x();
                    }
                    if (real_location.y() < y_min)
                    {
                        y_min = real_location.y();
                    }
                    else if (real_location.y() > y_max)
                    {
                        y_max = real_location.y();
                    }
                    if (real_location.z() < z_min)
                    {
                        z_min = real_location.z();
                    }
                    else if (real_location.z() > z_max)
                    {
                        z_max = real_location.z();
                    }
                }
                // If we haven't initialized the bounds yet, set them to the current position
                else
                {
                    x_min = real_location.x();
                    x_max = real_location.x();
                    y_min = real_location.y();
                    y_max = real_location.y();
                    z_min = real_location.z();
                    z_max = real_location.z();
                    xyz_bounds_initialized = true;
                }
            }
        }
        // Now that we've done that, we fill in the grid to store them
        // Key the center of the first cell off the the minimum value
        x_min -= (resolution * 0.5);
        y_min -= (resolution * 0.5);
        z_min -= (resolution * 0.5);
        // Add a 1-cell buffer to all sides
        x_min -= resolution;
        y_min -= resolution;
        z_min -= resolution;
        x_max += resolution;
        y_max += resolution;
        z_max += resolution;
        double grid_x_size = x_max - x_min;
        double grid_y_size = y_max - y_min;
        double grid_z_size = z_max - z_min;
        // The grid origin is the minimum point, with identity rotation
        Eigen::Translation3d grid_origin_translation(x_min, y_min, z_min);
        Eigen::Quaterniond grid_origin_rotation = Eigen::Quaterniond::Identity();
        Eigen::Affine3d grid_origin_transform = grid_origin_translation * grid_origin_rotation;
        // Make the grid
        sdf_tools::COLLISION_CELL default_cell(0.0);
        sdf_tools::CollisionMapGrid grid(grid_origin_transform, "nomdp_simulator", resolution, grid_x_size, grid_y_size, grid_z_size, default_cell);
        // Fill it in
        sdf_tools::COLLISION_CELL filled_cell(1.0);
        for (size_t idx = 0; idx < all_obstacle_cells.size(); idx++)
        {
            const Eigen::Vector3d& location = all_obstacle_cells[idx];
            grid.Set(location.x(), location.y(), location.z(), filled_cell);
        }
        // Set the environment
        return grid;
    }
}

std::pair<Eigen::Vector3d, bool> SimpleParticleContactSimulator::ForwardSimulationCallback(const Eigen::Vector3d& start_position, const Eigen::Vector3d& control_input, const u_int32_t num_microsteps) const
{
    UNUSED(num_microsteps);
    Eigen::Vector3d robot_position = start_position;
    // Step along the control input
    const double microstep_distance = GetResolution() * 0.25;
    u_int32_t number_microsteps = (u_int32_t)ceil(control_input.norm() / microstep_distance);
    Eigen::Vector3d control_input_step = control_input * (1.0 / (double)number_microsteps);
    bool collided = false;
    // Iterate
    for (u_int32_t micro_step = 0; micro_step < number_microsteps; micro_step++)
    {
        // Update the position of the robot
        robot_position += control_input_step;
        // Clamp to the limits of the environment
        robot_position.x() = std::min(robot_position.x(), env_max_x_);
        robot_position.x() = std::max(robot_position.x(), env_min_x_);
        robot_position.y() = std::min(robot_position.y(), env_max_y_);
        robot_position.y() = std::max(robot_position.y(), env_min_y_);
        robot_position.z() = std::min(robot_position.z(), env_max_z_);
        robot_position.z() = std::max(robot_position.z(), env_min_z_);
        // Check if it's in collision
        // If it's in free space, great
        if (environment_.Get(robot_position).first.occupancy < 0.5)
        {
            continue;
        }
        // If it's in collision, push it back out to the surface
        else
        {
            collided = true;
            bool in_collision = true;
            while (in_collision)
            {
                // Compute a correction vector
                Eigen::Vector3d correction_unit_vector;
                // We'd like to use a gradient from the environment SDF to help us out
                // Grab the local gradient
                Eigen::Vector3d local_gradient = EigenHelpers::StdVectorDoubleToEigenVector3d(environment_sdf_.GetGradient(robot_position, true));
                double gradient_norm = local_gradient.norm();
                // Make sure we have a useful gradient
                if (gradient_norm > 0.0)
                {
                    correction_unit_vector = local_gradient / gradient_norm;
                }
                // If we don't have a useful gradient, then we backtrack using the control input
                else
                {
                    correction_unit_vector = (control_input_step / control_input_step.norm()) * -1.0;
                }
                // Take a small step along the correction vector
                Eigen::Vector3d correction_step_vector = correction_unit_vector * microstep_distance;
                robot_position += correction_step_vector;
                // Update the collision check
                if (environment_.Get(robot_position).first.occupancy > 0.5)
                {
                    in_collision = true;
                }
                else
                {
                    in_collision = false;
                }
            }
        }
        assert(!isnan(robot_position.x()));
        assert(!isnan(robot_position.y()));
        assert(!isnan(robot_position.z()));
    }
    return std::pair<Eigen::Vector3d, bool>(robot_position, collided);
}

double SimpleParticleContactSimulator::ComputeTrajectoryCurvature(const EigenHelpers::VectorVector3d& trajectory) const
{
    if (trajectory.size() >= 3)
    {
        double max_curvature = 0.0;
        for (size_t idx = 1; idx < (trajectory.size() - 1); idx++)
        {
            Eigen::Vector3d next_step = trajectory[idx + 1] - trajectory[idx];
            Eigen::Vector3d prev_step = trajectory[idx] - trajectory[idx - 1];
            double step_curvature = acos((next_step.dot(prev_step)) / (next_step.norm() * prev_step.norm()));
            if (step_curvature > max_curvature)
            {
                max_curvature = step_curvature;
            }
        }
        return max_curvature;
    }
    else
    {
        return 0.0;
    }
}

std::pair<Eigen::Vector3d, bool> SimpleParticleContactSimulator::ForwardSimulatePointRobot(const Eigen::Vector3d& start_position, const Eigen::Vector3d& target_position, const POINT_ROBOT_CONFIG& robot_config, std::mt19937_64& rng, const u_int32_t forward_simulation_steps, const u_int32_t num_simulation_microsteps, double max_curvature, bool allow_contacts) const
{
    // Make a new robot
    simple_uncertainty_models::Simple3dRobot robot(start_position, robot_config.kp, robot_config.ki, robot_config.kd, robot_config.integral_clamp, robot_config.velocity_limit, robot_config.max_sensor_noise, robot_config.max_actuator_noise);
    // Make the simulation callback function
    std::function<std::pair<Eigen::Vector3d, bool>(const Eigen::Vector3d&, const Eigen::Vector3d&)> forward_simulation_callback_fn = std::bind(&SimpleParticleContactSimulator::ForwardSimulationCallback, this, std::placeholders::_1, std::placeholders::_2, num_simulation_microsteps);
    // Forward simulate for the provided number of steps
    EigenHelpers::VectorVector3d robot_trajectory(forward_simulation_steps);
    bool collided = false;
    for (u_int32_t step = 0; step < forward_simulation_steps; step++)
    {
        std::pair<Eigen::Vector3d, bool> result = robot.MoveTowardsTarget(target_position, 1.0, forward_simulation_callback_fn, rng);
        if (result.second)
        {
            collided = true;
        }
        robot_trajectory[step] = result.first;
    }
    // Get the ending position of the robot
    Eigen::Vector3d final_position = robot.GetPosition();
    bool succeeded = false;
    if (collided)
    {
        if (allow_contacts)
        {
            double traj_max_curvature = ComputeTrajectoryCurvature(robot_trajectory);
            if (traj_max_curvature < max_curvature)
            {
                succeeded = true;
            }
            else
            {
                succeeded = false;
            }
        }
        else
        {
            succeeded = false;
        }
    }
    else
    {
        succeeded = true;
    }
    return std::pair<Eigen::Vector3d, bool>(final_position, succeeded);
}

double NomdpPlanningSpace::StateDistance(const NomdpPlannerState& state1, const NomdpPlannerState& state2) const
{
    // Get the "space independent" expectation distance
    double expectation_distance = (state2.GetExpectation() - state1.GetExpectation()).norm() / step_size_;
    // Get the Pfeasibility(start -> state1)
    double feasibility_weight = (1.0 - state1.GetMotionPfeasibility()) * feasibility_alpha_ + (1.0 - feasibility_alpha_);
    // Get the "space independent" variance of state1
    Eigen::Vector3d raw_variances = state1.GetSpaceIndependentVariances();
    double raw_variance = raw_variances.x() * raw_variances.y() * raw_variances.z();
    // Turn the variance into a weight
    double variance_weight = erf(raw_variance) * variance_alpha_ + (1.0 - variance_alpha_);
    // Compute the actual distance
    double distance = (feasibility_weight * expectation_distance * variance_weight);
    return distance;
}

int64_t NomdpPlanningSpace::GetNearestNeighbor(const std::vector<simple_rrt_planner::SimpleRRTPlannerState<NomdpPlannerState, std::allocator<NomdpPlannerState>>>& planner_nodes, const NomdpPlannerState& random_state) const
{
    UNUSED(planner_nodes);
    // Get the nearest neighbor (ignoring the disabled states)
    int64_t best_index = -1;
    double best_distance = INFINITY;
#ifdef ENABLE_PARALLEL
    std::mutex nn_mutex;
    #pragma omp parallel for schedule(guided)
#endif
    for (size_t idx = 0; idx < nearest_neighbors_storage_.size(); idx++)
    {
        const simple_rrt_planner::SimpleRRTPlannerState<NomdpPlannerState>& current_state = nearest_neighbors_storage_[idx];
        // Only check against states enabled for NN checks
        if (current_state.GetValueImmutable().UseForNearestNeighbors())
        {
            double state_distance = StateDistance(current_state.GetValueImmutable(), random_state);
#ifdef ENABLE_PARALLEL
            std::lock_guard<std::mutex> lock(nn_mutex);
#endif
            if (state_distance < best_distance)
            {
                best_distance = state_distance;
                best_index = idx;
            }
        }
    }
    return best_index;
}

NomdpPlannerState NomdpPlanningSpace::SampleRandomTargetState()
{
    double random_x = x_distribution_(rng_);
    double random_y = y_distribution_(rng_);
    double random_z = z_distribution_(rng_);
    Eigen::Vector3d random_point(random_x, random_y, random_z);
    NomdpPlannerState random_state(random_point);
    return random_state;
}

std::pair<std::vector<NomdpPlannerState>, std::pair<EigenHelpers::VectorVector3d, std::vector<std::pair<Eigen::Vector3d, bool>>>> NomdpPlanningSpace::ForwardSimulateParticles(const NomdpPlannerState& nearest, const NomdpPlannerState& random)
{
    // First, compute a target state
    Eigen::Vector3d target_point = random.GetExpectation();
    if ((target_point - nearest.GetExpectation()).norm() > step_size_)
    {
        Eigen::Vector3d direction_vector = random.GetExpectation() - nearest.GetExpectation();
        Eigen::Vector3d direction_unit_vector = direction_vector / direction_vector.norm();
        Eigen::Vector3d step_vector = direction_unit_vector * step_size_;
        target_point = nearest.GetExpectation() + step_vector;
    }
    transition_id_++;
    Eigen::Vector3d control_input = target_point; // - nearest.GetExpectation();
    // Get the initial particles
    EigenHelpers::VectorVector3d initial_particles = nearest.ResampleParticles(num_particles_, rng_);
    // Forward propagate each of the particles
    std::vector<std::pair<Eigen::Vector3d, bool>> raw_propagated_points(num_particles_);
    // We want to parallelize this as much as possible!
#ifdef ENABLE_PARALLEL
    #pragma omp parallel for schedule(guided)
#endif
    for (size_t idx = 0; idx < num_particles_; idx++)
    {
        const Eigen::Vector3d& initial_particle = initial_particles[idx];
#ifdef ENABLE_PARALLEL
        int th_id = omp_get_thread_num();
        raw_propagated_points[idx] = simulator_.ForwardSimulatePointRobot(initial_particle, target_point, robot_config_, rngs_[th_id], 40, 10, max_robot_trajectory_curvature_, allow_contacts_);
#else
        raw_propagated_points[idx] = simulator_.ForwardSimulatePointRobot(initial_particle, target_point, robot_config_, rng_, 40, 10, max_robot_trajectory_curvature_, allow_contacts_);
#endif
    }
    // Collect the live particles
    EigenHelpers::VectorVector3d propagated_points;
    for (size_t idx = 0; idx < num_particles_; idx++)
    {
        if (raw_propagated_points[idx].second)
        {
            const Eigen::Vector3d& propagated_point = raw_propagated_points[idx].first;
            assert(!isnan(propagated_point.x()));
            assert(!isnan(propagated_point.y()));
            assert(!isnan(propagated_point.z()));
            propagated_points.push_back(propagated_point);
        }
    }
    // Cluster the live particles into (potentially) multiple states
    std::vector<EigenHelpers::VectorVector3d> particle_clusters = ClusterParticles(propagated_points);
    bool is_split_child = false;
    if (particle_clusters.size() > 1)
    {
        std::cout << "Transition produced " << particle_clusters.size() << " split states" << std::endl;
        is_split_child = true;
    }
    // Build the forward-propagated states
    std::vector<NomdpPlannerState> result_states(particle_clusters.size());
    for (size_t idx = 0; idx < particle_clusters.size(); idx++)
    {
        if (particle_clusters[idx].size() > 0)
        {
            state_counter_++;
            double edge_feasibility = (double)particle_clusters[idx].size() / (double)num_particles_;
            NomdpPlannerState propagated_state(particle_clusters[idx], edge_feasibility, nearest.GetMotionPfeasibility(), step_size_, control_input, transition_id_, is_split_child);
            result_states[idx] = propagated_state;
        }
    }
    return std::pair<std::vector<NomdpPlannerState>, std::pair<EigenHelpers::VectorVector3d, std::vector<std::pair<Eigen::Vector3d, bool>>>>(result_states, std::pair<EigenHelpers::VectorVector3d, std::vector<std::pair<Eigen::Vector3d, bool>>>(initial_particles, raw_propagated_points));
}

std::vector<EigenHelpers::VectorVector3d> NomdpPlanningSpace::ClusterParticles(const EigenHelpers::VectorVector3d& particles) const
{
    // Make sure there are particles to cluster
    if (particles.size() == 0)
    {
        return std::vector<EigenHelpers::VectorVector3d>();
    }
    else if (particles.size() == 1)
    {
        return std::vector<EigenHelpers::VectorVector3d>{particles};
    }
    // Dummy clustering that always produces a single cluster
    std::function<double(const Eigen::Vector3d&, const Eigen::Vector3d&)> distance_fn = [] (const Eigen::Vector3d& v1, const Eigen::Vector3d& v2) { return (v1 - v2).norm(); };
    double max_cluster_distance = step_size_ * 0.5;
    std::vector<EigenHelpers::VectorVector3d> clusters = clustering_.Cluster(particles, distance_fn, max_cluster_distance);
    return clusters;
}

#ifdef USE_ROS
visualization_msgs::MarkerArray NomdpPlanningSpace::DrawForwardPropagation(const EigenHelpers::VectorVector3d& start, const std::vector<std::pair<Eigen::Vector3d, bool>>& end, const bool is_split) const
{
    assert(start.size() == end.size());
    visualization_msgs::Marker propagation_display;
    propagation_display.pose = EigenHelpersConversions::EigenAffine3dToGeometryPose(Eigen::Affine3d::Identity());
    propagation_display.action = visualization_msgs::Marker::ADD;
    if (!is_split)
    {
        propagation_display.ns = "forward_propagation";
    }
    else
    {
        propagation_display.ns = "split_forward_propagation";
    }
    propagation_display.id = state_counter_;
    propagation_display.frame_locked = false;
    propagation_display.lifetime = ros::Duration(0.0);
    propagation_display.type = visualization_msgs::Marker::LINE_LIST;
    propagation_display.header.frame_id = simulator_.GetFrame();
    propagation_display.scale.x = simulator_.GetResolution() * 0.5;
    // Make the colors
    std_msgs::ColorRGBA alive_color;
    alive_color.r = 0.0;
    alive_color.g = 1.0;
    alive_color.b = 0.0;
    alive_color.a = 0.5;
    std_msgs::ColorRGBA dead_color;
    dead_color.r = 1.0;
    dead_color.g = 0.0;
    dead_color.b = 0.0;
    dead_color.a = 0.5;
    // Add each particle
    for (size_t idx = 0; idx < start.size(); idx++)
    {
        const Eigen::Vector3d& start_point = start[idx];
        const Eigen::Vector3d& end_point = end[idx].first;
        const bool alive = end[idx].second;
        propagation_display.points.push_back(EigenHelpersConversions::EigenVector3dToGeometryPoint(start_point));
        propagation_display.points.push_back(EigenHelpersConversions::EigenVector3dToGeometryPoint(end_point));
        propagation_display.colors.push_back(alive_color);
        if (alive)
        {
            propagation_display.colors.push_back(alive_color);
        }
        else
        {
            propagation_display.colors.push_back(dead_color);
        }
    }
    visualization_msgs::MarkerArray display_markers;
    display_markers.markers.push_back(propagation_display);
    return display_markers;
}
#endif

#ifdef USE_ROS
std::vector<NomdpPlannerState> NomdpPlanningSpace::PropagateForwardsAndDraw(const NomdpPlannerState& nearest, const NomdpPlannerState& random, ros::Publisher& display_pub)
{
    std::pair<std::vector<NomdpPlannerState>, std::pair<EigenHelpers::VectorVector3d, std::vector<std::pair<Eigen::Vector3d, bool>>>> propagated_state = ForwardSimulateParticles(nearest, random);
    // Draw the expansion
    visualization_msgs::MarkerArray propagation_display_rep = DrawForwardPropagation(propagated_state.second.first, propagated_state.second.second, (propagated_state.first.size() > 1));
    // Check if the expansion was useful
    if (propagated_state.first.size() > 0)
    {
        for (size_t idx = 0; idx < propagated_state.first.size(); idx++)
        {
            // Draw the variance
            visualization_msgs::Marker extents_marker;
            extents_marker.action = visualization_msgs::Marker::ADD;
            if (propagated_state.first.size() == 1)
            {
                extents_marker.ns = "extents_propagation";
            }
            else
            {
                extents_marker.ns = "split_extents_propagation";
            }
            extents_marker.id = state_counter_;
            extents_marker.frame_locked = false;
            extents_marker.lifetime = ros::Duration(0.0);
            extents_marker.type = visualization_msgs::Marker::SPHERE;
            extents_marker.header.frame_id = simulator_.GetFrame();
            Eigen::Vector3d extents = propagated_state.first[idx].GetParticleExtents(propagated_state.first[idx].GetExpectation());
            extents_marker.scale.x = std::max(extents.x() * 2.0, 0.001);
            extents_marker.scale.y = std::max(extents.y() * 2.0, 0.001);
            extents_marker.scale.z = std::max(extents.z() * 2.0, 0.001);
            extents_marker.pose.position = EigenHelpersConversions::EigenVector3dToGeometryPoint(propagated_state.first[idx].GetExpectation());
            extents_marker.pose.orientation = EigenHelpersConversions::EigenQuaterniondToGeometryQuaternion(Eigen::Quaterniond::Identity());
            double raw_variance = propagated_state.first[idx].GetSpaceIndependentVariance();
            extents_marker.color.r = erf(raw_variance) * variance_alpha_ + (1.0 - variance_alpha_);
            extents_marker.color.g = erf(raw_variance) * variance_alpha_ + (1.0 - variance_alpha_);
            extents_marker.color.b = erf(raw_variance) * variance_alpha_ + (1.0 - variance_alpha_);
            extents_marker.color.a = 0.33;
            propagation_display_rep.markers.push_back(extents_marker);
            visualization_msgs::Marker variance_marker;
            variance_marker.action = visualization_msgs::Marker::ADD;
            if (propagated_state.first.size() == 1)
            {
                variance_marker.ns = "variance_propagation";
            }
            else
            {
                variance_marker.ns = "split_variance_propagation";
            }
            variance_marker.id = state_counter_;
            variance_marker.frame_locked = false;
            variance_marker.lifetime = ros::Duration(0.0);
            variance_marker.type = visualization_msgs::Marker::SPHERE;
            variance_marker.header.frame_id = simulator_.GetFrame();
            variance_marker.scale.x = std::max(propagated_state.first[idx].GetVariances().x() * 2.0, 0.001);
            variance_marker.scale.y = std::max(propagated_state.first[idx].GetVariances().y() * 2.0, 0.001);
            variance_marker.scale.z = std::max(propagated_state.first[idx].GetVariances().z() * 2.0, 0.001);
            variance_marker.pose.position = EigenHelpersConversions::EigenVector3dToGeometryPoint(propagated_state.first[idx].GetExpectation());
            variance_marker.pose.orientation = EigenHelpersConversions::EigenQuaterniondToGeometryQuaternion(Eigen::Quaterniond::Identity());
            variance_marker.color.r = erf(raw_variance) * variance_alpha_ + (1.0 - variance_alpha_);
            variance_marker.color.g = erf(raw_variance) * variance_alpha_ + (1.0 - variance_alpha_);
            variance_marker.color.b = erf(raw_variance) * variance_alpha_ + (1.0 - variance_alpha_);
            variance_marker.color.a = 0.66;
            propagation_display_rep.markers.push_back(variance_marker);
            // Draw an arrow to the new expectation
            visualization_msgs::Marker expectation_marker;
            expectation_marker.action = visualization_msgs::Marker::ADD;
            if (propagated_state.first.size() == 1)
            {
                expectation_marker.ns = "expectation_propagation";
            }
            else
            {
                expectation_marker.ns = "split_expectation_propagation";
            }
            expectation_marker.id = state_counter_;
            expectation_marker.frame_locked = false;
            expectation_marker.lifetime = ros::Duration(0.0);
            expectation_marker.type = visualization_msgs::Marker::ARROW;
            expectation_marker.header.frame_id = simulator_.GetFrame();
            expectation_marker.scale.x = simulator_.GetResolution();
            expectation_marker.scale.y = simulator_.GetResolution() * 1.5;
            expectation_marker.scale.z = 0.0;
            expectation_marker.points.push_back(EigenHelpersConversions::EigenVector3dToGeometryPoint(nearest.GetExpectation()));
            expectation_marker.points.push_back(EigenHelpersConversions::EigenVector3dToGeometryPoint(propagated_state.first[idx].GetExpectation()));
            // Get motion feasibility
            double motion_Pfeasibility = propagated_state.first[idx].GetMotionPfeasibility();
            // Check agains the goal reached probability
            if (motion_Pfeasibility >= 0.0) //goal_probability_threshold_)
            {
                expectation_marker.color.r = (1.0 - motion_Pfeasibility);
                expectation_marker.color.g = (1.0 - motion_Pfeasibility);
                expectation_marker.color.b = (1.0 - motion_Pfeasibility);
                expectation_marker.color.a = 1.0;
                propagation_display_rep.markers.push_back(expectation_marker);
            }
            else
            {
                expectation_marker.color.r = 1.0;
                expectation_marker.color.g = 0.5;
                expectation_marker.color.b = 0.0;
                expectation_marker.color.a = 1.0;
                propagation_display_rep.markers.push_back(expectation_marker);
            }
        }
    }
    display_pub.publish(propagation_display_rep);
    return propagated_state.first;
}
#endif

std::vector<NomdpPlannerState> NomdpPlanningSpace::PropagateForwards(const NomdpPlannerState& nearest, const NomdpPlannerState& random)
{
    return ForwardSimulateParticles(nearest, random).first;
}

bool NomdpPlanningSpace::GoalReached(const NomdpPlannerState& state, const Eigen::Vector3d& goal_position) const
{
    // First, check if the expectation is within distance of the goal
    if ((state.GetExpectation() - goal_position).norm() > goal_distance_threshold_)
    {
        return false;
    }
    // Make sure that the state is within distance of the goal by probability
    else
    {
        double goal_probability = ComputeGoalReachedProbability(state, goal_position) * state.GetMotionPfeasibility();
        if (goal_probability < goal_probability_threshold_)
        {
            return false;
        }
        else
        {
            std::cout << "Goal reached with state " << PrettyPrint::PrettyPrint(state) << " with probability: " << goal_probability << std::endl;
            return true;
        }
    }
}

void NomdpPlanningSpace::GoalReachedCallback(simple_rrt_planner::SimpleRRTPlannerState<NomdpPlannerState>& new_goal, const Eigen::Vector3d& goal_position) const
{
    // Backtrack through the solution path until we reach the root of the current "goal branch"
    // A goal branch is the entire branch leading to the goal
    // Make sure the goal state isn't a branch root itself
    if (CheckIfGoalBranchRoot(new_goal))
    {
        std::cout << "Goal state is the root of its own goal branch, no need to blacklist" << std::endl;
    }
    else
    {
        int64_t current_index = new_goal.GetParentIndex();
        int64_t goal_branch_root_index = -1; // Initialize to an invalid index so we can detect later if it isn't valid
        while (current_index > 0)
        {
            // Get the current state that we're looking at
            simple_rrt_planner::SimpleRRTPlannerState<NomdpPlannerState>& current_state = nearest_neighbors_storage_[current_index];
            // Check if we've reached the root of the goal branch
            bool is_branch_root = CheckIfGoalBranchRoot(current_state);
            // If we haven't reached the root of goal branch
            if (!is_branch_root)
            {
                current_index = current_state.GetParentIndex();
            }
            else
            {
                goal_branch_root_index = current_index;
                break;
            }
        }
        std::cout << "Backtracked to state " << current_index << " for goal branch blacklisting" << std::endl;
        BlacklistGoalBranch(goal_branch_root_index);
        std::cout << "Goal branch blacklisting complete" << std::endl;
    }
    // Update the goal reached probability
    // Backtrack all the way to the goal, updating each state's goal_Pfeasbility
    // First, compute the goal state's goal reached probability
    double new_goal_Pfeasiblity = ComputeGoalReachedProbability(new_goal.GetValueImmutable(), goal_position);
    // Update the goal state
    new_goal.GetValueMutable().SetGoalPfeasibility(new_goal_Pfeasiblity);
    // Backtrack up the tree, updating states as we go
    int64_t current_index = new_goal.GetParentIndex();
    while (current_index >= 0)
    {
        // Get the current state that we're looking at
        simple_rrt_planner::SimpleRRTPlannerState<NomdpPlannerState>& current_state = nearest_neighbors_storage_[current_index];
        // Update the state
        UpdateNodeGoalReachedProbability(current_state);
        current_index = current_state.GetParentIndex();
    }
    // Get the goal reached probability that we use to decide when we're done
    total_goal_reached_probability_ = nearest_neighbors_storage_[0].GetValueImmutable().GetGoalPfeasibility();
    std::cout << "Updated goal reached probability to " << total_goal_reached_probability_ << std::endl;
}

void NomdpPlanningSpace::UpdateNodeGoalReachedProbability(simple_rrt_planner::SimpleRRTPlannerState<NomdpPlannerState>& current_node) const
{
    // Check all the children of the current node, and update the node's goal reached probability accordingly
    //
    // Naively, the goal reached probability of a node is the maximum of the child goal reached probabilities;
    // intuitively, the probability of reaching the goal is that of reaching the goal if we follow the best child.
    //
    // HOWEVER - the existence of "split" child states, where multiple states result from a single control input,
    // makes this more compilcated. For split child states, the goal reached probability of the split is the sum
    // over every split option of (split goal probability * probability of split)
    //
    // We can identify split nodes as children which share a transition id
    // First, we go through the children and separate them based on transition id (this puts all the children of a
    // split together in one place)
    std::map<u_int64_t, std::vector<int64_t>> effective_child_branches;
    for (size_t idx = 0; idx < current_node.GetChildIndices().size(); idx++)
    {
        const int64_t& current_child_index = current_node.GetChildIndices()[idx];
        const u_int64_t& child_transition_id = nearest_neighbors_storage_[current_child_index].GetValueImmutable().GetTransitionId();
        effective_child_branches[child_transition_id].push_back(current_child_index);
    }
    // Now that we have the transitions separated out, compute the goal probability of each transition
    std::vector<double> effective_child_branch_probabilities;
    for (auto itr = effective_child_branches.begin(); itr != effective_child_branches.end(); ++itr)
    {
        double transtion_goal_probability = ComputeTransitionGoalProbability(itr->second);
        effective_child_branch_probabilities.push_back(transtion_goal_probability);
    }
    // Now, get the highest transtion probability
    double max_transition_probability = 0.0;
    if (effective_child_branch_probabilities.size() > 0)
    {
        max_transition_probability = *std::max_element(effective_child_branch_probabilities.begin(), effective_child_branch_probabilities.end());
    }
    assert(max_transition_probability > 0.0);
    assert(max_transition_probability <= 1.0);
    // Update the current state
    current_node.GetValueMutable().SetGoalPfeasibility(max_transition_probability);
}

double NomdpPlanningSpace::ComputeTransitionGoalProbability(const std::vector<int64_t>& child_node_indices) const
{
    double total_transition_goal_probability = 0.0;
    for (size_t idx = 0; idx < child_node_indices.size(); idx++)
    {
        const int64_t& current_child_index = child_node_indices[idx];
        const NomdpPlannerState& current_child = nearest_neighbors_storage_[current_child_index].GetValueImmutable();
        total_transition_goal_probability += (current_child.GetGoalPfeasibility() * current_child.GetEdgePfeasibility());
    }
    return total_transition_goal_probability;
}

bool NomdpPlanningSpace::CheckIfGoalBranchRoot(const simple_rrt_planner::SimpleRRTPlannerState<NomdpPlannerState>& state) const
{
    // There are three ways a state can be the the root of a goal branch
    // 1) The transition leading to the state is low-probability
    bool has_low_probability_transition = (state.GetValueImmutable().GetEdgePfeasibility() < goal_probability_threshold_);
    // 2) The transition leading to the state is the result of a split
    bool is_child_of_split = state.GetValueImmutable().IsSplitChild();
    // 3) The parent of the current node is the root of the tree
    bool parent_is_root = (state.GetParentIndex() == 0);
    // If one or more condition is true, the state is a branch root
    if (has_low_probability_transition || is_child_of_split || parent_is_root)
    {
        return true;
    }
    else
    {
        return false;
    }
}

void NomdpPlanningSpace::BlacklistGoalBranch(const int64_t goal_branch_root_index) const
{
    if (goal_branch_root_index < 0)
    {
        ;
    }
    else if (goal_branch_root_index == 0)
    {
        std::cerr << "Blacklisting with goal branch root == tree root is not possible!" << std::endl;
    }
    else
    {
        //std::cout << "Blacklisting goal branch starting at index " << goal_branch_root_index << std::endl;
        // Get the current node
        simple_rrt_planner::SimpleRRTPlannerState<NomdpPlannerState>& current_state = nearest_neighbors_storage_[goal_branch_root_index];
        // Recursively blacklist it
        current_state.GetValueMutable().DisableForNearestNeighbors();
        assert(current_state.GetValueImmutable().UseForNearestNeighbors() == false);
        // Blacklist each child
        const std::vector<int64_t>& child_indices = current_state.GetChildIndices();
        for (size_t idx = 0; idx < child_indices.size(); idx++)
        {
            int64_t child_index = child_indices[idx];
            BlacklistGoalBranch(child_index);
        }
    }
}

#ifdef USE_ROS
std::pair<ExecutionPolicy<Eigen::Vector3d, Eigen::Vector3d>, std::map<std::string, double>> NomdpPlanningSpace::Plan(const Eigen::Vector3d& start, const Eigen::Vector3d& goal, const double goal_bias, const std::chrono::duration<double>& time_limit, ros::Publisher& display_pub)
{
    // Draw the environment, start, and goal
    visualization_msgs::Marker start_marker;
    start_marker.pose.position = EigenHelpersConversions::EigenVector3dToGeometryPoint(start);
    start_marker.pose.orientation = EigenHelpersConversions::EigenQuaterniondToGeometryQuaternion(Eigen::Quaterniond::Identity());
    start_marker.action = visualization_msgs::Marker::ADD;
    start_marker.ns = "start_state";
    start_marker.id = 1;
    start_marker.frame_locked = false;
    start_marker.lifetime = ros::Duration(0.0);
    start_marker.type = visualization_msgs::Marker::SPHERE;
    start_marker.header.frame_id = simulator_.GetFrame();
    start_marker.scale.x = simulator_.GetResolution();
    start_marker.scale.y = simulator_.GetResolution();
    start_marker.scale.z = simulator_.GetResolution();
    start_marker.color.r = 1.0;
    start_marker.color.g = 0.0;
    start_marker.color.b = 1.0;
    start_marker.color.a = 1.0;
    visualization_msgs::Marker goal_marker;
    goal_marker.pose.position = EigenHelpersConversions::EigenVector3dToGeometryPoint(goal);
    goal_marker.pose.orientation = EigenHelpersConversions::EigenQuaterniondToGeometryQuaternion(Eigen::Quaterniond::Identity());
    goal_marker.action = visualization_msgs::Marker::ADD;
    goal_marker.ns = "goal_state";
    goal_marker.id = 1;
    goal_marker.frame_locked = false;
    goal_marker.lifetime = ros::Duration(0.0);
    goal_marker.type = visualization_msgs::Marker::SPHERE;
    goal_marker.header.frame_id = simulator_.GetFrame();
    goal_marker.scale.x = goal_distance_threshold_ * 2.0;
    goal_marker.scale.y = goal_distance_threshold_ * 2.0;
    goal_marker.scale.z = goal_distance_threshold_ * 2.0;
    goal_marker.color.r = 0.0;
    goal_marker.color.g = 0.0;
    goal_marker.color.b = 1.0;
    goal_marker.color.a = 1.0;
    visualization_msgs::Marker env_marker = simulator_.ExportForDisplay();
    env_marker.ns = "environment";
    env_marker.id = 1;
    visualization_msgs::MarkerArray problem_display_rep;
    problem_display_rep.markers.push_back(env_marker);
    problem_display_rep.markers.push_back(start_marker);
    problem_display_rep.markers.push_back(goal_marker);
    display_pub.publish(problem_display_rep);
    int wait = 0;
    while (ros::ok() && wait < 10)
    {
        ros::spinOnce();
        ros::Rate(10.0).sleep();
        wait++;
    }
    NomdpPlannerState start_state(start);
    NomdpPlannerState goal_state(goal);
    // Bind the helper functions
    std::function<int64_t(const std::vector<simple_rrt_planner::SimpleRRTPlannerState<NomdpPlannerState>>&, const NomdpPlannerState&)> nearest_neighbor_fn = std::bind(&NomdpPlanningSpace::GetNearestNeighbor, this, std::placeholders::_1, std::placeholders::_2);
    std::function<bool(const NomdpPlannerState&)> goal_reached_fn = std::bind(&NomdpPlanningSpace::GoalReached, this, std::placeholders::_1, goal);
    std::function<void(simple_rrt_planner::SimpleRRTPlannerState<NomdpPlannerState>&)> goal_reached_callback = std::bind(&NomdpPlanningSpace::GoalReachedCallback, this, std::placeholders::_1, goal);
    std::function<NomdpPlannerState(void)> state_sampling_fn = std::bind(&NomdpPlanningSpace::SampleRandomTargetState, this);
    std::uniform_real_distribution<double> goal_bias_distribution(0.0, 1.0);
    std::function<NomdpPlannerState(void)> complete_sampling_fn = [&](void) { return ((goal_bias_distribution(rng_) > goal_bias) ? state_sampling_fn() : goal_state); };
    std::function<std::vector<NomdpPlannerState>(const NomdpPlannerState&, const NomdpPlannerState&)> forward_propagation_fn = std::bind(&NomdpPlanningSpace::PropagateForwardsAndDraw, this, std::placeholders::_1, std::placeholders::_2, display_pub);
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();
    std::function<bool(void)> termination_check_fn = std::bind(&NomdpPlanningSpace::PlannerTerminationCheck, this, start_time, time_limit);
    // Call the planner
    total_goal_reached_probability_ = 0.0;
    std::pair<std::vector<std::vector<NomdpPlannerState>>, std::map<std::string, double>> planning_results = planner_.PlanMultiPath(nearest_neighbors_storage_, start_state, nearest_neighbor_fn, goal_reached_fn, goal_reached_callback, complete_sampling_fn, forward_propagation_fn, termination_check_fn);
    ExecutionPolicy<Eigen::Vector3d, Eigen::Vector3d> policy = ExtractPolicy(nearest_neighbors_storage_, goal);
    // Draw the final path(s)
    for (size_t pidx = 0; pidx < planning_results.first.size(); pidx++)
    {
        std::vector<NomdpPlannerState> planned_path = planning_results.first[pidx];
        if (planned_path.size() >= 2)
        {
            double goal_reached_probability = ComputeGoalReachedProbability(planned_path[planned_path.size() - 1], goal) * planned_path[planned_path.size() - 1].GetMotionPfeasibility();
            visualization_msgs::MarkerArray path_display_rep;
            for (size_t idx = 1; idx < planned_path.size(); idx++)
            {
                visualization_msgs::Marker expectation_marker;
                expectation_marker.action = visualization_msgs::Marker::ADD;
                expectation_marker.ns = "final_path_" + std::to_string(pidx + 1);
                expectation_marker.id = idx;
                expectation_marker.frame_locked = false;
                expectation_marker.lifetime = ros::Duration(0.0);
                expectation_marker.type = visualization_msgs::Marker::ARROW;
                expectation_marker.header.frame_id = simulator_.GetFrame();
                expectation_marker.scale.x = simulator_.GetResolution() * 1.5;
                expectation_marker.scale.y = simulator_.GetResolution() * 2.25;
                expectation_marker.scale.z = 0.0;
                expectation_marker.points.push_back(EigenHelpersConversions::EigenVector3dToGeometryPoint(planned_path[idx - 1].GetExpectation()));
                expectation_marker.points.push_back(EigenHelpersConversions::EigenVector3dToGeometryPoint(planned_path[idx].GetExpectation()));
                expectation_marker.color.r = 1.0 - goal_reached_probability;
                expectation_marker.color.g = 0.0;
                expectation_marker.color.b = 0.0;
                expectation_marker.color.a = planned_path[idx].GetMotionPfeasibility();
                path_display_rep.markers.push_back(expectation_marker);
            }
            display_pub.publish(path_display_rep);
        }
    }
    return std::pair<ExecutionPolicy<Eigen::Vector3d, Eigen::Vector3d>, std::map<std::string, double>>(policy, planning_results.second);
}
#endif

std::pair<ExecutionPolicy<Eigen::Vector3d, Eigen::Vector3d>, std::map<std::string, double>> NomdpPlanningSpace::Plan(const Eigen::Vector3d& start, const Eigen::Vector3d& goal, const double goal_bias, const std::chrono::duration<double>& time_limit)
{
    NomdpPlannerState start_state(start);
    NomdpPlannerState goal_state(goal);
    // Bind the helper functions
    std::function<int64_t(const std::vector<simple_rrt_planner::SimpleRRTPlannerState<NomdpPlannerState>>&, const NomdpPlannerState&)> nearest_neighbor_fn = std::bind(&NomdpPlanningSpace::GetNearestNeighbor, this, std::placeholders::_1, std::placeholders::_2);
    std::function<bool(const NomdpPlannerState&)> goal_reached_fn = std::bind(&NomdpPlanningSpace::GoalReached, this, std::placeholders::_1, goal);
    std::function<void(simple_rrt_planner::SimpleRRTPlannerState<NomdpPlannerState>&)> goal_reached_callback = std::bind(&NomdpPlanningSpace::GoalReachedCallback, this, std::placeholders::_1, goal);
    std::function<NomdpPlannerState(void)> state_sampling_fn = std::bind(&NomdpPlanningSpace::SampleRandomTargetState, this);
    std::uniform_real_distribution<double> goal_bias_distribution(0.0, 1.0);
    std::function<NomdpPlannerState(void)> complete_sampling_fn = [&](void) { return ((goal_bias_distribution(rng_) > goal_bias) ? state_sampling_fn() : goal_state); };
    std::function<std::vector<NomdpPlannerState>(const NomdpPlannerState&, const NomdpPlannerState&)> forward_propagation_fn = std::bind(&NomdpPlanningSpace::PropagateForwards, this, std::placeholders::_1, std::placeholders::_2);
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();
    std::function<bool(void)> termination_check_fn = std::bind(&NomdpPlanningSpace::PlannerTerminationCheck, this, start_time, time_limit);
    // Call the planner
    total_goal_reached_probability_ = 0.0;
    std::pair<std::vector<std::vector<NomdpPlannerState>>, std::map<std::string, double>> planning_results = planner_.PlanMultiPath(nearest_neighbors_storage_, start_state, nearest_neighbor_fn, goal_reached_fn, goal_reached_callback, complete_sampling_fn, forward_propagation_fn, termination_check_fn);
    ExecutionPolicy<Eigen::Vector3d, Eigen::Vector3d> policy = ExtractPolicy(nearest_neighbors_storage_, goal);
    return std::pair<ExecutionPolicy<Eigen::Vector3d, Eigen::Vector3d>, std::map<std::string, double>>(policy, planning_results.second);
}

ExecutionPolicy<Eigen::Vector3d, Eigen::Vector3d> NomdpPlanningSpace::ExtractPolicy(const std::vector<simple_rrt_planner::SimpleRRTPlannerState<NomdpPlannerState, std::allocator<NomdpPlannerState>>>& planner_tree, const Eigen::Vector3d& goal_position) const
{
    std::function<double(const Eigen::Vector3d&, const Eigen::Vector3d&)> distance_fn = [] (const Eigen::Vector3d& v1, const Eigen::Vector3d& v2) { return (v1 - v2).norm(); };
    ExecutionPolicy<Eigen::Vector3d, Eigen::Vector3d> policy(distance_fn);
    // Go through each state in the tree
    for (size_t sdx = 0; sdx < planner_tree.size(); sdx++)
    {
        // Get the current state
        const simple_rrt_planner::SimpleRRTPlannerState<NomdpPlannerState, std::allocator<NomdpPlannerState>>& state = planner_tree[sdx];
        // Make sure the current state is on a branch that reaches the goal
        if (state.GetValueImmutable().GetGoalPfeasibility() > 0.0)
        {
            // This means at least one of our child states reaches the goal
            const std::vector<int64_t>& child_indices = state.GetChildIndices();
            // Pick the best child state
            int64_t best_child_index = -1;
            double best_child_value = 0.0;
            for (size_t cdx = 0; cdx < child_indices.size(); cdx++)
            {
                const int64_t& child_index = child_indices[cdx];
                const double child_value = planner_tree[child_index].GetValueImmutable().GetGoalPfeasibility();
                if (child_value > best_child_value)
                {
                    best_child_value = child_value;
                    best_child_index = child_index;
                }
            }
            if (best_child_index >= 0)
            {
                const simple_rrt_planner::SimpleRRTPlannerState<NomdpPlannerState, std::allocator<NomdpPlannerState>>& best_child_state = planner_tree[best_child_index];
                // Update the policy
                const Eigen::Vector3d& current_position = state.GetValueImmutable().GetExpectation();
                const Eigen::Vector3d& target_position = best_child_state.GetValueImmutable().GetControlInput();
                const double& confidence = best_child_value;
                policy.ExtendPolicy(current_position, target_position, confidence);
            }
            else
            {
                const Eigen::Vector3d& current_position = state.GetValueImmutable().GetExpectation();
                policy.ExtendPolicy(current_position, goal_position, 1.0);
            }
        }
    }
    return policy;
}

std::pair<EigenHelpers::VectorVector3d, bool> NomdpPlanningSpace::SimulateSinglePolicyExecution(const ExecutionPolicy<Eigen::Vector3d, Eigen::Vector3d>& policy, const Eigen::Vector3d& start, const Eigen::Vector3d& goal, const std::chrono::duration<double>& time_limit, std::mt19937_64& rng) const
{
    EigenHelpers::VectorVector3d trajectory;
    // Make a new robot
    simple_uncertainty_models::Simple3dRobot robot(start, robot_config_.kp, robot_config_.ki, robot_config_.kd, robot_config_.integral_clamp, robot_config_.velocity_limit, robot_config_.max_sensor_noise, robot_config_.max_actuator_noise);
    // Make the simulation callback function
    std::function<std::pair<Eigen::Vector3d, bool>(const Eigen::Vector3d&, const Eigen::Vector3d&)> forward_simulation_callback_fn = std::bind(&SimpleParticleContactSimulator::ForwardSimulationCallback, simulator_, std::placeholders::_1, std::placeholders::_2, 10u);
    // Keep track of where we are
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now();
    while (((std::chrono::time_point<std::chrono::high_resolution_clock>)std::chrono::high_resolution_clock::now() - start_time) < time_limit)
    {
        // Get the current position of the robot
        const Eigen::Vector3d step_start_position = robot.GetPosition();
        // Lookup an action from the policy
        std::pair<Eigen::Vector3d, std::pair<double, double>> policy_action = policy.GetAction(step_start_position, INFINITY);
        const Eigen::Vector3d& action = policy_action.first;
        // Execute forwards
        const double action_distance = (action - step_start_position).norm();
        u_int32_t forward_simulation_steps = (u_int32_t)ceil((action_distance / step_size_) * 40.0);
        for (u_int32_t step = 0; step < forward_simulation_steps; step++)
        {
            std::pair<Eigen::Vector3d, bool> result = robot.MoveTowardsTarget(action, 1.0, forward_simulation_callback_fn, rng);
            trajectory.push_back(result.first);
        }
        // Check if we've reached the goal
        const Eigen::Vector3d step_end_position = robot.GetPosition();
        if ((step_end_position - goal).norm() <= goal_distance_threshold_)
        {
            // We've reached the goal!
            return std::pair<EigenHelpers::VectorVector3d, bool>(trajectory, true);
        }
    }
    // If we get here, we haven't reached the goal!
    return std::pair<EigenHelpers::VectorVector3d, bool>(trajectory, false);
}

#ifdef USE_ROS
double NomdpPlanningSpace::SimulateExectionPolicy(const ExecutionPolicy<Eigen::Vector3d, Eigen::Vector3d>& policy, const Eigen::Vector3d& start, const Eigen::Vector3d& goal, const u_int32_t num_particles, const std::chrono::duration<double>& time_limit, ros::Publisher& display_pub) const
{
    std::vector<EigenHelpers::VectorVector3d> particle_executions(num_particles);
    u_int32_t reached_goal = 0;
    for (size_t idx = 0; idx < num_particles; idx++)
    {
        std::pair<EigenHelpers::VectorVector3d, bool> particle_execution = SimulateSinglePolicyExecution(policy, start, goal, time_limit, rng_);
        particle_executions[idx] = particle_execution.first;
        const EigenHelpers::VectorVector3d& particle_track = particle_execution.first;
        if (particle_execution.second)
        {
            reached_goal++;
        }
        visualization_msgs::Marker execution_display;
        execution_display.pose = EigenHelpersConversions::EigenAffine3dToGeometryPose(Eigen::Affine3d::Identity());
        execution_display.action = visualization_msgs::Marker::ADD;
        execution_display.ns = "policy_execution";
        execution_display.id = idx + 1;
        execution_display.frame_locked = false;
        execution_display.lifetime = ros::Duration(0.0);
        execution_display.type = visualization_msgs::Marker::SPHERE_LIST;
        execution_display.header.frame_id = simulator_.GetFrame();
        execution_display.scale.x = simulator_.GetResolution() * 0.5;
        // Make the colors
        std_msgs::ColorRGBA state_color;
        if (particle_execution.second)
        {
            state_color.r = 0.0;
            state_color.g = 1.0;
            state_color.b = 0.0;
            state_color.a = 1.0;
        }
        else
        {
            state_color.r = 1.0;
            state_color.g = 0.0;
            state_color.b = 0.0;
            state_color.a = 1.0;
        }
        // Add each particle
        for (size_t idx = 0; idx < particle_track.size(); idx++)
        {
            const Eigen::Vector3d& current_point = particle_track[idx];
            execution_display.points.push_back(EigenHelpersConversions::EigenVector3dToGeometryPoint(current_point));
            execution_display.colors.push_back(state_color);
        }
        visualization_msgs::MarkerArray display_markers;
        display_markers.markers.push_back(execution_display);
        display_pub.publish(display_markers);
    }
    return (double)reached_goal / (double)num_particles;
}
#endif

double NomdpPlanningSpace::SimulateExectionPolicy(const ExecutionPolicy<Eigen::Vector3d, Eigen::Vector3d>& policy, const Eigen::Vector3d& start, const Eigen::Vector3d& goal, const u_int32_t num_particles, const std::chrono::duration<double>& time_limit) const
{
    std::vector<EigenHelpers::VectorVector3d> particle_executions(num_particles);
#ifdef ENABLE_PARALLEL
    std::atomic<u_int32_t> reached_goal(0);
    #pragma omp parallel for schedule(guided)
#else
    u_int32_t reached_goal = 0;
#endif
    for (size_t idx = 0; idx < num_particles; idx++)
    {
#ifdef ENABLE_PARALLEL
        int th_id = omp_get_thread_num();
        std::pair<EigenHelpers::VectorVector3d, bool> particle_execution = SimulateSinglePolicyExecution(policy, start, goal, time_limit, rngs_[th_id]);
#else
        std::pair<EigenHelpers::VectorVector3d, bool> particle_execution = SimulateSinglePolicyExecution(policy, start, goal, time_limit, rng_);
#endif
        particle_executions[idx] = particle_execution.first;
        if (particle_execution.second)
        {
            reached_goal++;
        }
    }
    return (double)reached_goal / (double)num_particles;
}


