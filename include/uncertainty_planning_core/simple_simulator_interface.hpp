#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <functional>
#include <chrono>
#include <random>
#include <arc_utilities/eigen_helpers.hpp>
#include <arc_utilities/eigen_helpers_conversions.hpp>
#include <arc_utilities/voxel_grid.hpp>
#include <sdf_tools/tagged_object_collision_map.hpp>
#include <sdf_tools/sdf.hpp>
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <omp.h>

#ifndef SIMPLE_SIMULATOR_INTERFACE_HPP
#define SIMPLE_SIMULATOR_INTERFACE_HPP

namespace std
{
    template <>
    struct hash<std::pair<size_t, size_t>>
    {
        std::size_t operator()(const std::pair<size_t, size_t>& pair) const
        {
            using std::size_t;
            using std::hash;
            return (std::hash<size_t>()(pair.first) ^ (std::hash<size_t>()(pair.second) << 1));
        }
    };
}

namespace simple_simulator_interface
{
    template<typename Configuration, typename ConfigAlloc=std::allocator<Configuration>>
    struct ForwardSimulationContactResolverStepTrace
    {
        std::vector<Configuration, ConfigAlloc> contact_resolution_steps;
    };

    template<typename Configuration, typename ConfigAlloc=std::allocator<Configuration>>
    struct ForwardSimulationResolverTrace
    {
        Eigen::VectorXd control_input;
        Eigen::VectorXd control_input_step;
        std::vector<ForwardSimulationContactResolverStepTrace<Configuration, ConfigAlloc>> contact_resolver_steps;
    };

    template<typename Configuration, typename ConfigAlloc=std::allocator<Configuration>>
    struct ForwardSimulationStepTrace
    {
        std::vector<ForwardSimulationResolverTrace<Configuration, ConfigAlloc>> resolver_steps;

        inline void Reset()
        {
            resolver_steps.clear();
        }
    };

    template<typename Configuration, typename ConfigAlloc=std::allocator<Configuration>>
    inline std::vector<Configuration, ConfigAlloc> ExtractTrajectoryFromTrace(const ForwardSimulationStepTrace<Configuration, ConfigAlloc>& trace)
    {
        std::vector<Configuration, ConfigAlloc> execution_trajectory;
        execution_trajectory.reserve(trace.resolver_steps.size());
        // Each step corresponds to a controller interval timestep in the real world
        for (size_t step_idx = 0; step_idx < trace.resolver_steps.size(); step_idx++)
        {
            // Each step trace is the entire resolver history of the motion
            const ForwardSimulationResolverTrace<Configuration, ConfigAlloc>& step_trace = trace.resolver_steps[step_idx];
            // Get the current trace segment
            assert(step_trace.contact_resolver_steps.size() > 0);
            // The last contact resolution step is the final result of resolving the timestep
            const ForwardSimulationContactResolverStepTrace<Configuration, ConfigAlloc>& contact_resolution_trace = step_trace.contact_resolver_steps.back();
            // Get the last (collision-free resolved) config of the last resolution step
            assert(contact_resolution_trace.contact_resolution_steps.size() > 0);
            const Configuration& resolved_config = contact_resolution_trace.contact_resolution_steps.back();
            execution_trajectory.push_back(resolved_config);
        }
        execution_trajectory.shrink_to_fit();
        return execution_trajectory;
    }

    class SurfaceNormalGrid
    {
    protected:

        struct StoredSurfaceNormal
        {
        protected:

            Eigen::Vector4d entry_direction_;
            Eigen::Vector3d normal_;

        public:

            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            StoredSurfaceNormal(const Eigen::Vector3d& normal, const Eigen::Vector3d& direction) : normal_(EigenHelpers::SafeNormal(normal))
            {
                const Eigen::Vector4d direction4d(direction.x(), direction.y(), direction.z(), 0.0);
                entry_direction_ = (EigenHelpers::SafeNormal(direction4d));
            }

            StoredSurfaceNormal(const Eigen::Vector3d& normal, const Eigen::Vector4d& direction) : entry_direction_(EigenHelpers::SafeNormal(direction)), normal_(EigenHelpers::SafeNormal(normal)) {}

            StoredSurfaceNormal() : entry_direction_(Eigen::Vector4d(0.0, 0.0, 0.0, 0.0)), normal_(Eigen::Vector3d(0.0, 0.0, 0.0)) {}

            const Eigen::Vector4d& EntryDirection4d() const
            {
                return entry_direction_;
            }

            Eigen::Vector3d EntryDirection3d() const
            {
                return entry_direction_.block<3, 1>(0, 0);
            }

            const Eigen::Vector3d& Normal() const
            {
                return normal_;
            }
        };

        VoxelGrid::VoxelGrid<std::vector<StoredSurfaceNormal>> surface_normal_grid_;
        bool initialized_;

        static Eigen::Vector3d GetBestSurfaceNormal(const std::vector<StoredSurfaceNormal>& stored_surface_normals, const Eigen::Vector3d& direction)
        {
            assert(stored_surface_normals.size() > 0);
            const double direction_norm = direction.norm();
            assert(direction_norm > 0.0);
            const Eigen::Vector3d unit_direction = direction / direction_norm;
            int32_t best_stored_index = -1;
            double best_dot_product = -std::numeric_limits<double>::infinity();
            for (size_t idx = 0; idx < stored_surface_normals.size(); idx++)
            {
                const StoredSurfaceNormal& stored_surface_normal = stored_surface_normals[idx];
                const double dot_product = stored_surface_normal.EntryDirection3d().dot(unit_direction);
                if (dot_product > best_dot_product)
                {
                    best_dot_product = dot_product;
                    best_stored_index = (int32_t)idx;
                }
            }
            assert(best_stored_index >= 0);
            const Eigen::Vector3d& best_surface_normal = stored_surface_normals[(size_t)best_stored_index].Normal();
            return best_surface_normal;
        }

        static Eigen::Vector3d GetBestSurfaceNormal(const std::vector<StoredSurfaceNormal>& stored_surface_normals, const Eigen::Vector4d& direction)
        {
            assert(stored_surface_normals.size() > 0);
            const double direction_norm = direction.norm();
            assert(direction_norm > 0.0);
            const Eigen::Vector4d unit_direction = direction / direction_norm;
            int32_t best_stored_index = -1;
            double best_dot_product = -std::numeric_limits<double>::infinity();
            for (size_t idx = 0; idx < stored_surface_normals.size(); idx++)
            {
                const StoredSurfaceNormal& stored_surface_normal = stored_surface_normals[idx];
                const double dot_product = stored_surface_normal.EntryDirection4d().dot(unit_direction);
                if (dot_product > best_dot_product)
                {
                    best_dot_product = dot_product;
                    best_stored_index = (int32_t)idx;
                }
            }
            assert(best_stored_index >= 0);
            const Eigen::Vector3d& best_surface_normal = stored_surface_normals[(size_t)best_stored_index].Normal();
            return best_surface_normal;
        }

    public:

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SurfaceNormalGrid(const Eigen::Affine3d& origin_transform, const double resolution, const double x_size, const double y_size, const double z_size)
        {
            surface_normal_grid_ = VoxelGrid::VoxelGrid<std::vector<StoredSurfaceNormal>>(origin_transform, resolution, x_size, y_size, z_size, std::vector<StoredSurfaceNormal>());
            initialized_ = true;
        }

        SurfaceNormalGrid() : initialized_(false) {}

        bool IsInitialized() const
        {
            return initialized_;
        }

        inline std::pair<Eigen::Vector3d, bool> LookupSurfaceNormal(const double x, const double y, const double z, const Eigen::Vector3d& direction) const
        {
            assert(initialized_);
            const Eigen::Vector4d location(x, y, z, 1.0);
            return LookupSurfaceNormal(location, direction);
        }

        inline std::pair<Eigen::Vector3d, bool> LookupSurfaceNormal(const Eigen::Vector3d& location, const Eigen::Vector3d& direction) const
        {
            assert(initialized_);
            const std::vector<int64_t> indices = surface_normal_grid_.LocationToGridIndex3d(location);
            if (indices.size() == 3)
            {
                return LookupSurfaceNormal(indices[0], indices[1], indices[2], direction);
            }
            else
            {
                return std::pair<Eigen::Vector3d, bool>(Eigen::Vector3d(0.0, 0.0, 0.0), false);
            }
        }

        inline std::pair<Eigen::Vector3d, bool> LookupSurfaceNormal(const Eigen::Vector4d& location, const Eigen::Vector3d& direction) const
        {
            assert(initialized_);
            const std::vector<int64_t> indices = surface_normal_grid_.LocationToGridIndex4d(location);
            if (indices.size() == 3)
            {
                return LookupSurfaceNormal(indices[0], indices[1], indices[2], direction);
            }
            else
            {
                return std::pair<Eigen::Vector3d, bool>(Eigen::Vector3d(0.0, 0.0, 0.0), false);
            }
        }

        inline std::pair<Eigen::Vector3d, bool> LookupSurfaceNormal(const Eigen::Vector4d& location, const Eigen::Vector4d& direction) const
        {
            assert(initialized_);
            const std::vector<int64_t> indices = surface_normal_grid_.LocationToGridIndex4d(location);
            if (indices.size() == 3)
            {
                return LookupSurfaceNormal(indices[0], indices[1], indices[2], direction);
            }
            else
            {
                return std::pair<Eigen::Vector3d, bool>(Eigen::Vector3d(0.0, 0.0, 0.0), false);
            }
        }

        inline std::pair<Eigen::Vector3d, bool> LookupSurfaceNormal(const VoxelGrid::GRID_INDEX& index, const Eigen::Vector3d& direction) const
        {
            assert(initialized_);
            return LookupSurfaceNormal(index.x, index.y, index.z, direction);
        }

        inline std::pair<Eigen::Vector3d, bool> LookupSurfaceNormal(const int64_t x_index, const int64_t y_index, const int64_t z_index, const Eigen::Vector3d& direction) const
        {
            assert(initialized_);
            const std::pair<const std::vector<StoredSurfaceNormal>&, bool> lookup = surface_normal_grid_.GetImmutable(x_index, y_index, z_index);
            if (lookup.second)
            {
                const std::vector<StoredSurfaceNormal>& stored_surface_normals = lookup.first;
                if (stored_surface_normals.size() == 0)
                {
                    return std::pair<Eigen::Vector3d, bool>(Eigen::Vector3d(0.0, 0.0, 0.0), true);
                }
                else
                {
                    // We get the "best" match surface normal given our entry direction
                    return std::pair<Eigen::Vector3d, bool>(GetBestSurfaceNormal(stored_surface_normals, direction), true);
                }
            }
            else
            {
                return std::pair<Eigen::Vector3d, bool>(Eigen::Vector3d(0.0, 0.0, 0.0), false);
            }
        }

        inline std::pair<Eigen::Vector3d, bool> LookupSurfaceNormal(const int64_t x_index, const int64_t y_index, const int64_t z_index, const Eigen::Vector4d& direction) const
        {
            assert(initialized_);
            const std::pair<const std::vector<StoredSurfaceNormal>&, bool> lookup = surface_normal_grid_.GetImmutable(x_index, y_index, z_index);
            if (lookup.second)
            {
                const std::vector<StoredSurfaceNormal>& stored_surface_normals = lookup.first;
                if (stored_surface_normals.size() == 0)
                {
                    return std::pair<Eigen::Vector3d, bool>(Eigen::Vector3d(0.0, 0.0, 0.0), true);
                }
                else
                {
                    // We get the "best" match surface normal given our entry direction
                    return std::pair<Eigen::Vector3d, bool>(GetBestSurfaceNormal(stored_surface_normals, direction), true);
                }
            }
            else
            {
                return std::pair<Eigen::Vector3d, bool>(Eigen::Vector3d(0.0, 0.0, 0.0), false);
            }
        }

        inline bool InsertSurfaceNormal(const double x, const double y, const double z, const Eigen::Vector3d& surface_normal, const Eigen::Vector3d& entry_direction)
        {
            assert(initialized_);
            const Eigen::Vector3d location(x, y, z);
            return InsertSurfaceNormal(location, surface_normal, entry_direction);
        }

        inline bool InsertSurfaceNormal(const Eigen::Vector3d& location, const Eigen::Vector3d& surface_normal, const Eigen::Vector3d& entry_direction)
        {
            assert(initialized_);
            const std::vector<int64_t> indices = surface_normal_grid_.LocationToGridIndex3d(location);
            if (indices.size() == 3)
            {
                return InsertSurfaceNormal(indices[0], indices[1], indices[2], surface_normal, entry_direction);
            }
            else
            {
                return false;
            }
        }

        inline bool InsertSurfaceNormal(const VoxelGrid::GRID_INDEX& index, const Eigen::Vector3d& surface_normal, const Eigen::Vector3d& entry_direction)
        {
            assert(initialized_);
            return InsertSurfaceNormal(index.x, index.y, index.z, surface_normal, entry_direction);
        }

        inline bool InsertSurfaceNormal(const int64_t x_index, const int64_t y_index, const int64_t z_index, const Eigen::Vector3d& surface_normal, const Eigen::Vector3d& entry_direction)
        {
            assert(initialized_);
            std::pair<std::vector<StoredSurfaceNormal>&, bool> cell_query = surface_normal_grid_.GetMutable(x_index, y_index, z_index);
            if (cell_query.second)
            {
                std::vector<StoredSurfaceNormal>& cell_normals = cell_query.first;
                cell_normals.push_back(StoredSurfaceNormal(surface_normal, entry_direction));
                return true;
            }
            else
            {
                return false;
            }
        }

        inline bool ClearStoredSurfaceNormals(const double x, const double y, const double z)
        {
            assert(initialized_);
            const Eigen::Vector3d location(x, y, z);
            return ClearStoredSurfaceNormals(location);
        }

        inline bool ClearStoredSurfaceNormals(const Eigen::Vector3d& location)
        {
            assert(initialized_);
            const std::vector<int64_t> indices = surface_normal_grid_.LocationToGridIndex3d(location);
            if (indices.size() == 3)
            {
                return ClearStoredSurfaceNormals(indices[0], indices[1], indices[2]);
            }
            else
            {
                return false;
            }
        }

        inline bool ClearStoredSurfaceNormals(const VoxelGrid::GRID_INDEX& index)
        {
            assert(initialized_);
            return ClearStoredSurfaceNormals(index.x, index.y, index.z);
        }

        inline bool ClearStoredSurfaceNormals(const int64_t x_index, const int64_t y_index, const int64_t z_index)
        {
            assert(initialized_);
            std::pair<std::vector<StoredSurfaceNormal>&, bool> cell_query = surface_normal_grid_.GetMutable(x_index, y_index, z_index);
            if (cell_query.second)
            {
                std::vector<StoredSurfaceNormal>& cell_normals = cell_query.first;
                cell_normals.clear();
                return true;
            }
            else
            {
                return false;
            }
        }
    };

    template<typename Robot, typename Configuration, typename RNG, typename ConfigAlloc=std::allocator<Configuration>>
    class SimulatorInterface
    {
    protected:

        sdf_tools::TaggedObjectCollisionMapGrid environment_;
        sdf_tools::SignedDistanceField environment_sdf_;
        SurfaceNormalGrid surface_normals_grid_;
        std::map<uint32_t, uint32_t> convex_segment_counts_;
        int32_t debug_level_;

    public:

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SimulatorInterface(const sdf_tools::TaggedObjectCollisionMapGrid& environment, const sdf_tools::SignedDistanceField& environment_sdf, const SurfaceNormalGrid& surface_normals_grid, const int32_t debug_level)
        {
            environment_ = environment;
            environment_sdf_ = environment_sdf;
            surface_normals_grid_ = surface_normals_grid;
            convex_segment_counts_ = environment_.UpdateConvexSegments();
            debug_level_ = debug_level;
        }

        inline int32_t GetDebugLevel() const
        {
            return debug_level_;
        }

        inline int32_t SetDebugLevel(const int32_t debug_level)
        {
            debug_level_ = debug_level;
            return debug_level_;
        }

        inline Eigen::Affine3d GetOriginTransform() const
        {
            return environment_.GetOriginTransform();
        }

        inline std::string GetFrame() const
        {
            return environment_.GetFrame();
        }

        inline double GetResolution() const
        {
            return environment_.GetResolution();
        }

        inline const sdf_tools::TaggedObjectCollisionMapGrid& GetEnvironment() const
        {
            return environment_;
        }

        inline sdf_tools::TaggedObjectCollisionMapGrid& GetMutableEnvironment()
        {
            return environment_;
        }

        inline const sdf_tools::SignedDistanceField& GetEnvironmentSDF() const
        {
            return environment_sdf_;
        }

        inline sdf_tools::SignedDistanceField& GetMutableEnvironmentSDF()
        {
            return environment_sdf_;
        }

        inline visualization_msgs::Marker ExportEnvironmentForDisplay(const float alpha=1.0f) const
        {
            return environment_.ExportForDisplay(alpha);
        }

        inline visualization_msgs::Marker ExportSDFForDisplay(const float alpha=1.0f) const
        {
            return environment_sdf_.ExportForDisplay(alpha);
        }

        virtual visualization_msgs::MarkerArray ExportAllForDisplay() const
        {
            visualization_msgs::MarkerArray display_markers;
            visualization_msgs::Marker env_marker = environment_.ExportForDisplay();
            env_marker.id = 1;
            env_marker.ns = "sim_environment";
            display_markers.markers.push_back(env_marker);
            visualization_msgs::Marker components_marker = environment_.ExportConnectedComponentsForDisplay(false);
            components_marker.id = 1;
            components_marker.ns = "sim_environment_components";
            display_markers.markers.push_back(components_marker);
            visualization_msgs::Marker env_sdf_marker = environment_sdf_.ExportForDisplay(1.0f);
            env_sdf_marker.id = 1;
            env_sdf_marker.ns = "sim_environment_sdf";
            display_markers.markers.push_back(env_sdf_marker);
            // Draw all the convex segments for each object
            for (auto convex_segment_counts_itr = convex_segment_counts_.begin(); convex_segment_counts_itr != convex_segment_counts_.end(); ++convex_segment_counts_itr)
            {
                const uint32_t object_id = convex_segment_counts_itr->first;
                const uint32_t convex_segment_count = convex_segment_counts_itr->second;
                for (uint32_t convex_segment = 1; convex_segment <= convex_segment_count; convex_segment++)
                {
                    const visualization_msgs::Marker segment_marker = environment_.ExportConvexSegmentForDisplay(object_id, convex_segment);
                    display_markers.markers.push_back(segment_marker);
                }
            }
            return display_markers;
        }

        inline static std_msgs::ColorRGBA MakeColor(const float r, const float g, const float b, const float a)
        {
            std_msgs::ColorRGBA color;
            color.r = r;
            color.g = g;
            color.b = b;
            color.a = a;
            return color;
        }

        inline visualization_msgs::Marker DrawRobotConfiguration(const Robot& immutable_robot, const Configuration& configuration, const std_msgs::ColorRGBA& color) const
        {
            Robot robot = immutable_robot;
            std_msgs::ColorRGBA real_color = color;
            visualization_msgs::Marker configuration_marker;
            configuration_marker.action = visualization_msgs::Marker::ADD;
            configuration_marker.ns = "UNKNOWN";
            configuration_marker.id = 1;
            configuration_marker.frame_locked = false;
            configuration_marker.lifetime = ros::Duration(0.0);
            configuration_marker.type = visualization_msgs::Marker::SPHERE_LIST;
            configuration_marker.header.frame_id = this->GetFrame();
            configuration_marker.scale.x = this->GetResolution();
            configuration_marker.scale.y = this->GetResolution();
            configuration_marker.scale.z = this->GetResolution();
            const Eigen::Affine3d base_transform = Eigen::Affine3d::Identity();
            configuration_marker.pose = EigenHelpersConversions::EigenAffine3dToGeometryPose(base_transform);
            configuration_marker.color = real_color;
            // Make the indivudal points
            // Get the list of link name + link points for all the links of the robot
            const std::vector<std::pair<std::string, std::shared_ptr<EigenHelpers::VectorVector4d>>> robot_links_points = robot.GetRawLinksPoints();
            // Update the position of the robot
            robot.UpdatePosition(configuration);
            // Now, go through the links and points of the robot for collision checking
            for (size_t link_idx = 0; link_idx < robot_links_points.size(); link_idx++)
            {
                // Grab the link name and points
                const std::string& link_name = robot_links_points[link_idx].first;
                const EigenHelpers::VectorVector4d& link_points = *(robot_links_points[link_idx].second);
                // Get the transform of the current link
                const Eigen::Affine3d link_transform = robot.GetLinkTransform(link_name);
                // Now, go through the points of the link
                for (size_t point_idx = 0; point_idx < link_points.size(); point_idx++)
                {
                    // Transform the link point into the environment frame
                    const Eigen::Vector4d& link_relative_point = link_points[point_idx];
                    const Eigen::Vector4d environment_relative_point = link_transform * link_relative_point;
                    const geometry_msgs::Point marker_point = EigenHelpersConversions::EigenVector4dToGeometryPoint(environment_relative_point);
                    configuration_marker.points.push_back(marker_point);
                    if (link_relative_point.norm() == 0.0)
                    {
                        std_msgs::ColorRGBA black_color;
                        black_color.r = 0.0f;
                        black_color.g = 0.0f;
                        black_color.b = 0.0f;
                        black_color.a = 1.0f;
                        configuration_marker.colors.push_back(black_color);
                    }
                    else
                    {
                        configuration_marker.colors.push_back(real_color);
                    }
                }
            }
            return configuration_marker;
        }

        inline visualization_msgs::Marker DrawRobotControlInput(const Robot& immutable_robot, const Configuration& configuration, const Eigen::VectorXd& control_input, const std_msgs::ColorRGBA& color) const
        {
            Robot robot = immutable_robot;
            std_msgs::ColorRGBA real_color = color;
            visualization_msgs::Marker configuration_marker;
            configuration_marker.action = visualization_msgs::Marker::ADD;
            configuration_marker.ns = "UNKNOWN";
            configuration_marker.id = 1;
            configuration_marker.frame_locked = false;
            configuration_marker.lifetime = ros::Duration(0.0);
            configuration_marker.type = visualization_msgs::Marker::LINE_LIST;
            configuration_marker.header.frame_id = this->GetFrame();
            configuration_marker.scale.x = this->GetResolution() * 0.5;
            configuration_marker.scale.y = this->GetResolution() * 0.5;
            configuration_marker.scale.z = this->GetResolution() * 0.5;
            const Eigen::Affine3d base_transform = Eigen::Affine3d::Identity();
            configuration_marker.pose = EigenHelpersConversions::EigenAffine3dToGeometryPose(base_transform);
            configuration_marker.color = real_color;
            // Make the indivudal points
            // Get the list of link name + link points for all the links of the robot
            const std::vector<std::pair<std::string, std::shared_ptr<EigenHelpers::VectorVector4d>>> robot_links_points = robot.GetRawLinksPoints();
            // Now, go through the links and points of the robot for collision checking
            for (size_t link_idx = 0; link_idx < robot_links_points.size(); link_idx++)
            {
                // Grab the link name and points
                const std::string& link_name = robot_links_points[link_idx].first;
                const EigenHelpers::VectorVector4d& link_points = *(robot_links_points[link_idx].second);
                // Get the current transform
                // Update the position of the robot
                robot.UpdatePosition(configuration);
                // Get the transform of the current link
                const Eigen::Affine3d current_link_transform = robot.GetLinkTransform(link_name);
                // Apply the control input
                robot.ApplyControlInput(control_input);
                // Get the transform of the current link
                const Eigen::Affine3d current_plus_control_link_transform = robot.GetLinkTransform(link_name);
                // Now, go through the points of the link
                for (size_t point_idx = 0; point_idx < link_points.size(); point_idx++)
                {
                    // Transform the link point into the environment frame
                    const Eigen::Vector4d& link_relative_point = link_points[point_idx];
                    const Eigen::Vector4d environment_relative_current_point = current_link_transform * link_relative_point;
                    const Eigen::Vector4d environment_relative_current_plus_control_point = current_plus_control_link_transform * link_relative_point;
                    const geometry_msgs::Point current_marker_point = EigenHelpersConversions::EigenVector4dToGeometryPoint(environment_relative_current_point);
                    const geometry_msgs::Point current_plus_control_marker_point = EigenHelpersConversions::EigenVector4dToGeometryPoint(environment_relative_current_plus_control_point);
                    configuration_marker.points.push_back(current_marker_point);
                    configuration_marker.points.push_back(current_plus_control_marker_point);
                    configuration_marker.colors.push_back(real_color);
                    configuration_marker.colors.push_back(real_color);
                }
            }
            return configuration_marker;
        }

        virtual std::map<std::string, double> GetStatistics() const = 0;

        virtual void ResetStatistics() = 0;

        virtual bool CheckConfigCollision(const Robot& immutable_robot, const Configuration& config, const double inflation_ratio=0.0) const = 0;

        virtual std::pair<Configuration, bool> ForwardSimulateMutableRobot(Robot& mutable_robot, const Configuration& target_position, RNG& rng, const double forward_simulation_time, const double simulation_shortcut_distance, const bool use_individual_jacobians, const bool allow_contacts, ForwardSimulationStepTrace<Configuration, ConfigAlloc>& trace, const bool enable_tracing, ros::Publisher& display_debug_publisher) const = 0;

        virtual std::pair<Configuration, bool> ForwardSimulateRobot(const Robot& immutable_robot, const Configuration& start_position, const Configuration& target_position, RNG& rng, const double forward_simulation_time, const double simulation_shortcut_distance, const bool use_individual_jacobians, const bool allow_contacts, ForwardSimulationStepTrace<Configuration, ConfigAlloc>& trace, const bool enable_tracing, ros::Publisher& display_debug_publisher) const = 0;

        virtual std::vector<std::pair<Configuration, bool>> ForwardSimulateRobots(const Robot& immutable_robot, const std::vector<Configuration, ConfigAlloc>& start_positions, const std::vector<Configuration, ConfigAlloc>& target_positions, std::vector<RNG>& rng, const double forward_simulation_time, const double simulation_shortcut_distance, const bool use_individual_jacobians, const bool allow_contacts, ros::Publisher& display_debug_publisher) const = 0;
    };
}

#endif // SIMPLE_SIMULATOR_INTERFACE_HPP
