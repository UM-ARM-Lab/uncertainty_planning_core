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
#include <arc_utilities/voxel_grid.hpp>
#include <sdf_tools/tagged_object_collision_map.hpp>
#include <sdf_tools/sdf.hpp>
#include <nomdp_planning/simple_pid_controller.hpp>
#include <nomdp_planning/simple_uncertainty_models.hpp>
#include <nomdp_planning/nomdp_planner_state.hpp>

#ifndef SIMPLE_PARTICLE_CONTACT_SIMULATOR_HPP
#define SIMPLE_PARTICLE_CONTACT_SIMULATOR_HPP

#ifndef DISABLE_ROS_INTERFACE
    #define USE_ROS
#endif

#ifdef USE_ROS
    #include <ros/ros.h>
    #include <visualization_msgs/MarkerArray.h>
#endif

#ifndef MAX_RESOLVER_ITERATIONS
    #define MAX_RESOLVER_ITERATIONS 30
#endif

#ifndef RESOLVE_CORRECTION_STEP_SCALING_DECAY_ITERATIONS
    #define RESOLVE_CORRECTION_STEP_SCALING_DECAY_ITERATIONS 5
#endif

#ifndef RESOLVE_CORRECTION_STEP_SCALING_DECAY_RATE
    #define RESOLVE_CORRECTION_STEP_SCALING_DECAY_RATE 0.75
#endif

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

namespace nomdp_planning_tools
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
    };

    struct OBSTACLE_CONFIG
    {
        uint32_t object_id;
        uint8_t r;
        uint8_t g;
        uint8_t b;
        uint8_t a;
        Eigen::Affine3d pose;
        Eigen::Vector3d extents;

        OBSTACLE_CONFIG(const uint32_t in_object_id, const Eigen::Affine3d& in_pose, const Eigen::Vector3d& in_extents, const uint8_t in_r, const uint8_t in_g, const uint8_t in_b, const uint8_t in_a) : r(in_r), g(in_g), b(in_b), a(in_a), pose(in_pose), extents(in_extents)
        {
            assert(in_object_id > 0);
            object_id = in_object_id;
        }

        OBSTACLE_CONFIG(const uint32_t in_object_id, const Eigen::Vector3d& in_translation, const Eigen::Quaterniond& in_orientation, const Eigen::Vector3d& in_extents, const uint8_t in_r, const uint8_t in_g, const uint8_t in_b, const uint8_t in_a) : r(in_r), g(in_g), b(in_b), a(in_a)
        {
            assert(in_object_id > 0);
            object_id = in_object_id;
            pose = (Eigen::Translation3d)in_translation * in_orientation;
            extents = in_extents;
        }

        OBSTACLE_CONFIG() : object_id(0u), r(0u), g(0u), b(0u), a(0u), pose(Eigen::Affine3d::Identity()), extents(0.0, 0.0, 0.0) {}
    };

    class SurfaceNormalGrid
    {
    protected:

        struct StoredSurfaceNormal
        {
            Eigen::Vector3d normal;
            Eigen::Vector3d entry_direction;

            StoredSurfaceNormal(const Eigen::Vector3d& in_normal, const Eigen::Vector3d& in_direction) : normal(EigenHelpers::SafeNormal(in_normal)), entry_direction(EigenHelpers::SafeNormal(in_direction)) {}

            StoredSurfaceNormal() : normal(Eigen::Vector3d(0.0, 0.0, 0.0)), entry_direction(Eigen::Vector3d(0.0, 0.0, 0.0)) {}
        };

        bool initialized_;
        VoxelGrid::VoxelGrid<std::vector<StoredSurfaceNormal>> surface_normal_grid_;

        static Eigen::Vector3d GetBestSurfaceNormal(const std::vector<StoredSurfaceNormal>& stored_surface_normals, const Eigen::Vector3d& direction)
        {
            assert(stored_surface_normals.size() > 0);
            const double direction_norm = direction.norm();
            assert(direction_norm > 0.0);
            const Eigen::Vector3d unit_direction = direction / direction_norm;
            int32_t best_stored_index = -1;
            double best_dot_product = -INFINITY;
            for (size_t idx = 0; idx < stored_surface_normals.size(); idx++)
            {
                const StoredSurfaceNormal& stored_surface_normal = stored_surface_normals[idx];
                const double dot_product = stored_surface_normal.entry_direction.dot(unit_direction);
                if (dot_product > best_dot_product)
                {
                    best_dot_product = dot_product;
                    best_stored_index = (int32_t)idx;
                }
            }
            assert(best_stored_index >= 0);
            const Eigen::Vector3d& best_surface_normal = stored_surface_normals[best_stored_index].normal;
            return best_surface_normal;
        }

    public:

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
            const Eigen::Vector3d location(x, y, z);
            return LookupSurfaceNormal(location, direction);
        }

        inline std::pair<Eigen::Vector3d, bool> LookupSurfaceNormal(const Eigen::Vector3d& location, const Eigen::Vector3d& direction) const
        {
            assert(initialized_);
            const std::vector<int64_t> indices = surface_normal_grid_.LocationToGridIndex(location);
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

        inline bool InsertSurfaceNormal(const double x, const double y, const double z, const Eigen::Vector3d& surface_normal, const Eigen::Vector3d& entry_direction)
        {
            assert(initialized_);
            const Eigen::Vector3d location(x, y, z);
            return InsertSurfaceNormal(location, surface_normal, entry_direction);
        }

        inline bool InsertSurfaceNormal(const Eigen::Vector3d& location, const Eigen::Vector3d& surface_normal, const Eigen::Vector3d& entry_direction)
        {
            assert(initialized_);
            const std::vector<int64_t> indices = surface_normal_grid_.LocationToGridIndex(location);
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
            const std::vector<int64_t> indices = surface_normal_grid_.LocationToGridIndex(location);
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

    class SimpleParticleContactSimulator
    {
    protected:

        struct RawCellSurfaceNormal
        {
            Eigen::Vector3d normal;
            Eigen::Vector3d entry_direction;

            RawCellSurfaceNormal(const Eigen::Vector3d& in_normal, const Eigen::Vector3d& in_direction) : normal(in_normal), entry_direction(in_direction) {}

            RawCellSurfaceNormal() : normal(Eigen::Vector3d(0.0, 0.0, 0.0)), entry_direction(Eigen::Vector3d(0.0, 0.0, 0.0)) {}
        };

        bool initialized_;
        double contact_distance_threshold_;
        sdf_tools::TaggedObjectCollisionMapGrid environment_;
        sdf_tools::SignedDistanceField environment_sdf_;
        SurfaceNormalGrid surface_normals_grid_;
        std::map<uint32_t, sdf_tools::SignedDistanceField> per_object_sdfs_;
        std::map<uint32_t, uint32_t> convex_segment_counts_;
        VoxelGrid::VoxelGrid<uint64_t> fingerprint_grid_;
        mutable std::atomic<uint64_t> successful_resolves_;
        mutable std::atomic<uint64_t> unsuccessful_resolves_;
        mutable std::atomic<uint64_t> unsuccessful_env_collision_resolves_;
        mutable std::atomic<uint64_t> unsuccessful_self_collision_resolves_;

        /* Discretize a cuboid obstacle to resolution-sized cells */
        inline std::vector<std::pair<Eigen::Vector3d, sdf_tools::TAGGED_OBJECT_COLLISION_CELL>> DiscretizeObstacle(const OBSTACLE_CONFIG& obstacle, const double resolution) const
        {
            const double effective_resolution = resolution * 0.5;
            std::vector<std::pair<Eigen::Vector3d, sdf_tools::TAGGED_OBJECT_COLLISION_CELL>> cells;
            // Make the cell for the object
            sdf_tools::TAGGED_OBJECT_COLLISION_CELL object_cell(1.0, obstacle.object_id);
            // Generate all cells for the object
            int32_t x_cells = (int32_t)(obstacle.extents.x() * 2.0 * (1.0 / effective_resolution));
            int32_t y_cells = (int32_t)(obstacle.extents.y() * 2.0 * (1.0 / effective_resolution));
            int32_t z_cells = (int32_t)(obstacle.extents.z() * 2.0 * (1.0 / effective_resolution));
            for (int32_t xidx = 0; xidx < x_cells; xidx++)
            {
                for (int32_t yidx = 0; yidx < y_cells; yidx++)
                {
                    for (int32_t zidx = 0; zidx < z_cells; zidx++)
                    {
                        double x_location = -(obstacle.extents.x() - (resolution * 0.5)) + (effective_resolution * xidx);
                        double y_location = -(obstacle.extents.y() - (resolution * 0.5)) + (effective_resolution * yidx);
                        double z_location = -(obstacle.extents.z() - (resolution * 0.5)) + (effective_resolution * zidx);
                        Eigen::Vector3d cell_location(x_location, y_location, z_location);
                        cells.push_back(std::pair<Eigen::Vector3d, sdf_tools::TAGGED_OBJECT_COLLISION_CELL>(cell_location, object_cell));
                    }
                }
            }
            return cells;
        }

        /* Build certain special case environments */
        inline sdf_tools::TaggedObjectCollisionMapGrid BuildEnvironment(const std::string& environment_id, const double resolution) const
        {
            std::cout << "Generating the " << environment_id << " environment" << std::endl;
            if (environment_id == "nested_corners")
            {
                double grid_x_size = 10.0;
                double grid_y_size = 10.0;
                double grid_z_size = 10.0;
                // The grid origin is the minimum point, with identity rotation
                Eigen::Translation3d grid_origin_translation(0.0, 0.0, 0.0);
                Eigen::Quaterniond grid_origin_rotation = Eigen::Quaterniond::Identity();
                Eigen::Affine3d grid_origin_transform = grid_origin_translation * grid_origin_rotation;
                // Make the grid
                sdf_tools::TAGGED_OBJECT_COLLISION_CELL default_cell;
                sdf_tools::TaggedObjectCollisionMapGrid grid(grid_origin_transform, "nomdp_simulator", resolution, grid_x_size, grid_y_size, grid_z_size, default_cell);
                for (int64_t x_idx = 0; x_idx < grid.GetNumXCells(); x_idx++)
                {
                    for (int64_t y_idx = 0; y_idx < grid.GetNumYCells(); y_idx++)
                    {
                        for (int64_t z_idx = 0; z_idx < grid.GetNumZCells(); z_idx++)
                        {
                            const Eigen::Vector3d location = EigenHelpers::StdVectorDoubleToEigenVector3d(grid.GridIndexToLocation(x_idx, y_idx, z_idx));
                            const double& x = location.x();
                            const double& y = location.y();
                            const double& z = location.z();
                            // Set the object we belong to
                            // We assume that all objects are convex, so we can set the convex region as 1
                            // "Bottom bottom"
                            if (x > 1.0 && x <= 9.0 && y > 1.0 && y <= 9.0 && z > 1.0 && z<= 1.5)
                            {
                                const sdf_tools::TAGGED_OBJECT_COLLISION_CELL object_cell(1.0, 1u, 0u, 1u);
                                grid.Set(x_idx, y_idx, z_idx, object_cell);
                            }
                            // "Right right"
                            if (x > 1.0 && x <= 9.0 && y > 1.0 && y <= 1.5 && z > 1.0 && z<= 9.0)
                            {
                                const sdf_tools::TAGGED_OBJECT_COLLISION_CELL object_cell(1.0, 2u, 0u, 1u);
                                grid.Set(x_idx, y_idx, z_idx, object_cell);
                            }
                            // "Back back"
                            if (x > 1.0 && x <= 1.5 && y > 1.0 && y <= 9.0 && z > 1.0 && z<= 9.0)
                            {
                                const sdf_tools::TAGGED_OBJECT_COLLISION_CELL object_cell(1.0, 3u, 0u, 1u);
                                grid.Set(x_idx, y_idx, z_idx, object_cell);
                            }
                            // "Top bottom"
                            if (x > 2.0 && x <= 7.0 && y > 2.0 && y <= 7.0 && z > 2.0 && z<= 2.5)
                            {
                                const sdf_tools::TAGGED_OBJECT_COLLISION_CELL object_cell(1.0, 4u, 0u, 1u);
                                grid.Set(x_idx, y_idx, z_idx, object_cell);
                            }
                            // "Left right"
                            if (x > 2.0 && x <= 7.0 && y > 2.0 && y <= 2.5 && z > 2.0 && z<= 7.0)
                            {
                                const sdf_tools::TAGGED_OBJECT_COLLISION_CELL object_cell(1.0, 5u, 0u, 1u);
                                grid.Set(x_idx, y_idx, z_idx, object_cell);
                            }
                            // "Front back"
                            if (x > 2.0 && x <= 2.5 && y > 2.0 && y <= 7.0 && z > 2.0 && z<= 7.0)
                            {
                                const sdf_tools::TAGGED_OBJECT_COLLISION_CELL object_cell(1.0, 6u, 0u, 1u);
                                grid.Set(x_idx, y_idx, z_idx, object_cell);
                            }
                            // Set the free-space convex segment we belong to (if necessary)
                            if (grid.GetImmutable(x_idx, y_idx, z_idx).first.occupancy < 0.5)
                            {
                                // There are 13 convex regions, we can belong to multiple regions, so we check for each
                                // First three
                                if (x <= 1.0)
                                {
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(1u);
                                }
                                if (y <= 1.0)
                                {
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(2u);
                                }
                                if (z <= 1.0)
                                {
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(3u);
                                }
                                // Second three
                                if ((x > 1.5) && (x <= 2.0) && (y > 1.5) && (z > 1.5))
                                {
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(4u);
                                }
                                if ((x > 1.5) && (y > 1.5) && (y <= 2.0) && (z > 1.5))
                                {
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(5u);
                                }
                                if ((x > 1.5) && (y > 1.5) && (z > 1.5) && (z <= 2.0))
                                {
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(6u);
                                }
                                // Third three
                                if (x > 9.0)
                                {
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(7u);
                                }
                                if (y > 9.0)
                                {
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(8u);
                                }
                                if (z > 9.0)
                                {
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(9u);
                                }
                                // Fourth three
                                if ((x > 7.0) && (y > 1.5) && (z > 1.5))
                                {
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(10u);
                                }
                                if ((x > 1.5) && (y > 7.0) && (z > 1.5))
                                {
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(11u);
                                }
                                if ((x > 1.5) && (y > 1.5) && (z > 7.0))
                                {
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(12u);
                                }
                                // Last one
                                if ((x > 2.5) && (y > 2.5) && (z > 2.5))
                                {
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(13u);
                                }
                            }
                        }
                    }
                }
                return grid;
            }
            else if (environment_id == "peg_in_hole")
            {
                double grid_x_size = 12.0;
                double grid_y_size = 12.0;
                double grid_z_size = 12.0;
                // The grid origin is the minimum point, with identity rotation
                Eigen::Translation3d grid_origin_translation(-1.0, -1.0, -1.0);
                Eigen::Quaterniond grid_origin_rotation = Eigen::Quaterniond::Identity();
                Eigen::Affine3d grid_origin_transform = grid_origin_translation * grid_origin_rotation;
                // Make the grid
                sdf_tools::TAGGED_OBJECT_COLLISION_CELL default_cell(1.0f, 0u, 0u, 0u);
                sdf_tools::TaggedObjectCollisionMapGrid grid(grid_origin_transform, "nomdp_simulator", resolution, grid_x_size, grid_y_size, grid_z_size, default_cell);
                for (int64_t x_idx = 0; x_idx < grid.GetNumXCells(); x_idx++)
                {
                    for (int64_t y_idx = 0; y_idx < grid.GetNumYCells(); y_idx++)
                    {
                        for (int64_t z_idx = 0; z_idx < grid.GetNumZCells(); z_idx++)
                        {
                            const Eigen::Vector3d location = EigenHelpers::StdVectorDoubleToEigenVector3d(grid.GridIndexToLocation(x_idx, y_idx, z_idx));
                            const double& x = location.x();
                            const double& y = location.y();
                            const double& z = location.z();
                            // Set the object we belong to
                            // We assume that all objects are convex, so we can set the convex region as 1
                            // "Bottom bottom"
                            if (x_idx <= 8 || y_idx <= 8 || z_idx <= 8 || x_idx >= (grid.GetNumXCells() - 8)  || y_idx >= (grid.GetNumYCells() - 8) || z_idx >= (grid.GetNumZCells() - 8))
                            {
                                const sdf_tools::TAGGED_OBJECT_COLLISION_CELL buffer_cell(1.0f, 0u, 0u, 0u);
                                grid.Set(x_idx, y_idx, z_idx, buffer_cell);
                            }
                            else
                            {
                                if (z <= 3.0)
                                {
                                    if (x <= 2.0 || y <= 2.0)
                                    {
                                        const sdf_tools::TAGGED_OBJECT_COLLISION_CELL object_cell(1.0, 1u, 0u, 1u);
                                        grid.Set(x_idx, y_idx, z_idx, object_cell);
                                    }
                                    else if (x > 2.5 || y > 2.5)
                                    {
                                        const sdf_tools::TAGGED_OBJECT_COLLISION_CELL object_cell(1.0, 1u, 0u, 1u);
                                        grid.Set(x_idx, y_idx, z_idx, object_cell);
                                    }
                                    else if (z <= resolution)
                                    {
                                        const sdf_tools::TAGGED_OBJECT_COLLISION_CELL object_cell(1.0, 1u, 0u, 1u);
                                        grid.Set(x_idx, y_idx, z_idx, object_cell);
                                    }
                                    else
                                    {
                                        const sdf_tools::TAGGED_OBJECT_COLLISION_CELL free_cell(0.0f, 0u, 0u, 0u);
                                        grid.Set(x_idx, y_idx, z_idx, free_cell);
                                    }
                                }
                                else
                                {
                                    const sdf_tools::TAGGED_OBJECT_COLLISION_CELL free_cell(0.0f, 0u, 0u, 0u);
                                    grid.Set(x_idx, y_idx, z_idx, free_cell);
                                }
                            }
                            // Set the free-space convex segment we belong to (if necessary)
                            if (grid.GetImmutable(x_idx, y_idx, z_idx).first.occupancy < 0.5)
                            {
                                // There are 2 convex regions, we can belong to multiple regions, so we check for each
                                if (z > 3.0)
                                {
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(1u);
                                }
                                if (x > 2.0 && y > 2.0 && x <= 2.5 && y <= 2.5 && z > resolution)
                                {
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(2u);
                                }
                            }
                        }
                    }
                }
                return grid;
            }
            else if (environment_id == "se3_cluttered")
            {
                double grid_x_size = 12.0;
                double grid_y_size = 12.0;
                double grid_z_size = 12.0;
                // The grid origin is the minimum point, with identity rotation
                Eigen::Translation3d grid_origin_translation(-1.0, -1.0, -1.0);
                Eigen::Quaterniond grid_origin_rotation = Eigen::Quaterniond::Identity();
                Eigen::Affine3d grid_origin_transform = grid_origin_translation * grid_origin_rotation;
                // Make the grid
                sdf_tools::TAGGED_OBJECT_COLLISION_CELL default_cell(0.0f, 0u, 0u, 0u);
                sdf_tools::TaggedObjectCollisionMapGrid grid(grid_origin_transform, "nomdp_simulator", resolution, grid_x_size, grid_y_size, grid_z_size, default_cell);
                for (int64_t x_idx = 0; x_idx < grid.GetNumXCells(); x_idx++)
                {
                    for (int64_t y_idx = 0; y_idx < grid.GetNumYCells(); y_idx++)
                    {
                        for (int64_t z_idx = 0; z_idx < grid.GetNumZCells(); z_idx++)
                        {
                            const Eigen::Vector3d location = EigenHelpers::StdVectorDoubleToEigenVector3d(grid.GridIndexToLocation(x_idx, y_idx, z_idx));
                            const double& x = location.x();
                            const double& y = location.y();
                            const double& z = location.z();
                            // Make some of the exterior walls opaque
                            if (x_idx <= 8 || y_idx <= 8 || z_idx <= 8 || x_idx >= (grid.GetNumXCells() - 8) || y_idx >= (grid.GetNumYCells() - 8))
                            {
                                const sdf_tools::TAGGED_OBJECT_COLLISION_CELL buffer_cell(1.0f, 1u, 0u, 0u);
                                grid.Set(x_idx, y_idx, z_idx, buffer_cell);
                            }
                            else if (z_idx >= (grid.GetNumZCells() - 8))
                            {
                                const sdf_tools::TAGGED_OBJECT_COLLISION_CELL buffer_cell(1.0f, 0u, 0u, 0u);
                                grid.Set(x_idx, y_idx, z_idx, buffer_cell);
                            }
                            else
                            {
                                // Set the interior 10x10x10 meter area
                                if (x > 1.0 && x <= 3.0 && y > 1.0 && y <= 3.0)
                                {
                                    const sdf_tools::TAGGED_OBJECT_COLLISION_CELL object_cell(1.0f, 2u, 0u, 0u);
                                    grid.Set(x_idx, y_idx, z_idx, object_cell);
                                }
                                else if (x > 5.0 && x <= 8.0 && y > 0.0 && y <= 2.0)
                                {
                                    const sdf_tools::TAGGED_OBJECT_COLLISION_CELL object_cell(1.0f, 3u, 0u, 0u);
                                    grid.Set(x_idx, y_idx, z_idx, object_cell);
                                }
                                else if (x > 4.0 && x <= 9.0 && y > 3.0 && y <= 5.0)
                                {
                                    const sdf_tools::TAGGED_OBJECT_COLLISION_CELL object_cell(1.0f, 4u, 0u, 0u);
                                    grid.Set(x_idx, y_idx, z_idx, object_cell);
                                }
                                else if (x > 0.0 && x <= 2.0 && y > 4.0 && y <= 7.0)
                                {
                                    const sdf_tools::TAGGED_OBJECT_COLLISION_CELL object_cell(1.0f, 5u, 0u, 0u);
                                    grid.Set(x_idx, y_idx, z_idx, object_cell);
                                }
                                else if (x > 3.0 && x <= 6.0&& y > 6.0 && y <= 8.0)
                                {
                                    const sdf_tools::TAGGED_OBJECT_COLLISION_CELL object_cell(1.0f, 6u, 0u, 0u);
                                    grid.Set(x_idx, y_idx, z_idx, object_cell);
                                }
                                else if (x > 0.0 && x <= 5.0 && y > 9.0 && y <= 10.0)
                                {
                                    const sdf_tools::TAGGED_OBJECT_COLLISION_CELL object_cell(1.0f, 7u, 0u, 0u);
                                    grid.Set(x_idx, y_idx, z_idx, object_cell);
                                }
                                else if (x > 8.0 && x <= 10.0 && y > 6.0 && y <= 10.0)
                                {
                                    const sdf_tools::TAGGED_OBJECT_COLLISION_CELL object_cell(1.0f, 8u, 0u, 0u);
                                    grid.Set(x_idx, y_idx, z_idx, object_cell);
                                }
                                else
                                {
                                    // Set the convex regions
                                    if (x > 0.0 && x <= 5.0 && y > 0.0 && y <= 1.0)
                                    {
                                        grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(1u);
                                    }
                                    if (x > 0.0 && x <= 1.0 && y > 0.0 && y <= 4.0)
                                    {
                                        grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(2u);
                                    }
                                    if (x > 0.0 && x <= 4.0 && y > 3.0 && y <= 4.0)
                                    {
                                        grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(3u);
                                    }
                                    if (x > 3.0 && x <= 4.0 && y > 0.0 && y <= 6.0)
                                    {
                                        grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(4u);
                                    }
                                    if (x > 3.0 && x <= 5.0 && y > 0.0 && y <= 3.0)
                                    {
                                        grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(5u);
                                    }
                                    if (x > 3.0 && x <= 10.0 && y > 2.0 && y <= 3.0)
                                    {
                                        grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(6u);
                                    }
                                    if (x > 8.0 && x <= 10.0 && y > 0.0 && y <= 3.0)
                                    {
                                        grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(7u);
                                    }
                                    if (x > 9.0 && x <= 10.0 && y > 0.0 && y <= 6.0)
                                    {
                                        grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(8u);
                                    }
                                    if (x > 2.0 && x <= 10.0 && y > 5.0 && y <= 6.0)
                                    {
                                        grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(9u);
                                    }
                                    if (x > 2.0 && x <= 3.0 && y > 3.0 && y <= 9.0)
                                    {
                                        grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(10u);
                                    }
                                    if (x > 6.0 && x <= 8.0 && y > 5.0 && y <= 10.0)
                                    {
                                        grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(11u);
                                    }
                                    if (x > 0.0 && x <= 3.0 && y > 7.0 && y <= 9.0)
                                    {
                                        grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(12u);
                                    }
                                    if (x > 5.0 && x <= 8.0 && y > 8.0 && y <= 10.0)
                                    {
                                        grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(13u);
                                    }
                                    if (x > 0.0 && x <= 8.0 && y > 8.0 && y <= 9.0)
                                    {
                                        grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(14u);
                                    }
                                    if (x > 2.0 && x <= 4.0 && y > 3.0 && y <= 6.0)
                                    {
                                        grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(15u);
                                    }
                                }
                            }
                        }
                    }
                }
                return grid;
            }
            else if (environment_id == "se3_box_puzzle")
            {
                double grid_x_size = 12.0;
                double grid_y_size = 12.0;
                double grid_z_size = 12.0;
                // The grid origin is the minimum point, with identity rotation
                Eigen::Translation3d grid_origin_translation(-1.0, -1.0, -1.0);
                Eigen::Quaterniond grid_origin_rotation = Eigen::Quaterniond::Identity();
                Eigen::Affine3d grid_origin_transform = grid_origin_translation * grid_origin_rotation;
                // Make the grid
                sdf_tools::TAGGED_OBJECT_COLLISION_CELL default_cell(0.0f, 0u, 0u, 0u);
                sdf_tools::TaggedObjectCollisionMapGrid grid(grid_origin_transform, "nomdp_simulator", resolution, grid_x_size, grid_y_size, grid_z_size, default_cell);
                for (int64_t x_idx = 0; x_idx < grid.GetNumXCells(); x_idx++)
                {
                    for (int64_t y_idx = 0; y_idx < grid.GetNumYCells(); y_idx++)
                    {
                        for (int64_t z_idx = 0; z_idx < grid.GetNumZCells(); z_idx++)
                        {
                            const Eigen::Vector3d location = EigenHelpers::StdVectorDoubleToEigenVector3d(grid.GridIndexToLocation(x_idx, y_idx, z_idx));
                            const double& x = location.x();
                            const double& y = location.y();
                            const double& z = location.z();
                            // Set the object we belong to
                            // Make some of the exterior walls opaque
                            if (x_idx <= 8 || y_idx <= 8 || z_idx <= 8 || x_idx >= (grid.GetNumXCells() - 8) || y_idx >= (grid.GetNumYCells() - 8) || z_idx >= (grid.GetNumZCells() - 8))
                            {
                                const sdf_tools::TAGGED_OBJECT_COLLISION_CELL buffer_cell(1.0f, 0u, 0u, 0u);
                                grid.Set(x_idx, y_idx, z_idx, buffer_cell);
                            }
                            else
                            {
                                // Make central planes(s)
                                if (x > 4.5 && x <= 5.5)
                                {
                                    const sdf_tools::TAGGED_OBJECT_COLLISION_CELL object_cell(1.0f, 1u, 0u, 0u);
                                    grid.Set(x_idx, y_idx, z_idx, object_cell);
                                }
                                else if (y > 4.5 && y <= 5.5)
                                {
                                    const sdf_tools::TAGGED_OBJECT_COLLISION_CELL object_cell(1.0f, 2u, 0u, 0u);
                                    grid.Set(x_idx, y_idx, z_idx, object_cell);
                                }
                                else if (z > 4.5 && z <= 5.5)
                                {
                                    const sdf_tools::TAGGED_OBJECT_COLLISION_CELL object_cell(1.0f, 3u, 0u, 0u);
                                    grid.Set(x_idx, y_idx, z_idx, object_cell);
                                }
                                // Make the 8 interior voids
                                else if (x <= 4.5 && y <= 4.5 && z <= 4.5)
                                {
                                    const sdf_tools::TAGGED_OBJECT_COLLISION_CELL object_cell(0.0f, 0u, 0u, 1u);
                                    grid.Set(x_idx, y_idx, z_idx, object_cell);
                                }
                                else if (x <= 4.5 && y <= 4.5 && z > 5.5)
                                {
                                    const sdf_tools::TAGGED_OBJECT_COLLISION_CELL object_cell(0.0f, 0u, 0u, 2u);
                                    grid.Set(x_idx, y_idx, z_idx, object_cell);
                                }
                                else if (x <= 4.5 && y > 5.5 && z <= 4.5)
                                {
                                    const sdf_tools::TAGGED_OBJECT_COLLISION_CELL object_cell(0.0f, 0u, 0u, 4u);
                                    grid.Set(x_idx, y_idx, z_idx, object_cell);
                                }
                                else if (x <= 4.5 && y > 5.5 && z > 5.5)
                                {
                                    const sdf_tools::TAGGED_OBJECT_COLLISION_CELL object_cell(0.0f, 0u, 0u, 8u);
                                    grid.Set(x_idx, y_idx, z_idx, object_cell);
                                }
                                else if (x > 5.5 && y <= 4.5 && z <= 4.5)
                                {
                                    const sdf_tools::TAGGED_OBJECT_COLLISION_CELL object_cell(0.0f, 0u, 0u, 16u);
                                    grid.Set(x_idx, y_idx, z_idx, object_cell);
                                }
                                else if (x > 5.5 && y <= 4.5 && z > 5.5)
                                {
                                    const sdf_tools::TAGGED_OBJECT_COLLISION_CELL object_cell(0.0f, 0u, 0u, 32u);
                                    grid.Set(x_idx, y_idx, z_idx, object_cell);
                                }
                                else if (x > 5.5 && y > 5.5 && z <= 4.5)
                                {
                                    const sdf_tools::TAGGED_OBJECT_COLLISION_CELL object_cell(0.0f, 0u, 0u, 64u);
                                    grid.Set(x_idx, y_idx, z_idx, object_cell);
                                }
                                else if (x > 5.5 && y > 5.5 && z > 5.5)
                                {
                                    const sdf_tools::TAGGED_OBJECT_COLLISION_CELL object_cell(0.0f, 0u, 0u, 128u);
                                    grid.Set(x_idx, y_idx, z_idx, object_cell);
                                }
                                // Make the four x-axis oriented interior channels
                                if (y > 5.5 && y <= 7.0 && z > 5.5 && z <= 7.0)
                                {
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.occupancy = 0.0f;
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.object_id = 0u;
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.component = 0u;
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(9u);
                                }
                                else if (y > 3.0 && y <= 4.5 && z > 5.5 && z <= 7.0)
                                {
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.occupancy = 0.0f;
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.object_id = 0u;
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.component = 0u;
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(10u);
                                }
                                else if (y > 5.5 && y <= 7.0 && z > 3.0 && z <= 4.5)
                                {
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.occupancy = 0.0f;
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.object_id = 0u;
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.component = 0u;
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(11u);
                                }
                                else if (y > 3.0 && y <= 4.5 && z > 3.0 && z <= 4.5)
                                {
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.occupancy = 0.0f;
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.object_id = 0u;
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.component = 0u;
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(12u);
                                }
                                // Make the two z-axis oriented interior channels
                                if (y > 5.5 && y <= 7.0 && x > 5.5 && x <= 7.0)
                                {
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.occupancy = 0.0f;
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.object_id = 0u;
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.component = 0u;
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(13u);
                                }
                                else if (y > 3.0 && y <= 4.5 && x > 5.5 && x <= 7.0)
                                {
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.occupancy = 0.0f;
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.object_id = 0u;
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.component = 0u;
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(14u);
                                }
                                // Add the 1 y-axis oriented interior channel
                                if (x > 3.0 && x <= 4.5 && z > 3.0 && z <= 4.5)
                                {
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.occupancy = 0.0f;
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.object_id = 0u;
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.component = 0u;
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(15u);
                                }
                            }
                        }
                    }
                }
                return grid;
            }
            else if (environment_id == "baxter_env")
            {
                double grid_x_size = 2.0;
                double grid_y_size = 2.0;
                double grid_z_size = 2.0;
                // The grid origin is the minimum point, with identity rotation
                Eigen::Translation3d grid_origin_translation(-0.5, -1.5, -0.5);
                Eigen::Quaterniond grid_origin_rotation = Eigen::Quaterniond::Identity();
                Eigen::Affine3d grid_origin_transform = grid_origin_translation * grid_origin_rotation;
                // Make the grid
                sdf_tools::TAGGED_OBJECT_COLLISION_CELL default_cell;
                sdf_tools::TaggedObjectCollisionMapGrid grid(grid_origin_transform, "nomdp_simulator", resolution, grid_x_size, grid_y_size, grid_z_size, default_cell);
                for (int64_t x_idx = 0; x_idx < grid.GetNumXCells(); x_idx++)
                {
                    for (int64_t y_idx = 0; y_idx < grid.GetNumYCells(); y_idx++)
                    {
                        for (int64_t z_idx = 0; z_idx < grid.GetNumZCells(); z_idx++)
                        {
                            const Eigen::Vector3d location = EigenHelpers::StdVectorDoubleToEigenVector3d(grid.GridIndexToLocation(x_idx, y_idx, z_idx));
                            const double& x = location.x();
                            const double& y = location.y();
                            const double& z = location.z();
                            // Buffer
                            if (x_idx <= 8 || y_idx <= 8 || z_idx <= 8 || x_idx >= (grid.GetNumXCells() - 8)  || y_idx >= (grid.GetNumYCells() - 8) || z_idx >= (grid.GetNumZCells() - 8))
                            {
                                const sdf_tools::TAGGED_OBJECT_COLLISION_CELL buffer_cell(1.0f, 0u, 0u, 0u);
                                grid.Set(x_idx, y_idx, z_idx, buffer_cell);
                            }
                            // Set the object we belong to
                            if (z_idx < 10)
                            {
                                const sdf_tools::TAGGED_OBJECT_COLLISION_CELL object_cell(1.0, 1u, 0u, 1u);
                                grid.Set(x_idx, y_idx, z_idx, object_cell);
                            }
                            // Set the free-space convex segment we belong to (if necessary)
                            else if (grid.GetImmutable(x_idx, y_idx, z_idx).first.occupancy < 0.5)
                            {
                                const sdf_tools::TAGGED_OBJECT_COLLISION_CELL free_cell(0.0f, 0u, 0u, 1u);
                                grid.Set(x_idx, y_idx, z_idx, free_cell);
                            }
                        }
                    }
                }
                return grid;
            }
            else if (environment_id == "blocked_peg_in_hole")
            {
                double grid_x_size = 10.0;
                double grid_y_size = 10.0;
                double grid_z_size = 10.0;
                // The grid origin is the minimum point, with identity rotation
                Eigen::Translation3d grid_origin_translation(0.0, 0.0, 0.0);
                Eigen::Quaterniond grid_origin_rotation = Eigen::Quaterniond::Identity();
                Eigen::Affine3d grid_origin_transform = grid_origin_translation * grid_origin_rotation;
                // Make the grid
                sdf_tools::TAGGED_OBJECT_COLLISION_CELL default_cell;
                sdf_tools::TaggedObjectCollisionMapGrid grid(grid_origin_transform, "nomdp_simulator", resolution, grid_x_size, grid_y_size, grid_z_size, default_cell);
                for (int64_t x_idx = 0; x_idx < grid.GetNumXCells(); x_idx++)
                {
                    for (int64_t y_idx = 0; y_idx < grid.GetNumYCells(); y_idx++)
                    {
                        for (int64_t z_idx = 0; z_idx < grid.GetNumZCells(); z_idx++)
                        {
                            const Eigen::Vector3d location = EigenHelpers::StdVectorDoubleToEigenVector3d(grid.GridIndexToLocation(x_idx, y_idx, z_idx));
                            const double& x = location.x();
                            const double& y = location.y();
                            const double& z = location.z();
                            // Set the object we belong to
                            // We assume that all objects are convex, so we can set the convex region as 1
                            // "Bottom bottom"
                            if (z <= 3.0)
                            {
                                if (x <= 2.0 || y <= 2.0)
                                {
                                    const sdf_tools::TAGGED_OBJECT_COLLISION_CELL object_cell(1.0, 1u, 0u, 1u);
                                    grid.Set(x_idx, y_idx, z_idx, object_cell);
                                }
                                else if (x > 2.5 || y > 2.5)
                                {
                                    const sdf_tools::TAGGED_OBJECT_COLLISION_CELL object_cell(1.0, 1u, 0u, 1u);
                                    grid.Set(x_idx, y_idx, z_idx, object_cell);
                                }
                                else if (z <= resolution)
                                {
                                    const sdf_tools::TAGGED_OBJECT_COLLISION_CELL object_cell(1.0, 1u, 0u, 1u);
                                    grid.Set(x_idx, y_idx, z_idx, object_cell);
                                }
                            }
                            else if (z <= 4.5)
                            {
                                if (y > 1.5 && y <= 5.0 && x > 3.0 && x <= 3.5)
                                {
                                    const sdf_tools::TAGGED_OBJECT_COLLISION_CELL object_cell(1.0, 1u, 0u, 1u);
                                    grid.Set(x_idx, y_idx, z_idx, object_cell);
                                }
                            }
                            // Set the free-space convex segment we belong to (if necessary)
                            if (grid.GetImmutable(x_idx, y_idx, z_idx).first.occupancy < 0.5)
                            {
                                // There are 2 convex regions, we can belong to multiple regions, so we check for each
                                if (z > 3.0)
                                {
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(1u);
                                }
                                if (x > 2.0 && y > 2.0 && x <= 2.5 && y <= 2.5 && z > resolution)
                                {
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(2u);
                                }

                            }
                        }
                    }
                }
                return grid;
            }
            else if (environment_id == "inset_peg_in_hole")
            {
                double grid_x_size = 10.0;
                double grid_y_size = 10.0;
                double grid_z_size = 10.0;
                // The grid origin is the minimum point, with identity rotation
                Eigen::Translation3d grid_origin_translation(0.0, 0.0, 0.0);
                Eigen::Quaterniond grid_origin_rotation = Eigen::Quaterniond::Identity();
                Eigen::Affine3d grid_origin_transform = grid_origin_translation * grid_origin_rotation;
                // Make the grid
                sdf_tools::TAGGED_OBJECT_COLLISION_CELL default_cell;
                sdf_tools::TaggedObjectCollisionMapGrid grid(grid_origin_transform, "nomdp_simulator", resolution, grid_x_size, grid_y_size, grid_z_size, default_cell);
                for (int64_t x_idx = 0; x_idx < grid.GetNumXCells(); x_idx++)
                {
                    for (int64_t y_idx = 0; y_idx < grid.GetNumYCells(); y_idx++)
                    {
                        for (int64_t z_idx = 0; z_idx < grid.GetNumZCells(); z_idx++)
                        {
                            const Eigen::Vector3d location = EigenHelpers::StdVectorDoubleToEigenVector3d(grid.GridIndexToLocation(x_idx, y_idx, z_idx));
                            const double& x = location.x();
                            const double& y = location.y();
                            const double& z = location.z();
                            // Set the object we belong to
                            // We assume that all objects are convex, so we can set the convex region as 1
                            // "Bottom bottom"
                            if (z <= 2.0)
                            {
                                if (x <= 2.0 || y <= 2.0)
                                {
                                    const sdf_tools::TAGGED_OBJECT_COLLISION_CELL object_cell(1.0, 1u, 0u, 1u);
                                    grid.Set(x_idx, y_idx, z_idx, object_cell);
                                }
                                else if ((x > 2.5 || y > 2.5) && (z <= 1.0))
                                {
                                    const sdf_tools::TAGGED_OBJECT_COLLISION_CELL object_cell(1.0, 1u, 0u, 1u);
                                    grid.Set(x_idx, y_idx, z_idx, object_cell);
                                }
                                else if (z <= resolution)
                                {
                                    const sdf_tools::TAGGED_OBJECT_COLLISION_CELL object_cell(1.0, 1u, 0u, 1u);
                                    grid.Set(x_idx, y_idx, z_idx, object_cell);
                                }
                            }
                            // Set the free-space convex segment we belong to (if necessary)
                            if (grid.GetImmutable(x_idx, y_idx, z_idx).first.occupancy < 0.5)
                            {
                                // There are 2 convex regions, we can belong to multiple regions, so we check for each
                                if (z > 2.0)
                                {
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(1u);
                                }
                                if (x > 2.0 && y > 2.0 && x <= 2.5 && y <= 2.5 && z > resolution)
                                {
                                    grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(2u);
                                }

                            }
                        }
                    }
                }
                return grid;
            }
            else if (environment_id == "se2_maze")
            {
                double grid_x_size = 10.0;
                double grid_y_size = 10.0;
                double grid_z_size = 1.0;
                // The grid origin is the minimum point, with identity rotation
                Eigen::Translation3d grid_origin_translation(0.0, 0.0, -0.5);
                Eigen::Quaterniond grid_origin_rotation = Eigen::Quaterniond::Identity();
                Eigen::Affine3d grid_origin_transform = grid_origin_translation * grid_origin_rotation;
                // Make the grid
                sdf_tools::TAGGED_OBJECT_COLLISION_CELL default_cell(1.0, 1u, 0u, 1u); // Everything is filled by default
                sdf_tools::TaggedObjectCollisionMapGrid grid(grid_origin_transform, "nomdp_simulator", resolution, grid_x_size, grid_y_size, grid_z_size, default_cell);
                for (int64_t x_idx = 0; x_idx < grid.GetNumXCells(); x_idx++)
                {
                    for (int64_t y_idx = 0; y_idx < grid.GetNumYCells(); y_idx++)
                    {
                        for (int64_t z_idx = 0; z_idx < grid.GetNumZCells(); z_idx++)
                        {
                            const Eigen::Vector3d location = EigenHelpers::StdVectorDoubleToEigenVector3d(grid.GridIndexToLocation(x_idx, y_idx, z_idx));
                            const double& x = location.x();
                            const double& y = location.y();
                            //const double& z = location.z();
                            // Check if the cell belongs to any of the regions of freespace
                            if (x > 0.5 && y > 0.5 && x <= 3.0 && y <= 9.5)
                            {
                                grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(1u);
                                grid.GetMutable(x_idx, y_idx, z_idx).first.occupancy = 0.0f;
                                grid.GetMutable(x_idx, y_idx, z_idx).first.object_id = 0u;
                            }
                            if (x > 3.5 && y > 0.5 && x <= 6.5 && y <= 9.5)
                            {
                                grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(2u);
                                grid.GetMutable(x_idx, y_idx, z_idx).first.occupancy = 0.0f;
                                grid.GetMutable(x_idx, y_idx, z_idx).first.object_id = 0u;
                            }
                            if (x > 7.0 && y > 0.5 && x <= 9.5 && y <= 9.5)
                            {
                                grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(3u);
                                grid.GetMutable(x_idx, y_idx, z_idx).first.occupancy = 0.0f;
                                grid.GetMutable(x_idx, y_idx, z_idx).first.object_id = 0u;
                            }
                            if (x > 0.5 && y > 7.0 && x <= 9.5 && y <= 9.0)
                            {
                                grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(4u);
                                grid.GetMutable(x_idx, y_idx, z_idx).first.occupancy = 0.0f;
                                grid.GetMutable(x_idx, y_idx, z_idx).first.object_id = 0u;
                            }
                            if (x > 0.5 && y > 1.0 && x <= 9.5 && y <= 3.0)
                            {
                                grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(5u);
                                grid.GetMutable(x_idx, y_idx, z_idx).first.occupancy = 0.0f;
                                grid.GetMutable(x_idx, y_idx, z_idx).first.object_id = 0u;
                            }
                        }
                    }
                }
                return grid;
            }
            else if (environment_id == "se2_maze_blocked")
            {
                double grid_x_size = 10.0;
                double grid_y_size = 10.0;
                double grid_z_size = 1.0;
                // The grid origin is the minimum point, with identity rotation
                Eigen::Translation3d grid_origin_translation(0.0, 0.0, -0.5);
                Eigen::Quaterniond grid_origin_rotation = Eigen::Quaterniond::Identity();
                Eigen::Affine3d grid_origin_transform = grid_origin_translation * grid_origin_rotation;
                // Make the grid
                sdf_tools::TAGGED_OBJECT_COLLISION_CELL default_cell(1.0, 1u, 0u, 1u); // Everything is filled by default
                sdf_tools::TaggedObjectCollisionMapGrid grid(grid_origin_transform, "nomdp_simulator", resolution, grid_x_size, grid_y_size, grid_z_size, default_cell);
                for (int64_t x_idx = 0; x_idx < grid.GetNumXCells(); x_idx++)
                {
                    for (int64_t y_idx = 0; y_idx < grid.GetNumYCells(); y_idx++)
                    {
                        for (int64_t z_idx = 0; z_idx < grid.GetNumZCells(); z_idx++)
                        {
                            const Eigen::Vector3d location = EigenHelpers::StdVectorDoubleToEigenVector3d(grid.GridIndexToLocation(x_idx, y_idx, z_idx));
                            const double& x = location.x();
                            const double& y = location.y();
                            //const double& z = location.z();
                            // Check if the cell belongs to any of the regions of freespace
                            if (x > 0.5 && y > 0.5 && x <= 3.0 && y <= 9.5)
                            {
                                grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(1u);
                                grid.GetMutable(x_idx, y_idx, z_idx).first.occupancy = 0.0f;
                                grid.GetMutable(x_idx, y_idx, z_idx).first.object_id = 0u;
                            }
                            if (x > 3.5 && y > 0.5 && x <= 6.5 && y <= 9.5)
                            {
                                grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(2u);
                                grid.GetMutable(x_idx, y_idx, z_idx).first.occupancy = 0.0f;
                                grid.GetMutable(x_idx, y_idx, z_idx).first.object_id = 0u;
                            }
                            if (x > 7.0 && y > 0.5 && x <= 9.5 && y <= 9.5)
                            {
                                grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(3u);
                                grid.GetMutable(x_idx, y_idx, z_idx).first.occupancy = 0.0f;
                                grid.GetMutable(x_idx, y_idx, z_idx).first.object_id = 0u;
                            }
                            if (x > 0.5 && y > 7.0 && x <= 9.5 && y <= 9.0)
                            {
                                grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(4u);
                                grid.GetMutable(x_idx, y_idx, z_idx).first.occupancy = 0.0f;
                                grid.GetMutable(x_idx, y_idx, z_idx).first.object_id = 0u;
                            }
                            if (x > 0.5 && y > 1.0 && x <= 9.5 && y <= 3.0)
                            {
                                grid.GetMutable(x_idx, y_idx, z_idx).first.AddToConvexSegment(5u);
                                grid.GetMutable(x_idx, y_idx, z_idx).first.occupancy = 0.0f;
                                grid.GetMutable(x_idx, y_idx, z_idx).first.object_id = 0u;
                            }
                            // Re-add the obstacle in the middle
                            if (x > 3.5 && y > 4.75 && x <= 6.5 && y <= 5.25)
                            {
                                grid.GetMutable(x_idx, y_idx, z_idx).first.occupancy = 1.0f;
                                grid.GetMutable(x_idx, y_idx, z_idx).first.object_id = 2u;
                            }
                        }
                    }
                }
                return grid;
            }
            else if (environment_id == "se2_cluttered")
            {
                throw std::invalid_argument("Not implemented yet");
            }
            else if (environment_id == "noisy_arm_cluttered")
            {
                throw std::invalid_argument("Not implemented yet");
            }
            else
            {
                throw std::invalid_argument("Unrecognized environment ID");
            }
        }

        /* Build a new environment from the provided obstacles */
        inline sdf_tools::TaggedObjectCollisionMapGrid BuildEnvironment(const std::vector<OBSTACLE_CONFIG>& obstacles, const double resolution) const
        {
            if (obstacles.empty())
            {
                std::cerr << "No obstacles provided, generating the default environment" << std::endl;
                double grid_x_size = 10.0;
                double grid_y_size = 10.0;
                double grid_z_size = 10.0;
                // The grid origin is the minimum point, with identity rotation
                Eigen::Translation3d grid_origin_translation(0.0, 0.0, 0.0);
                Eigen::Quaterniond grid_origin_rotation = Eigen::Quaterniond::Identity();
                Eigen::Affine3d grid_origin_transform = grid_origin_translation * grid_origin_rotation;
                // Make the grid
                sdf_tools::TAGGED_OBJECT_COLLISION_CELL default_cell;
                sdf_tools::TaggedObjectCollisionMapGrid grid(grid_origin_transform, "nomdp_simulator", resolution, grid_x_size, grid_y_size, grid_z_size, default_cell);
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
                std::vector<std::pair<Eigen::Vector3d, sdf_tools::TAGGED_OBJECT_COLLISION_CELL>> all_obstacle_cells;
                for (size_t idx = 0; idx < obstacles.size(); idx++)
                {
                    const OBSTACLE_CONFIG& obstacle = obstacles[idx];
                    std::vector<std::pair<Eigen::Vector3d, sdf_tools::TAGGED_OBJECT_COLLISION_CELL>> obstacle_cells = DiscretizeObstacle(obstacle, resolution);
                    for (size_t cidx = 0; cidx < obstacle_cells.size(); cidx++)
                    {
                        const Eigen::Vector3d& relative_location = obstacle_cells[cidx].first;
                        Eigen::Vector3d real_location = obstacle.pose * relative_location;
                        all_obstacle_cells.push_back(std::pair<Eigen::Vector3d, sdf_tools::TAGGED_OBJECT_COLLISION_CELL>(real_location, obstacle_cells[cidx].second));
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
                x_min -= (resolution * 3.0);
                y_min -= (resolution * 3.0);
                z_min -= (resolution * 3.0);
                x_max += (resolution * 3.0);
                y_max += (resolution * 3.0);
                z_max += (resolution * 3.0);
                double grid_x_size = x_max - x_min;
                double grid_y_size = y_max - y_min;
                double grid_z_size = z_max - z_min;
                // The grid origin is the minimum point, with identity rotation
                Eigen::Translation3d grid_origin_translation(x_min, y_min, z_min);
                Eigen::Quaterniond grid_origin_rotation = Eigen::Quaterniond::Identity();
                Eigen::Affine3d grid_origin_transform = grid_origin_translation * grid_origin_rotation;
                // Make the grid
                sdf_tools::TAGGED_OBJECT_COLLISION_CELL default_cell;
                sdf_tools::TaggedObjectCollisionMapGrid grid(grid_origin_transform, "nomdp_simulator", resolution, grid_x_size, grid_y_size, grid_z_size, default_cell);
                // Fill it in
                for (size_t idx = 0; idx < all_obstacle_cells.size(); idx++)
                {
                    const Eigen::Vector3d& location = all_obstacle_cells[idx].first;
                    const sdf_tools::TAGGED_OBJECT_COLLISION_CELL& cell = all_obstacle_cells[idx].second;
                    grid.Set(location.x(), location.y(), location.z(), cell);
                }
                // Set the environment
                return grid;
            }
        }

        inline void UpdateSurfaceNormalGridCell(const std::vector<RawCellSurfaceNormal>& raw_surface_normals, const Eigen::Affine3d& transform, const Eigen::Vector3d& cell_location, const sdf_tools::SignedDistanceField& environment_sdf, SurfaceNormalGrid& surface_normals_grid) const
        {
            const Eigen::Vector3d world_location = transform * cell_location;
            // Let's check the penetration distance. We only want to update cells that are *actually* on the surface
            const float distance = environment_sdf.Get(world_location);
            // If we're within one cell of the surface, we update
            if (distance > -(environment_sdf.GetResolution() * 1.5))
            {
                // First, we clear any stored surface normals
                surface_normals_grid.ClearStoredSurfaceNormals(world_location);
                for (size_t idx = 0; idx < raw_surface_normals.size(); idx++)
                {
                    const RawCellSurfaceNormal& current_surface_normal = raw_surface_normals[idx];
                    const Eigen::Vector3d& raw_surface_normal = current_surface_normal.normal;
                    const Eigen::Vector3d& raw_entry_direction = current_surface_normal.entry_direction;
                    const Eigen::Vector3d real_surface_normal = (Eigen::Vector3d)(transform.rotation() * raw_surface_normal);
                    const Eigen::Vector3d real_entry_direction = (Eigen::Vector3d)(transform.rotation() * raw_entry_direction);
                    surface_normals_grid.InsertSurfaceNormal(world_location, real_surface_normal, real_entry_direction);
                }
            }
            else
            {
                // Do nothing otherwise
                ;
            }
        }

        inline SurfaceNormalGrid BuildSurfaceNormalsGrid(const std::string& environment_id, const sdf_tools::SignedDistanceField& environment_sdf) const
        {
            // Make the grid
            SurfaceNormalGrid surface_normals_grid(environment_sdf.GetOriginTransform(), environment_sdf.GetResolution(), environment_sdf.GetXSize(), environment_sdf.GetYSize(), environment_sdf.GetZSize());
            // The naive start is to fill the surface normals grid with the gradient values from the SDF
            for (int64_t x_idx = 0; x_idx < environment_sdf.GetNumXCells(); x_idx++)
            {
                for (int64_t y_idx = 0; y_idx < environment_sdf.GetNumYCells(); y_idx++)
                {
                    for (int64_t z_idx = 0; z_idx < environment_sdf.GetNumZCells(); z_idx++)
                    {
                        const float distance = environment_sdf.Get(x_idx, y_idx, z_idx);
                        if (distance < 0.0)
                        {
                            const Eigen::Vector3d gradient = EigenHelpers::StdVectorDoubleToEigenVector3d(environment_sdf.GetGradient(x_idx, y_idx, z_idx, true));
                            surface_normals_grid.InsertSurfaceNormal(x_idx, y_idx, z_idx, gradient, Eigen::Vector3d(0.0, 0.0, 0.0));
                        }
                    }
                }
            }
            // Now, as a second pass, we add environment-specific true surface normal(s)
            if (environment_id == "nested_corners")
            {
                ;
            }
            else if (environment_id == "peg_in_hole")
            {
                ;
            }
            else if (environment_id == "se3_cluttered")
            {
                ;
            }
            else if (environment_id == "se3_box_puzzle")
            {
                ;
            }
            else if (environment_id == "baxter_env")
            {
                ;
            }
            else if (environment_id == "blocked_peg_in_hole")
            {
                ;
            }
            else if (environment_id == "inset_peg_in_hole")
            {
                ;
            }
            else if (environment_id == "se3_cluttered")
            {
                ;
            }
            else if (environment_id == "se2_maze")
            {
                ;
            }
            else if (environment_id == "se2_maze_blocked")
            {
                ;
            }
            else if (environment_id == "noisy_arm_cluttered")
            {
                ;
            }
            else
            {
                throw std::invalid_argument("Unrecognized environment ID");
            }
            return surface_normals_grid;
        }

        inline SurfaceNormalGrid BuildSurfaceNormalsGrid(const std::vector<OBSTACLE_CONFIG>& obstacles, const sdf_tools::SignedDistanceField& environment_sdf) const
        {
            // Make the grid
            SurfaceNormalGrid surface_normals_grid(environment_sdf.GetOriginTransform(), environment_sdf.GetResolution(), environment_sdf.GetXSize(), environment_sdf.GetYSize(), environment_sdf.GetZSize());
            // The naive start is to fill the surface normals grid with the gradient values from the SDF
            for (int64_t x_idx = 0; x_idx < environment_sdf.GetNumXCells(); x_idx++)
            {
                for (int64_t y_idx = 0; y_idx < environment_sdf.GetNumYCells(); y_idx++)
                {
                    for (int64_t z_idx = 0; z_idx < environment_sdf.GetNumZCells(); z_idx++)
                    {
                        const float distance = environment_sdf.Get(x_idx, y_idx, z_idx);
                        if (distance < 0.0)
                        {
                            const Eigen::Vector3d gradient = EigenHelpers::StdVectorDoubleToEigenVector3d(environment_sdf.GetGradient(x_idx, y_idx, z_idx, true));
                            surface_normals_grid.InsertSurfaceNormal(x_idx, y_idx, z_idx, gradient, Eigen::Vector3d(0.0, 0.0, 0.0));
                        }
                    }
                }
            }
            // Now, as a second pass, we go through the objects and compute the true surface normal(s) for every object
            for (size_t idx = 0; idx < obstacles.size(); idx++)
            {
                const OBSTACLE_CONFIG& current_obstacle = obstacles[idx];
                const double effective_resolution = environment_sdf.GetResolution() * 0.5;
                // Generate all cells for the object
                int32_t x_cells = (int32_t)(current_obstacle.extents.x() * 2.0 * (1.0 / effective_resolution));
                int32_t y_cells = (int32_t)(current_obstacle.extents.y() * 2.0 * (1.0 / effective_resolution));
                int32_t z_cells = (int32_t)(current_obstacle.extents.z() * 2.0 * (1.0 / effective_resolution));
                for (int32_t xidx = 0; xidx < x_cells; xidx++)
                {
                    for (int32_t yidx = 0; yidx < y_cells; yidx++)
                    {
                        for (int32_t zidx = 0; zidx < z_cells; zidx++)
                        {
                            // If we're on the edge of the obstacle
                            if ((xidx == 0) || (yidx == 0) || (zidx == 0) || (xidx == (x_cells - 1)) || (yidx == (y_cells - 1)) || (zidx == (z_cells - 1)))
                            {
                                double x_location = -(current_obstacle.extents.x() - effective_resolution) + (effective_resolution * xidx);
                                double y_location = -(current_obstacle.extents.y() - effective_resolution) + (effective_resolution * yidx);
                                double z_location = -(current_obstacle.extents.z() - effective_resolution) + (effective_resolution * zidx);
                                const Eigen::Vector3d local_cell_location(x_location, y_location, z_location);
                                // Go through all 26 cases
                                // Start with the 8 corners
                                if ((xidx == 0) && (yidx == 0) && (zidx == 0))
                                {
                                    const RawCellSurfaceNormal normal1(-Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitX());
                                    const RawCellSurfaceNormal normal2(-Eigen::Vector3d::UnitY(), Eigen::Vector3d::UnitY());
                                    const RawCellSurfaceNormal normal3(-Eigen::Vector3d::UnitZ(), Eigen::Vector3d::UnitZ());
                                    UpdateSurfaceNormalGridCell(std::vector<RawCellSurfaceNormal>{normal1, normal2, normal3}, current_obstacle.pose, local_cell_location, environment_sdf, surface_normals_grid);
                                }
                                else if ((xidx == 0) && (yidx == 0) && (zidx == (z_cells - 1)))
                                {
                                    const RawCellSurfaceNormal normal1(-Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitX());
                                    const RawCellSurfaceNormal normal2(-Eigen::Vector3d::UnitY(), Eigen::Vector3d::UnitY());
                                    const RawCellSurfaceNormal normal3(Eigen::Vector3d::UnitZ(), -Eigen::Vector3d::UnitZ());
                                    UpdateSurfaceNormalGridCell(std::vector<RawCellSurfaceNormal>{normal1, normal2, normal3}, current_obstacle.pose, local_cell_location, environment_sdf, surface_normals_grid);
                                }
                                else if ((xidx == 0) && (yidx == (y_cells - 1)) && (zidx == 0))
                                {
                                    const RawCellSurfaceNormal normal1(-Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitX());
                                    const RawCellSurfaceNormal normal2(Eigen::Vector3d::UnitY(), -Eigen::Vector3d::UnitY());
                                    const RawCellSurfaceNormal normal3(-Eigen::Vector3d::UnitZ(), Eigen::Vector3d::UnitZ());
                                    UpdateSurfaceNormalGridCell(std::vector<RawCellSurfaceNormal>{normal1, normal2, normal3}, current_obstacle.pose, local_cell_location, environment_sdf, surface_normals_grid);
                                }
                                else if ((xidx == 0) && (yidx == (y_cells - 1)) && (zidx == (z_cells - 1)))
                                {
                                    const RawCellSurfaceNormal normal1(-Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitX());
                                    const RawCellSurfaceNormal normal2(Eigen::Vector3d::UnitY(), -Eigen::Vector3d::UnitY());
                                    const RawCellSurfaceNormal normal3(Eigen::Vector3d::UnitZ(), -Eigen::Vector3d::UnitZ());
                                    UpdateSurfaceNormalGridCell(std::vector<RawCellSurfaceNormal>{normal1, normal2, normal3}, current_obstacle.pose, local_cell_location, environment_sdf, surface_normals_grid);
                                }
                                else if ((xidx == (x_cells - 1)) && (yidx == 0) && (zidx == 0))
                                {
                                    const RawCellSurfaceNormal normal1(Eigen::Vector3d::UnitX(), -Eigen::Vector3d::UnitX());
                                    const RawCellSurfaceNormal normal2(-Eigen::Vector3d::UnitY(), Eigen::Vector3d::UnitY());
                                    const RawCellSurfaceNormal normal3(-Eigen::Vector3d::UnitZ(), Eigen::Vector3d::UnitZ());
                                    UpdateSurfaceNormalGridCell(std::vector<RawCellSurfaceNormal>{normal1, normal2, normal3}, current_obstacle.pose, local_cell_location, environment_sdf, surface_normals_grid);
                                }
                                else if ((xidx == (x_cells - 1)) && (yidx == 0) && (zidx == (z_cells - 1)))
                                {
                                    const RawCellSurfaceNormal normal1(Eigen::Vector3d::UnitX(), -Eigen::Vector3d::UnitX());
                                    const RawCellSurfaceNormal normal2(-Eigen::Vector3d::UnitY(), Eigen::Vector3d::UnitY());
                                    const RawCellSurfaceNormal normal3(Eigen::Vector3d::UnitZ(), -Eigen::Vector3d::UnitZ());
                                    UpdateSurfaceNormalGridCell(std::vector<RawCellSurfaceNormal>{normal1, normal2, normal3}, current_obstacle.pose, local_cell_location, environment_sdf, surface_normals_grid);
                                }
                                else if ((xidx == (x_cells - 1)) && (yidx == (y_cells - 1)) && (zidx == 0))
                                {
                                    const RawCellSurfaceNormal normal1(Eigen::Vector3d::UnitX(), -Eigen::Vector3d::UnitX());
                                    const RawCellSurfaceNormal normal2(Eigen::Vector3d::UnitY(), -Eigen::Vector3d::UnitY());
                                    const RawCellSurfaceNormal normal3(-Eigen::Vector3d::UnitZ(), Eigen::Vector3d::UnitZ());
                                    UpdateSurfaceNormalGridCell(std::vector<RawCellSurfaceNormal>{normal1, normal2, normal3}, current_obstacle.pose, local_cell_location, environment_sdf, surface_normals_grid);
                                }
                                else if ((xidx == (x_cells - 1)) && (yidx == (y_cells - 1)) && (zidx == (z_cells - 1)))
                                {
                                    const RawCellSurfaceNormal normal1(Eigen::Vector3d::UnitX(), -Eigen::Vector3d::UnitX());
                                    const RawCellSurfaceNormal normal2(Eigen::Vector3d::UnitY(), -Eigen::Vector3d::UnitY());
                                    const RawCellSurfaceNormal normal3(Eigen::Vector3d::UnitZ(), -Eigen::Vector3d::UnitZ());
                                    UpdateSurfaceNormalGridCell(std::vector<RawCellSurfaceNormal>{normal1, normal2, normal3}, current_obstacle.pose, local_cell_location, environment_sdf, surface_normals_grid);
                                }
                                // Next, let's cover the 12 edges
                                else if ((xidx == 0) && (yidx == 0))
                                {
                                    const RawCellSurfaceNormal normal1(-Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitX());
                                    const RawCellSurfaceNormal normal2(-Eigen::Vector3d::UnitY(), Eigen::Vector3d::UnitY());
                                    UpdateSurfaceNormalGridCell(std::vector<RawCellSurfaceNormal>{normal1, normal2}, current_obstacle.pose, local_cell_location, environment_sdf, surface_normals_grid);
                                }
                                else if ((xidx == 0) && (yidx == (y_cells - 1)))
                                {
                                    const RawCellSurfaceNormal normal1(-Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitX());
                                    const RawCellSurfaceNormal normal2(Eigen::Vector3d::UnitY(), -Eigen::Vector3d::UnitY());
                                    UpdateSurfaceNormalGridCell(std::vector<RawCellSurfaceNormal>{normal1, normal2}, current_obstacle.pose, local_cell_location, environment_sdf, surface_normals_grid);
                                }
                                else if ((xidx == (x_cells - 1)) && (yidx == 0))
                                {
                                    const RawCellSurfaceNormal normal1(Eigen::Vector3d::UnitX(), -Eigen::Vector3d::UnitX());
                                    const RawCellSurfaceNormal normal2(-Eigen::Vector3d::UnitY(), Eigen::Vector3d::UnitY());
                                    UpdateSurfaceNormalGridCell(std::vector<RawCellSurfaceNormal>{normal1, normal2}, current_obstacle.pose, local_cell_location, environment_sdf, surface_normals_grid);
                                }
                                else if ((xidx == (x_cells - 1)) && (yidx == (y_cells - 1)))
                                {
                                    const RawCellSurfaceNormal normal1(Eigen::Vector3d::UnitX(), -Eigen::Vector3d::UnitX());
                                    const RawCellSurfaceNormal normal2(Eigen::Vector3d::UnitY(), -Eigen::Vector3d::UnitY());
                                    UpdateSurfaceNormalGridCell(std::vector<RawCellSurfaceNormal>{normal1, normal2}, current_obstacle.pose, local_cell_location, environment_sdf, surface_normals_grid);
                                }
                                else if ((xidx == 0) && (zidx == 0))
                                {
                                    const RawCellSurfaceNormal normal1(-Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitX());
                                    const RawCellSurfaceNormal normal2(-Eigen::Vector3d::UnitZ(), Eigen::Vector3d::UnitZ());
                                    UpdateSurfaceNormalGridCell(std::vector<RawCellSurfaceNormal>{normal1, normal2}, current_obstacle.pose, local_cell_location, environment_sdf, surface_normals_grid);
                                }
                                else if ((xidx == 0) && (zidx == (z_cells - 1)))
                                {
                                    const RawCellSurfaceNormal normal1(-Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitX());
                                    const RawCellSurfaceNormal normal2(Eigen::Vector3d::UnitZ(), -Eigen::Vector3d::UnitZ());
                                    UpdateSurfaceNormalGridCell(std::vector<RawCellSurfaceNormal>{normal1, normal2}, current_obstacle.pose, local_cell_location, environment_sdf, surface_normals_grid);
                                }
                                else if ((xidx == (x_cells - 1)) && (zidx == 0))
                                {
                                    const RawCellSurfaceNormal normal1(Eigen::Vector3d::UnitX(), -Eigen::Vector3d::UnitX());
                                    const RawCellSurfaceNormal normal2(-Eigen::Vector3d::UnitZ(), Eigen::Vector3d::UnitZ());
                                    UpdateSurfaceNormalGridCell(std::vector<RawCellSurfaceNormal>{normal1, normal2}, current_obstacle.pose, local_cell_location, environment_sdf, surface_normals_grid);
                                }
                                else if ((xidx == (x_cells - 1)) && (zidx == (z_cells - 1)))
                                {
                                    const RawCellSurfaceNormal normal1(Eigen::Vector3d::UnitX(), -Eigen::Vector3d::UnitX());
                                    const RawCellSurfaceNormal normal2(Eigen::Vector3d::UnitZ(), -Eigen::Vector3d::UnitZ());
                                    UpdateSurfaceNormalGridCell(std::vector<RawCellSurfaceNormal>{normal1, normal2}, current_obstacle.pose, local_cell_location, environment_sdf, surface_normals_grid);
                                }
                                else if ((yidx == 0) && (zidx == 0))
                                {
                                    const RawCellSurfaceNormal normal1(-Eigen::Vector3d::UnitY(), Eigen::Vector3d::UnitY());
                                    const RawCellSurfaceNormal normal2(-Eigen::Vector3d::UnitZ(), Eigen::Vector3d::UnitZ());
                                    UpdateSurfaceNormalGridCell(std::vector<RawCellSurfaceNormal>{normal1, normal2}, current_obstacle.pose, local_cell_location, environment_sdf, surface_normals_grid);
                                }
                                else if ((yidx == 0) && (zidx == (z_cells - 1)))
                                {
                                    const RawCellSurfaceNormal normal1(-Eigen::Vector3d::UnitY(), Eigen::Vector3d::UnitY());
                                    const RawCellSurfaceNormal normal2(Eigen::Vector3d::UnitZ(), -Eigen::Vector3d::UnitZ());
                                    UpdateSurfaceNormalGridCell(std::vector<RawCellSurfaceNormal>{normal1, normal2}, current_obstacle.pose, local_cell_location, environment_sdf, surface_normals_grid);
                                }
                                else if ((yidx == (y_cells - 1)) && (zidx == 0))
                                {
                                    const RawCellSurfaceNormal normal1(Eigen::Vector3d::UnitY(), -Eigen::Vector3d::UnitY());
                                    const RawCellSurfaceNormal normal2(-Eigen::Vector3d::UnitZ(), Eigen::Vector3d::UnitZ());
                                    UpdateSurfaceNormalGridCell(std::vector<RawCellSurfaceNormal>{normal1, normal2}, current_obstacle.pose, local_cell_location, environment_sdf, surface_normals_grid);
                                }
                                else if ((yidx == (y_cells - 1)) && (zidx == (z_cells - 1)))
                                {
                                    const RawCellSurfaceNormal normal1(Eigen::Vector3d::UnitY(), -Eigen::Vector3d::UnitY());
                                    const RawCellSurfaceNormal normal2(Eigen::Vector3d::UnitZ(), -Eigen::Vector3d::UnitZ());
                                    UpdateSurfaceNormalGridCell(std::vector<RawCellSurfaceNormal>{normal1, normal2}, current_obstacle.pose, local_cell_location, environment_sdf, surface_normals_grid);
                                }
                                // Finally, let's cover the 6 faces
                                else if (xidx == 0)
                                {
                                    const RawCellSurfaceNormal normal(-Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitX());
                                    UpdateSurfaceNormalGridCell(std::vector<RawCellSurfaceNormal>{normal}, current_obstacle.pose, local_cell_location, environment_sdf, surface_normals_grid);
                                }
                                else if (xidx == (x_cells - 1))
                                {
                                    const RawCellSurfaceNormal normal(Eigen::Vector3d::UnitX(), -Eigen::Vector3d::UnitX());
                                    UpdateSurfaceNormalGridCell(std::vector<RawCellSurfaceNormal>{normal}, current_obstacle.pose, local_cell_location, environment_sdf, surface_normals_grid);
                                }
                                else if (yidx == 0)
                                {
                                    const RawCellSurfaceNormal normal(-Eigen::Vector3d::UnitY(), Eigen::Vector3d::UnitY());
                                    UpdateSurfaceNormalGridCell(std::vector<RawCellSurfaceNormal>{normal}, current_obstacle.pose, local_cell_location, environment_sdf, surface_normals_grid);
                                }
                                else if (yidx == (y_cells - 1))
                                {
                                    const RawCellSurfaceNormal normal(Eigen::Vector3d::UnitY(), -Eigen::Vector3d::UnitY());
                                    UpdateSurfaceNormalGridCell(std::vector<RawCellSurfaceNormal>{normal}, current_obstacle.pose, local_cell_location, environment_sdf, surface_normals_grid);
                                }
                                else if (zidx == 0)
                                {
                                    const RawCellSurfaceNormal normal(-Eigen::Vector3d::UnitZ(), Eigen::Vector3d::UnitZ());
                                    UpdateSurfaceNormalGridCell(std::vector<RawCellSurfaceNormal>{normal}, current_obstacle.pose, local_cell_location, environment_sdf, surface_normals_grid);
                                }
                                else if (zidx == (z_cells - 1))
                                {
                                    const RawCellSurfaceNormal normal(Eigen::Vector3d::UnitZ(), -Eigen::Vector3d::UnitZ());
                                    UpdateSurfaceNormalGridCell(std::vector<RawCellSurfaceNormal>{normal}, current_obstacle.pose, local_cell_location, environment_sdf, surface_normals_grid);
                                }
                            }
                        }
                    }
                }
            }
            return surface_normals_grid;
        }

    public:

        inline SimpleParticleContactSimulator(const std::vector<OBSTACLE_CONFIG>& environment_objects, const double environment_resolution)
        {
            contact_distance_threshold_ = 0.0 * environment_resolution;
            // Build the environment
            environment_ = BuildEnvironment(environment_objects, environment_resolution);
            // Build the SDF
            environment_sdf_ = environment_.ExtractSignedDistanceField(INFINITY, std::vector<uint32_t>()).first;
            // Build the surface normals map
            surface_normals_grid_ = BuildSurfaceNormalsGrid(environment_objects, environment_sdf_);
            // Update & mark connected components
            //environment_.UpdateConnectedComponents();
            // Update the convex segments
            convex_segment_counts_ = environment_.UpdateConvexSegments();
//            // Make the SDF for each object
//            for (size_t idx = 0; idx < environment_objects.size(); idx++)
//            {
//                const OBSTACLE_CONFIG& current_object = environment_objects[idx];
//                per_object_sdfs_[current_object.object_id] = environment_.ExtractSignedDistanceField(INFINITY, std::vector<uint32_t>{current_object.object_id}).first;
//            }
            ResetResolveStatistics();
            initialized_ = true;
        }

        inline SimpleParticleContactSimulator(const std::string& environment_id, const double environment_resolution)
        {
            contact_distance_threshold_ = 0.0 * environment_resolution;
            // Build the environment
            environment_ = BuildEnvironment(environment_id, environment_resolution);
            // Build the SDF
            environment_sdf_ = environment_.ExtractSignedDistanceField(INFINITY, std::vector<uint32_t>()).first;
            // Build the surface normals map
            surface_normals_grid_ = BuildSurfaceNormalsGrid(environment_id, environment_sdf_);
            // Update & mark connected components
            //environment_.UpdateConnectedComponents();
            // Update the convex segments
            convex_segment_counts_ = environment_.UpdateConvexSegments();
//            // Make the SDF for each object
//            // First. we need to collect all the object IDs (since we don't have objects to start with)
//            std::map<uint32_t, uint32_t> object_ids;
//            for (int64_t x_idx = 0; x_idx < environment_.GetNumXCells(); x_idx++)
//            {
//                for (int64_t y_idx = 0; y_idx < environment_.GetNumYCells(); y_idx++)
//                {
//                    for (int64_t z_idx = 0; z_idx < environment_.GetNumZCells(); z_idx++)
//                    {
//                        const sdf_tools::TAGGED_OBJECT_COLLISION_CELL& env_cell = environment_.GetImmutable(x_idx, y_idx, z_idx).first;
//                        const uint32_t& env_cell_object_id = env_cell.object_id;
//                        if (env_cell_object_id > 0)
//                        {
//                            object_ids[env_cell_object_id] = 1u;
//                        }
//                    }
//                }
//            }
//            for (auto itr = object_ids.begin(); itr != object_ids.end(); ++itr)
//            {
//                const uint32_t object_id = itr->first;
//                per_object_sdfs_[object_id] = environment_.ExtractSignedDistanceField(INFINITY, std::vector<uint32_t>{object_id}).first;
//            }
            ResetResolveStatistics();
            initialized_ = true;
        }

        inline SimpleParticleContactSimulator()
        {
            ResetResolveStatistics();
            initialized_ = false;
        }

        inline bool IsInitialized() const
        {
            return initialized_;
        }

        inline std::pair<std::pair<uint64_t, uint64_t>, std::pair<uint64_t, uint64_t>> GetResolveStatistics() const
        {
            return std::make_pair(std::make_pair(successful_resolves_.load(), unsuccessful_resolves_.load()), std::make_pair(unsuccessful_self_collision_resolves_.load(), unsuccessful_env_collision_resolves_.load()));
        }

        inline void ResetResolveStatistics() const
        {
            successful_resolves_.store(0);
            unsuccessful_resolves_.store(0);
            unsuccessful_self_collision_resolves_.store(0);
            unsuccessful_env_collision_resolves_.store(0);
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

    #ifdef USE_ROS
        inline visualization_msgs::Marker ExportForDisplay() const
        {
            return environment_.ExportForDisplay();
        }

        inline visualization_msgs::Marker ExportSDFForDisplay() const
        {
            return environment_sdf_.ExportForDisplay(1.0f);
        }

//        inline visualization_msgs::Marker ExportObjectSDFForDisplay(const uint32_t object_id) const
//        {
//            auto found_itr = per_object_sdfs_.find(object_id);
//            if (found_itr != per_object_sdfs_.end())
//            {
//                return found_itr->second.ExportForDisplay(1.0f);
//            }
//            else
//            {
//                return visualization_msgs::Marker();
//            }
//        }

        inline visualization_msgs::MarkerArray ExportAllForDisplay() const
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
//            for (auto object_sdfs_itr = per_object_sdfs_.begin(); object_sdfs_itr != per_object_sdfs_.end(); ++object_sdfs_itr)
//            {
//                const uint32_t object_id = object_sdfs_itr->first;
//                const sdf_tools::SignedDistanceField& object_sdf = object_sdfs_itr->second;
//                visualization_msgs::Marker object_sdf_marker = object_sdf.ExportForDisplay(1.0f);
//                object_sdf_marker.id = 1;
//                object_sdf_marker.ns = "object_" + std::to_string(object_id) + "_sdf";
//                display_markers.markers.push_back(object_sdf_marker);
//            }
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
    #endif

        template<typename Robot, typename Configuration, typename RNG, typename ConfigAlloc=std::allocator<Configuration>>
        inline std::pair<Configuration, bool> ForwardSimulateRobot(Robot robot, const Configuration& start_position, const Configuration& target_position, RNG& rng, const uint32_t forward_simulation_steps, const double simulation_shortcut_distance, const bool use_individual_jacobians, const bool allow_contacts, const bool enable_contact_manifold_target_adjustment, ForwardSimulationStepTrace<Configuration, ConfigAlloc>& trace, const bool enable_tracing) const
        {
            //std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
            UNUSED(enable_contact_manifold_target_adjustment);
            // Configure the robot
            robot.UpdatePosition(start_position);
            // Forward simulate for the provided number of steps
            bool collided = false;
            for (uint32_t step = 0; step < forward_simulation_steps; step++)
            {
                // Step forward via the simulator
                // Have robot compute next control input first
                // Then, in a second function *not* in a callback, apply that control input
                const Eigen::VectorXd control_action = robot.GenerateControlAction(target_position, rng);
                const std::pair<Configuration, std::pair<bool, bool>> result = ResolveForwardSimulation<Robot, Configuration>(robot, control_action, rng, use_individual_jacobians, allow_contacts, trace, enable_tracing);
                const Configuration& resolved_configuration = result.first;
                const bool resolve_collided = result.second.first;
                if ((allow_contacts == true) || (resolve_collided == false))
                {
                    robot.UpdatePosition(resolved_configuration);
                    // Check if we've collided with the environment
                    if (resolve_collided)
                    {
                        collided = true;
                    }
                    // Last, but not least, check if we've gotten close enough the target state to short-circut the simulation
                    const double target_distance = robot.ComputeDistanceTo(target_position);
                    if (target_distance < simulation_shortcut_distance)
                    {
                        break;
                    }
                }
                else
                {
                    assert(resolve_collided == true);
                    // If we don't allow contacts, we don't update the position and we stop simulating
                    break;
                }
            }
            // Return the ending position of the robot and if it has collided during simulation
            const Configuration reached_position = robot.GetPosition();
            //std::cout << "Forward simulated from\nNearest: " << PrettyPrint::PrettyPrint(start_position) << "\nTarget: " << PrettyPrint::PrettyPrint(target_position) << "\nReached: " << PrettyPrint::PrettyPrint(reached_position) << std::endl;
            return std::pair<Configuration, bool>(reached_position, collided);
        }

        template<typename Robot>
        inline bool CheckEnvironmentCollision(const Robot& robot, const std::vector<std::pair<std::string, EigenHelpers::VectorVector3d>>& robot_links_points) const
        {
            //bool collided = false;
            // Now, go through the links and points of the robot for collision checking
            for (size_t link_idx = 0; link_idx < robot_links_points.size(); link_idx++)
            {
                // Grab the link name and points
                const std::string& link_name = robot_links_points[link_idx].first;
                const EigenHelpers::VectorVector3d link_points = robot_links_points[link_idx].second;
                // Get the transform of the current link
                const Eigen::Affine3d link_transform = robot.GetLinkTransform(link_name);
                // Now, go through the points of the link
                for (size_t point_idx = 0; point_idx < link_points.size(); point_idx++)
                {
                    // Transform the link point into the environment frame
                    const Eigen::Vector3d& link_relative_point = link_points[point_idx];
                    const Eigen::Vector3d environment_relative_point = link_transform * link_relative_point;
                    assert(environment_sdf_.CheckInBounds(environment_relative_point));
                    // Check for collisions
                    const float distance = environment_sdf_.Get(environment_relative_point);
                    // We only work with points in collision
                    if (distance < contact_distance_threshold_)
                    {
                        return true;
                        //collided = true;
                        //const std::string msg = "Collision on link " + PrettyPrint::PrettyPrint(link_idx) + " with point " + PrettyPrint::PrettyPrint(point_idx) + " at " + PrettyPrint::PrettyPrint(environment_relative_point) + " with penetration " + PrettyPrint::PrettyPrint(distance) + " and gradient " + PrettyPrint::PrettyPrint(environment_sdf_.GetGradient(environment_relative_point, true));
                        //std::cerr << msg << std::endl;
                    }
                }
            }
            return false;
            //return collided;
        }

        template<typename Robot>
        inline std::map<std::pair<size_t, size_t>, Eigen::Vector3d> ExtractSelfCollidingPoints(const Robot& previous_robot, const Robot& current_robot, const std::vector<std::pair<std::string, EigenHelpers::VectorVector3d>>& robot_links_points, const std::vector<std::pair<size_t, size_t>>& candidate_points) const
        {
            if (candidate_points.size() > 1)
            {
                // Now, we separate the points by link
                std::map<size_t, std::vector<size_t>> point_self_collision_check_map;
                for (size_t idx = 0; idx < candidate_points.size(); idx++)
                {
                    const std::pair<size_t, size_t>& point = candidate_points[idx];
                    point_self_collision_check_map[point.first].push_back(point.second);
                }
                //std::cout << "Considering " << point_self_collision_check_map.size() << " separate links with self-colliding points" << std::endl;
                // Let's see how many links we have - we only care if multiple links are involved
                if (point_self_collision_check_map.size() >= 2)
                {
                    // For each link, figure out which *other* links it is colliding with
                    std::map<size_t, std::vector<size_t>> link_collisions;
                    for (auto fitr = point_self_collision_check_map.begin(); fitr != point_self_collision_check_map.end(); ++fitr)
                    {
                        for (auto sitr = point_self_collision_check_map.begin(); sitr != point_self_collision_check_map.end(); ++sitr)
                        {
                            if (fitr != sitr)
                            {
                                const size_t fitr_link = fitr->first;
                                const size_t sitr_link = sitr->first;
                                const bool self_collision_allowed = current_robot.CheckIfSelfCollisionAllowed(fitr_link, sitr_link);
                                if (self_collision_allowed == false)
                                {
                                    //const std::string msg = "Self collisions not allowed between " + std::to_string(fitr_link) + " and " + std::to_string(sitr_link);
                                    //std::cout << msg << std::endl;
                                    link_collisions[fitr_link].push_back(sitr_link);
                                }
                            }
                        }
                    }
                    if (link_collisions.size() < 2)
                    {
                        return std::map<std::pair<size_t, size_t>, Eigen::Vector3d>();
                    }
                    //std::cout << "Self collisions: " << PrettyPrint::PrettyPrint(link_collisions) << std::endl;
                    // We go through each link and compute an "input momentum" vector that reflects the contribution of the particular link to the collision
                    // We can assume that point motion has occurred over unit time, so motion = velocity, and that each point has unit mass, so momentum = velocity.
                    // Thus, the momentum = motion for each particle
                    std::map<size_t, Eigen::Vector3d> link_momentum_vectors;
                    for (auto link_itr = point_self_collision_check_map.begin(); link_itr != point_self_collision_check_map.end(); ++link_itr)
                    {
                        const size_t link_idx = link_itr->first;
                        // Skip links we already filtered out due to allowed self collision
                        if (link_collisions.find(link_idx) != link_collisions.end())
                        {
                            const std::string& link_name = robot_links_points[link_itr->first].first;
                            const Eigen::Affine3d previous_link_transform = previous_robot.GetLinkTransform(link_name);
                            const Eigen::Affine3d current_link_transform = current_robot.GetLinkTransform(link_name);
                            const std::vector<size_t>& link_points = link_itr->second;
                            Eigen::Vector3d link_momentum_vector(0.0, 0.0, 0.0);
                            for (size_t idx = 0; idx < link_points.size(); idx++)
                            {
                                const size_t link_point = link_points[idx];
                                const Eigen::Vector3d& link_relative_point = robot_links_points[link_idx].second[link_point];
                                const Eigen::Vector3d previous_point_location = previous_link_transform * link_relative_point;
                                const Eigen::Vector3d current_point_location = current_link_transform * link_relative_point;
                                const Eigen::Vector3d point_motion = current_point_location - previous_point_location;
                                if (point_motion.norm() <= std::numeric_limits<double>::epsilon())
                                {
                                    //const std::string msg = "Point motion would be zero (link " + std::to_string(link_idx) + ", point " + std::to_string(link_point) + ")\nPrevious location: " + PrettyPrint::PrettyPrint(previous_point_location) + "\nCurrent location: " + PrettyPrint::PrettyPrint(current_point_location);
                                    //std::cout << msg << std::endl;
                                }
                                link_momentum_vector = link_momentum_vector + point_motion;
                            }
                            link_momentum_vectors[link_idx] = link_momentum_vector;
                        }
                    }
                    //std::cout << "Link momentum vectors:\n" << PrettyPrint::PrettyPrint(link_momentum_vectors) << std::endl;
                    // Store the corrections we compute
                    std::map<std::pair<size_t, size_t>, Eigen::Vector3d> self_colliding_points_with_corrections;
                    // Now, for each link, we compute a correction for each colliding point on the link
                    for (auto link_itr = link_collisions.begin(); link_itr != link_collisions.end(); ++link_itr)
                    {
                        const size_t link_idx = link_itr->first;
                        const std::string& link_name = robot_links_points[link_idx].first;
                        const Eigen::Affine3d previous_link_transform = previous_robot.GetLinkTransform(link_name);
                        const Eigen::Vector3d& link_point = robot_links_points[link_idx].second[point_self_collision_check_map[link_idx].front()];
                        const Eigen::Vector3d link_point_location = previous_link_transform * link_point;
                        const std::vector<size_t>& colliding_links = link_itr->second;
                        const auto link_momentum_vector_query = link_momentum_vectors.find(link_idx);
                        assert(link_momentum_vector_query != link_momentum_vectors.end());
                        const Eigen::Vector3d link_momentum_vector = link_momentum_vector_query->second;
                        const Eigen::Vector3d link_velocity = link_momentum_vector / (double)point_self_collision_check_map[link_idx].size();
                        // We compute a whole-link correction
                        // For the purposes of simulation, we assume an elastic collision - i.e. momentum must be conserved
                        Eigen::MatrixXd contact_matrix = Eigen::MatrixXd::Zero((colliding_links.size() + 1) * 3, colliding_links.size() * 3);
                        // For each link, fill in the contact matrix
                        for (int64_t link = 1; link <= (int64_t)colliding_links.size(); link++)
                        {
                            int64_t collision_number = link - 1;
                            // Our current link gets -I
                            contact_matrix.block<3, 3>(0, (collision_number * 3)) = (Eigen::MatrixXd::Identity(3, 3) * -1.0);
                            // The other link gets +I
                            contact_matrix.block<3, 3>((link * 3), (collision_number * 3)) = Eigen::MatrixXd::Identity(3, 3);
                        }
                        //std::cout << "Contact matrix:\n" << PrettyPrint::PrettyPrint(contact_matrix) << std::endl;
                        // Generate the contact normal matrix
                        Eigen::MatrixXd contact_normal_matrix = Eigen::MatrixXd::Zero(colliding_links.size() * 3, colliding_links.size());
                        for (int64_t collision = 0; collision < (int64_t)colliding_links.size(); collision++)
                        {
                            const size_t other_link_idx = colliding_links[collision];
                            const std::string& other_link_name = robot_links_points[other_link_idx].first;
                            const Eigen::Affine3d previous_other_link_transform = previous_robot.GetLinkTransform(other_link_name);
                            const Eigen::Vector3d& other_link_point = robot_links_points[other_link_idx].second[point_self_collision_check_map[other_link_idx].front()];
                            const Eigen::Vector3d other_link_point_location = previous_other_link_transform * other_link_point;
                            // Compute the contact normal
                            //const Eigen::Vector3d other_link_velocity = link_momentum_vectors[other_link_idx] / (double)point_self_collision_check_map[other_link_idx].size();
                            //const Eigen::Vector3d current_link_position = link_velocity * -1.0;
                            //const Eigen::Vector3d current_other_link_position = other_link_velocity * -1.0;
                            //const Eigen::Vector3d contact_direction = current_other_link_position - current_link_position;
                            const Eigen::Vector3d contact_direction = other_link_point_location - link_point_location;
                            const Eigen::Vector3d contact_normal = EigenHelpers::SafeNormal(contact_direction);
                            //const std::string msg = "Contact normal: " + PrettyPrint::PrettyPrint(contact_normal) + "\nCurrent link position: " + PrettyPrint::PrettyPrint(link_point_location) + "\nOther link position: " + PrettyPrint::PrettyPrint(other_link_point_location);
                            //std::cout << msg << std::endl;
                            contact_normal_matrix.block<3, 1>((collision * 3), collision) = contact_normal;
                        }
                        //std::cout << "Contact normal matrix:\n" << PrettyPrint::PrettyPrint(contact_normal_matrix) << std::endl;
                        // Generate the mass matrix
                        Eigen::MatrixXd mass_matrix = Eigen::MatrixXd::Zero((colliding_links.size() + 1) * 3, (colliding_links.size() + 1) * 3);
                        // Add the mass of our link
                        mass_matrix.block<3, 3>(0, 0) = Eigen::MatrixXd::Identity(3, 3) * (double)point_self_collision_check_map[link_idx].size();
                        // Add the mass of the other links
                        for (int64_t link = 1; link <= (int64_t)colliding_links.size(); link++)
                        {
                            const size_t other_link_idx = colliding_links[link - 1];
                            mass_matrix.block<3, 3>((link * 3), (link * 3)) = Eigen::MatrixXd::Identity(3, 3) * (double)point_self_collision_check_map[other_link_idx].size();
                        }
                        //std::cout << "Mass matrix:\n" << PrettyPrint::PrettyPrint(mass_matrix) << std::endl;
                        // Generate the velocity matrix
                        Eigen::MatrixXd velocity_matrix = Eigen::MatrixXd::Zero((colliding_links.size() + 1) * 3, 1);
                        velocity_matrix.block<3, 1>(0, 0) = link_velocity;
                        for (int64_t link = 1; link <= (int64_t)colliding_links.size(); link++)
                        {
                            const size_t other_link_idx = colliding_links[link - 1];
                            const Eigen::Vector3d other_link_velocity = link_momentum_vectors[other_link_idx] / (double)point_self_collision_check_map[other_link_idx].size();
                            velocity_matrix.block<3, 1>((link * 3), 0) = other_link_velocity;
                        }
                        //std::cout << "Velocity matrix:\n" << PrettyPrint::PrettyPrint(velocity_matrix) << std::endl;
                        // Compute the impulse corrections
                        // Yes, this is ugly. This is to suppress a warning on type conversion related to Eigen operations
                        #pragma GCC diagnostic push
                        #pragma GCC diagnostic ignored "-Wconversion"
                        const Eigen::MatrixXd impulses = (contact_normal_matrix.transpose() * contact_matrix.transpose() * mass_matrix.inverse() * contact_matrix * contact_normal_matrix).inverse() * contact_normal_matrix.transpose() * contact_matrix.transpose() * velocity_matrix;
                        //std::cout << "Impulses:\n" << PrettyPrint::PrettyPrint(impulses) << std::endl;
                        // Compute the new velocities
                        const Eigen::MatrixXd velocity_delta = (mass_matrix.inverse() * contact_matrix * contact_normal_matrix * impulses) * -1.0;
                        //std::cout << "New velocities:\n" << velocity_delta << std::endl;
                        #pragma GCC diagnostic pop
                        // Extract the correction just for our current link
                        const Eigen::Vector3d link_correction_velocity = velocity_delta.block<3, 1>(0, 0);
                        // We then distribute that correction over the points on that link that have contributed to the collision
                        const std::vector<size_t>& link_points = point_self_collision_check_map[link_idx];
                        for (size_t idx = 0; idx < link_points.size(); idx++)
                        {
                            const size_t point_idx = link_points[idx];
                            const std::pair<size_t, size_t> point_id(link_idx, point_idx);
                            //std::cout << "Link correction velocity: " << PrettyPrint::PrettyPrint(link_correction_velocity) << std::endl;
                            const Eigen::Vector3d point_correction = link_correction_velocity / (double)(link_points.size());
                            assert(isnan(point_correction.x()) == false);
                            assert(isnan(point_correction.y()) == false);
                            assert(isnan(point_correction.z()) == false);
                            //std::cout << "Correction (new): " << PrettyPrint::PrettyPrint(point_id) << " - " << PrettyPrint::PrettyPrint(point_correction) << std::endl;
                            self_colliding_points_with_corrections[point_id] = point_correction;
                        }
                    }
                    /* This is the old, but "working" version
                    // Now, we go through every pair and check if self-collisions are allowed
                    for (auto fitr = point_self_collision_check_map.begin(); fitr != point_self_collision_check_map.end(); ++fitr)
                    {
                        for (auto sitr = point_self_collision_check_map.begin(); sitr != point_self_collision_check_map.end(); ++sitr)
                        {
                            if (fitr != sitr)
                            {
                                const size_t fitr_link = fitr->first;
                                const size_t sitr_link = sitr->first;
                                const bool self_collision_allowed = robot.CheckIfSelfCollisionAllowed(fitr_link, sitr_link);
                                // If self-collision is not allowed, we have detected a self collision and we are done
                                if (self_collision_allowed == false)
                                {
                                    //std::cout << "Processing self-collision between " << robot_links_points[fitr_link].first << " (" << fitr_link << ") and " << robot_links_points[sitr_link].first << " (" << sitr_link << ")" << std::endl;
                                    // Add all the fitr points
                                    for (size_t idx = 0; idx < fitr->second.size(); idx++)
                                    {
                                        const size_t point_idx = fitr->second[idx];
                                        const std::pair<size_t, size_t> point_id(fitr_link, point_idx);
                                        const std::string& link_name = robot_links_points[point_id.first].first;
                                        const Eigen::Vector3d& link_relative_point = robot_links_points[point_id.first].second[point_id.second];
                                        const Eigen::Matrix<double, 3, Eigen::Dynamic> point_jacobian = robot.ComputeLinkPointJacobian(link_name, link_relative_point);
                                        //std::cout << "Point Jacobian:\n" << point_jacobian << std::endl;
                                        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> raw_point_motion = point_jacobian * control_input_step;
                                        //std::cout << "Point motion:\n" << raw_point_motion << std::endl;
                                        const Eigen::Vector3d point_motion = raw_point_motion.col(0);
                                        const Eigen::Vector3d point_correction = point_motion * -1.0;
                                        std::cout << "Correction (old): " << PrettyPrint::PrettyPrint(point_id) << "\n" << point_correction << std::endl;
                                        //self_colliding_points_with_corrections[point_id] = point_correction;
                                    }
                                    // We're done with the current fitr
                                    break;
                                }
                                else
                                {
                                    //std::cout << "Ignoring allowed self-collision between " << robot_links_points[fitr_link].first << " (" << fitr_link << ") and " << robot_links_points[sitr_link].first << " (" << sitr_link << ")" << std::endl;
                                }
                            }
                        }
                    }
                    */
                    return self_colliding_points_with_corrections;
                }
                // One link cannot be self-colliding
                else
                {
                    return std::map<std::pair<size_t, size_t>, Eigen::Vector3d>();
                }
            }
            // One point cannot be self-colliding
            else
            {
                return std::map<std::pair<size_t, size_t>, Eigen::Vector3d>();
            }
        }

        inline std::vector<int64_t> LocationToExtendedGridIndex(const Eigen::Vector3d& location) const
        {
            assert(initialized_);
            const Eigen::Vector3d point_in_grid_frame = environment_.GetOriginTransform().inverse() * location;
            const int64_t x_cell = (int64_t)(point_in_grid_frame.x() / environment_.GetResolution());
            const int64_t y_cell = (int64_t)(point_in_grid_frame.y() / environment_.GetResolution());
            const int64_t z_cell = (int64_t)(point_in_grid_frame.z() / environment_.GetResolution());
            return std::vector<int64_t>{x_cell, y_cell, z_cell};
        }

        template<typename Robot>
        inline std::unordered_map<std::pair<size_t, size_t>, Eigen::Vector3d> CollectSelfCollisions(const Robot& previous_robot, const Robot& current_robot, const std::vector<std::pair<std::string, EigenHelpers::VectorVector3d>>& robot_links_points) const
        {
            // Note that robots with only one link *cannot* self-collide!
            if (robot_links_points.size() == 1)
            {
                return std::unordered_map<std::pair<size_t, size_t>, Eigen::Vector3d>();
            }
            else if (robot_links_points.size() == 2)
            {
                // If the robot is only two links, and self-collision between them is allowed, we can avoid checks
                if (current_robot.CheckIfSelfCollisionAllowed(0, 1))
                {
                    return std::unordered_map<std::pair<size_t, size_t>, Eigen::Vector3d>();
                }
            }
            // We use a hastable to detect self-collisions
            std::unordered_map<VoxelGrid::GRID_INDEX, std::vector<std::pair<size_t, size_t>>> self_collision_check_map;
            // Now, go through the links and points of the robot for collision checking
            for (size_t link_idx = 0; link_idx < robot_links_points.size(); link_idx++)
            {
                // Grab the link name and points
                const std::string& link_name = robot_links_points[link_idx].first;
                const EigenHelpers::VectorVector3d link_points = robot_links_points[link_idx].second;
                // Get the transform of the current link
                const Eigen::Affine3d link_transform = current_robot.GetLinkTransform(link_name);
                // Now, go through the points of the link
                for (size_t point_idx = 0; point_idx < link_points.size(); point_idx++)
                {
                    // Transform the link point into the environment frame
                    const Eigen::Vector3d& link_relative_point = link_points[point_idx];
                    const Eigen::Vector3d environment_relative_point = link_transform * link_relative_point;
                    // Get the corresponding index
                    const std::vector<int64_t> index = LocationToExtendedGridIndex(environment_relative_point);
                    assert(index.size() == 3);
                    VoxelGrid::GRID_INDEX point_index(index[0], index[1], index[2]);
                    // Insert the index into the map
                    self_collision_check_map[point_index].push_back(std::pair<size_t, size_t>(link_idx, point_idx));
                }
            }
            // Now, we go through the map and see if any points overlap
            // Store "true" self-collisions
            std::unordered_map<std::pair<size_t, size_t>, Eigen::Vector3d> self_collisions;
            for (auto itr = self_collision_check_map.begin(); itr != self_collision_check_map.end(); ++itr)
            {
                //const VoxelGrid::GRID_INDEX& location = itr->first;
                const std::vector<std::pair<size_t, size_t>>& candidate_points = itr->second;
                //std::cout << "Candidate points: " << PrettyPrint::PrettyPrint(candidate_points) << std::endl;
                const std::map<std::pair<size_t, size_t>, Eigen::Vector3d> self_colliding_points = ExtractSelfCollidingPoints(previous_robot, current_robot, robot_links_points, candidate_points);
                //std::cout << "Extracted points: " << PrettyPrint::PrettyPrint(self_colliding_points) << std::endl;
                for (auto spcitr = self_colliding_points.begin(); spcitr != self_colliding_points.end(); ++spcitr)
                {
                    const std::pair<size_t, size_t>& self_colliding_point = spcitr->first;
                    const Eigen::Vector3d& correction = spcitr->second;
                    self_collisions[self_colliding_point] = correction;
                }
            }
            // If we haven't already returned, we are self-collision-free
            return self_collisions;
        }

        template<typename Robot, typename Configuration>
        inline std::pair<bool, std::unordered_map<std::pair<size_t, size_t>, Eigen::Vector3d>> CheckCollision(const Robot& robot, const Configuration& previous_config, const Configuration& current_config, const std::vector<std::pair<std::string, EigenHelpers::VectorVector3d>>& robot_links_points) const
        {
            Robot current_robot = robot;
            Robot previous_robot = robot;
            // We need our own copies with a set config to use for kinematics!
            current_robot.UpdatePosition(current_config);
            previous_robot.UpdatePosition(previous_config);
            const bool env_collision = CheckEnvironmentCollision(current_robot, robot_links_points);
            const std::unordered_map<std::pair<size_t, size_t>, Eigen::Vector3d> self_collisions = CollectSelfCollisions(previous_robot, current_robot, robot_links_points);
            //std::cout << self_collisions.size() << " self-colliding points to resolve" << std::endl;
            if (env_collision || (self_collisions.size() > 0))
            {
                return std::pair<bool, std::unordered_map<std::pair<size_t, size_t>, Eigen::Vector3d>>(true, self_collisions);
            }
            else
            {
                return std::pair<bool, std::unordered_map<std::pair<size_t, size_t>, Eigen::Vector3d>>(false, self_collisions);
            }
        }

        template<typename Robot, typename Configuration, typename RNG, typename ConfigAlloc=std::allocator<Configuration>>
        inline std::pair<Configuration, std::pair<bool, bool>> ResolveForwardSimulation(Robot robot, const Eigen::VectorXd& control_input, RNG& rng, const bool use_individual_jacobians, const bool allow_contacts, ForwardSimulationStepTrace<Configuration, ConfigAlloc>& trace, const bool enable_tracing) const
        {
            //std::cout << "Resolving control input: " << PrettyPrint::PrettyPrint(control_input) << std::endl;
            // Get the list of link name + link points for all the links of the robot
            const std::vector<std::pair<std::string, EigenHelpers::VectorVector3d>>& robot_links_points = robot.GetRawLinksPoints();
            // Step along the control input
            const double step_norm = control_input.norm();
            const double max_robot_motion_per_step = robot.GetMaxMotionPerStep();
            const double step_motion_estimate = step_norm * max_robot_motion_per_step;
            const double allowed_microstep_distance = GetResolution() * 0.5;
            const uint32_t number_microsteps = std::max(1u, ((uint32_t)ceil(step_motion_estimate / allowed_microstep_distance)));
            assert(number_microsteps > 0);
            //std::cout << "Resolving simulation step with estimated motion " << step_motion_estimate << " in " << number_microsteps << " microsteps" << std::endl;
            const Eigen::VectorXd control_input_step = control_input * (1.0 / (double)number_microsteps);
            //std::cout << "Control input step: " << PrettyPrint::PrettyPrint(control_input_step) << std::endl;
            bool collided = false;
            if (enable_tracing)
            {
                trace.resolver_steps.emplace_back();
                trace.resolver_steps.back().control_input = control_input;
                trace.resolver_steps.back().control_input_step = control_input_step;
            }
            // Iterate
            for (uint32_t micro_step = 0; micro_step < number_microsteps; micro_step++)
            {
                if (enable_tracing)
                {
                    trace.resolver_steps.back().contact_resolver_steps.emplace_back();
                }
                // Store the previous configuration of the robot
                const Configuration previous_configuration = robot.GetPosition();
                //std::cout << "\x1b[35;1m Pre-action configuration: " << PrettyPrint::PrettyPrint(previous_configuration) << " \x1b[0m" << std::endl;
                // Update the position of the robot
                robot.ApplyControlInput(control_input_step, rng);
                const Configuration post_action_configuration = robot.GetPosition(rng);
                robot.UpdatePosition(post_action_configuration);
                //std::cout << "\x1b[33;1m Post-action configuration: " << PrettyPrint::PrettyPrint(post_action_configuration) << " \x1b[0m" << std::endl;
                std::pair<bool, std::unordered_map<std::pair<size_t, size_t>, Eigen::Vector3d>> collision_check = CheckCollision(robot, previous_configuration, post_action_configuration, robot_links_points);
                std::unordered_map<std::pair<size_t, size_t>, Eigen::Vector3d>& self_collision_map = collision_check.second;
                bool in_collision = collision_check.first;
                if (in_collision)
                {
                    collided = true;
                }
                if (enable_tracing)
                {
                    trace.resolver_steps.back().contact_resolver_steps.back().contact_resolution_steps.push_back(post_action_configuration);
                }
                // Now, we know if a collision has happened
                if (in_collision && allow_contacts)
                {
                    Configuration active_configuration = post_action_configuration;
                    uint32_t resolver_iterations = 0;
                    double correction_step_scaling = 1.0;
                    while (in_collision)
                    {
                        const Eigen::VectorXd raw_correction_step = use_individual_jacobians ? ComputeResolverCorrectionStepIndividualJacobians(robot, previous_configuration, active_configuration, robot_links_points, self_collision_map) : ComputeResolverCorrectionStepStackedJacobian(robot, previous_configuration, active_configuration, robot_links_points, self_collision_map);
                        //std::cout << "Raw Cstep: " << raw_correction_step << std::endl;
                        // Scale down the size of the correction step
                        const double correction_step_norm = raw_correction_step.norm();
                        const double correction_step_motion_estimate = correction_step_norm * max_robot_motion_per_step;
                        const double step_fraction = std::max((correction_step_motion_estimate / allowed_microstep_distance), 1.0);
                        const Eigen::VectorXd real_correction_step = (raw_correction_step / step_fraction) * correction_step_scaling;
                        //const Eigen::VectorXd real_correction_step = robot.ProcessCorrectionAction(raw_correction_step);
                        //std::cout << "Real Cstep: " << real_correction_step << std::endl;
                        // Apply correction step
                        robot.ApplyControlInput(real_correction_step);
                        const Configuration post_resolve_configuration = robot.GetPosition();
                        //std::cout << "\x1b[36;1m Post-resolve step configuration: " << PrettyPrint::PrettyPrint(post_resolve_configuration) << " \x1b[0m"  << std::endl;
                        active_configuration = post_resolve_configuration;
                        // Check to see if we're still in collision
                        const std::pair<bool, std::unordered_map<std::pair<size_t, size_t>, Eigen::Vector3d>> new_collision_check = CheckCollision(robot, previous_configuration, active_configuration, robot_links_points);
                        // Update the self-collision map
                        self_collision_map = new_collision_check.second;
                        // Update the collision check variable
                        in_collision = new_collision_check.first;
                        resolver_iterations++;
                        // Update tracing
                        if (enable_tracing)
                        {
                            trace.resolver_steps.back().contact_resolver_steps.back().contact_resolution_steps.push_back(active_configuration);
                        }
                        if (resolver_iterations > MAX_RESOLVER_ITERATIONS)
                        {
//                            const std::string msg = "\x1b[31;1m Resolver iterations > " + std::to_string(MAX_RESOLVER_ITERATIONS) + ", terminating microstep+resolver at configuration " + PrettyPrint::PrettyPrint(active_configuration) + " and returning previous configuration " + PrettyPrint::PrettyPrint(previous_configuration) + "\nCollision check results:\n" + PrettyPrint::PrettyPrint(new_collision_check, false, "\n") + " \x1b[0m";
//                            std::cerr << msg << std::endl;
                            if (enable_tracing)
                            {
                                trace.resolver_steps.back().contact_resolver_steps.back().contact_resolution_steps.push_back(previous_configuration);
                            }
                            unsuccessful_resolves_.fetch_add(1);
                            if (self_collision_map.size() > 0)
                            {
                                unsuccessful_self_collision_resolves_.fetch_add(1);
                            }
                            else
                            {
                                unsuccessful_env_collision_resolves_.fetch_add(1);
                            }
                            return std::make_pair(previous_configuration, std::make_pair(true, true));
                        }
                        if ((resolver_iterations % RESOLVE_CORRECTION_STEP_SCALING_DECAY_ITERATIONS) == 0)
                        {
                            correction_step_scaling = correction_step_scaling * RESOLVE_CORRECTION_STEP_SCALING_DECAY_RATE;
                        }
                    }
                }
                else if (in_collision && (allow_contacts == false))
                {
                    if (enable_tracing)
                    {
                        trace.resolver_steps.back().contact_resolver_steps.back().contact_resolution_steps.push_back(previous_configuration);
                    }
                    successful_resolves_.fetch_add(1);
                    return std::make_pair(previous_configuration, std::make_pair(true, false));
                }
                else
                {
                    continue;
                }
            }
            //std::cout << "\x1b[32;1m Post-action resolution configuration: " << PrettyPrint::PrettyPrint(robot.GetPosition()) << " \x1b[0m" << std::endl;
            successful_resolves_.fetch_add(1);
            return std::make_pair(robot.GetPosition(), std::make_pair(collided, false));
        }

        template<typename Robot, typename Configuration>
        inline Eigen::VectorXd ComputeResolverCorrectionStepIndividualJacobians(const Robot& robot, const Configuration& previous_config, const Configuration& current_config, const std::vector<std::pair<std::string, EigenHelpers::VectorVector3d>>& robot_links_points, const std::unordered_map<std::pair<size_t, size_t>, Eigen::Vector3d>& self_collision_map) const
        {
            Robot current_robot = robot;
            Robot previous_robot = robot;
            // We need our own copy with a set config to use for kinematics!
            current_robot.UpdatePosition(current_config);
            previous_robot.UpdatePosition(previous_config);
            // In case a collision has occured, we need to compute a "collision gradient" that will push the robot out of collision
            Eigen::VectorXd raw_correction_step;
            // Now, go through the links and points of the robot to build up the xgradient and Jacobian
            for (size_t link_idx = 0; link_idx < robot_links_points.size(); link_idx++)
            {
                // Grab the link name and points
                const std::string& link_name = robot_links_points[link_idx].first;
                const EigenHelpers::VectorVector3d link_points = robot_links_points[link_idx].second;
                // Get the transform of the current link
                const Eigen::Affine3d previous_link_transform = previous_robot.GetLinkTransform(link_name);
                const Eigen::Affine3d current_link_transform = current_robot.GetLinkTransform(link_name);
                // Now, go through the points of the link
                for (size_t point_idx = 0; point_idx < link_points.size(); point_idx++)
                {
                    // Check if we have a self-collision correction
                    std::pair<bool, Eigen::Vector3d> self_collision_correction(false, Eigen::Vector3d(0.0, 0.0, 0.0));
                    auto self_collision_check = self_collision_map.find(std::pair<size_t, size_t>(link_idx, point_idx));
                    if (self_collision_check != self_collision_map.end())
                    {
                        self_collision_correction.first = true;
                        self_collision_correction.second = self_collision_check->second;
                    }
                    // Check against the environment
                    std::pair<bool, Eigen::Vector3d> env_collision_correction(false, Eigen::Vector3d(0.0, 0.0, 0.0));
                    const Eigen::Vector3d& link_relative_point = link_points[point_idx];
                    // Get the Jacobian for the current point
                    const Eigen::Matrix<double, 3, Eigen::Dynamic> point_jacobian = current_robot.ComputeLinkPointJacobian(link_name, link_relative_point);
                    //std::cout << "Point jacobian: " << point_jacobian << std::endl;
                    // Transform the link point into the environment frame
                    const Eigen::Vector3d previous_point_location = previous_link_transform * link_relative_point;
                    const Eigen::Vector3d current_point_location = current_link_transform * link_relative_point;
                    // We only work with points in the SDF
                    assert(environment_sdf_.CheckInBounds(current_point_location));
                    const float distance = environment_sdf_.Get(current_point_location);
                    // We only work with points in collision
                    if (distance < contact_distance_threshold_)
                    {
                        // We query the surface normal map for the gradient to move out of contact using the particle motion
                        const Eigen::Vector3d point_motion = current_point_location - previous_point_location;
                        const Eigen::Vector3d normed_point_motion = EigenHelpers::SafeNormal(point_motion);
                        // Query the surface normal map
                        const std::pair<Eigen::Vector3d, bool> surface_normal_query = surface_normals_grid_.LookupSurfaceNormal(current_point_location, normed_point_motion);
                        assert(surface_normal_query.second);
                        const Eigen::Vector3d& raw_gradient = surface_normal_query.first;
                        const Eigen::Vector3d normed_point_gradient = EigenHelpers::SafeNormal(raw_gradient);
                        // We use the penetration distance as a scale
                        const double penetration_distance = fabs(contact_distance_threshold_ - distance);
                        const Eigen::Vector3d scaled_gradient = normed_point_gradient * penetration_distance;
                        //std::cout << "Point gradient: " << scaled_gradient << std::endl;
                        env_collision_correction.first = true;
                        env_collision_correction.second = scaled_gradient;
                    }
                    // We only add a correction for the point if necessary
                    if (self_collision_correction.first || env_collision_correction.first)
                    {
                        // Assemble the workspace correction vector
                        Eigen::Vector3d point_correction(0.0, 0.0, 0.0);
                        if (self_collision_correction.first)
                        {
                            point_correction = point_correction + self_collision_correction.second;
                        }
                        if (env_collision_correction.first)
                        {
                            point_correction = point_correction + env_collision_correction.second;
                        }
                        // Compute the correction step
//                        // Naive Pinv version
//                        // Invert the point jacobian
//                        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> point_jacobian_pinv = EigenHelpers::Pinv(point_jacobian, EigenHelpers::SuggestedRcond(), true);
//                        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> raw_correction = point_jacobian_pinv * point_correction;
                        // We could use the naive Pinv(J) * pdot, but instead we solve the Ax = b (Jqdot = pdot) problem directly using one of the solvers in Eigen
                        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> raw_correction = point_jacobian.colPivHouseholderQr().solve(point_correction);
                        // Extract the c-space correction
                        const Eigen::VectorXd point_correction_step = raw_correction.col(0);
                        if (raw_correction_step.size() == 0)
                        {
                            raw_correction_step = point_correction_step;
                        }
                        else
                        {
                            raw_correction_step = raw_correction_step + point_correction_step;
                        }
                    }
                }
            }
            return raw_correction_step;
        }

        template<typename Robot, typename Configuration>
        inline Eigen::VectorXd ComputeResolverCorrectionStepStackedJacobian(const Robot& robot, const Configuration& previous_config, const Configuration& current_config, const std::vector<std::pair<std::string, EigenHelpers::VectorVector3d>>& robot_links_points, const std::unordered_map<std::pair<size_t, size_t>, Eigen::Vector3d>& self_collision_map) const
        {
            Robot current_robot = robot;
            Robot previous_robot = robot;
            // We need our own copy with a set config to use for kinematics!
            current_robot.UpdatePosition(current_config);
            previous_robot.UpdatePosition(previous_config);
            // In case a collision has occured, we need to compute a "collision gradient" that will push the robot out of collision
            // The "collision gradient" is of the form qgradient = J(q)+ * xgradient
            // Make space for the xgradient and Jacobian
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> robot_jacobians;
            Eigen::Matrix<double, Eigen::Dynamic, 1> point_corrections;
            // Now, go through the links and points of the robot to build up the xgradient and Jacobian
            for (size_t link_idx = 0; link_idx < robot_links_points.size(); link_idx++)
            {
                // Grab the link name and points
                const std::string& link_name = robot_links_points[link_idx].first;
                const EigenHelpers::VectorVector3d link_points = robot_links_points[link_idx].second;
                // Get the transform of the current link
                const Eigen::Affine3d previous_link_transform = previous_robot.GetLinkTransform(link_name);
                const Eigen::Affine3d current_link_transform = current_robot.GetLinkTransform(link_name);
                // Now, go through the points of the link
                for (size_t point_idx = 0; point_idx < link_points.size(); point_idx++)
                {
                    // Check if we have a self-collision correction
                    std::pair<bool, Eigen::Vector3d> self_collision_correction(false, Eigen::Vector3d(0.0, 0.0, 0.0));
                    auto self_collision_check = self_collision_map.find(std::pair<size_t, size_t>(link_idx, point_idx));
                    if (self_collision_check != self_collision_map.end())
                    {
                        self_collision_correction.first = true;
                        self_collision_correction.second = self_collision_check->second;
                    }
                    // Check against the environment
                    std::pair<bool, Eigen::Vector3d> env_collision_correction(false, Eigen::Vector3d(0.0, 0.0, 0.0));
                    const Eigen::Vector3d& link_relative_point = link_points[point_idx];
                    // Get the Jacobian for the current point
                    const Eigen::Matrix<double, 3, Eigen::Dynamic> point_jacobian = current_robot.ComputeLinkPointJacobian(link_name, link_relative_point);
                    // Transform the link point into the environment frame
                    const Eigen::Vector3d previous_point_location = previous_link_transform * link_relative_point;
                    const Eigen::Vector3d current_point_location = current_link_transform * link_relative_point;
                    // We only work with points in the SDF
                    if (environment_sdf_.CheckInBounds(current_point_location) == false)
                    {
                        const std::string msg = "Point out of bounds: " + PrettyPrint::PrettyPrint(current_point_location);
                        std::cout << msg << std::endl;
                        assert(false);
                    }
                    const float distance = environment_sdf_.Get(current_point_location);
                    // We only work with points in collision
                    if (distance < contact_distance_threshold_)
                    {
                        // We query the surface normal map for the gradient to move out of contact using the particle motion
                        const Eigen::Vector3d point_motion = current_point_location - previous_point_location;
                        const Eigen::Vector3d normed_point_motion = EigenHelpers::SafeNormal(point_motion);
                        // Query the surface normal map
                        const std::pair<Eigen::Vector3d, bool> surface_normal_query = surface_normals_grid_.LookupSurfaceNormal(current_point_location, normed_point_motion);
                        assert(surface_normal_query.second);
                        const Eigen::Vector3d& raw_gradient = surface_normal_query.first;
                        const Eigen::Vector3d normed_point_gradient = EigenHelpers::SafeNormal(raw_gradient);
                        // We use the penetration distance as a scale
                        const double penetration_distance = fabs(contact_distance_threshold_ - distance);
                        const Eigen::Vector3d scaled_gradient = normed_point_gradient * penetration_distance;
                        //std::cout << "Point gradient: " << scaled_gradient << std::endl;
                        env_collision_correction.first = true;
                        env_collision_correction.second = scaled_gradient;
                    }
                    // We only add a correction for the point if necessary
                    if (self_collision_correction.first || env_collision_correction.first)
                    {
                        // Append the new point jacobian to the matrix of jacobians
                        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> extended_robot_jacobians;
                        extended_robot_jacobians.resize(robot_jacobians.rows() + 3, point_jacobian.cols());
                        if (robot_jacobians.cols() > 0)
                        {
                            extended_robot_jacobians << robot_jacobians,point_jacobian;
                        }
                        else
                        {
                            extended_robot_jacobians << point_jacobian;
                        }
                        robot_jacobians = extended_robot_jacobians;
                        // Assemble the workspace correction vector
                        Eigen::Vector3d point_correction(0.0, 0.0, 0.0);
                        if (self_collision_correction.first)
                        {
                            //std::cout << "Self-collision correction: " << PrettyPrint::PrettyPrint(self_collision_correction.second) << std::endl;
                            point_correction = point_correction + self_collision_correction.second;
                        }
                        if (env_collision_correction.first)
                        {
                            //std::cout << "Env-collision correction: " << PrettyPrint::PrettyPrint(env_collision_correction.second) << std::endl;
                            point_correction = point_correction + env_collision_correction.second;
                        }
//                        // Weight the correction based on the distance from the joint axis (note, the joint axis is at the origin of the link!)
//                        const double dist_to_joint_axis = link_relative_point.norm();
//                        point_correction = point_correction * dist_to_joint_axis;
                        // Append the new workspace correction vector to the matrix of correction vectors
                        Eigen::Matrix<double, Eigen::Dynamic, 1> extended_point_corrections;
                        extended_point_corrections.resize(point_corrections.rows() + 3, Eigen::NoChange);
                        extended_point_corrections << point_corrections,point_correction;
                        point_corrections = extended_point_corrections;
                        //std::cout << "Point jacobian:\n" << point_jacobian << std::endl;
                        //std::cout << "Point correction: " << PrettyPrint::PrettyPrint(point_correction) << std::endl;
                    }
                }
            }
            // Compute the correction step
//                        // Naive Pinv version
//                        // Invert the robot jacobians
//            const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> robot_jacobians_pinv = EigenHelpers::Pinv(robot_jacobians, EigenHelpers::SuggestedRcond());
//            const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> raw_correction = robot_jacobians_pinv * point_corrections;
            // We could use the naive Pinv(J) * pdot, but instead we solve the Ax = b (Jqdot = pdot) problem directly using one of the solvers in Eigen
            const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> raw_correction = robot_jacobians.colPivHouseholderQr().solve(point_corrections);
            // Extract the c-space correction
            const Eigen::VectorXd raw_correction_step = raw_correction.col(0);
            return raw_correction_step;
        }
    };
}

#endif // SIMPLE_PARTICLE_CONTACT_SIMULATOR_HPP
