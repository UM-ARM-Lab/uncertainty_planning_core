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

namespace nomdp_planning_tools
{
    struct OBSTACLE_CONFIG
    {
        u_int32_t object_id;
        u_int8_t r;
        u_int8_t g;
        u_int8_t b;
        u_int8_t a;
        Eigen::Affine3d pose;
        Eigen::Vector3d extents;

        OBSTACLE_CONFIG(const u_int32_t in_object_id, const Eigen::Affine3d& in_pose, const Eigen::Vector3d& in_extents, const u_int8_t in_r, const u_int8_t in_g, const u_int8_t in_b, const u_int8_t in_a) : r(in_r), g(in_g), b(in_b), a(in_a), pose(in_pose), extents(in_extents)
        {
            assert(in_object_id > 0);
            object_id = in_object_id;
        }

        OBSTACLE_CONFIG(const u_int32_t in_object_id, const Eigen::Vector3d& in_translation, const Eigen::Quaterniond& in_orientation, const Eigen::Vector3d& in_extents, const u_int8_t in_r, const u_int8_t in_g, const u_int8_t in_b, const u_int8_t in_a) : r(in_r), g(in_g), b(in_b), a(in_a)
        {
            assert(in_object_id > 0);
            object_id = in_object_id;
            pose = (Eigen::Translation3d)in_translation * in_orientation;
            extents = in_extents;
        }

        OBSTACLE_CONFIG() : object_id(0u), r(0u), g(0u), b(0u), a(0u), pose(Eigen::Affine3d::Identity()), extents(0.0, 0.0, 0.0) {}
    };

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

    class SurfaceNormalGrid
    {
    protected:

        struct StoredSurfaceNormal
        {
            Eigen::Vector3d normal;
            Eigen::Vector3d entry_direction;

            StoredSurfaceNormal(const Eigen::Vector3d& in_normal, const Eigen::Vector3d& in_direction) : normal(EigenHelpers::SafeNorm(in_normal)), entry_direction(EigenHelpers::SafeNorm(in_direction)) {}

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
        std::map<u_int32_t, sdf_tools::SignedDistanceField> per_object_sdfs_;
        VoxelGrid::VoxelGrid<u_int64_t> fingerprint_grid_;

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
            else
            {
                //std::cout << "Rebuilding the environment with " << obstacles.size() << " obstacles" << std::endl;
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

        inline SurfaceNormalGrid BuildSurfaceNormalsGrid(const std::vector<OBSTACLE_CONFIG>& obstacles, const double resolution, const sdf_tools::SignedDistanceField& environment_sdf) const
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
                const double effective_resolution = resolution * 0.5;
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
                                double x_location = -(current_obstacle.extents.x() - (resolution * 0.5)) + (effective_resolution * xidx);
                                double y_location = -(current_obstacle.extents.y() - (resolution * 0.5)) + (effective_resolution * yidx);
                                double z_location = -(current_obstacle.extents.z() - (resolution * 0.5)) + (effective_resolution * zidx);
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
            environment_sdf_ = environment_.ExtractSignedDistanceField(INFINITY, std::vector<u_int32_t>()).first;
            // Build the surface normals map
            surface_normals_grid_ = BuildSurfaceNormalsGrid(environment_objects, environment_resolution, environment_sdf_);
            // Update & mark connected components
            environment_.UpdateConnectedComponents();
            // Update the convex segments
            environment_.UpdateConvexSegments();
            // Make the SDF for each object
            for (size_t idx = 0; idx < environment_objects.size(); idx++)
            {
                const OBSTACLE_CONFIG& current_object = environment_objects[idx];
                per_object_sdfs_[current_object.object_id] = environment_.ExtractSignedDistanceField(INFINITY, std::vector<u_int32_t>{current_object.object_id}).first;
            }
            initialized_ = true;
        }

        inline SimpleParticleContactSimulator() : initialized_(false) {}

        inline bool IsInitialized() const
        {
            return initialized_;
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

        inline visualization_msgs::Marker ExportObjectSDFForDisplay(const u_int32_t object_id) const
        {
            auto found_itr = per_object_sdfs_.find(object_id);
            if (found_itr != per_object_sdfs_.end())
            {
                return found_itr->second.ExportForDisplay(1.0f);
            }
            else
            {
                return visualization_msgs::Marker();
            }
        }

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
            for (auto object_sdfs_itr = per_object_sdfs_.begin(); object_sdfs_itr != per_object_sdfs_.end(); ++object_sdfs_itr)
            {
                const u_int32_t object_id = object_sdfs_itr->first;
                const sdf_tools::SignedDistanceField& object_sdf = object_sdfs_itr->second;
                visualization_msgs::Marker object_sdf_marker = object_sdf.ExportForDisplay(1.0f);
                object_sdf_marker.id = 1;
                object_sdf_marker.ns = "object_" + std::to_string(object_id) + "_sdf";
                display_markers.markers.push_back(object_sdf_marker);
            }
            // Draw all the 13 convex components of the free space (in the future, this will be better)
            for (u_int32_t convex_segment = 1; convex_segment <= 13; convex_segment++)
            {
                const visualization_msgs::Marker segment_marker = environment_.ExportConvexSegmentForDisplay(0u, convex_segment);
                display_markers.markers.push_back(segment_marker);
            }
            return display_markers;
        }
    #endif

        template<typename Robot, typename Configuration, typename RNG, typename ConfigAlloc=std::allocator<Configuration>>
        inline std::pair<Configuration, bool> ForwardSimulatePointRobot(Robot robot, const Configuration& start_position, const Configuration& target_position, RNG& rng, const u_int32_t forward_simulation_steps, const double simulation_shortcut_distance, ForwardSimulationStepTrace<Configuration, ConfigAlloc>& trace, const bool enable_tracing=false) const
        {
            // Configure the robot
            robot.UpdatePosition(start_position);
            // Forward simulate for the provided number of steps
            bool collided = false;
            for (u_int32_t step = 0; step < forward_simulation_steps; step++)
            {
                // Step forward via the simulator
                // Have robot compute next control input first
                // Then, in a second funtion *not* in a callback, apply that control input
                const Eigen::VectorXd control_action = robot.GenerateControlAction(target_position, rng);
                std::pair<Configuration, bool> result = ResolveForwardSimulation<Robot, Configuration>(robot, control_action, rng, trace, enable_tracing);
                robot.UpdatePosition(result.first);
                // Check if we've collided with the environment
                if (result.second)
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
            // Return the ending position of the robot and if it has collided during simulation
            return std::pair<Configuration, bool>(robot.GetPosition(), collided);
        }

        template<typename Robot>
        inline bool CheckCollision(const Robot& robot, const std::vector<std::pair<std::string, EigenHelpers::VectorVector3d>>& robot_links_points) const
        {
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
                    // Check for collisions
                    // We only work with points in the SDF
                    if (environment_sdf_.CheckInBounds(environment_relative_point))
                    {
                        const float distance = environment_sdf_.Get(environment_relative_point);
                        // We only work with points in collision
                        if (distance < contact_distance_threshold_)
                        {
                            return true;
                        }
                    }
                }
            }
            return false;
        }

        template<typename Robot, typename Configuration, typename RNG, typename ConfigAlloc=std::allocator<Configuration>>
        inline std::pair<Configuration, bool> ResolveForwardSimulation(Robot& robot, const Eigen::VectorXd& control_input, RNG& rng, ForwardSimulationStepTrace<Configuration, ConfigAlloc>& trace, const bool enable_tracing) const
        {
            // Get the list of link name + link points for all the links of the robot
            const std::vector<std::pair<std::string, EigenHelpers::VectorVector3d>>& robot_links_points = robot.GetRawLinksPoints();
            // Step along the control input
            const double microstep_distance = GetResolution() * 0.25;
            u_int32_t number_microsteps = (u_int32_t)ceil(control_input.norm() / microstep_distance);
            const Eigen::VectorXd control_input_step = control_input * (1.0 / (double)number_microsteps);
            bool collided = false;
            if (enable_tracing)
            {
                trace.resolver_steps.emplace_back();
                trace.resolver_steps.back().control_input = control_input;
                trace.resolver_steps.back().control_input_step = control_input_step;
            }
            // Iterate
            for (u_int32_t micro_step = 0; micro_step < number_microsteps; micro_step++)
            {
                if (enable_tracing)
                {
                    trace.resolver_steps.back().contact_resolver_steps.emplace_back();
                }
                // Update the position of the robot
                robot.ApplyControlInput(control_input_step, rng);
                collided = CheckCollision(robot, robot_links_points);
                if (enable_tracing)
                {
                    trace.resolver_steps.back().contact_resolver_steps.back().contact_resolution_steps.push_back(robot.GetPosition());
                }
                // Now, we know if a collision has happened
                if (collided)
                {
                    bool in_collision = true;
                    u_int32_t resolver_iterations = 0;
                    while (in_collision)
                    {
                        // In case a collision has occured, we need to compute a "collision gradient" that will push the robot out of collision
                        // The "collision gradient" is of the form qgradient = J(q)+ * xgradient
//                        // Make space for the xgradient and Jacobian
//                        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> robot_jacobians;
//                        Eigen::Matrix<double, Eigen::Dynamic, 1> point_gradients;
                        Eigen::VectorXd raw_correction_step;
                        // Now, go through the links and points of the robot to build up the xgradient and Jacobian
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
                                const Eigen::Vector3d& link_relative_point = link_points[point_idx];
                                // Transform the link point into the environment frame
                                const Eigen::Vector3d environment_relative_point = link_transform * link_relative_point;
                                // We only work with points in the SDF
                                if (environment_sdf_.CheckInBounds(environment_relative_point))
                                {
                                    const float distance = environment_sdf_.Get(environment_relative_point);
                                    // We only work with points in collision
                                    if (distance < contact_distance_threshold_)
                                    {
                                        // Get the Jacobian for the current point
                                        const Eigen::Matrix<double, 3, Eigen::Dynamic> point_jacobian = robot.ComputeLinkPointJacobian(link_name, link_relative_point);
                                        std::cout << "Point jacobian: " << point_jacobian << std::endl;
                                        // Invert
                                        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> point_jacobian_pinv = EigenHelpers::Pinv(point_jacobian, EigenHelpers::SuggestedRcond());
//                                        // Make a new, larger, matrix to store the extended jacobians
//                                        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> extended_robot_jacobians;
//                                        extended_robot_jacobians.resize(robot_jacobians.rows() + 3, point_jacobian.cols());
//                                        // Append the active jacobian to the matrix of jacobians
//                                        if (robot_jacobians.cols() > 0)
//                                        {
//                                            extended_robot_jacobians << robot_jacobians,point_jacobian;
//                                        }
//                                        else
//                                        {
//                                            extended_robot_jacobians << point_jacobian;
//                                        }
//                                        robot_jacobians = extended_robot_jacobians;
                                        // We query the surface normal map for the gradient to move out of contact using the particle motion
                                        // Instead of storing these for every point, we just compute them as needed via differential kinematics - i.e. the Jacobian
                                        // xdot = J(q, p) * qdot
                                        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> raw_point_motion = point_jacobian * control_input_step;
                                        const Eigen::Vector3d point_motion = raw_point_motion.col(0);
                                        const Eigen::Vector3d normed_point_motion = EigenHelpers::SafeNorm(point_motion);
                                        // Query the surface normal map
                                        const std::pair<Eigen::Vector3d, bool> surface_normal_query = surface_normals_grid_.LookupSurfaceNormal(environment_relative_point, normed_point_motion);
                                        assert(surface_normal_query.second);
                                        const Eigen::Vector3d& raw_gradient = surface_normal_query.first;
                                        const Eigen::Vector3d normed_point_gradient = EigenHelpers::SafeNorm(raw_gradient);
                                        // We use the penetration distance as a scale
                                        const double penetration_distance = fabs(contact_distance_threshold_ - distance);
                                        const Eigen::Vector3d scaled_gradient = normed_point_gradient * penetration_distance;
                                        std::cout << "Point gradient: " << scaled_gradient << std::endl;
//                                            // Make a new, larger, matrix to store the extended gradients
//                                            Eigen::Matrix<double, Eigen::Dynamic, 1> extended_point_gradients;
//                                            extended_point_gradients.resize(point_gradients.rows() + 3, Eigen::NoChange);
//                                            // Append the gradient to the vector of gradients
//                                            extended_point_gradients << point_gradients,scaled_gradient;
//                                            point_gradients = extended_point_gradients;
                                        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> raw_correction = point_jacobian_pinv * scaled_gradient;
                                        const Eigen::VectorXd point_correction = raw_correction.col(0);
                                        if (raw_correction_step.size() == 0)
                                        {
                                            raw_correction_step = point_correction;
                                        }
                                        else
                                        {
                                            raw_correction_step = raw_correction_step + point_correction;
                                        }
                                    }
                                }
                            }
                        }
//                        // Now, we've collected the Jacobians and point gradients for everything
//                        // Do qgrad = J(q)+ * xgrad
//                        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> jacobian_pinv = EigenHelpers::Pinv(robot_jacobians, EigenHelpers::SuggestedRcond());
//                        const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> raw_cspace_gradient = jacobian_pinv * point_gradients;
//                        // Process it to make a correction step
//                        // The "robot" knows how to do this
//                        const Eigen::VectorXd raw_correction_step = raw_cspace_gradient.col(0);
                        std::cout << "Raw Cstep: " << raw_correction_step << std::endl;
                        const Eigen::VectorXd real_correction_step = robot.ProcessCorrectionAction(raw_correction_step);
                        std::cout << "Real Cstep: " << real_correction_step << std::endl;
                        // Apply correction step
                        robot.ApplyControlInput(real_correction_step);
                        // Check to see if we're still in collision
                        bool still_in_collision = CheckCollision(robot, robot_links_points);
                        // Update the collision check variable
                        in_collision = still_in_collision;
                        resolver_iterations++;
                        if (resolver_iterations > 100)
                        {
                            std::cerr << "Resolver iterations = " << resolver_iterations << ", is the simulator stuck?" << std::endl;
                        }
                        if (enable_tracing)
                        {
                            trace.resolver_steps.back().contact_resolver_steps.back().contact_resolution_steps.push_back(robot.GetPosition());
                        }
                    }
                }
                else
                {
                    continue;
                }
            }
            return std::pair<Configuration, bool>(robot.GetPosition(), collided);
        }
    };
}

#endif // SIMPLE_PARTICLE_CONTACT_SIMULATOR_HPP
