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
#include <memory>
#include <arc_utilities/eigen_helpers.hpp>
#include <arc_utilities/simple_robot_model_interface.hpp>
#include <uncertainty_planning_core/simple_simulator_interface.hpp>
#include <visualization_msgs/MarkerArray.h>

#ifndef SIMPLE_OUTCOME_CLUSTERING_INTERFACE_HPP
#define SIMPLE_OUTCOME_CLUSTERING_INTERFACE_HPP

namespace simple_outcome_clustering_interface
{
    template<typename Configuration, typename ConfigAlloc=std::allocator<Configuration>>
    class OutcomeClusteringInterface
    {
    protected:

        typedef simple_robot_model_interface::SimpleRobotModelInterface<Configuration, ConfigAlloc> Robot;
        typedef simple_simulator_interface::SimulationResult<Configuration> SimulationResult;

    public:

        virtual ~OutcomeClusteringInterface() {}

        virtual int32_t GetDebugLevel() const = 0;

        virtual int32_t SetDebugLevel(const int32_t debug_level) = 0;

        virtual std::map<std::string, double> GetStatistics() const = 0;

        virtual void ResetStatistics() = 0;

        virtual std::vector<std::vector<size_t>> ClusterParticles(const std::shared_ptr<Robot>& robot, const std::vector<SimulationResult>& particles, const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn) = 0;

        virtual std::vector<uint8_t> IdentifyClusterMembers(const std::shared_ptr<Robot>& robot, const std::vector<Configuration, ConfigAlloc>& cluster, const std::vector<SimulationResult>& particles, const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn) = 0;
    };
}

#endif // SIMPLE_OUTCOME_CLUSTERING_INTERFACE_HPP
