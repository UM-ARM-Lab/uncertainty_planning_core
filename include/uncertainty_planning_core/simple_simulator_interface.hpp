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
#include <arc_utilities/simple_robot_model_interface.hpp>
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <omp.h>

#ifndef SIMPLE_SIMULATOR_INTERFACE_HPP
#define SIMPLE_SIMULATOR_INTERFACE_HPP

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

    template<typename Configuration, typename RNG, typename ConfigAlloc=std::allocator<Configuration>>
    class SimulatorInterface
    {
    protected:

        typedef simple_robot_model_interface::SimpleRobotModelInterface<Configuration, ConfigAlloc> Robot;

    public:

        SimulatorInterface() {}

        virtual int32_t GetDebugLevel() const = 0;

        virtual int32_t SetDebugLevel(const int32_t debug_level) = 0;

        virtual RNG& GetRandomGenerator() = 0;

        virtual std::string GetFrame() const = 0;

        virtual visualization_msgs::MarkerArray MakeEnvironmentDisplayRep() const = 0;

        inline static std_msgs::ColorRGBA MakeColor(const float r, const float g, const float b, const float a)
        {
            std_msgs::ColorRGBA color;
            color.r = r;
            color.g = g;
            color.b = b;
            color.a = a;
            return color;
        }

        virtual visualization_msgs::MarkerArray MakeConfigurationDisplayRep(const std::shared_ptr<Robot>& immutable_robot, const Configuration& configuration, const std_msgs::ColorRGBA& color, const int32_t starting_index, const std::string& config_marker_ns) const = 0;

        virtual visualization_msgs::MarkerArray MakeControlInputDisplayRep(const std::shared_ptr<Robot>& immutable_robot, const Configuration& configuration, const Eigen::VectorXd& control_input, const std_msgs::ColorRGBA& color, const int32_t starting_index, const std::string& control_input_marker_ns) const = 0;

        virtual Eigen::Vector4d Get3dPointForConfig(const std::shared_ptr<Robot>& immutable_robot, const Configuration& config) const = 0;

        virtual std::map<std::string, double> GetStatistics() const = 0;

        virtual void ResetStatistics() = 0;

        virtual bool CheckConfigCollision(const std::shared_ptr<Robot>& immutable_robot, const Configuration& config, const double inflation_ratio=0.0) const = 0;

        virtual std::pair<Configuration, std::pair<bool, bool>> ForwardSimulateMutableRobot(const std::shared_ptr<Robot>& mutable_robot, const Configuration& target_position, const bool allow_contacts, ForwardSimulationStepTrace<Configuration, ConfigAlloc>& trace, const bool enable_tracing, const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn) = 0;

        virtual std::pair<Configuration, std::pair<bool, bool>> ForwardSimulateRobot(const std::shared_ptr<Robot>& immutable_robot, const Configuration& start_position, const Configuration& target_position, const bool allow_contacts, ForwardSimulationStepTrace<Configuration, ConfigAlloc>& trace, const bool enable_tracing, const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn) = 0;

        virtual std::vector<std::pair<Configuration, std::pair<bool, bool>>> ForwardSimulateRobots(const std::shared_ptr<Robot>& immutable_robot, const std::vector<Configuration, ConfigAlloc>& start_positions, const std::vector<Configuration, ConfigAlloc>& target_positions, const bool allow_contacts, const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn) = 0;

        virtual std::pair<Configuration, std::pair<bool, bool>> ReverseSimulateMutableRobot(const std::shared_ptr<Robot>& mutable_robot, const Configuration& target_position, const bool allow_contacts, ForwardSimulationStepTrace<Configuration, ConfigAlloc>& trace, const bool enable_tracing, const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn) = 0;

        virtual std::pair<Configuration, std::pair<bool, bool>> ReverseSimulateRobot(const std::shared_ptr<Robot>& immutable_robot, const Configuration& start_position, const Configuration& target_position, const bool allow_contacts, ForwardSimulationStepTrace<Configuration, ConfigAlloc>& trace, const bool enable_tracing, const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn) = 0;

        virtual std::vector<std::pair<Configuration, std::pair<bool, bool>>> ReverseSimulateRobots(const std::shared_ptr<Robot>& immutable_robot, const std::vector<Configuration, ConfigAlloc>& start_positions, const std::vector<Configuration, ConfigAlloc>& target_positions, const bool allow_contacts, const std::function<void(const visualization_msgs::MarkerArray&)>& display_fn) = 0;
    };
}

#endif // SIMPLE_SIMULATOR_INTERFACE_HPP
