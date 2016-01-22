#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <functional>
#include <arc_utilities/pretty_print.hpp>
#include <arc_utilities/simple_rrt_planner.hpp>
#include <nomdp_planning/nomdp_planner_state.hpp>

#ifndef EXECUTION_POLICY_HPP
#define EXECUTION_POLICY_HPP

namespace execution_policy
{
    template<typename Configuration, typename AverageFn, typename DistanceFn, typename DimDistanceFn, typename ConfigAlloc=std::allocator<Configuration>>
    class ExecutionPolicy
    {
    protected:

        bool initialized_;
        std::vector<simple_rrt_planner::SimpleRRTPlannerState<nomdp_planning_tools::NomdpPlannerState<Configuration, AverageFn, DistanceFn, DimDistanceFn, ConfigAlloc>>> policy_node_tree_;
        std::function<double(const std::vector<Configuration, ConfigAlloc>&, const std::vector<Configuration, ConfigAlloc>&)> distance_fn_;

    public:

        inline ExecutionPolicy(const std::function<double(const std::vector<Configuration, ConfigAlloc>&, const std::vector<Configuration, ConfigAlloc>&)>& distance_fn) : initialized_(true), distance_fn_(distance_fn) {}

        inline ExecutionPolicy() : initialized_(false), distance_fn_([] (const std::vector<Configuration, ConfigAlloc>&, const std::vector<Configuration, ConfigAlloc>&) { return INFINITY; }) {}

        inline bool IsInitialized() const
        {
            return initialized_;
        }

        inline bool InitializePolicyTree(const std::vector<simple_rrt_planner::SimpleRRTPlannerState<nomdp_planning_tools::NomdpPlannerState<Configuration, AverageFn, DistanceFn, DimDistanceFn, ConfigAlloc>>>& policy_tree)
        {
            assert(initialized_);
            if (simple_rrt_planner::SimpleHybridRRTPlanner::CheckTreeLinkage(policy_tree))
            {
                policy_node_tree_ = policy_tree;
                return true;
            }
            else
            {
                std::cerr << "Provided policy tree has invalid linkage" << std::endl;
                return false;
            }
        }

        inline const simple_rrt_planner::SimpleRRTPlannerState<nomdp_planning_tools::NomdpPlannerState<Configuration, AverageFn, DistanceFn, DimDistanceFn, ConfigAlloc>>& GetRawPolicy() const
        {
            assert(initialized_);
            return policy_node_tree_;
        }
    };
}

template<typename Configuration, typename AverageFn, typename DistanceFn, typename DimDistanceFn, typename ConfigAlloc=std::allocator<Configuration>>
std::ostream& operator<<(std::ostream& strm, const execution_policy::ExecutionPolicy<Configuration, AverageFn, DimDistanceFn, ConfigAlloc>& policy)
{
    const std::vector<simple_rrt_planner::SimpleRRTPlannerState<nomdp_planning_tools::NomdpPlannerState<Configuration, AverageFn, DistanceFn, DimDistanceFn, ConfigAlloc>>>& raw_policy_tree = policy.GetRawPolicy();
    strm << "Execution Policy - Policy: ";
    for (size_t idx = 0; idx < raw_policy_tree.size(); idx++)
    {
        const simple_rrt_planner::SimpleRRTPlannerState<nomdp_planning_tools::NomdpPlannerState<Configuration, AverageFn, DistanceFn, DimDistanceFn, ConfigAlloc>>& policy_tree_state = raw_policy_tree[idx];
        const int64_t parent_index = policy_tree_state.GetParentIndex();
        const std::vector<int64_t>& child_indices = policy_tree_state.GetChildIndices();
        const nomdp_planning_tools::NomdpPlannerState<Configuration, AverageFn, DistanceFn, DimDistanceFn, ConfigAlloc>& policy_state = policy_tree_state.GetValueImmutable();
        strm << "\nState # " << idx << " with parent " << parent_index << " and children " << PrettyPrint::PrettyPrint(child_indices, true) << " - value: " << PrettyPrint::PrettyPrint(policy_state);
    }
    return strm;
}

#endif // EXECUTION_POLICY_HPP
