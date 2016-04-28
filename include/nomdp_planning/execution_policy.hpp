#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <functional>
#include <queue>
#include <arc_utilities/pretty_print.hpp>
#include <arc_utilities/simple_rrt_planner.hpp>
#include <nomdp_planning/nomdp_planner_state.hpp>

#ifdef USE_ROS
    #include <arc_utilities/zlib_helpers.hpp>
#endif

#ifndef EXECUTION_POLICY_HPP
#define EXECUTION_POLICY_HPP

namespace execution_policy
{
    class GraphEdge
    {
    protected:

        int64_t from_index_;
        int64_t to_index_;
        double weight_;

    public:

        static inline uint64_t Serialize(const GraphEdge& edge, std::vector<uint8_t>& buffer)
        {
            return edge.SerializeSelf(buffer);
        }

        static inline std::pair<GraphEdge, uint64_t> Deserialize(const std::vector<uint8_t>& buffer, const uint64_t current)
        {
            GraphEdge temp_edge;
            const uint64_t bytes_read = temp_edge.DeserializeSelf(buffer, current);
            return std::make_pair(temp_edge, bytes_read);
        }

        GraphEdge(const int64_t from_index, const int64_t to_index, const double weight) : from_index_(from_index), to_index_(to_index), weight_(weight) {}

        GraphEdge() : from_index_(-1), to_index_(-1), weight_(0.0) {}

        inline uint64_t SerializeSelf(std::vector<uint8_t>& buffer) const
        {
            const uint64_t start_buffer_size = buffer.size();
            arc_helpers::SerializeFixedSizePOD<int64_t>(from_index_, buffer);
            arc_helpers::SerializeFixedSizePOD<int64_t>(to_index_, buffer);
            arc_helpers::SerializeFixedSizePOD<double>(weight_, buffer);
            // Figure out how many bytes were written
            const uint64_t end_buffer_size = buffer.size();
            const uint64_t bytes_written = end_buffer_size - start_buffer_size;
            return bytes_written;
        }

        inline uint64_t DeserializeSelf(const std::vector<uint8_t>& buffer, const uint64_t current)
        {
            assert(current < buffer.size());
            uint64_t current_position = current;
            const std::pair<int64_t, uint64_t> deserialized_from_index = arc_helpers::DeserializeFixedSizePOD<double>(buffer, current_position);
            from_index_ = deserialized_from_index.first;
            current_position += deserialized_from_index.second;
            const std::pair<int64_t, uint64_t> deserialized_to_index = arc_helpers::DeserializeFixedSizePOD<double>(buffer, current_position);
            to_index_ = deserialized_to_index.first;
            current_position += deserialized_to_index.second;
            const std::pair<double, uint64_t> deserialized_weight = arc_helpers::DeserializeFixedSizePOD<double>(buffer, current_position);
            weight_ = deserialized_weight.first;
            current_position += deserialized_weight.second;
            // Figure out how many bytes were read
            const uint64_t bytes_read = current_position - current;
            return bytes_read;
        }

        inline bool operator==(const GraphEdge& other) const
        {
            return (from_index_ == other.GetFromIndex() && to_index_ == other.GetToIndex() && weight_ == other.GetWeight());
        }

        inline int64_t GetFromIndex() const
        {
            return from_index_;
        }

        inline int64_t GetToIndex() const
        {
            return to_index_;
        }

        inline double GetWeight() const
        {
            return weight_;
        }

        inline void SetFromIndex(const int64_t new_from_index)
        {
            from_index_ = new_from_index;
        }

        inline void SetToIndex(const int64_t new_to_index)
        {
            to_index_ = new_to_index;
        }

        inline void SetWeight(const double new_weight)
        {
            weight_ = new_weight;
        }
    };

    template<typename T, typename Allocator=std::allocator<T>>
    class GraphNode
    {
    protected:

        T value_;
        double distance_;
        std::vector<GraphEdge> in_edges_;
        std::vector<GraphEdge> out_edges_;

    public:

        static inline uint64_t Serialize(const GraphNode<T, Allocator>& node, std::vector<uint8_t>& buffer, const std::function<uint64_t(const T&, std::vector<uint8_t>&)>& value_serializer)
        {
            return node.SerializeSelf(buffer, value_serializer);
        }

        static inline std::pair<GraphNode<T, Allocator>, uint64_t> Deserialize(const std::vector<uint8_t>& buffer, const uint64_t current, const std::function<std::pair<T, uint64_t>(const std::vector<uint8_t>&, const uint64_t)>& value_deserializer)
        {
            GraphNode<T, Allocator> temp_node;
            const uint64_t bytes_read = temp_node.DeserializeSelf(buffer, current, value_deserializer);
            return std::make_pair(temp_node, bytes_read);
        }

        GraphNode(const T& value, const double distance, const std::vector<GraphEdge>& new_in_edges, const std::vector<GraphEdge>& new_out_edges) : value_(value), distance_(distance), in_edges_(new_in_edges), out_edges_(new_out_edges) {}

        GraphNode(const T& value) : value_(value), distance_(INFINITY) {}

        GraphNode() : distance_(INFINITY) {}


        inline uint64_t SerializeSelf(std::vector<uint8_t>& buffer, const std::function<uint64_t(const T&, std::vector<uint8_t>&)>& value_serializer) const
        {
            const uint64_t start_buffer_size = buffer.size();
            // Serialize the value
            value_serializer(value_, buffer);
            // Serialize the distance
            arc_helpers::SerializeFixedSizePOD<double>(distance_, buffer);
            // Serialize the in edges
            arc_helpers::SerializeVector<GraphEdge>(in_edges_, buffer, GraphEdge::Serialize);
            // Serialize the in edges
            arc_helpers::SerializeVector<GraphEdge>(out_edges_, buffer, GraphEdge::Serialize);
            // Figure out how many bytes were written
            const uint64_t end_buffer_size = buffer.size();
            const uint64_t bytes_written = end_buffer_size - start_buffer_size;
            return bytes_written;
        }

        inline uint64_t DeserializeSelf(const std::vector<uint8_t>& buffer, const uint64_t current, const std::function<std::pair<T, uint64_t>(const std::vector<uint8_t>&, const uint64_t)>& value_deserializer)
        {
            uint64_t current_position = current;
            // Deserialize the value
            const std::pair<T, uint64_t> value_deserialized = value_deserializer(buffer, current_position);
            value_ = value_deserialized.first;
            current_position += value_deserialized.second;
            // Deserialize the distace
            const std::pair<double, uint64_t> distance_deserialized = arc_helpers::DeserializeFixedSizePOD<double>(buffer, current_position);
            distance_ = distance_deserialized.first;
            current_position += distance_deserialized.second;
            // Deserialize the in edges
            const std::pair<std::vector<GraphEdge>, uint64_t> in_edges_deserialized = arc_helpers::DeserializeVector<GraphEdge>(buffer, current_position, GraphEdge::Deserialize);
            in_edges_ = in_edges_deserialized.first;
            current_position += in_edges_deserialized.second;
            // Deserialize the out edges
            const std::pair<std::vector<GraphEdge>, uint64_t> out_edges_deserialized = arc_helpers::DeserializeVector<GraphEdge>(buffer, current_position, GraphEdge::Deserialize);
            out_edges_ = out_edges_deserialized.first;
            current_position += out_edges_deserialized.second;
            // Figure out how many bytes were read
            const uint64_t bytes_read = current_position - current;
            return bytes_read;
        }

        const T& GetValueImmutable() const
        {
            return value_;
        }

        T& GetValueMutable()
        {
            return value_;
        }

        inline void AddInEdge(const GraphEdge& new_in_edge)
        {
            in_edges_.push_back(new_in_edge);
        }

        inline void AddOutEdge(const GraphEdge& new_out_edge)
        {
            out_edges_.push_back(new_out_edge);
        }

        inline void AddEdgePair(const GraphEdge& new_in_edge, const GraphEdge& new_out_edge)
        {
            AddInEdge(new_in_edge);
            AddOutEdge(new_out_edge);
        }

        inline double GetDistance() const
        {
            return distance_;
        }

        inline void SetDistance(const double distance)
        {
            distance_ = distance;
        }

        inline const std::vector<GraphEdge>& GetInEdgesImmutable() const
        {
            return in_edges_;
        }

        inline std::vector<GraphEdge>& GetInEdgesMutable()
        {
            return in_edges_;
        }

        inline const std::vector<GraphEdge>& GetOutEdgesImmutable() const
        {
            return out_edges_;
        }

        inline std::vector<GraphEdge>& GetOutEdgesMutable()
        {
            return out_edges_;
        }

        inline void SetInEdges(const std::vector<GraphEdge>& new_in_edges)
        {
            in_edges_ = new_in_edges;
        }

        inline void SetOutEdges(const std::vector<GraphEdge>& new_out_edges)
        {
            out_edges_ = new_out_edges;
        }
    };

    template<typename T, typename Allocator=std::allocator<T>>
    class Graph
    {
    protected:

        std::vector<GraphNode<T, Allocator>> nodes_;

    public:

        static inline uint64_t Serialize(const Graph<T, Allocator>& graph, std::vector<uint8_t>& buffer, const std::function<uint64_t(const T&, std::vector<uint8_t>&)>& value_serializer)
        {
            return graph.SerializeSelf(buffer, value_serializer);
        }

        static inline std::pair<Graph<T, Allocator>, uint64_t> Deserialize(const std::vector<uint8_t>& buffer, const uint64_t current, const std::function<std::pair<T, uint64_t>(const std::vector<uint8_t>&, const uint64_t)>& value_deserializer)
        {
            Graph<T, Allocator> temp_graph;
            const uint64_t bytes_read = temp_graph.DeserializeSelf(buffer, current, value_deserializer);
            return std::make_pair(temp_graph, bytes_read);
        }

        Graph(const std::vector<GraphNode<T, Allocator>>& nodes)
        {
            if (CheckGraphLinkage(nodes))
            {
                nodes_ = nodes;
            }
            else
            {
                throw std::invalid_argument("Invalid graph linkage");
            }
        }

        Graph(const size_t expected_size)
        {
            nodes_.reserve(expected_size);
        }

        Graph() {}

        inline uint64_t SerializeSelf(std::vector<uint8_t>& buffer, const std::function<uint64_t(const T&, std::vector<uint8_t>&)>& value_serializer) const
        {
            const uint64_t start_buffer_size = buffer.size();
            std::function<uint64_t(const GraphNode<T, Allocator>&, std::vector<uint8_t>&)> graph_state_serializer = std::bind(GraphNode<T, Allocator>::Serialize, std::placeholders::_1, std::placeholders::_2, value_serializer);
            arc_helpers::SerializeVector<GraphNode<T, Allocator>>(nodes_, buffer, graph_state_serializer);
            // Figure out how many bytes were written
            const uint64_t end_buffer_size = buffer.size();
            const uint64_t bytes_written = end_buffer_size - start_buffer_size;
            return bytes_written;
        }

        inline uint64_t DeserializeSelf(const std::vector<uint8_t>& buffer, const uint64_t current, const std::function<std::pair<T, uint64_t>(const std::vector<uint8_t>&, const uint64_t)>& value_deserializer)
        {
            std::function<std::pair<GraphNode<T, Allocator>, uint64_t>(const std::vector<uint8_t>&, const uint64_t)> graph_state_deserializer = std::bind(GraphNode<T, Allocator>::Deserialize, std::placeholders::_1, std::placeholders::_2, value_deserializer);
            const std::pair<std::vector<GraphNode<T, Allocator>>, uint64_t> deserialized_nodes = arc_helpers::DeserializeVector<GraphNode<T, Allocator>>(buffer, current, graph_state_deserializer);
            nodes_ = deserialized_nodes.first;
            return deserialized_nodes.second;
        }

        inline void ShrinkToFit()
        {
            nodes_.shrink_to_fit();
        }

        inline bool CheckGraphLinkage() const
        {
            return CheckGraphLinkage(GetNodesImmutable());
        }

        inline static bool CheckGraphLinkage(const Graph<T, Allocator>& graph)
        {
            return CheckGraphLinkage(graph.GetNodesImmutable());
        }

        inline static bool CheckGraphLinkage(const std::vector<GraphNode<T, Allocator>>& nodes)
        {
            // Go through every node and make sure the edges are valid
            for (size_t idx = 0; idx < nodes.size(); idx++)
            {
                const GraphNode<T, Allocator>& current_node = nodes[idx];
                const std::vector<GraphEdge>& in_edges = current_node.GetInEdgesImmutable();
                const std::vector<GraphEdge>& out_edges = current_node.GetOutEdgesImmutable();
                for (size_t in_edge_idx = 0; in_edge_idx < in_edges.size(); in_edge_idx++)
                {
                    const GraphEdge& current_edge = in_edges[in_edge_idx];
                    // Check from index
                    const int64_t from_index = current_edge.GetFromIndex();
                    if (from_index < 0 || from_index >= nodes.size())
                    {
                        return false;
                    }
                    // Check to index
                    const int64_t to_index = current_edge.GetToIndex();
                    if (to_index < 0 || to_index >= nodes.size())
                    {
                        return false;
                    }
                    else if (to_index != idx)
                    {
                        return false;
                    }
                    // Check edge validity
                    if (from_index == to_index)
                    {
                        return false;
                    }
                }
                for (size_t out_edge_idx = 0; out_edge_idx < out_edges.size(); out_edge_idx++)
                {
                    const GraphEdge& current_edge = out_edges[out_edge_idx];
                    // Check from index
                    const int64_t from_index = current_edge.GetFromIndex();
                    if (from_index < 0 || from_index >= nodes.size())
                    {
                        return false;
                    }
                    else if (from_index != idx)
                    {
                        return false;
                    }
                    // Check to index
                    const int64_t to_index = current_edge.GetToIndex();
                    if (to_index < 0 || to_index >= nodes.size())
                    {
                        return false;
                    }
                    // Check edge validity
                    if (from_index == to_index)
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        inline const std::vector<GraphNode<T, Allocator>>& GetNodesImmutable() const
        {
            return nodes_;
        }

        inline std::vector<GraphNode<T, Allocator>>& GetNodesMutable()
        {
            return nodes_;
        }

        inline const GraphNode<T, Allocator>& GetNodeImmutable(const int64_t index) const
        {
            assert(index >= 0);
            assert(index < nodes_.size());
            return nodes_[index];
        }

        inline GraphNode<T, Allocator>& GetNodeMutable(const int64_t index)
        {
            assert(index >= 0);
            assert(index < nodes_.size());
            return nodes_[index];
        }

        inline int64_t AddNode(const GraphNode<T, Allocator>& new_node)
        {
            nodes_.push_back(new_node);
            return (int64_t)(nodes_.size() - 1);
        }

        inline void AddEdgeBetweenNodes(const int64_t from_index, const int64_t to_index, const double edge_weight)
        {
            assert(from_index >= 0);
            assert(from_index < nodes_.size());
            assert(to_index >= 0);
            assert(to_index < nodes_.size());
            assert(from_index != to_index);
            const GraphEdge new_edge(from_index, to_index, edge_weight);
            GetNodeMutable(from_index).AddOutEdge(new_edge);
            GetNodeMutable(to_index).AddInEdge(new_edge);
        }

        inline void AddEdgesBetweenNodes(const int64_t first_index, const int64_t second_index, const double edge_weight)
        {
            assert(first_index >= 0);
            assert(first_index < nodes_.size());
            assert(second_index >= 0);
            assert(second_index < nodes_.size());
            assert(first_index != second_index);
            const GraphEdge first_edge(first_index, second_index, edge_weight);
            GetNodeMutable(first_index).AddOutEdge(first_edge);
            GetNodeMutable(second_index).AddInEdge(first_edge);
            const GraphEdge second_edge(second_index, first_index, edge_weight);
            GetNodeMutable(second_index).AddOutEdge(second_edge);
            GetNodeMutable(first_index).AddInEdge(second_edge);
        }
    };

    template<typename T, typename Allocator=std::allocator<T>>
    class SimpleDijkstrasAlgorithm
    {
    protected:

        class CompareIndexFn
        {
        public:

            constexpr bool operator()(const std::pair<int64_t, double>& lhs, const std::pair<int64_t, double>& rhs) const
            {
                return lhs.second > rhs.second;
            }
        };

        SimpleDijkstrasAlgorithm() {}

    public:

        inline static std::pair<Graph<T, Allocator>, std::vector<int64_t>> PerformDijkstrasAlgorithm(const Graph<T, Allocator>& graph, const int64_t start_index)
        {
            assert(start_index >= 0);
            assert(start_index < graph.GetNodesImmutable().size());
            Graph<T, Allocator> working_copy = graph;
            // Setup
            std::vector<int64_t> previous_index_map(working_copy.GetNodesImmutable().size(), -1);
            std::priority_queue<std::pair<int64_t, double>, std::vector<std::pair<int64_t, double>>, CompareIndexFn> queue;
            std::unordered_map<int64_t, uint32_t> explored(graph.GetNodesImmutable().size());
            for (size_t idx = 0; idx < working_copy.GetNodesImmutable().size(); idx++)
            {
                working_copy.GetNodeMutable(idx).SetDistance(INFINITY);
                queue.push(std::make_pair((int64_t)idx, INFINITY));
            }
            working_copy.GetNodeMutable(start_index).SetDistance(0.0);
            previous_index_map[start_index] = start_index;
            queue.push(std::make_pair(start_index, 0.0));
            while (queue.size() > 0)
            {
                const std::pair<int64_t, double> top_node = queue.top();
                const int64_t& top_node_index = top_node.first;
                const double& top_node_distance = top_node.second;
                queue.pop();
                if (explored[top_node.first] > 0)
                {
                    // We've already been here
                    continue;
                }
                else
                {
                    // Note that we've been here
                    explored[top_node.first] = 1;
                    // Get our neighbors
                    const std::vector<GraphEdge>& neighbor_edges = working_copy.GetNodeImmutable(top_node_index).GetInEdgesImmutable(); // Previously, this was Out edges
                    // Go through our neighbors
                    for (size_t neighbor_idx = 0; neighbor_idx < neighbor_edges.size(); neighbor_idx++)
                    {
                        const int64_t neighbor_index = neighbor_edges[neighbor_idx].GetFromIndex(); // Previously, this was To index
                        const double neighbor_edge_weight = neighbor_edges[neighbor_idx].GetWeight();
                        const double new_neighbor_distance = top_node_distance + neighbor_edge_weight;
                        // Check against the neighbor
                        const double stored_neighbor_distance = working_copy.GetNodeImmutable(neighbor_index).GetDistance();
                        if (new_neighbor_distance < stored_neighbor_distance)
                        {
                            // We've found a better way to get to this node
                            // Check if it's already been explored
                            if (explored[neighbor_index] > 0)
                            {
                                // If it's already been explored, we just update it in place
                                working_copy.GetNodeMutable(neighbor_index).SetDistance(new_neighbor_distance);
                            }
                            else
                            {
                                // If it hasn't been explored, we need to update it and add it to the queue
                                working_copy.GetNodeMutable(neighbor_index).SetDistance(new_neighbor_distance);
                                queue.push(std::make_pair(neighbor_index, new_neighbor_distance));
                            }
                            // Update that we're the best previous node
                            previous_index_map[neighbor_index] = top_node_index;
                        }
                        else
                        {
                            // Do nothing
                            continue;
                        }
                    }
                }
            }
            return std::pair<Graph<T, Allocator>, std::vector<int64_t>>(working_copy, previous_index_map);
        }
    };

    template<typename Configuration, typename ConfigSerializer, typename AverageFn, typename DistanceFn, typename DimDistanceFn, typename ConfigAlloc=std::allocator<Configuration>>
    class PolicyGraphBuilder
    {
    private:

        // Typedef so we don't hate ourselves
        typedef nomdp_planning_tools::NomdpPlannerState<Configuration, ConfigSerializer, AverageFn, DistanceFn, DimDistanceFn, ConfigAlloc> NomdpPlanningState;
        typedef simple_rrt_planner::SimpleRRTPlannerState<NomdpPlanningState, std::allocator<NomdpPlanningState>> NomdpPlanningTreeState;
        typedef std::vector<NomdpPlanningTreeState> NomdpPlanningTree;
        typedef GraphNode<NomdpPlanningState, std::allocator<NomdpPlanningState>> PolicyGraphNode;
        typedef Graph<NomdpPlanningState, std::allocator<NomdpPlanningState>> PolicyGraph;

        PolicyGraphBuilder() {}

    public:

        inline static PolicyGraph BuildPolicyGraphFromPlannerTree(const NomdpPlanningTree& planner_tree, const NomdpPlanningState& goal_state)
        {
            assert (simple_rrt_planner::SimpleHybridRRTPlanner::CheckTreeLinkage(planner_tree));
            PolicyGraph policy_graph(planner_tree.size() + 1);
            // First pass, add all the nodes to the graph
            for (size_t idx = 0; idx < planner_tree.size(); idx++)
            {
                const NomdpPlanningTreeState& current_tree_state = planner_tree[idx];
                const NomdpPlanningState& current_planner_state = current_tree_state.GetValueImmutable();
                policy_graph.AddNode(PolicyGraphNode(current_planner_state));
            }
            policy_graph.AddNode(PolicyGraphNode(goal_state));
            policy_graph.ShrinkToFit();
            const int64_t goal_idx = policy_graph.GetNodesImmutable().size() - 1;
            // Second pass, add all the edges
            for (size_t idx = 0; idx < planner_tree.size(); idx++)
            {
                const NomdpPlanningTreeState& current_tree_state = planner_tree[idx];
                const NomdpPlanningState& current_planner_state = current_tree_state.GetValueImmutable();
                const int64_t parent_index = current_tree_state.GetParentIndex();
                const std::vector<int64_t>& child_indices = current_tree_state.GetChildIndices();
                if (parent_index >= 0)
                {
                    const double edge_weight = current_planner_state.GetReverseEdgePfeasibility();
                    policy_graph.AddEdgeBetweenNodes(idx, parent_index, edge_weight);
                }
                for (size_t child_index_idx = 0; child_index_idx < child_indices.size(); child_index_idx++)
                {
                    const int64_t child_index = child_indices[child_index_idx];
                    const double edge_weight = planner_tree[child_index].GetValueImmutable().GetEffectiveEdgePfeasibility();
                    policy_graph.AddEdgeBetweenNodes(idx, child_index, edge_weight);
                }
                // Detect if we are a goal state and add edges to the goal
                if (child_indices.size() == 0 && current_planner_state.GetGoalPfeasibility() > 0.0)
                {
                    const double edge_weight = current_planner_state.GetGoalPfeasibility();
                    policy_graph.AddEdgesBetweenNodes(idx, goal_idx, edge_weight);
                }
            }
            assert(policy_graph.CheckGraphLinkage());
            return policy_graph;
        }

        inline static uint32_t ComputeEstimatedEdgeAttemptCount(const PolicyGraph& graph, const GraphEdge& current_edge, const double conformant_planning_threshold, const uint32_t edge_repeat_threshold)
        {
            // First, safety checks to make sure the graph + edge are valid
            const int64_t from_index = current_edge.GetFromIndex();
            const int64_t to_index = current_edge.GetToIndex();
            assert(from_index >= 0 && from_index < graph.GetNodesImmutable().size());
            assert(to_index >= 0 && to_index < graph.GetNodesImmutable().size());
            assert(from_index != to_index);
            // Now, we estimate the number of executions of the edge necessary to reach (1) the conformant planning threshold or (2) we reach the edge repeat threshold
            // If we're going forwards
            if (from_index < to_index)
            {
                const PolicyGraphNode& from_node = graph.GetNodeImmutable(from_index);
                const NomdpPlanningState& from_node_value = from_node.GetValueImmutable();
                // Get the other child states for our action (if there are any)
                const std::vector<GraphEdge>& all_child_edges = from_node.GetOutEdgesImmutable();
                std::vector<GraphEdge> same_action_other_child_edges;
                same_action_other_child_edges.reserve(all_child_edges.size());
                for (size_t idx = 0; idx < all_child_edges.size(); idx++)
                {
                    const GraphEdge& other_child_edge = all_child_edges[idx];
                    const int64_t child_index = other_child_edge.GetToIndex();
                    const PolicyGraphNode& child_node = graph.GetNodeImmutable(child_index);
                    const NomdpPlanningState& child_node_value = child_node.GetValueImmutable();
                    // Only other child nodes with the same transition ID, but different state IDs are kept
                    if (child_node_value.GetTransitionId() == from_node_value.GetTransitionId() && child_node_value.GetStateId() != from_node_value.GetStateId())
                    {
                        same_action_other_child_edges.push_back(other_child_edge);
                    }
                }
                same_action_other_child_edges.shrink_to_fit();
                // If we aren't a child of a split, we're done
                if (same_action_other_child_edges.size() == 0)
                {
                    return 1u;
                }
                // Compute the retry count
                double percent_active = 1.0;
                double p_reached = 0.0;
                for (uint32_t try_attempt = 1; try_attempt <= edge_repeat_threshold; try_attempt++)
                {
                    // How many particles got to our state on this attempt?
                    p_reached += (percent_active * from_node_value.GetRawEdgePfeasibility());
                    // See if we've reached our threshold
                    if (p_reached >= conformant_planning_threshold)
                    {
                        return try_attempt;
                    }
                    // Update the percent of particles that are still usefully active
                    double updated_percent_active = 0.0;
                    for (size_t other_idx = 0; other_idx < same_action_other_child_edges.size(); other_idx++)
                    {
                        const GraphEdge& other_child_edge = same_action_other_child_edges[other_idx];
                        const int64_t child_index = other_child_edge.GetToIndex();
                        const PolicyGraphNode& child_node = graph.GetNodeImmutable(child_index);
                        const NomdpPlanningState& child_node_value = child_node.GetValueImmutable();
                        const double p_reached_other = percent_active * child_node_value.GetRawEdgePfeasibility();
                        const double p_returned_to_parent = p_reached_other * child_node_value.GetReverseEdgePfeasibility();
                        updated_percent_active += p_returned_to_parent;
                    }
                    percent_active = updated_percent_active;
                }
                return edge_repeat_threshold;
            }
            // If we're going backwards
            else if (from_index > to_index)
            {
                // We don't yet have a meaningful way to estimate retries of reverse actions
                return 1u;
            }
            // Can't happen
            else
            {
                return 1u;
            }
        }

        inline static PolicyGraph ComputeTrueEdgeWeights(const PolicyGraph& initial_graph, const double marginal_edge_weight, const double conformant_planning_threshold, const uint32_t edge_attempt_threshold)
        {
            PolicyGraph updated_graph = initial_graph;
            for (size_t idx = 0; idx < updated_graph.GetNodesImmutable().size(); idx++)
            {
                PolicyGraphNode& current_node = updated_graph.GetNodeMutable(idx);
                // Update all edges going out of the node
                std::vector<GraphEdge>& current_out_edges = current_node.GetOutEdgesMutable();
                for (size_t out_edge_index = 0; out_edge_index < current_out_edges.size(); out_edge_index++)
                {
                    GraphEdge& current_out_edge = current_out_edges[out_edge_index];
                    // The current edge weight is the probability of that edge
                    const double current_edge_weight = current_out_edge.GetWeight();
                    // If the edge has positive probability, we need to consider the estimated retry count of the edge
                    if (current_edge_weight > 0.0)
                    {
                        const uint32_t estimated_attempt_count = ComputeEstimatedEdgeAttemptCount(initial_graph, current_out_edge, conformant_planning_threshold, edge_attempt_threshold);
                        std::cout << "Estimated attempt count at " << estimated_attempt_count << " out of " << edge_attempt_threshold << " possible attempts" << std::endl;
                        const double edge_attempt_weight = marginal_edge_weight * (double)estimated_attempt_count;
                        const double edge_probability_weight = ((1.0 - current_edge_weight) * (1.0 - marginal_edge_weight)) + marginal_edge_weight;
                        const double new_edge_weight = edge_probability_weight * edge_attempt_weight;
                        current_out_edge.SetWeight(new_edge_weight);
                    }
                    // If the edge is zero probability (here for linkage only)
                    else
                    {
                        // We set the weight to infinity to remove it from consideration
                        const double new_edge_weight = INFINITY;
                        current_out_edge.SetWeight(new_edge_weight);
                    }
                }
            }
            return updated_graph;
        }

        inline static std::pair<PolicyGraph, std::vector<int64_t>> ComputeNodeDistances(const PolicyGraph& initial_graph, const int64_t start_index)
        {
            assert(start_index >= 0);
            assert(start_index < initial_graph.GetNodesImmutable().size());
            const std::pair<PolicyGraph, std::vector<int64_t>> search_results = SimpleDijkstrasAlgorithm<NomdpPlanningState, std::allocator<NomdpPlanningState>>::PerformDijkstrasAlgorithm(initial_graph, start_index);
            std::cout << "Previous index map: " << PrettyPrint::PrettyPrint(search_results.second) << std::endl;
            for (size_t idx = 0; idx < search_results.second.size(); idx++)
            {
                const int64_t previous_index = search_results.second[idx];
                assert(previous_index >= 0 && previous_index < search_results.second.size());
            }
            return search_results;
        }
    };

    template<typename Configuration, typename ConfigSerializer, typename AverageFn, typename DistanceFn, typename DimDistanceFn, typename ConfigAlloc=std::allocator<Configuration>>
    class ExecutionPolicy
    {
    protected:

        // Typedef so we don't hate ourselves
        typedef nomdp_planning_tools::NomdpPlannerState<Configuration, ConfigSerializer, AverageFn, DistanceFn, DimDistanceFn, ConfigAlloc> NomdpPlanningState;
        typedef GraphNode<NomdpPlanningState, std::allocator<NomdpPlanningState>> PolicyGraphNode;
        typedef Graph<NomdpPlanningState, std::allocator<NomdpPlanningState>> PolicyGraph;

        bool initialized_;
        PolicyGraph policy_graph_;
        std::vector<int64_t> previous_index_map_;

    public:

        static inline uint64_t Serialize(const ExecutionPolicy<Configuration, ConfigSerializer, AverageFn, DistanceFn, DimDistanceFn, ConfigAlloc>& policy, std::vector<uint8_t>& buffer)
        {
            return policy.SerializeSelf(buffer);
        }

        static inline std::pair<ExecutionPolicy<Configuration, ConfigSerializer, AverageFn, DistanceFn, DimDistanceFn, ConfigAlloc>, uint64_t> Deserialize(const std::vector<uint8_t>& buffer, const uint64_t current)
        {
            ExecutionPolicy<Configuration, ConfigSerializer, AverageFn, DistanceFn, DimDistanceFn, ConfigAlloc> temp_policy;
            const uint64_t bytes_read = temp_policy.DeserializeSelf(buffer, current);
            return std::make_pair(temp_policy, bytes_read);
        }

        inline ExecutionPolicy(const PolicyGraph& policy_graph, const std::vector<int64_t>& previous_index_map)
        {
            if (InitializePolicyGraph(policy_graph, previous_index_map))
            {
                initialized_ = true;
            }
            else
            {
                throw std::invalid_argument("Policy graph is not valid");
            }
        }

        inline ExecutionPolicy() : initialized_(false) {}

        uint64_t SerializeSelf(std::vector<uint8_t>& buffer) const
        {
            const uint64_t start_buffer_size = buffer.size();
            // Serialize the initialized
            arc_helpers::SerializeFixedSizePOD<uint8_t>((uint8_t)initialized_, buffer);
            // Serialize the graph
            PolicyGraph::Serialize(policy_graph_, buffer, NomdpPlanningState::Serialize);
            // Serialize the previous index map
            arc_helpers::SerializeVector<int64_t>(previous_index_map_, buffer, arc_helpers::SerializeFixedSizePOD<int64_t>);
            // Figure out how many bytes were written
            const uint64_t end_buffer_size = buffer.size();
            const uint64_t bytes_written = end_buffer_size - start_buffer_size;
            return bytes_written;
        }

        uint64_t DeserializeSelf(const std::vector<uint8_t>& buffer, const uint64_t current)
        {
            uint64_t current_position = current;
            // Deserialize the initialized
            const std::pair<uint8_t, uint64_t> initialized_deserialized = arc_helpers::DeserializeFixedSizePOD<uint8_t>(buffer, current_position);
            initialized_ = (bool)initialized_deserialized.first;
            current_position += initialized_deserialized.second;
            // Deserialize the graph
            const std::pair<PolicyGraph, uint64_t> policy_graph_deserialized = PolicyGraph::Deserialize(buffer, current_position, NomdpPlanningState::Deserialize);
            policy_graph_ = policy_graph_deserialized.first;
            current_position += policy_graph_deserialized.second;
            // Deserialize the child indices
            const std::pair<std::vector<int64_t>, uint64_t> previous_indices_deserialized = arc_helpers::DeserializeVector<int64_t>(buffer, current_position, arc_helpers::DeserializeFixedSizePOD<int64_t>);
            previous_index_map_ = previous_indices_deserialized.first;
            current_position += previous_indices_deserialized.second;
            // Figure out how many bytes were read
            const uint64_t bytes_read = current_position - current;
            return bytes_read;
        }

        inline bool IsInitialized() const
        {
            return initialized_;
        }

        inline bool InitializePolicyGraph(const PolicyGraph& policy_graph, const std::vector<int64_t>& previous_index_map)
        {
            if (policy_graph.CheckGraphLinkage())
            {
                if (policy_graph.GetNodesImmutable().size() == previous_index_map.size())
                {
                    policy_graph_ = policy_graph;
                    previous_index_map_ = previous_index_map;
                    return true;
                }
                else
                {
                    std::cerr << "Provided policy graph is a different size than the provided previous index map" << std::endl;
                    return false;
                }
            }
            else
            {
                std::cerr << "Provided policy graph has invalid linkage" << std::endl;
                return false;
            }
        }

        inline const PolicyGraph& GetRawPolicy() const
        {
            assert(initialized_);
            return policy_graph_;
        }

        inline const std::vector<int64_t>& GetRawPreviousIndexMap() const
        {
            assert(initialized_);
            return previous_index_map_;
        }

        inline Configuration QueryBestAction(const Configuration& current_config, const std::function<double(const Configuration&, const NomdpPlanningState&)>& distance_fn)
        {
            assert(initialized_);
            // Search the policy graph for the best-match node
            int64_t current_best_index = -1;
            double current_best_distance = INFINITY;
            for (size_t idx = 0; idx < policy_graph_.GetNodesImmutable().size(); idx++)
            {
                const PolicyGraphNode& current_node = policy_graph_.GetNodeImmutable(idx);
                const NomdpPlanningState& current_node_value = current_node.GetValueImmutable();
                const double state_distance = distance_fn(current_config, current_node_value);
                if (state_distance < current_best_distance)
                {
                    current_best_distance = state_distance;
                    current_best_index = idx;
                }
            }
            const int64_t best_index = current_best_index;
            const double best_distance = current_best_distance;
            assert(best_index >= 0);
            assert(best_distance < INFINITY);
            // Get the best node
            const PolicyGraphNode& best_node = policy_graph_.GetNodeImmutable(best_index);
            const NomdpPlanningState& best_node_state = best_node.GetValueImmutable();
            // Get the previous node, as indicated by Dijkstra's algorithm
            const int64_t previous_node_index = previous_index_map_[best_index];
            if (previous_node_index >= 0)
            {
                const PolicyGraphNode& previous_node = policy_graph_.GetNodeImmutable(previous_node_index);
                const NomdpPlanningState& previous_node_state = previous_node.GetValueImmutable();
                // Figure out the correct action to take
                const uint64_t best_node_state_id = best_node_state.GetStateId();
                const uint64_t previous_node_state_id = previous_node_state.GetStateId();
                // If the "previous" node that we want to go to is a downstream state, we get the action of the downstream state
                if (best_node_state_id < previous_node_state_id)
                {
                    return previous_node_state.GetCommand();
                }
                // If the "previous" node that we want to go to is an upstream state, we get the expectation of the upstream state
                else if (previous_node_state_id < best_node_state_id)
                {
                    return previous_node_state.GetExpectation();
                }
                // The closest node is the goal
                else
                {
                    return best_node_state.GetExpectation();
                }
            }
            else
            {
                throw std::invalid_argument("This should be impossible");
            }
        }
    };
}

template<typename Configuration, typename ConfigSerializer, typename AverageFn, typename DistanceFn, typename DimDistanceFn, typename ConfigAlloc=std::allocator<Configuration>>
std::ostream& operator<<(std::ostream& strm, const execution_policy::ExecutionPolicy<Configuration, ConfigSerializer, AverageFn, DimDistanceFn, ConfigAlloc>& policy)
{
    const std::vector<simple_rrt_planner::SimpleRRTPlannerState<nomdp_planning_tools::NomdpPlannerState<Configuration, ConfigSerializer, AverageFn, DistanceFn, DimDistanceFn, ConfigAlloc>>>& raw_policy_tree = policy.GetRawPolicy();
    strm << "Execution Policy - Policy: ";
    for (size_t idx = 0; idx < raw_policy_tree.size(); idx++)
    {
        const simple_rrt_planner::SimpleRRTPlannerState<nomdp_planning_tools::NomdpPlannerState<Configuration, ConfigSerializer, AverageFn, DistanceFn, DimDistanceFn, ConfigAlloc>>& policy_tree_state = raw_policy_tree[idx];
        const int64_t parent_index = policy_tree_state.GetParentIndex();
        const std::vector<int64_t>& child_indices = policy_tree_state.GetChildIndices();
        const nomdp_planning_tools::NomdpPlannerState<Configuration, ConfigSerializer, AverageFn, DistanceFn, DimDistanceFn, ConfigAlloc>& policy_state = policy_tree_state.GetValueImmutable();
        strm << "\nState # " << idx << " with parent " << parent_index << " and children " << PrettyPrint::PrettyPrint(child_indices, true) << " - value: " << PrettyPrint::PrettyPrint(policy_state);
    }
    return strm;
}

#endif // EXECUTION_POLICY_HPP
