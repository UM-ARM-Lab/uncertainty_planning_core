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
#include <uncertainty_planning_core/uncertainty_planner_state.hpp>

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

        inline GraphEdge(const int64_t from_index, const int64_t to_index, const double weight) : from_index_(from_index), to_index_(to_index), weight_(weight) {}

        inline GraphEdge() : from_index_(-1), to_index_(-1), weight_(0.0) {}

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
            const std::pair<int64_t, uint64_t> deserialized_from_index = arc_helpers::DeserializeFixedSizePOD<int64_t>(buffer, current_position);
            from_index_ = deserialized_from_index.first;
            current_position += deserialized_from_index.second;
            const std::pair<int64_t, uint64_t> deserialized_to_index = arc_helpers::DeserializeFixedSizePOD<int64_t>(buffer, current_position);
            to_index_ = deserialized_to_index.first;
            current_position += deserialized_to_index.second;
            const std::pair<double, uint64_t> deserialized_weight = arc_helpers::DeserializeFixedSizePOD<double>(buffer, current_position);
            weight_ = deserialized_weight.first;
            current_position += deserialized_weight.second;
            // Figure out how many bytes were read
            const uint64_t bytes_read = current_position - current;
            return bytes_read;
        }

        inline std::string Print() const
        {
            return std::string("(" + std::to_string(from_index_) + "->" + std::to_string(to_index_) + ") : " + std::to_string(weight_));
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

        inline GraphNode(const T& value, const double distance, const std::vector<GraphEdge>& new_in_edges, const std::vector<GraphEdge>& new_out_edges) : value_(value), distance_(distance), in_edges_(new_in_edges), out_edges_(new_out_edges) {}

        inline GraphNode(const T& value) : value_(value), distance_(INFINITY) {}

        inline GraphNode() : distance_(INFINITY) {}

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
            assert(current < buffer.size());
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

        inline std::string Print() const
        {
            std::ostringstream strm;
            strm << "Node : " << distance_ << " In Edges : ";
            if (in_edges_.size() > 0)
            {
                strm << in_edges_[0].Print();
                for (size_t idx = 1; idx < in_edges_.size(); idx++)
                {
                    strm << ", " << in_edges_[idx].Print();
                }
            }
            strm << " Out Edges : ";
            if (out_edges_.size() > 0)
            {
                strm << out_edges_[0].Print();
                for (size_t idx = 1; idx < out_edges_.size(); idx++)
                {
                    strm << ", " << out_edges_[idx].Print();
                }
            }
            return strm.str();
        }

        inline const T& GetValueImmutable() const
        {
            return value_;
        }

        inline T& GetValueMutable()
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

        inline Graph(const std::vector<GraphNode<T, Allocator>>& nodes)
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

        inline Graph(const size_t expected_size)
        {
            nodes_.reserve(expected_size);
        }

        inline Graph() {}

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

        inline std::string Print() const
        {
            std::ostringstream strm;
            strm << "Graph - Nodes : ";
            if (nodes_.size() > 0)
            {
                strm << nodes_[0].Print();
                for (size_t idx = 1; idx < nodes_.size(); idx++)
                {
                    strm << "\n" << nodes_[idx].Print();
                }
            }
            return strm.str();
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
                // Check the in edges first
                const std::vector<GraphEdge>& in_edges = current_node.GetInEdgesImmutable();
                for (size_t in_edge_idx = 0; in_edge_idx < in_edges.size(); in_edge_idx++)
                {
                    const GraphEdge& current_edge = in_edges[in_edge_idx];
                    // Check from index to make sure it's in bounds
                    const int64_t from_index = current_edge.GetFromIndex();
                    if (from_index < 0 || from_index >= (int64_t)nodes.size())
                    {
                        return false;
                    }
                    // Check to index to make sure it matches our own index
                    const int64_t to_index = current_edge.GetToIndex();
                    if (to_index != (int64_t)idx)
                    {
                        return false;
                    }
                    // Check edge validity (edges to ourself are not allowed)
                    if (from_index == to_index)
                    {
                        return false;
                    }
                    // Check to make sure that the from index node is linked to us
                    const GraphNode<T, Allocator>& from_node = nodes[(size_t)from_index];
                    const std::vector<GraphEdge>& from_node_out_edges = from_node.GetOutEdgesImmutable();
                    bool from_node_connection_valid = false;
                    // Make sure at least one out edge of the from index node corresponds to the current node
                    for (size_t from_node_out_edge_idx = 0; from_node_out_edge_idx < from_node_out_edges.size(); from_node_out_edge_idx++)
                    {
                        const GraphEdge& current_from_node_out_edge = from_node_out_edges[from_node_out_edge_idx];
                        if (current_from_node_out_edge.GetToIndex() == (int64_t)idx)
                        {
                            from_node_connection_valid = true;
                        }
                    }
                    if (from_node_connection_valid == false)
                    {
                        return false;
                    }
                }
                // Check the out edges second
                const std::vector<GraphEdge>& out_edges = current_node.GetOutEdgesImmutable();
                for (size_t out_edge_idx = 0; out_edge_idx < out_edges.size(); out_edge_idx++)
                {
                    const GraphEdge& current_edge = out_edges[out_edge_idx];
                    // Check from index to make sure it matches our own index
                    const int64_t from_index = current_edge.GetFromIndex();
                    if (from_index != (int64_t)idx)
                    {
                        return false;
                    }
                    // Check to index to make sure it's in bounds
                    const int64_t to_index = current_edge.GetToIndex();
                    if (to_index < 0 || to_index >= (int64_t)nodes.size())
                    {
                        return false;
                    }
                    // Check edge validity (edges to ourself are not allowed)
                    if (from_index == to_index)
                    {
                        return false;
                    }
                    // Check to make sure that the to index node is linked to us
                    const GraphNode<T, Allocator>& to_node = nodes[(size_t)to_index];
                    const std::vector<GraphEdge>& to_node_in_edges = to_node.GetInEdgesImmutable();
                    bool to_node_connection_valid = false;
                    // Make sure at least one in edge of the to index node corresponds to the current node
                    for (size_t to_node_in_edge_idx = 0; to_node_in_edge_idx < to_node_in_edges.size(); to_node_in_edge_idx++)
                    {
                        const GraphEdge& current_to_node_in_edge = to_node_in_edges[to_node_in_edge_idx];
                        if (current_to_node_in_edge.GetFromIndex() == (int64_t)idx)
                        {
                            to_node_connection_valid = true;
                        }
                    }
                    if (to_node_connection_valid == false)
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
            assert(index < (int64_t)nodes_.size());
            return nodes_[(size_t)index];
        }

        inline GraphNode<T, Allocator>& GetNodeMutable(const int64_t index)
        {
            assert(index >= 0);
            assert(index < (int64_t)nodes_.size());
            return nodes_[(size_t)index];
        }

        inline int64_t AddNode(const GraphNode<T, Allocator>& new_node)
        {
            nodes_.push_back(new_node);
            return (int64_t)(nodes_.size() - 1);
        }

        inline void AddEdgeBetweenNodes(const int64_t from_index, const int64_t to_index, const double edge_weight)
        {
            assert(from_index >= 0);
            assert(from_index < (int64_t)nodes_.size());
            assert(to_index >= 0);
            assert(to_index < (int64_t)nodes_.size());
            assert(from_index != to_index);
            const GraphEdge new_edge(from_index, to_index, edge_weight);
            GetNodeMutable(from_index).AddOutEdge(new_edge);
            GetNodeMutable(to_index).AddInEdge(new_edge);
        }

        inline void AddEdgesBetweenNodes(const int64_t first_index, const int64_t second_index, const double edge_weight)
        {
            assert(first_index >= 0);
            assert(first_index < (int64_t)nodes_.size());
            assert(second_index >= 0);
            assert(second_index < (int64_t)nodes_.size());
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

        inline static std::pair<Graph<T, Allocator>, std::pair<std::vector<int64_t>, std::vector<double>>> PerformDijkstrasAlgorithm(const Graph<T, Allocator>& graph, const int64_t start_index)
        {
            assert(start_index >= (int64_t)0);
            assert(start_index < (int64_t)graph.GetNodesImmutable().size());
            Graph<T, Allocator> working_copy = graph;
            // Setup
            std::vector<int64_t> previous_index_map(working_copy.GetNodesImmutable().size(), -1);
            std::vector<double> distances(working_copy.GetNodesImmutable().size(), INFINITY);
            std::priority_queue<std::pair<int64_t, double>, std::vector<std::pair<int64_t, double>>, CompareIndexFn> queue;
            std::unordered_map<int64_t, uint32_t> explored(working_copy.GetNodesImmutable().size());
            for (size_t idx = 0; idx < working_copy.GetNodesImmutable().size(); idx++)
            {
                working_copy.GetNodeMutable((int64_t)idx).SetDistance(INFINITY);
                queue.push(std::make_pair((int64_t)idx, INFINITY));
            }
            working_copy.GetNodeMutable(start_index).SetDistance(0.0);
            previous_index_map[(size_t)start_index] = start_index;
            distances[(size_t)start_index] = 0.0;
            queue.push(std::make_pair(start_index, 0.0));
            while (queue.size() > 0)
            {
                const std::pair<int64_t, double> top_node = queue.top();
                const int64_t& top_node_index = top_node.first;
                const double& top_node_distance = top_node.second;
                queue.pop();
                if (explored[top_node_index] > 0)
                {
                    // We've already been here
                    continue;
                }
                else
                {
                    // Note that we've been here
                    explored[top_node_index] = 1;
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
                            previous_index_map[(size_t)neighbor_index] = top_node_index;
                            distances[(size_t)neighbor_index] = new_neighbor_distance;
                        }
                        else
                        {
                            // Do nothing
                            continue;
                        }
                    }
                }
            }
            return std::make_pair(working_copy, std::make_pair(previous_index_map, distances));
        }
    };

    template<typename Configuration, typename ConfigSerializer, typename AverageFn, typename DistanceFn, typename DimDistanceFn, typename ConfigAlloc=std::allocator<Configuration>>
    class PolicyGraphBuilder
    {
    private:

        // Typedef so we don't hate ourselves
        typedef uncertainty_planning_tools::UncertaintyPlannerState<Configuration, ConfigSerializer, AverageFn, DistanceFn, DimDistanceFn, ConfigAlloc> NomdpPlanningState;
        typedef simple_rrt_planner::SimpleRRTPlannerState<NomdpPlanningState, std::allocator<NomdpPlanningState>> NomdpPlanningTreeState;
        typedef std::vector<NomdpPlanningTreeState> NomdpPlanningTree;
        typedef GraphNode<NomdpPlanningState, std::allocator<NomdpPlanningState>> PolicyGraphNode;
        typedef Graph<NomdpPlanningState, std::allocator<NomdpPlanningState>> PolicyGraph;

        PolicyGraphBuilder() {}

    public:

        inline static PolicyGraph BuildPolicyGraphFromPlannerTree(const NomdpPlanningTree& planner_tree, const NomdpPlanningState& goal_state)
        {
            assert(simple_rrt_planner::SimpleHybridRRTPlanner::CheckTreeLinkage(planner_tree));
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
            // We just added the goal as the last node, so we know it has the last index
            const int64_t goal_idx = (int64_t)policy_graph.GetNodesImmutable().size() - 1;
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
                    policy_graph.AddEdgeBetweenNodes((int64_t)idx, parent_index, edge_weight);
                }
                for (size_t child_index_idx = 0; child_index_idx < child_indices.size(); child_index_idx++)
                {
                    const int64_t child_index = child_indices[child_index_idx];
                    const double edge_weight = planner_tree[(size_t)child_index].GetValueImmutable().GetEffectiveEdgePfeasibility();
                    policy_graph.AddEdgeBetweenNodes((int64_t)idx, child_index, edge_weight);
                }
                // Detect if we are a goal state and add edges to the goal
                if (child_indices.size() == 0 && current_planner_state.GetGoalPfeasibility() > 0.0)
                {
                    const double edge_weight = current_planner_state.GetGoalPfeasibility();
                    policy_graph.AddEdgesBetweenNodes((int64_t)idx, goal_idx, edge_weight);
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
            assert(from_index >= 0 && from_index < (int64_t)graph.GetNodesImmutable().size());
            assert(to_index >= 0 && to_index < (int64_t)graph.GetNodesImmutable().size());
            assert(from_index != to_index);
            // Now, we estimate the number of executions of the edge necessary to reach (1) the conformant planning threshold or (2) we reach the edge repeat threshold
            // If we're going forwards
            if (from_index < to_index)
            {
                const PolicyGraphNode& from_node = graph.GetNodeImmutable(from_index);
                const PolicyGraphNode& to_node = graph.GetNodeImmutable(to_index);
                const NomdpPlanningState& to_node_value = to_node.GetValueImmutable();
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
                    if (child_node_value.GetTransitionId() == to_node_value.GetTransitionId() && child_node_value.GetStateId() != to_node_value.GetStateId())
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
                    p_reached += (percent_active * to_node_value.GetRawEdgePfeasibility());
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
                assert(false);
            }
        }

        inline static PolicyGraph ComputeTrueEdgeWeights(const PolicyGraph& initial_graph, const double marginal_edge_weight, const double conformant_planning_threshold, const uint32_t edge_attempt_threshold)
        {
            PolicyGraph updated_graph = initial_graph;
            for (size_t idx = 0; idx < updated_graph.GetNodesImmutable().size(); idx++)
            {
                PolicyGraphNode& current_node = updated_graph.GetNodeMutable((int64_t)idx);
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
                        const uint32_t estimated_attempt_count = ComputeEstimatedEdgeAttemptCount(updated_graph, current_out_edge, conformant_planning_threshold, edge_attempt_threshold);
                        //std::cout << "Estimated attempt count at " << estimated_attempt_count << " out of " << edge_attempt_threshold << " possible attempts" << std::endl;
                        //std::cout << "Current edge probability " << current_edge_weight << std::endl;
                        const double edge_probability_weight = (current_edge_weight >= std::numeric_limits<double>::epsilon()) ? 1.0 / current_edge_weight : INFINITY;
                        const double edge_attempt_weight = marginal_edge_weight * (double)estimated_attempt_count;
                        const double new_edge_weight = edge_probability_weight * edge_attempt_weight;
                        //std::cout << "Assigned new edge weight " << new_edge_weight << " from 1/P(edge) " << edge_probability_weight << " and edge attempt weight " << edge_attempt_weight << std::endl;
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
                // Update all edges going into the node
                std::vector<GraphEdge>& current_in_edges = current_node.GetInEdgesMutable();
                for (size_t out_edge_index = 0; out_edge_index < current_in_edges.size(); out_edge_index++)
                {
                    GraphEdge& current_in_edge = current_in_edges[out_edge_index];
                    // The current edge weight is the probability of that edge
                    const double current_edge_weight = current_in_edge.GetWeight();
                    // If the edge has positive probability, we need to consider the estimated retry count of the edge
                    if (current_edge_weight > 0.0)
                    {
                        const uint32_t estimated_attempt_count = ComputeEstimatedEdgeAttemptCount(updated_graph, current_in_edge, conformant_planning_threshold, edge_attempt_threshold);
                        //std::cout << "Estimated attempt count at " << estimated_attempt_count << " out of " << edge_attempt_threshold << " possible attempts" << std::endl;
                        //std::cout << "Current edge probability " << current_edge_weight << std::endl;
                        const double edge_probability_weight = (current_edge_weight >= std::numeric_limits<double>::epsilon()) ? 1.0 / current_edge_weight : INFINITY;
                        const double edge_attempt_weight = marginal_edge_weight * (double)estimated_attempt_count;
                        const double new_edge_weight = edge_probability_weight * edge_attempt_weight;
                        //std::cout << "Assigned new edge weight " << new_edge_weight << " from 1/P(edge) " << edge_probability_weight << " and edge attempt weight " << edge_attempt_weight << std::endl;
                        current_in_edge.SetWeight(new_edge_weight);
                    }
                    // If the edge is zero probability (here for linkage only)
                    else
                    {
                        // We set the weight to infinity to remove it from consideration
                        const double new_edge_weight = INFINITY;
                        current_in_edge.SetWeight(new_edge_weight);
                    }
                }
            }
            return updated_graph;
        }

        inline static std::pair<PolicyGraph, std::pair<std::vector<int64_t>, std::vector<double>>> ComputeNodeDistances(const PolicyGraph& initial_graph, const int64_t start_index)
        {
            assert(start_index >= 0);
            assert(start_index < (int64_t)initial_graph.GetNodesImmutable().size());
            const std::pair<PolicyGraph, std::pair<std::vector<int64_t>, std::vector<double>>> complete_search_results = SimpleDijkstrasAlgorithm<NomdpPlanningState, std::allocator<NomdpPlanningState>>::PerformDijkstrasAlgorithm(initial_graph, start_index);
            //std::cout << "Previous index map: " << PrettyPrint::PrettyPrint(search_results.second) << std::endl;
            for (size_t idx = 0; idx < complete_search_results.second.first.size(); idx++)
            {
                const int64_t previous_index = complete_search_results.second.first[idx];
                assert(previous_index >= 0 && previous_index < (int64_t)complete_search_results.second.first.size());
            }
            return complete_search_results;
        }
    };

    template<typename Configuration, typename ConfigSerializer, typename AverageFn, typename DistanceFn, typename DimDistanceFn, typename ConfigAlloc=std::allocator<Configuration>>
    class ExecutionPolicy
    {
    protected:

        // Typedef so we don't hate ourselves
        typedef uncertainty_planning_tools::UncertaintyPlannerState<Configuration, ConfigSerializer, AverageFn, DistanceFn, DimDistanceFn, ConfigAlloc> NomdpPlanningState;
        typedef simple_rrt_planner::SimpleRRTPlannerState<NomdpPlanningState, std::allocator<NomdpPlanningState>> NomdpPlanningTreeState;
        typedef std::vector<NomdpPlanningTreeState> NomdpPlanningTree;
        typedef GraphNode<NomdpPlanningState, std::allocator<NomdpPlanningState>> PolicyGraphNode;
        typedef Graph<NomdpPlanningState, std::allocator<NomdpPlanningState>> PolicyGraph;
        typedef PolicyGraphBuilder<Configuration, ConfigSerializer, AverageFn, DistanceFn, DimDistanceFn, ConfigAlloc> ExecutionPolicyGraphBuilder;

        bool initialized_;
        // Raw data used to rebuild the policy graph
        NomdpPlanningTree planner_tree_;
        Configuration goal_;
        double marginal_edge_weight_;
        double conformant_planning_threshold_;
        uint32_t edge_attempt_threshold_;
        uint32_t policy_action_attempt_count_;
        // Actual policy graph
        PolicyGraph policy_graph_;
        std::vector<int64_t> previous_index_map_;

    public:

        static inline uint32_t AddWithOverflowClamp(const uint32_t original, const uint32_t additional)
        {
            if (additional == 0u)
            {
                return original;
            }
            if ((original + additional) <= original)
            {
                std::cout << "@@@ WARNING - CLAMPING ON OVERFLOW OF UINT32_T @@@" << std::endl;
                return std::numeric_limits<uint32_t>::max();
            }
            else
            {
                return original + additional;
            }
        }

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

        inline ExecutionPolicy(const NomdpPlanningTree& planner_tree, const Configuration& goal, const double marginal_edge_weight, const double conformant_planning_threshold, const uint32_t edge_attempt_threshold, const uint32_t policy_action_attempt_count) : initialized_(true), planner_tree_(planner_tree), goal_(goal), marginal_edge_weight_(marginal_edge_weight), conformant_planning_threshold_(conformant_planning_threshold), edge_attempt_threshold_(edge_attempt_threshold), policy_action_attempt_count_(policy_action_attempt_count)
        {
            RebuildPolicyGraph();
        }

        inline ExecutionPolicy() : initialized_(false) {}

        inline std::pair<PolicyGraph, std::vector<int64_t>> BuildPolicyGraphComponentsFromTree(const NomdpPlanningTree& planner_tree, const Configuration& goal, const double marginal_edge_weight, const double conformant_planning_threshold, const uint32_t edge_attempt_threshold) const
        {
            const PolicyGraph preliminary_policy_graph = ExecutionPolicyGraphBuilder::BuildPolicyGraphFromPlannerTree(planner_tree, NomdpPlanningState(goal));
            //std::cout << "Preliminary graph " << preliminary_policy_graph.Print() << std::endl;
            //std::cout << "Policy graph has " << preliminary_policy_graph.GetNodesImmutable().size() << " graph nodes" << std::endl;
            const PolicyGraph intermediate_policy_graph = ExecutionPolicyGraphBuilder::ComputeTrueEdgeWeights(preliminary_policy_graph, marginal_edge_weight, conformant_planning_threshold, edge_attempt_threshold);
            //std::cout << "Intermediate graph " << intermediate_policy_graph.Print() << std::endl;
            //std::cout << "Computed true edge weights for " << intermediate_policy_graph.GetNodesImmutable().size() << " policy graph nodes" << std::endl;
            const std::pair<PolicyGraph, std::pair<std::vector<int64_t>, std::vector<double>>> processed_policy_graph_components = ExecutionPolicyGraphBuilder::ComputeNodeDistances(intermediate_policy_graph, (int64_t)intermediate_policy_graph.GetNodesImmutable().size() - 1);
            //std::cout << "Processed policy graph into graph with " << processed_policy_graph_components.first.GetNodesImmutable().size() << " policy nodes and previous index map with " << processed_policy_graph_components.second.size() << " entries" << std::endl;
            //std::cout << "(Re)Built policy graph from planner tree\n" << PrintTree(planner_tree, processed_policy_graph_components.second.first, processed_policy_graph_components.second.second) << std::endl;
            return std::make_pair(processed_policy_graph_components.first, processed_policy_graph_components.second.first);
        }

        inline std::string PrintTree(const NomdpPlanningTree& planning_tree, const std::vector<int64_t>& previous_index_map, const std::vector<double>& dijkstras_distances) const
        {
            assert(planning_tree.size() > 1);
            std::ostringstream strm;
            strm << "Planning tree with " << planning_tree.size() << " states:";
            for (size_t idx = 1; idx < planning_tree.size(); idx++)
            {
                const int64_t previous_index = previous_index_map[idx];
                const double distance = dijkstras_distances[idx];
                const NomdpPlanningTreeState& current_tree_state = planning_tree[idx];
                const int64_t parent_index = current_tree_state.GetParentIndex();
                const NomdpPlanningState& current_state = current_tree_state.GetValueImmutable();
                const double raw_edge_probability = current_state.GetRawEdgePfeasibility();
                const double effective_edge_probability = current_state.GetEffectiveEdgePfeasibility();
                const double reverse_edge_probability = current_state.GetReverseEdgePfeasibility();
                const double goal_proability = current_state.GetGoalPfeasibility();
                strm << "\nState " << idx << " with P(" << parent_index << "->" << idx << ") = " << raw_edge_probability << "/" << effective_edge_probability << " [raw/effective] and P(" << idx << "->" << parent_index << ") = " << reverse_edge_probability << " and P(->goal) = " << goal_proability << " and Previous = ";
                if (previous_index == planning_tree.size())
                {
                    strm << "(goal) with distance = " << distance;
                }
                else
                {
                    strm << previous_index << " with distance = " << distance;
                }
            }
            return strm.str();
        }

        inline void RebuildPolicyGraph()
        {
            const std::pair<PolicyGraph, std::vector<int64_t>> processed_policy_graph_components = BuildPolicyGraphComponentsFromTree(planner_tree_, goal_, marginal_edge_weight_, conformant_planning_threshold_, edge_attempt_threshold_);
            policy_graph_ = processed_policy_graph_components.first;
            previous_index_map_ = processed_policy_graph_components.second;
        }

        inline uint64_t SerializeSelf(std::vector<uint8_t>& buffer) const
        {
            const uint64_t start_buffer_size = buffer.size();
            // Serialize the initialized
            arc_helpers::SerializeFixedSizePOD<uint8_t>((uint8_t)initialized_, buffer);
            // Serialize the planner tree
            std::function<uint64_t(const NomdpPlanningTreeState&, std::vector<uint8_t>&)> planning_tree_state_serializer_fn = [] (const NomdpPlanningTreeState& state, std::vector<uint8_t>& buffer) { return NomdpPlanningTreeState::Serialize(state, buffer, NomdpPlanningState::Serialize); };
            arc_helpers::SerializeVector(planner_tree_, buffer, planning_tree_state_serializer_fn);
            // Serialize the goal
            ConfigSerializer::Serialize(goal_, buffer);
            // Serialize the marginal edge weight
            arc_helpers::SerializeFixedSizePOD<double>(marginal_edge_weight_, buffer);
            // Serialize the conformant planning threshold
            arc_helpers::SerializeFixedSizePOD<double>(conformant_planning_threshold_, buffer);
            // Serialize the edge attempt threshold
            arc_helpers::SerializeFixedSizePOD<uint32_t>(edge_attempt_threshold_, buffer);
            // Serialize the policy action attempt count
            arc_helpers::SerializeFixedSizePOD<uint32_t>(policy_action_attempt_count_, buffer);
            // Figure out how many bytes were written
            const uint64_t end_buffer_size = buffer.size();
            const uint64_t bytes_written = end_buffer_size - start_buffer_size;
            return bytes_written;
        }

        inline uint64_t DeserializeSelf(const std::vector<uint8_t>& buffer, const uint64_t current)
        {
            uint64_t current_position = current;
            // Deserialize the initialized
            const std::pair<uint8_t, uint64_t> initialized_deserialized = arc_helpers::DeserializeFixedSizePOD<uint8_t>(buffer, current_position);
            initialized_ = (bool)initialized_deserialized.first;
            current_position += initialized_deserialized.second;
            // Deserialize the planner tree
            std::function<std::pair<NomdpPlanningTreeState, uint64_t>(const std::vector<uint8_t>&, const uint64_t)> planning_tree_state_deserializer_fn = [] (const std::vector<uint8_t>& buffer, const uint64_t current) { return NomdpPlanningTreeState::Deserialize(buffer, current, NomdpPlanningState::Deserialize); };
            const std::pair<NomdpPlanningTree, uint64_t> planner_tree_deserialized = arc_helpers::DeserializeVector<NomdpPlanningTreeState>(buffer, current_position, planning_tree_state_deserializer_fn);
            planner_tree_ = planner_tree_deserialized.first;
            current_position += planner_tree_deserialized.second;
            // Deserialize the goal
            const std::pair<Configuration, uint64_t> goal_deserialized = ConfigSerializer::Deserialize(buffer, current_position);
            goal_ = goal_deserialized.first;
            current_position += goal_deserialized.second;
            // Deserialize the marginal edge weight
            const std::pair<double, uint64_t> marginal_edge_weight_deserialized = arc_helpers::DeserializeFixedSizePOD<double>(buffer, current_position);
            marginal_edge_weight_ = marginal_edge_weight_deserialized.first;
            current_position += marginal_edge_weight_deserialized.second;
            // Deserialize the conformant planning threshold
            const std::pair<double, uint64_t> conformant_planning_threshold_deserialized = arc_helpers::DeserializeFixedSizePOD<double>(buffer, current_position);
            conformant_planning_threshold_ = conformant_planning_threshold_deserialized.first;
            current_position += conformant_planning_threshold_deserialized.second;
            // Deserialize the edge attempt threshold
            const std::pair<uint32_t, uint64_t> edge_attempt_threshold_deserialized = arc_helpers::DeserializeFixedSizePOD<uint32_t>(buffer, current_position);
            edge_attempt_threshold_ = edge_attempt_threshold_deserialized.first;
            current_position += edge_attempt_threshold_deserialized.second;
            // Deserialize the policy action attempt count
            const std::pair<uint32_t, uint64_t> policy_action_attempt_count_deserialized = arc_helpers::DeserializeFixedSizePOD<uint32_t>(buffer, current_position);
            policy_action_attempt_count_ = policy_action_attempt_count_deserialized.first;
            current_position += policy_action_attempt_count_deserialized.second;
            // Rebuild the policy graph
            RebuildPolicyGraph();
            // Figure out how many bytes were read
            const uint64_t bytes_read = current_position - current;
            return bytes_read;
        }

        inline bool IsInitialized() const
        {
            return initialized_;
        }

        inline const NomdpPlanningTree& GetRawPlannerTree() const
        {
            assert(initialized_);
            return planner_tree_;
        }

        inline const Configuration& GetRawGoalConfiguration() const
        {
            assert(initialized_);
            return goal_;
        }

        inline double GetMarginalEdgeWeight() const
        {
            return marginal_edge_weight_;
        }

        inline double GetConformantPlanningThreshold() const
        {
            return conformant_planning_threshold_;
        }

        inline uint32_t GetEdgeAttemptThreshold() const
        {
            return edge_attempt_threshold_;
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

        inline uint32_t GetPolicyActionAttemptCount() const
        {
            return policy_action_attempt_count_;
        }

        inline void SetPolicyActionAttemptCount(const uint32_t new_count)
        {
            policy_action_attempt_count_ = new_count;
        }

        inline std::pair<std::pair<int64_t, uint64_t>, std::pair<Configuration, Configuration>> QueryNextAction(const int64_t current_state_index) const
        {
            assert(current_state_index >= 0);
            assert(current_state_index < (int64_t)previous_index_map_.size());
            const PolicyGraphNode& result_state_policy_node = policy_graph_.GetNodeImmutable(current_state_index);
            const NomdpPlanningState& result_state = result_state_policy_node.GetValueImmutable();
            // Get the action to take
            // Get the previous node, as indicated by Dijkstra's algorithm
            const int64_t target_state_index = previous_index_map_[(size_t)current_state_index];
            if (target_state_index < 0)
            {
                throw std::invalid_argument("Policy no longer has a solution");
            }
            const PolicyGraphNode& target_state_policy_node = policy_graph_.GetNodeImmutable(target_state_index);
            const NomdpPlanningState& target_state = target_state_policy_node.GetValueImmutable();
            // Figure out the correct action to take
            const uint64_t result_state_id = result_state.GetStateId();
            const uint64_t target_state_id = target_state.GetStateId();
            //std::cout << "==========\nGenerating next action to go from current state " << result_state_id << " to target state " << target_state_id << std::endl;
            // If the "previous" node that we want to go to is a downstream state, we get the action of the downstream state
            if (result_state_id < target_state_id)
            {
                std::cout << "Returning forward action for current state " << current_state_index << ", transition ID " << target_state.GetTransitionId() << std::endl;
                return std::make_pair(std::make_pair(current_state_index, target_state.GetTransitionId()), std::make_pair(target_state.GetCommand(), target_state.GetExpectation()));
            }
            // If the "previous" node that we want to go to is an upstream state, we get the expectation of the upstream state
            else if (target_state_id < result_state_id)
            {
                std::cout << "Returning reverse action for current state " << current_state_index << ", transition ID " << result_state.GetReverseTransitionId() << std::endl;
                return std::make_pair(std::make_pair(current_state_index, result_state.GetReverseTransitionId()), std::make_pair(target_state.GetExpectation(), target_state.GetExpectation()));
            }
            else
            {
                assert(false);
            }
        }

        inline std::pair<std::pair<int64_t, uint64_t>, std::pair<Configuration, Configuration>> QueryBestAction(const uint64_t performed_transition_id, const Configuration& current_config, const std::function<std::vector<std::vector<size_t>>(const std::vector<Configuration, ConfigAlloc>&, const Configuration&)>& particle_clustering_fn)
        {
            assert(initialized_);
            // If we're just starting out
            if (performed_transition_id == 0)
            {
                return QueryStartBestAction(current_config, particle_clustering_fn);
            }
            else
            {
                return QueryNormalBestAction(performed_transition_id, current_config, particle_clustering_fn);
            }
        }

        inline std::pair<std::pair<int64_t, uint64_t>, std::pair<Configuration, Configuration>> QueryStartBestAction(const Configuration& current_config, const std::function<std::vector<std::vector<size_t>>(const std::vector<Configuration, ConfigAlloc>&, const Configuration&)>& particle_clustering_fn)
        {
            assert(initialized_);
            // Get the starting state
            const PolicyGraphNode& first_node = policy_graph_.GetNodeImmutable(0);
            const NomdpPlanningState& first_node_state = first_node.GetValueImmutable();
            const Configuration first_node_config = first_node_state.GetExpectation();
            // Make sure we are close enough to the start state
            const std::vector<std::vector<size_t>> start_check_clusters = particle_clustering_fn(first_node_state.GetParticlePositionsImmutable().first, current_config);
            // If we're close enough to the start
            if (start_check_clusters.size() == 1)
            {
                return QueryNextAction(0);
            }
            // If not
            else
            {
                // Return the start state
                return std::make_pair(std::make_pair(-1, 0), std::make_pair(first_node_config, first_node_config));
            }
        }

        inline std::pair<std::pair<int64_t, uint64_t>, std::pair<Configuration, Configuration>> QueryNormalBestAction(const uint64_t performed_transition_id, const Configuration& current_config, const std::function<std::vector<std::vector<size_t>>(const std::vector<Configuration, ConfigAlloc>&, const Configuration&)>& particle_clustering_fn)
        {
            std::cout << "++++++++++\nQuerying the policy with performed transition ID " << performed_transition_id << "..." << std::endl;
            assert(performed_transition_id > 0);
            // Collect the possible states that could have resulted from the transition we just performed
            std::vector<std::pair<int64_t, bool>> expected_possible_result_states;
            int64_t previous_state_index = -1;
            // Go through the entire tree and retrieve all states with matching transition IDs
            for (int64_t idx = 0; idx < (int64_t)planner_tree_.size(); idx++)
            {
                const NomdpPlanningTreeState& candidate_tree_state = planner_tree_[(size_t)idx];
                const NomdpPlanningState& candidate_state = candidate_tree_state.GetValueImmutable();
                if (candidate_state.GetTransitionId() == performed_transition_id)
                {
                    const int64_t parent_state_idx = candidate_tree_state.GetParentIndex();
                    expected_possible_result_states.push_back(std::make_pair(idx, false));
                    if (previous_state_index == -1)
                    {
                        previous_state_index = parent_state_idx;
                    }
                    else
                    {
                        assert(previous_state_index == parent_state_idx);
                    }
                }
                else if (candidate_state.GetReverseTransitionId() == performed_transition_id)
                {
                    expected_possible_result_states.push_back(std::make_pair(idx, true));
                    if (previous_state_index == -1)
                    {
                        previous_state_index = idx;
                    }
                    else
                    {
                        assert(previous_state_index == idx);
                    }
                }
            }
            assert(expected_possible_result_states.size() > 0);
            //std::cout << "Result state could match " << expected_possible_result_states.size() << " states" << std::endl;
            //std::cout << "Expected possible result states: " << PrettyPrint::PrettyPrint(expected_possible_result_states) << std::endl;
            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Check if the current config matches one or more of the expected result states
            std::vector<std::pair<int64_t, bool>> expected_result_state_matches;
            for (size_t idx = 0; idx < expected_possible_result_states.size(); idx++)
            {
                const std::pair<int64_t, bool>& possible_match = expected_possible_result_states[idx];
                const int64_t possible_match_state_idx = (possible_match.second) ? planner_tree_[(size_t)(possible_match.first)].GetParentIndex() : possible_match.first;
                const NomdpPlanningTreeState& possible_match_tree_state = planner_tree_[(size_t)possible_match_state_idx];
                const NomdpPlanningState& possible_match_state = possible_match_tree_state.GetValueImmutable();
                const std::vector<Configuration, ConfigAlloc>& possible_match_node_particles = possible_match_state.GetParticlePositionsImmutable().first;
                const std::vector<std::vector<size_t>> possible_match_check_clusters = particle_clustering_fn(possible_match_node_particles, current_config);
                // If the current config is part of the cluster
                if (possible_match_check_clusters.size() == 1)
                {
                    const Configuration possible_match_state_expectation = possible_match_state.GetExpectation();
                    std::cout << "Possible result state matches with distance " << DistanceFn::Distance(current_config, possible_match_state_expectation) << " and expectation " << PrettyPrint::PrettyPrint(possible_match_state_expectation) << std::endl;
                    expected_result_state_matches.push_back(possible_match);
                }
            }
            //std::cout << "Result state matches: " << PrettyPrint::PrettyPrint(expected_result_state_matches) << std::endl;
            // If any child states matched
            if (expected_result_state_matches.size() > 0)
            {
                const int64_t result_state_index = UpdateNodeCountsAndTree(expected_possible_result_states, expected_result_state_matches);
                /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                // Now that we've updated the tree, we can rebuild and query for the action to take
                // The rebuild and action query process is the same in all cases
                RebuildPolicyGraph();
                return QueryNextAction(result_state_index);
            }
            else
            {
                std::cout << "Result state matched none of the " << expected_possible_result_states.size() << " expected results, adding a new state" << std::endl;
                // Compute the parameters of the new node
                const uint64_t new_child_state_id = planner_tree_.size() + UINT64_C(1000000000);
                std::vector<Configuration, ConfigAlloc> new_child_particles;
                new_child_particles.push_back(current_config);
                // These will get updated in the recursive call (that's why the attempt/reached counts are zero)
                const uint32_t reached_count = 0u;
                const double effective_edge_Pfeasibility = 0.0;
                const double parent_motion_Pfeasibility = planner_tree_[(size_t)previous_state_index].GetValueImmutable().GetMotionPfeasibility();
                const double step_size = planner_tree_[(size_t)previous_state_index].GetValueImmutable().GetStepSize();
                // Basic prior assuming that actions are reversible
                const uint32_t reverse_attempt_count = 1u;
                const uint32_t reverse_reached_count = 1u;
                // More fixed params
                const uint64_t transition_id = performed_transition_id;
                // Get a new transition ID for the reverse
                const uint64_t reverse_transition_id = planner_tree_.size() + UINT64_C(1000000000);
                // Get some params
                const uint64_t previous_state_reverse_transition_id = planner_tree_[(size_t)previous_state_index].GetValueImmutable().GetReverseTransitionId();
                const bool desired_transition_is_reversal = (performed_transition_id == previous_state_reverse_transition_id) ? true : false;
                Configuration command;
                uint32_t attempt_count = 0u;
                uint64_t split_id = 0;
                // If the action was a reversal, then grab the expectation of the previous state's parent
                if (desired_transition_is_reversal)
                {
                    const int64_t parent_index = planner_tree_[(size_t)previous_state_index].GetParentIndex();
                    const NomdpPlanningTreeState& parent_tree_state = planner_tree_[(size_t)parent_index];
                    const NomdpPlanningState& parent_state = parent_tree_state.GetValueImmutable();
                    command = parent_state.GetExpectation();
                    attempt_count = planner_tree_[(size_t)previous_state_index].GetValueImmutable().GetReverseAttemptAndReachedCounts().first;
                    split_id = performed_transition_id; // Split IDs aren't actually used, other than > 0 meaning children of splits
                }
                // If not, then grab the expectation & split ID from one of the children
                else
                {
                    const std::pair<int64_t, bool>& first_possible_match = expected_possible_result_states.front();
                    assert(first_possible_match.second == false); // Reversals will not result in a parent index lookup
                    const int64_t child_state_idx = first_possible_match.first;
                    const NomdpPlanningTreeState& child_tree_state = planner_tree_[(size_t)child_state_idx];
                    const NomdpPlanningState& child_state = child_tree_state.GetValueImmutable();
                    command = child_state.GetCommand();
                    attempt_count = child_state.GetAttemptAndReachedCounts().first;
                    split_id = child_state.GetSplitId();
                }
                // Put together the new state
                const NomdpPlanningState new_child_state(new_child_state_id, new_child_particles, attempt_count, reached_count, effective_edge_Pfeasibility, reverse_attempt_count, reverse_reached_count, parent_motion_Pfeasibility, step_size, command, transition_id, reverse_transition_id, split_id);
                // We add a new child state to the graph
                const NomdpPlanningTreeState new_child_tree_state(new_child_state, previous_state_index);
                planner_tree_.push_back(new_child_tree_state);
                // Add the linkage to the parent (link to the last state we just added)
                const int64_t new_state_index = (int64_t)planner_tree_.size() - 1;
                // NOTE - by adding to the tree, we have broken any references already held
                // so we can't use the previous_index_tree_state any more!
                planner_tree_[(size_t)previous_state_index].AddChildIndex(new_state_index);
                //std::cout << "Added new state to tree with index " << new_state_index << ", transition ID " << transition_id << ", reverse transition ID " << reverse_transition_id << ", and state ID " << new_child_state_id << std::endl;
                // Update the policy graph with the new state
                RebuildPolicyGraph();
                // To get the action, we recursively call this function (this time there will be an exact matching child state!)
                return QueryNormalBestAction(performed_transition_id, current_config, particle_clustering_fn);
            }
        }

        inline int64_t UpdateNodeCountsAndTree(const std::vector<std::pair<int64_t, bool>>& expected_possible_result_states, const std::vector<std::pair<int64_t, bool>>& expected_result_state_matches)
        {
            // If there was one possible result state and it matches
            // This should be the most likely case, and requires the least editing of the tree
            if (expected_possible_result_states.size() == 1 && expected_result_state_matches.size() == 1)
            {
                std::cout << "Result state matched single expected result" << std::endl;
                const std::pair<int64_t, bool>& result_match = expected_result_state_matches[0];
                // Update the attempt/reached counts
                // If the transition was a forward transition, we update the result state
                if (result_match.second == false)
                {
                    NomdpPlanningTreeState& result_tree_state = planner_tree_[(size_t)result_match.first];
                    NomdpPlanningState& result_state = result_tree_state.GetValueMutable();
                    const std::pair<uint32_t, uint32_t> counts = result_state.GetAttemptAndReachedCounts();
                    const uint32_t attempt_count = AddWithOverflowClamp(counts.first, policy_action_attempt_count_);
                    const uint32_t reached_count = AddWithOverflowClamp(counts.second, policy_action_attempt_count_);
                    result_state.UpdateAttemptAndReachedCounts(attempt_count, reached_count);
                    //std::cout << "Forward motion - updated counts from " << PrettyPrint::PrettyPrint(counts) << " to " << PrettyPrint::PrettyPrint(result_state.GetAttemptAndReachedCounts()) << std::endl;
                    return result_match.first;
                }
                else
                {
                    NomdpPlanningTreeState& result_child_tree_state = planner_tree_[(size_t)result_match.first];
                    NomdpPlanningState& result_child_state = result_child_tree_state.GetValueMutable();
                    const std::pair<uint32_t, uint32_t> counts = result_child_state.GetReverseAttemptAndReachedCounts();
                    const uint32_t attempt_count = AddWithOverflowClamp(counts.first, policy_action_attempt_count_);
                    const uint32_t reached_count = AddWithOverflowClamp(counts.second, policy_action_attempt_count_);
                    result_child_state.UpdateReverseAttemptAndReachedCounts(attempt_count, reached_count);
                    //std::cout << "Reverse motion - updated counts from " << PrettyPrint::PrettyPrint(counts) << " to " << PrettyPrint::PrettyPrint(result_child_state.GetReverseAttemptAndReachedCounts()) << std::endl;
                    return result_child_tree_state.GetParentIndex();
                }
            }
            else
            {
                std::cout << "Result state matched " << expected_result_state_matches.size() << " of " << expected_possible_result_states.size() << " expected results" << std::endl;
                /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                // Select the current best-distance result state as THE result state
                std::pair<int64_t, bool> best_result_state(-1, false);
                double best_distance = INFINITY;
                for (size_t idx = 0; idx < expected_result_state_matches.size(); idx++)
                {
                    const std::pair<int64_t, bool>& result_match = expected_result_state_matches[idx];
                    const int64_t result_match_state_idx = (result_match.second) ? planner_tree_[(size_t)result_match.first].GetParentIndex() : result_match.first;
                    const PolicyGraphNode& result_match_node = policy_graph_.GetNodeImmutable(result_match_state_idx);
                    const double result_match_distance = result_match_node.GetDistance();
                    if (result_match_distance < best_distance)
                    {
                        best_result_state = result_match;
                        best_distance = result_match_distance;
                    }
                }
                assert(best_result_state.first >= 0);
                const int64_t result_state_index = (best_result_state.second) ? planner_tree_[(size_t)best_result_state.first].GetParentIndex() : best_result_state.first;
                if (best_result_state.second == false)
                {
                    std::cout << "Selected best match result state (forward movement): " << result_state_index << std::endl;
                }
                else
                {
                    std::cout << "Selected best match result state (reverse movement): " << result_state_index << std::endl;
                }
                /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                // Update the attempt/reached counts for all *POSSIBLE* result states
                for (size_t idx = 0; idx < expected_possible_result_states.size(); idx++)
                {
                    const std::pair<int64_t, bool>& possible_result_match = expected_possible_result_states[idx];
                    NomdpPlanningTreeState& possible_result_tree_state = planner_tree_[(size_t)possible_result_match.first];
                    NomdpPlanningState& possible_result_state = possible_result_tree_state.GetValueMutable();
                    if (possible_result_match.second == false)
                    {
                        const std::pair<uint32_t, uint32_t> counts = possible_result_state.GetAttemptAndReachedCounts();
                        if ((possible_result_match.first == best_result_state.first) && (possible_result_match.second == best_result_state.second))
                        {
                            const uint32_t attempt_count = AddWithOverflowClamp(counts.first, policy_action_attempt_count_);
                            const uint32_t reached_count = AddWithOverflowClamp(counts.second, policy_action_attempt_count_);
                            possible_result_state.UpdateAttemptAndReachedCounts(attempt_count, reached_count);
                        }
                        else
                        {
                            const uint32_t attempt_count = AddWithOverflowClamp(counts.first, policy_action_attempt_count_);
                            const uint32_t reached_count = AddWithOverflowClamp(counts.second, 0u);
                            possible_result_state.UpdateAttemptAndReachedCounts(attempt_count, reached_count);
                        }
                        //std::cout << "Updated forward counts for state " << possible_result_match.first << " from " << PrettyPrint::PrettyPrint(counts) << " to " << PrettyPrint::PrettyPrint(possible_result_state.GetAttemptAndReachedCounts()) << std::endl;
                    }
                    else
                    {
                        const std::pair<uint32_t, uint32_t> reverse_counts = possible_result_state.GetReverseAttemptAndReachedCounts();
                        if ((possible_result_match.first == best_result_state.first) && (possible_result_match.second == best_result_state.second))
                        {
                            const uint32_t attempt_count = AddWithOverflowClamp(reverse_counts.first, policy_action_attempt_count_);
                            const uint32_t reached_count = AddWithOverflowClamp(reverse_counts.second, policy_action_attempt_count_);
                            possible_result_state.UpdateReverseAttemptAndReachedCounts(attempt_count, reached_count);
                        }
                        else
                        {
                            const uint32_t attempt_count = AddWithOverflowClamp(reverse_counts.first, policy_action_attempt_count_);
                            const uint32_t reached_count = AddWithOverflowClamp(reverse_counts.second, 0u);
                            possible_result_state.UpdateReverseAttemptAndReachedCounts(attempt_count, reached_count);
                        }
                        //std::cout << "Updated reverse counts for state " << possible_result_match.first << " from " << PrettyPrint::PrettyPrint(reverse_counts) << " to " << PrettyPrint::PrettyPrint(possible_result_state.GetReverseAttemptAndReachedCounts()) << std::endl;
                    }
                }
                /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                // Update the effective edge probabilities for the current transition
                UpdatePlannerTreeProbabilities();
                // Return the matching result state
                return result_state_index;
            }
        }

        inline void UpdatePlannerTreeProbabilities()
        {
            // Fuck it, let's update the entire tree. This is slower than it could be, but I don't want to miss anything
            UpdateChildTransitionProbabilities(0);
            // Backtrack up the tree and update P(->goal) probabilities
            for (int64_t idx = ((int64_t)planner_tree_.size() - 1); idx >= 0; idx--)
            {
                UpdateStateGoalReachedProbability(idx);
            }
            // Forward pass through the tree to update P(->goal) for leaf nodes
            for (size_t idx = 1; idx < planner_tree_.size(); idx++)
            {
                // Get the current state
                NomdpPlanningTreeState& current_state = planner_tree_[idx];
                const int64_t parent_index = current_state.GetParentIndex();
                // Get the parent state
                const NomdpPlanningTreeState& parent_state = planner_tree_[(size_t)parent_index];
                // If the current state is on a goal branch
                if (current_state.GetValueImmutable().GetGoalPfeasibility() > 0.0)
                {
                    continue;
                }
                // If we are a non-goal child of a goal branch state
                else if (parent_state.GetValueImmutable().GetGoalPfeasibility() > 0.0)
                {
                    // Update P(goal reached) based on our ability to reverse to the goal branch
                    const double parent_pgoalreached = parent_state.GetValueImmutable().GetGoalPfeasibility();
                    const double new_pgoalreached = -(parent_pgoalreached * current_state.GetValueImmutable().GetReverseEdgePfeasibility()); // We use negative goal reached probabilities to signal probability due to reversing
                    current_state.GetValueMutable().SetGoalPfeasibility(new_pgoalreached);
                }
            }
        }

        inline void UpdateChildTransitionProbabilities(const int64_t current_state_index)
        {
            // Gather all the children, split them by transition, and recompute the P(->)estimated edge probabilities
            const NomdpPlanningTreeState& current_tree_state = planner_tree_[(size_t)current_state_index];
            const std::vector<int64_t>& child_state_indices = current_tree_state.GetChildIndices();
            // Split them by transition IDs
            std::map<uint64_t, std::vector<int64_t>> transition_children_map;
            for (size_t idx = 0; idx < child_state_indices.size(); idx++)
            {
                const int64_t child_state_index = child_state_indices[idx];
                const NomdpPlanningTreeState& child_tree_state = planner_tree_[(size_t)child_state_index];
                const NomdpPlanningState& child_state = child_tree_state.GetValueImmutable();
                const uint64_t child_state_transition_id = child_state.GetTransitionId();
                transition_children_map[child_state_transition_id].push_back(child_state_index);
            }
            // Compute updated probabilites for each group
            for (auto itr = transition_children_map.begin(); itr != transition_children_map.end(); ++itr)
            {
                const std::vector<int64_t> transition_child_indices = itr->second;
                UpdateEstimatedEffectiveProbabilities(transition_child_indices);
            }
            // Perform the same update on all of our children
            for (size_t idx = 0; idx < child_state_indices.size(); idx++)
            {
                const int64_t child_state_index = child_state_indices[idx];
                UpdateChildTransitionProbabilities(child_state_index);
            }
        }

        inline void UpdateEstimatedEffectiveProbabilities(const std::vector<int64_t>& transition_child_states)
        {
            // Now that we have the forward-propagated states, we go back and update their effective edge P(feasibility)
            for (size_t idx = 0; idx < transition_child_states.size(); idx++)
            {
                const int64_t current_state_index = transition_child_states[idx];
                NomdpPlanningTreeState& current_tree_state = planner_tree_[(size_t)current_state_index];
                NomdpPlanningState& current_state = current_tree_state.GetValueMutable();
                double percent_active = 1.0;
                double p_reached = 0.0;
                for (uint32_t try_attempt = 0; try_attempt < edge_attempt_threshold_; try_attempt++)
                {
                    // How many particles got to our state on this attempt?
                    p_reached += (percent_active * current_state.GetRawEdgePfeasibility());
                    // Update the percent of particles that are still usefully active
                    double updated_percent_active = 0.0;
                    for (size_t other_idx = 0; other_idx < transition_child_states.size(); other_idx++)
                    {
                        if (other_idx != idx)
                        {
                            const int64_t other_state_index = transition_child_states[other_idx];
                            const NomdpPlanningTreeState& other_tree_state = planner_tree_[(size_t)other_state_index];
                            const NomdpPlanningState& other_state = other_tree_state.GetValueImmutable();
                            const double p_reached_other = percent_active * other_state.GetRawEdgePfeasibility();
                            const double p_returned_to_parent = p_reached_other * other_state.GetReverseEdgePfeasibility();
                            updated_percent_active += p_returned_to_parent;
                        }
                    }
                    percent_active = updated_percent_active;
                }
                assert(p_reached >= 0.0);
                if (p_reached > 1.0)
                {
                    std::cout << "WARNING - P(reached) = " << p_reached << " > 1.0 (probably numerical error)" << std::endl;
                    assert(p_reached <= 1.001);
                    p_reached = 1.0;
                }
                assert(p_reached <= 1.0);
                //std::cout << "Computed effective edge P(feasibility) of " << p_reached << " for " << edge_attempt_threshold_ << " try/retry attempts" << std::endl;
                current_state.SetEffectiveEdgePfeasibility(p_reached);
            }
        }

        inline void UpdateStateGoalReachedProbability(const int64_t current_state_index)
        {
            NomdpPlanningTreeState& current_tree_state = planner_tree_[(size_t)current_state_index];
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
            std::map<uint64_t, std::vector<int64_t>> effective_child_branches;
            for (size_t idx = 0; idx < current_tree_state.GetChildIndices().size(); idx++)
            {
                const int64_t& current_child_index = current_tree_state.GetChildIndices()[idx];
                const uint64_t& child_transition_id = planner_tree_[(size_t)current_child_index].GetValueImmutable().GetTransitionId();
                effective_child_branches[child_transition_id].push_back(current_child_index);
            }
            //std::cout << "Gathered effective child branches: " << PrettyPrint::PrettyPrint(effective_child_branches) << std::endl;
            // Now that we have the transitions separated out, compute the goal probability of each transition
            std::vector<double> effective_child_branch_probabilities;
            for (auto itr = effective_child_branches.begin(); itr != effective_child_branches.end(); ++itr)
            {
                double transtion_goal_probability = ComputeTransitionGoalProbability(itr->second, edge_attempt_threshold_);
                effective_child_branch_probabilities.push_back(transtion_goal_probability);
            }
            //std::cout << "Computed effective child branch probabilities: " << PrettyPrint::PrettyPrint(effective_child_branch_probabilities) << std::endl;
            // Now, get the highest transtion probability
            if (effective_child_branch_probabilities.size() > 0)
            {
                const double max_transition_probability = *std::max_element(effective_child_branch_probabilities.begin(), effective_child_branch_probabilities.end());
                assert(max_transition_probability >= 0.0);
                assert(max_transition_probability <= 1.0);
                // Update the current state
                //std::cout << "Updating P(goal reached) to " << max_transition_probability << std::endl;
                current_tree_state.GetValueMutable().SetGoalPfeasibility(max_transition_probability);
            }
            else
            {
                if (current_tree_state.GetValueMutable().GetGoalPfeasibility() > 0.0)
                {
                    //std::cout << "Not updating P(goal reached) for an assumed goal state" << std::endl;
                }
                else
                {
                    //std::cout << "Not updating P(goal reached) for a state with no children" << std::endl;
                }
            }
        }

        inline double ComputeTransitionGoalProbability(const std::vector<int64_t>& child_node_indices, const uint32_t edge_attempt_threshold) const
        {
            std::vector<NomdpPlanningState> child_states(child_node_indices.size());
            for (size_t idx = 0; idx < child_node_indices.size(); idx++)
            {
                // Get the current child
                const int64_t& current_child_index = child_node_indices[idx];
                const NomdpPlanningState& current_child = planner_tree_[(size_t)current_child_index].GetValueImmutable();
                child_states[idx] = current_child;
            }
            return ComputeTransitionGoalProbability(child_states, edge_attempt_threshold);
        }

        inline double ComputeTransitionGoalProbability(const std::vector<NomdpPlanningState>& child_nodes, const uint32_t planner_action_try_attempts) const
        {
            // Let's handle the special cases first
            // The most common case - a non-split transition
            if (child_nodes.size() == 1)
            {
                const NomdpPlanningState& current_child = child_nodes.front();
                return (current_child.GetGoalPfeasibility() * current_child.GetEffectiveEdgePfeasibility());
            }
            // IMPOSSIBLE (but we handle it just to be sure)
            else if (child_nodes.size() == 0)
            {
                return 0.0;
            }
            // Let's handle the split case(s)
            else
            {
                // We do this the right way
                std::vector<double> child_goal_reached_probabilities(child_nodes.size(), 0.0);
                // For each child state, we compute the probability that we'll end up at each of the result states, accounting for try/retry with reversibility
                // This lets us compare child states as if they were separate actions, so the overall P(goal reached) = max(child) P(goal reached | child)
                for (size_t idx = 0; idx < child_nodes.size(); idx++)
                {
                    // Get the current child
                    const NomdpPlanningState& current_child = child_nodes[idx];
                    // For the selected child, we keep track of the probability that we reach the goal directly via the child state AND the probability that we reach the goal from unintended other child states
                    double percent_active = 1.0;
                    double p_we_reached_goal = 0.0;
                    double p_others_reached_goal = 0.0;
                    for (uint32_t try_attempt = 0; try_attempt < planner_action_try_attempts; try_attempt++)
                    {
                        // How many particles got to our state on this attempt?
                        const double p_reached = percent_active * current_child.GetRawEdgePfeasibility();
                        //std::cout << "P(reached) target child " << p_reached << std::endl;
                        // Children with negative P(goal feasibility) cannot reach the goal directly, and thus get P(goal reached)=0 here
                        const double raw_child_goal_Pfeasibility = current_child.GetGoalPfeasibility();
                        const double child_goal_Pfeasibility = (raw_child_goal_Pfeasibility > 0.0) ? raw_child_goal_Pfeasibility : 0.0;
                        const double p_we_reached = p_reached * child_goal_Pfeasibility;
                        //std::cout << "P(we reached goal) in this attempt " << p_we_reached << std::endl;
                        p_we_reached_goal += p_we_reached;
                        //std::cout << "P(we reached goal) so far " << p_we_reached_goal << std::endl;
                        // Update the percent of particles that are still usefully active
                        // and the probability that the goal was reached via a different child
                        double updated_percent_active = 0.0;
                        double p_others_reached = 0.0;
                        for (size_t other_idx = 0; other_idx < child_nodes.size(); other_idx++)
                        {
                            if (other_idx != idx)
                            {
                                // Get the other child
                                const NomdpPlanningState& other_child = child_nodes[other_idx];
                                const double p_reached_other = percent_active * other_child.GetRawEdgePfeasibility();
                                //std::cout << "P(reached) other child " << p_reached_other << std::endl;
                                const double p_returned_to_parent = p_reached_other * other_child.GetReverseEdgePfeasibility();
                                const double p_stuck_at_other = p_reached_other * (1.0 - other_child.GetReverseEdgePfeasibility());
                                // Children with negative P(goal feasibility) cannot reach the goal directly, and thus get P(goal reached)=0 here
                                const double raw_other_child_goal_Pfeasibility = other_child.GetGoalPfeasibility();
                                const double other_child_goal_Pfeasibility = (raw_other_child_goal_Pfeasibility > 0.0) ? raw_other_child_goal_Pfeasibility : 0.0;
                                const double p_reached_goal_from_other = p_stuck_at_other * other_child_goal_Pfeasibility;
                                p_others_reached += p_reached_goal_from_other;
                                updated_percent_active += p_returned_to_parent;
                            }
                        }
                        //std::cout << "P(others reached goal) in this attempt " << p_others_reached << std::endl;
                        p_others_reached_goal += p_others_reached;
                        //std::cout << "P(others reached goal) so far " << p_others_reached_goal << std::endl;
                        percent_active = updated_percent_active;
                    }
                    double p_reached_goal = p_we_reached_goal + p_others_reached_goal;
                    //std::cout << "Computed new P(goal reached) " << p_reached_goal << " for child " << idx + 1 << " of " << child_nodes.size() << std::endl;
                    assert(p_reached_goal >= 0.0);
                    if (p_reached_goal > 1.0)
                    {
                        std::cout << "WARNING - P(reached) = " << p_reached_goal << " > 1.0 (probably numerical error)" << std::endl;
                        assert(p_reached_goal <= 1.001);
                        p_reached_goal = 1.0;
                    }
                    assert(p_reached_goal <= 1.0);
                    child_goal_reached_probabilities[idx] = p_reached_goal;
                }
                const double max_child_transition_goal_reached_probability = *std::max_element(child_goal_reached_probabilities.begin(), child_goal_reached_probabilities.end());
                //std::cout << "Computed new P(goal reached) " << max_child_transition_goal_reached_probability << " from " << child_nodes.size() << " child states " << std::endl;
                return max_child_transition_goal_reached_probability;
            }
        }
    };
}

template<typename Configuration, typename ConfigSerializer, typename AverageFn, typename DistanceFn, typename DimDistanceFn, typename ConfigAlloc=std::allocator<Configuration>>
std::ostream& operator<<(std::ostream& strm, const execution_policy::ExecutionPolicy<Configuration, ConfigSerializer, AverageFn, DimDistanceFn, ConfigAlloc>& policy)
{
    const std::vector<simple_rrt_planner::SimpleRRTPlannerState<uncertainty_planning_tools::UncertaintyPlannerState<Configuration, ConfigSerializer, AverageFn, DistanceFn, DimDistanceFn, ConfigAlloc>>>& raw_policy_tree = policy.GetRawPolicy();
    strm << "Execution Policy - Policy: ";
    for (size_t idx = 0; idx < raw_policy_tree.size(); idx++)
    {
        const simple_rrt_planner::SimpleRRTPlannerState<uncertainty_planning_tools::UncertaintyPlannerState<Configuration, ConfigSerializer, AverageFn, DistanceFn, DimDistanceFn, ConfigAlloc>>& policy_tree_state = raw_policy_tree[idx];
        const int64_t parent_index = policy_tree_state.GetParentIndex();
        const std::vector<int64_t>& child_indices = policy_tree_state.GetChildIndices();
        const uncertainty_planning_tools::UncertaintyPlannerState<Configuration, ConfigSerializer, AverageFn, DistanceFn, DimDistanceFn, ConfigAlloc>& policy_state = policy_tree_state.GetValueImmutable();
        strm << "\nState # " << idx << " with parent " << parent_index << " and children " << PrettyPrint::PrettyPrint(child_indices, true) << " - value: " << PrettyPrint::PrettyPrint(policy_state);
    }
    return strm;
}

#endif // EXECUTION_POLICY_HPP
