#include <stdio.h>
#include <vector>
#include <map>
#include <random>
#include <Eigen/Geometry>
#include <arc_utilities/arc_helpers.hpp>
#include <arc_utilities/eigen_helpers.hpp>
#include <nomdp_planning/simple_pid_controller.hpp>
#include <nomdp_planning/simple_uncertainty_models.hpp>

#ifndef SIMPLELINKED_ROBOT_HELPERS_HPP
#define SIMPLELINKED_ROBOT_HELPERS_HPP

namespace simplelinked_robot_helpers
{
    class SimpleJointModel
    {
    public:

        enum JOINT_TYPE {PRISMATIC, REVOLUTE, CONTINUOUS, FIXED};

    protected:

        std::pair<double, double> limits_;
        double value_;
        JOINT_TYPE type_;

    public:

        static inline uint64_t Serialize(const SimpleJointModel& model, std::vector<uint8_t>& buffer)
        {
            return model.SerializeSelf(buffer);
        }

        static inline std::pair<SimpleJointModel, uint64_t> Deserialize(const std::vector<uint8_t>& buffer, const uint64_t current)
        {
            SimpleJointModel temp_model;
            const uint64_t bytes_read = temp_model.DeserializeSelf(buffer, current);
            return std::make_pair(temp_model, bytes_read);
        }

        inline SimpleJointModel(const std::pair<double, double>& limits, const double value, const JOINT_TYPE type)
        {
            type_ = type;
            if (IsContinuous())
            {
                limits_ = std::pair<double, double>(-M_PI, M_PI);
            }
            else
            {
                assert(limits.first <= limits.second);
                limits_ = limits;
            }
            SetValue(value);
        }

        inline SimpleJointModel()
        {
            limits_ = std::pair<double, double>(0.0, 0.0);
            type_ = JOINT_TYPE::FIXED;
            SetValue(0.0);
        }

        inline uint64_t SerializeSelf(std::vector<uint8_t>& buffer) const
        {
            const uint64_t start_buffer_size = buffer.size();
            arc_helpers::SerializeFixedSizePOD<double>(limits_.first, buffer);
            arc_helpers::SerializeFixedSizePOD<double>(limits_.second, buffer);
            arc_helpers::SerializeFixedSizePOD<double>(value_, buffer);
            arc_helpers::SerializeFixedSizePOD<JOINT_TYPE>(type_, buffer);
            // Figure out how many bytes were written
            const uint64_t end_buffer_size = buffer.size();
            const uint64_t bytes_written = end_buffer_size - start_buffer_size;
            return bytes_written;
        }

        inline uint64_t DeserializeSelf(const std::vector<uint8_t>& buffer, const uint64_t current)
        {
            assert(current < buffer.size());
            uint64_t current_position = current;
            const std::pair<double, uint64_t> deserialized_limits_first = arc_helpers::DeserializeFixedSizePOD<double>(buffer, current_position);
            limits_.first = deserialized_limits_first.first;
            current_position += deserialized_limits_first.second;
            const std::pair<double, uint64_t> deserialized_limits_second = arc_helpers::DeserializeFixedSizePOD<double>(buffer, current_position);
            limits_.second = deserialized_limits_second.first;
            current_position += deserialized_limits_second.second;
            const std::pair<double, uint64_t> deserialized_value = arc_helpers::DeserializeFixedSizePOD<double>(buffer, current_position);
            value_ = deserialized_value.first;
            current_position += deserialized_value.second;
            const std::pair<JOINT_TYPE, uint64_t> deserialized_type = arc_helpers::DeserializeFixedSizePOD<JOINT_TYPE>(buffer, current_position);
            type_ = deserialized_type.first;
            current_position += deserialized_type.second;
            // Figure out how many bytes were read
            const uint64_t bytes_read = current_position - current;
            return bytes_read;
        }

        inline bool IsContinuous() const
        {
            if (type_ == JOINT_TYPE::CONTINUOUS)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        inline bool IsRevolute() const
        {
            if ((type_ == JOINT_TYPE::REVOLUTE) || (type_ == JOINT_TYPE::CONTINUOUS))
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        inline bool IsPrismatic() const
        {
            if (type_ == JOINT_TYPE::PRISMATIC)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        inline bool IsFixed() const
        {
            if (type_ == JOINT_TYPE::FIXED)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        inline double GetValue() const
        {
            return value_;
        }

        inline std::pair<double, double> GetLimits() const
        {
            return limits_;
        }

        inline bool InLimits(const double value) const
        {
            if (IsContinuous())
            {
                 return true;
            }
            else
            {
                if ((value < limits_.first) || (value > limits_.second))
                {
                    return false;
                }
                else
                {
                    return true;
                }
            }
        }

        inline double EnforceLimits(const double value) const
        {
            if (IsContinuous())
            {
                return EigenHelpers::EnforceContinuousRevoluteBounds(value);
            }
            else
            {
                if (value < limits_.first)
                {
                    return limits_.first;
                }
                else if (value > limits_.second)
                {
                    return limits_.second;
                }
                else
                {
                    return value;
                }
            }
        }

        inline void SetValue(const double value)
        {
            const double real_value = EnforceLimits(value);
            value_ = real_value;
        }

        inline double SignedDistance(const double v1, const double v2) const
        {
            if (IsContinuous())
            {
                return EigenHelpers::ContinuousRevoluteSignedDistance(v1, v2);
            }
            else
            {
                return (v2 - v1);
            }
        }

        inline double SignedDistance(const double v) const
        {
            return SignedDistance(GetValue(), v);
        }

        inline double Distance(const double v1, const double v2) const
        {
            return fabs(SignedDistance(v1, v2));
        }

        inline double Distance(const double v) const
        {
            return fabs(SignedDistance(GetValue(), v));
        }

        inline SimpleJointModel CopyWithNewValue(const double value) const
        {
            return SimpleJointModel(limits_, value, type_);
        }
    };

    class SimpleLinkedConfigurationSerializer
    {
    public:

        static inline std::string TypeName()
        {
            return std::string("SimpleLinkedConfigurationSerializer");
        }

        static inline uint64_t Serialize(const std::vector<SimpleJointModel>& value, std::vector<uint8_t>& buffer)
        {
            return arc_helpers::SerializeVector<SimpleJointModel>(value, buffer, SimpleJointModel::Serialize);
        }

        static inline std::pair<std::vector<SimpleJointModel>, uint64_t> Deserialize(const std::vector<uint8_t>& buffer, const uint64_t current)
        {
            return arc_helpers::DeserializeVector<SimpleJointModel>(buffer, current, SimpleJointModel::Deserialize);
        }
    };

    // Typedef to make our life a bit easier
    typedef std::vector<SimpleJointModel> SimpleLinkedConfiguration;

    class SimpleLinkedBaseSampler
    {
    protected:

        std::vector<std::uniform_real_distribution<double>> distributions_;
        SimpleLinkedConfiguration representative_configuration_;

    public:

        SimpleLinkedBaseSampler(const SimpleLinkedConfiguration& representative_configuration)
        {
            representative_configuration_ = representative_configuration;
            for (size_t idx = 0; idx < representative_configuration_.size(); idx++)
            {
                const SimpleJointModel& current_joint = representative_configuration_[idx];
                const std::pair<double, double> limits = current_joint.GetLimits();
                distributions_.push_back(std::uniform_real_distribution<double>(limits.first, limits.second));
            }
        }

        template<typename Generator>
        SimpleLinkedConfiguration Sample(Generator& prng)
        {
            SimpleLinkedConfiguration sampled;
            sampled.reserve(representative_configuration_.size());
            for (size_t idx = 0; idx < representative_configuration_.size(); idx++)
            {
                const SimpleJointModel& current_joint = representative_configuration_[idx];
                const double sampled_val = distributions_[idx](prng);
                sampled.push_back(current_joint.CopyWithNewValue(sampled_val));
            }
            return sampled;
        }
    };

    class SimpleLinkedInterpolator
    {
    public:

        SimpleLinkedConfiguration operator()(const SimpleLinkedConfiguration& q1, const SimpleLinkedConfiguration& q2, const double ratio) const
        {
            return Interpolate(q1, q2, ratio);
        }

        static SimpleLinkedConfiguration Interpolate(const SimpleLinkedConfiguration& q1, const SimpleLinkedConfiguration& q2, const double ratio)
        {
            assert(q1.size() == q2.size());
            // Make the interpolated config
            SimpleLinkedConfiguration interpolated;
            interpolated.reserve(q1.size());
            for (size_t idx = 0; idx < q1.size(); idx++)
            {
                const SimpleJointModel& j1 = q1[idx];
                const SimpleJointModel& j2 = q2[idx];
                assert(j1.IsContinuous() == j2.IsContinuous());
                if (j1.IsContinuous())
                {
                    const double interpolated_value = EigenHelpers::InterpolateContinuousRevolute(j1.GetValue(), j2.GetValue(), ratio);
                    interpolated.push_back(j1.CopyWithNewValue(interpolated_value));
                }
                else
                {
                    const double interpolated_value = EigenHelpers::Interpolate(j1.GetValue(), j2.GetValue(), ratio);
                    interpolated.push_back(j1.CopyWithNewValue(interpolated_value));
                }
            }
            return interpolated;
        }
    };

    class SimpleLinkedAverager
    {
    protected:

        static SimpleLinkedConfiguration GetRepresentativeConfiguration(const std::vector<SimpleLinkedConfiguration>& vec)
        {
            assert(vec.size() > 0);
            const SimpleLinkedConfiguration& representative_config = vec.front();
            for (size_t idx = 0; idx < vec.size(); idx++)
            {
                const SimpleLinkedConfiguration& current_config = vec[idx];
                assert(representative_config.size() == current_config.size());
                for (size_t jdx = 0; jdx < representative_config.size(); jdx++)
                {
                    const SimpleJointModel& jr = representative_config[jdx];
                    const SimpleJointModel& jc = current_config[jdx];
                    assert(jr.IsContinuous() == jc.IsContinuous());
                }
            }
            return representative_config;
        }

    public:

        SimpleLinkedConfiguration operator()(const std::vector<SimpleLinkedConfiguration>& vec) const
        {
            return Average(vec);
        }

        static SimpleLinkedConfiguration Average(const std::vector<SimpleLinkedConfiguration>& vec)
        {
            if (vec.size() > 0)
            {
                const SimpleLinkedConfiguration representative_configuration = GetRepresentativeConfiguration(vec);
                const size_t config_size = representative_configuration.size();
                // Separate the joint values
                std::vector<std::vector<double>> raw_values(config_size);
                for (size_t idx = 0; idx < vec.size(); idx++)
                {
                    const SimpleLinkedConfiguration& q = vec[idx];
                    for (size_t qdx = 0; qdx < config_size; qdx++)
                    {
                        const double jval = q[qdx].GetValue();
                        raw_values[qdx].push_back(jval);
                    }
                }
                // Average each joint
                SimpleLinkedConfiguration average_config;
                average_config.reserve(config_size);
                for (size_t jdx = 0; jdx < config_size; jdx++)
                {
                    const std::vector<double>& values = raw_values[jdx];
                    const SimpleJointModel& representative_joint = representative_configuration[jdx];
                    if (representative_joint.IsContinuous())
                    {
                        const double average_value = EigenHelpers::AverageContinuousRevolute(values);
                        average_config.push_back(representative_joint.CopyWithNewValue(average_value));
                    }
                    else
                    {
                        const double average_value = EigenHelpers::AverageStdVectorDouble(values);
                        average_config.push_back(representative_joint.CopyWithNewValue(average_value));
                    }
                }
                return average_config;
            }
            else
            {
                return SimpleLinkedConfiguration();
            }
        }

        static inline std::string TypeName()
        {
            return std::string("SimpleLinkedAverager");
        }
    };

    class SimpleLinkedDimDistancer
    {
    public:

        Eigen::VectorXd operator()(const SimpleLinkedConfiguration& q1, const SimpleLinkedConfiguration& q2) const
        {
            return Distance(q1, q2);
        }

        static Eigen::VectorXd Distance(const SimpleLinkedConfiguration& q1, const SimpleLinkedConfiguration& q2)
        {
            return RawDistance(q1, q2).cwiseAbs();
        }

        static double RawJointDistance(const SimpleJointModel& j1, const SimpleJointModel& j2)
        {
            assert(j1.IsContinuous() == j2.IsContinuous());
            return j1.SignedDistance(j1.GetValue(), j2.GetValue());
        }

        static Eigen::VectorXd RawDistance(const SimpleLinkedConfiguration& q1, const SimpleLinkedConfiguration& q2)
        {
            assert(q1.size() == q2.size());
            Eigen::VectorXd distances = Eigen::VectorXd::Zero(q1.size());
            for (size_t idx = 0; idx < q1.size(); idx++)
            {
                const SimpleJointModel& j1 = q1[idx];
                const SimpleJointModel& j2 = q2[idx];
                distances((int64_t)idx) = RawJointDistance(j1, j2);
            }
            return distances;
        }

        static inline std::string TypeName()
        {
            return std::string("SimpleLinkedDimDistancer");
        }
    };

    class SimpleLinkedDistancer
    {
    public:

        double operator()(const SimpleLinkedConfiguration& q1, const SimpleLinkedConfiguration& q2) const
        {
            return Distance(q1, q2);
        }

        static double Distance(const SimpleLinkedConfiguration& q1, const SimpleLinkedConfiguration& q2)
        {
            const Eigen::VectorXd dim_distances = SimpleLinkedDimDistancer::Distance(q1, q2);
            return dim_distances.norm();
        }

        static inline std::string TypeName()
        {
            return std::string("SimpleLinkedDistancer");
        }
    };

    struct ROBOT_CONFIG
    {
        double kp;
        double ki;
        double kd;
        double integral_clamp;
        double velocity_limit;
        double max_sensor_noise;
        double max_actuator_noise;

        ROBOT_CONFIG()
        {
            kp = 0.0;
            ki = 0.0;
            kd = 0.0;
            integral_clamp = 0.0;
            velocity_limit = 0.0;
            max_sensor_noise = 0.0;
            max_actuator_noise = 0.0;
        }

        ROBOT_CONFIG(const double in_kp, const double in_ki, const double in_kd, const double in_integral_clamp, const double in_velocity_limit, const double in_max_sensor_noise, const double in_max_actuator_noise)
        {
            kp = in_kp;
            ki = in_ki;
            kd = in_kd;
            integral_clamp = in_integral_clamp;
            velocity_limit = in_velocity_limit;
            max_sensor_noise = in_max_sensor_noise;
            max_actuator_noise = in_max_actuator_noise;
        }
    };

    struct JointControllerGroup
    {
        simple_pid_controller::SimplePIDController controller;
        simple_uncertainty_models::SimpleUncertainSensor sensor;
        simple_uncertainty_models::SimpleUncertainVelocityActuator actuator;

        JointControllerGroup(const ROBOT_CONFIG& config)
        {
            controller = simple_pid_controller::SimplePIDController(config.kp, config.ki, config.kd, config.integral_clamp);
            sensor = simple_uncertainty_models::SimpleUncertainSensor(-config.max_sensor_noise, config.max_sensor_noise);
            actuator = simple_uncertainty_models::SimpleUncertainVelocityActuator(-config.max_actuator_noise, config.max_actuator_noise, config.velocity_limit);
        }

        JointControllerGroup()
        {
            controller = simple_pid_controller::SimplePIDController(0.0, 0.0, 0.0, 0.0);
            sensor = simple_uncertainty_models::SimpleUncertainSensor(0.0, 0.0);
            actuator = simple_uncertainty_models::SimpleUncertainVelocityActuator(0.0, 0.0, 0.0);
        }
    };

    struct RobotLink
    {
        Eigen::Affine3d link_transform;
        EigenHelpers::VectorVector3d link_points;
        std::string link_name;
    };

    struct RobotJoint
    {
        int64_t parent_link_index;
        int64_t child_link_index;
        Eigen::Affine3d joint_transform;
        Eigen::Vector3d joint_axis;
        SimpleJointModel joint_model;
        JointControllerGroup joint_controller;
    };

    class SimpleLinkedRobot
    {
    protected:

        bool initialized_;
        size_t num_active_joints_;
        double max_motion_per_unit_step_;
        Eigen::Affine3d base_transform_;
        Eigen::MatrixXi self_collision_map_;
        std::vector<RobotLink> links_;
        std::vector<RobotJoint> joints_;

        inline void UpdateTransforms()
        {
            // Update the transform for the first link
            links_.front().link_transform = base_transform_;
            // Go out the kinematic chain
            for (size_t idx = 0; idx < joints_.size(); idx++)
            {
                // Get the current joint
                const RobotJoint& current_joint = joints_[idx];
                // Get the parent link
                const RobotLink& parent_link = links_[current_joint.parent_link_index];
                // Get the child link
                RobotLink& child_link = links_[current_joint.child_link_index];
                // Get the parent_link transform
                const Eigen::Affine3d parent_transform = parent_link.link_transform;
                // Get the parent_link->joint transform
                const Eigen::Affine3d parent_to_joint_transform = current_joint.joint_transform;
                // Compute the base->joint_transform
                const Eigen::Affine3d complete_transform = parent_transform * parent_to_joint_transform;
                // Compute the joint transform
                if (current_joint.joint_model.IsRevolute())
                {
                    const Eigen::Affine3d joint_transform = Eigen::Translation3d(0.0, 0.0, 0.0) * Eigen::Quaterniond(Eigen::AngleAxisd(current_joint.joint_model.GetValue(), current_joint.joint_axis));
                    const Eigen::Affine3d child_transform = complete_transform * joint_transform;
                    child_link.link_transform = child_transform;
                }
                else if (current_joint.joint_model.IsPrismatic())
                {
                    const Eigen::Translation3d joint_translation = (Eigen::Translation3d)(current_joint.joint_axis * current_joint.joint_model.GetValue());
                    const Eigen::Affine3d joint_transform = joint_translation * Eigen::Quaterniond::Identity();
                    const Eigen::Affine3d child_transform = complete_transform * joint_transform;
                    child_link.link_transform = child_transform;
                }
                else
                {
                    // Joint is fixed
                    child_link.link_transform = complete_transform;
                }
            }
        }

        inline void SetConfig(const SimpleLinkedConfiguration& new_config)
        {
            assert(new_config.size() == num_active_joints_);
            size_t config_idx = 0u;
            for (size_t idx = 0; idx < joints_.size(); idx++)
            {
                RobotJoint& current_joint = joints_[idx];
                // Skip fixed joints
                if (current_joint.joint_model.IsFixed())
                {
                    continue;
                }
                else
                {
                    assert(config_idx < new_config.size());
                    const SimpleJointModel& new_joint = new_config[config_idx];
                    current_joint.joint_model.SetValue(new_joint.GetValue());
                    config_idx++;
                }
            }
            // Update forward kinematics
            UpdateTransforms();
        }

        inline void ResetControllers()
        {
            for (size_t idx = 0; idx < joints_.size(); idx++)
            {
                RobotJoint& current_joint = joints_[idx];
                current_joint.joint_controller.controller.Zero();
            }
        }

        inline size_t GetNumActiveJoints() const
        {
            size_t num_active_joints = 0u;
            for (size_t idx = 0; idx < joints_.size(); idx++)
            {
                const RobotJoint& current_joint = joints_[idx];
                // Skip fixed joints
                if (!(current_joint.joint_model.IsFixed()))
                {
                    num_active_joints++;
                }
            }
            return num_active_joints;
        }

    public:

        static inline bool SanityCheckRobotModel(const std::vector<RobotLink>& links, const std::vector<RobotJoint>& joints)
        {
            if (links.size() != (joints.size() + 1))
            {
                std::cerr << links.size() << " links are not enough for " << joints.size() << " joints" << std::endl;
                return false;
            }
            // We need to make sure that every joint references valid links, and that the links+joints form a valid kinematic chain
            int64_t last_child_index = 0u;
            for (size_t idx = 0; idx < joints.size(); idx++)
            {
                const RobotJoint& current_joint = joints[idx];
                if (current_joint.parent_link_index != last_child_index)
                {
                    std::cerr << "Parent link index must be the same as the previous joint's child link index" << std::endl;
                    return false;
                }
                if (current_joint.child_link_index != (current_joint.parent_link_index + 1))
                {
                    std::cerr << "Parent and child links must have successive indices" << std::endl;
                    return false;
                }
                if ((current_joint.child_link_index >= (int64_t)links.size()) || (current_joint.child_link_index < 0))
                {
                    std::cerr << "Invalid child link index" << std::endl;
                    return false;
                }
                last_child_index = current_joint.child_link_index;
                // Last, check the joint axes
                const double joint_axis_norm = current_joint.joint_axis.norm();
                const double error = fabs(joint_axis_norm - 1.0);
                if (error > std::numeric_limits<double>::epsilon())
                {
                    std::cerr << "Joint axis is not a unit vector" << std::endl;
                    return false;
                }
            }
            // Make sure the links all have unique names
            std::map<std::string, uint32_t> name_check_map;
            for (size_t idx = 0; idx < links.size(); idx++)
            {
                const std::string& link_name = links[idx].link_name;
                name_check_map[link_name]++;
                if (name_check_map[link_name] > 1)
                {
                    std::cerr << "Link " << link_name << " is not unique" << std::endl;
                    return false;
                }
            }
            return true;
        }

        static inline Eigen::MatrixXi GenerateAllowedSelfColllisionMap(const size_t num_links, const std::vector<std::pair<size_t, size_t>>& allowed_self_collisions)
        {
            Eigen::MatrixXi allowed_self_collision_map = Eigen::MatrixXi::Identity(num_links, num_links);
            for (size_t idx = 0; idx < allowed_self_collisions.size(); idx++)
            {
                const std::pair<size_t, size_t>& allowed_self_collision = allowed_self_collisions[idx];
                const int64_t first_link_index = (int64_t)allowed_self_collision.first;
                const int64_t second_link_index = (int64_t)allowed_self_collision.second;
                // Insert it both ways
                allowed_self_collision_map(first_link_index, second_link_index) = 1;
                allowed_self_collision_map(second_link_index, first_link_index) = 1;
            }
            return allowed_self_collision_map;
        }

        static inline double ComputeMaxMotionPerStep(SimpleLinkedRobot robot)
        {
            double max_motion = 0.0;
            const std::vector<std::pair<std::string, EigenHelpers::VectorVector3d>> robot_links_points = robot.GetRawLinksPoints();
            // Generate motion primitives
            std::vector<Eigen::VectorXd> motion_primitives;
            motion_primitives.reserve(robot.GetNumActiveJoints() * 2);
            for (size_t joint_idx = 0; joint_idx < robot.GetNumActiveJoints(); joint_idx++)
            {
                Eigen::VectorXd raw_motion_plus = Eigen::VectorXd::Zero(robot.GetNumActiveJoints());
                raw_motion_plus(joint_idx) = 1.0;
                motion_primitives.push_back(raw_motion_plus);
                Eigen::VectorXd raw_motion_neg = Eigen::VectorXd::Zero(robot.GetNumActiveJoints());
                raw_motion_neg(joint_idx) = -1.0;
                motion_primitives.push_back(raw_motion_neg);
            }
            // Go through the robot model & compute how much it moves
            for (size_t link_idx = 0; link_idx < robot_links_points.size(); link_idx++)
            {
                // Grab the link name and points
                const std::string& link_name = robot_links_points[link_idx].first;
                const EigenHelpers::VectorVector3d link_points = robot_links_points[link_idx].second;
                // Now, go through the points of the link
                for (size_t point_idx = 0; point_idx < link_points.size(); point_idx++)
                {
                    const Eigen::Vector3d& link_relative_point = link_points[point_idx];
                    // Get the Jacobian for the current point
                    const Eigen::Matrix<double, 3, Eigen::Dynamic> point_jacobian = robot.ComputeLinkPointJacobian(link_name, link_relative_point);
                    // Compute max point motion
                    for (size_t motion_idx = 0; motion_idx < motion_primitives.size(); motion_idx++)
                    {
                        const Eigen::VectorXd& current_motion = motion_primitives[motion_idx];
                        const double point_motion = (point_jacobian * current_motion).row(0).norm();
                        if (point_motion > max_motion)
                        {
                            max_motion = point_motion;
                        }
                    }
                }
            }
            return max_motion;
        }

        inline SimpleLinkedRobot(const Eigen::Affine3d& base_transform, const std::vector<RobotLink>& links, const std::vector<RobotJoint>& joints, const std::vector<std::pair<size_t, size_t>>& allowed_self_collisions, const SimpleLinkedConfiguration& initial_position)
        {
            base_transform_ = base_transform;
            // We take a list of robot links and a list of robot joints, but we have to sanity-check them first
            links_ = links;
            joints_ = joints;
            if (!SanityCheckRobotModel(links_, joints_))
            {
                throw std::invalid_argument("Attempted to construct a SimpleLinkedRobot with an invalid robot model");
            }
            num_active_joints_ = GetNumActiveJoints();
            max_motion_per_unit_step_ = ComputeMaxMotionPerStep(*this);
            // Generate the self colllision map
            self_collision_map_ = GenerateAllowedSelfColllisionMap(links_.size(), allowed_self_collisions);
            UpdatePosition(initial_position);
            initialized_ = true;
        }

        inline SimpleLinkedRobot(const std::vector<RobotLink>& links, const std::vector<RobotJoint>& joints, const std::vector<std::pair<size_t, size_t>>& allowed_self_collisions, const SimpleLinkedConfiguration& initial_position)
        {
            base_transform_ = Eigen::Affine3d::Identity();
            // We take a list of robot links and a list of robot joints, but we have to sanity-check them first
            links_ = links;
            joints_ = joints;
            if (!SanityCheckRobotModel(links_, joints_))
            {
                throw std::invalid_argument("Attempted to construct a SimpleLinkedRobot with an invalid robot model");
            }
            num_active_joints_ = GetNumActiveJoints();
            // Generate the self colllision map
            self_collision_map_ = GenerateAllowedSelfColllisionMap(links_.size(), allowed_self_collisions);
            UpdatePosition(initial_position);
            initialized_ = true;
        }

        inline void SetBaseTransform(const Eigen::Affine3d& base_transform)
        {
            base_transform_ = base_transform;
            UpdateTransforms();
        }

        inline bool CheckIfSelfCollisionAllowed(const size_t link1_index, const size_t link2_index) const
        {
            assert(link1_index < links_.size());
            assert(link2_index < links_.size());
            if (link1_index == link2_index)
            {
                return true;
            }
            int32_t stored = self_collision_map_((int64_t)link1_index, (int64_t)link2_index);
            if (stored > 0)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        inline std::vector<std::pair<std::string, EigenHelpers::VectorVector3d>> GetRawLinksPoints() const
        {
            std::vector<std::pair<std::string, EigenHelpers::VectorVector3d>> links_points(links_.size());
            for (size_t idx = 0; idx < links_.size(); idx++)
            {
                const RobotLink& current_link = links_[idx];
                links_points[idx].first = current_link.link_name;
                links_points[idx].second = current_link.link_points;
            }
            return links_points;
        }

        inline void UpdatePosition(const SimpleLinkedConfiguration& position)
        {
            SetConfig(position);
            ResetControllers();
        }

        inline Eigen::Affine3d GetLinkTransform(const std::string& link_name) const
        {
            for (size_t idx = 0; idx < links_.size(); idx++)
            {
                const RobotLink& current_link = links_[idx];
                if (current_link.link_name == link_name)
                {
                    return current_link.link_transform;
                }
            }
            throw std::invalid_argument("Invalid link_name");
        }

        inline SimpleLinkedConfiguration GetPosition() const
        {
            SimpleLinkedConfiguration configuration;
            configuration.reserve(num_active_joints_);
            for (size_t idx = 0; idx < joints_.size(); idx++)
            {
                const RobotJoint& current_joint = joints_[idx];
                // Skip fixed joints
                if (current_joint.joint_model.IsFixed())
                {
                    continue;
                }
                else
                {
                    const SimpleJointModel& current_joint_model = current_joint.joint_model;
                    configuration.push_back(current_joint_model);
                }
            }
            configuration.shrink_to_fit();
            return configuration;
        }

        inline double ComputeDistanceTo(const SimpleLinkedConfiguration& target) const
        {
            return SimpleLinkedDistancer::Distance(GetPosition(), target);
        }

        template<typename PRNG>
        inline Eigen::VectorXd GenerateControlAction(const SimpleLinkedConfiguration& target, PRNG& rng)
        {
            // Get the current position
            const SimpleLinkedConfiguration current = GetPosition();
            // Get the current error
            const Eigen::VectorXd current_error = SimpleLinkedDimDistancer::RawDistance(current, target);
            // Make the control action
            Eigen::VectorXd control_action = Eigen::VectorXd::Zero(current_error.size());
            int64_t control_idx = 0;
            for (size_t idx = 0; idx < joints_.size(); idx++)
            {
                RobotJoint& current_joint = joints_[idx];
                // Skip fixed joints
                if (current_joint.joint_model.IsFixed())
                {
                    continue;
                }
                else
                {
                    const double joint_error = current_error(control_idx);
                    const double joint_term = current_joint.joint_controller.controller.ComputeFeedbackTerm(joint_error, 1.0);
                    const double joint_control = current_joint.joint_controller.actuator.GetControlValue(joint_term, rng);
                    control_action(control_idx) = joint_control;
                    control_idx++;
                }
            }
            return control_action;
        }

        template<typename PRNG>
        inline void ApplyControlInput(const Eigen::VectorXd& input, PRNG& rng)
        {
            assert((size_t)input.size() == num_active_joints_);
            SimpleLinkedConfiguration new_config;
            new_config.reserve(num_active_joints_);
            int64_t input_idx = 0u;
            for (size_t idx = 0; idx < joints_.size(); idx++)
            {
                const RobotJoint& current_joint = joints_[idx];
                // Skip fixed joints
                if (current_joint.joint_model.IsFixed())
                {
                    continue;
                }
                else
                {
                    assert(input_idx < input.size());
                    const double input_val = input(input_idx);
                    const double current_val = current_joint.joint_model.GetValue();
                    const double raw_new_val = current_val + input_val;
                    const double noisy_new_val = current_joint.joint_controller.sensor.GetSensorValue(raw_new_val, rng);
                    new_config.push_back(current_joint.joint_model.CopyWithNewValue(noisy_new_val));
                    input_idx++;
                }
            }
            new_config.shrink_to_fit();
            // Update config
            SetConfig(new_config);
        }

        inline void ApplyControlInput(const Eigen::VectorXd& input)
        {
            assert((size_t)input.size() == num_active_joints_);
            SimpleLinkedConfiguration new_config;
            new_config.reserve(num_active_joints_);
            int64_t input_idx = 0u;
            for (size_t idx = 0; idx < joints_.size(); idx++)
            {
                const RobotJoint& current_joint = joints_[idx];
                // Skip fixed joints
                if (current_joint.joint_model.IsFixed())
                {
                    continue;
                }
                else
                {
                    assert(input_idx < input.size());
                    const double input_val = input(input_idx);
                    const double current_val = current_joint.joint_model.GetValue();
                    const double raw_new_val = current_val + input_val;
                    new_config.push_back(current_joint.joint_model.CopyWithNewValue(raw_new_val));
                    input_idx++;
                }
            }
            new_config.shrink_to_fit();
            // Update config
            SetConfig(new_config);
        }

        inline Eigen::Matrix<double, 6, Eigen::Dynamic> ComputeFullLinkPointJacobian(const std::string& link_name, const Eigen::Vector3d& link_relative_point) const
        {
            // Get the link transform (by extension, this ensures we've been given a valid link)
            const Eigen::Affine3d link_transform = GetLinkTransform(link_name);
            // Transform the point into world frame
            const Eigen::Vector3d world_point = link_transform * link_relative_point;
            // Make the jacobian storage
            Eigen::Matrix<double, 6, Eigen::Dynamic> jacobian = Eigen::Matrix<double, 6, Eigen::Dynamic>::Zero(6, num_active_joints_);
            // First, check if the link name is a valid link name
            bool link_found = false;
            for (size_t idx = 0; idx < links_.size(); idx++)
            {
                const std::string& current_link_name = links_[idx].link_name;
                if (current_link_name == link_name)
                {
                    link_found = true;
                    break;
                }
            }
            if (link_found == false)
            {
                std::cerr << "Link " << link_name << " does not exist" << std::endl;
                return jacobian;
            }
            // Second, check if the link name is the first (root) link
            if (links_[0].link_name == link_name)
            {
                // If so, return zeros for the jacobian, because the point *CANNOT MOVE*
                return jacobian;
            }
            // Go through the kinematic chain and compute the jacobian
            int64_t joint_idx = 0;
            for (size_t idx = 0; idx < joints_.size(); idx++)
            {
                // Get the current joint
                const RobotJoint& current_joint = joints_[idx];
                // Get the child link
                const RobotLink& child_link = links_[current_joint.child_link_index];
                // Get the transform of the current joint
                const Eigen::Affine3d joint_transform = child_link.link_transform;
                // Compute the jacobian for the current joint
                if (current_joint.joint_model.IsRevolute())
                {
                    const Eigen::Vector3d joint_axis = (Eigen::Vector3d)(joint_transform.rotation() * current_joint.joint_axis);
                    jacobian.block<3,1>(0, joint_idx) = jacobian.block<3,1>(0, joint_idx) + joint_axis.cross(world_point - joint_transform.translation());
                    jacobian.block<3,1>(3, joint_idx) = jacobian.block<3,1>(3, joint_idx) + joint_axis;
                    // Increment our joint count
                    joint_idx++;
                }
                else if (current_joint.joint_model.IsPrismatic())
                {
                    const Eigen::Vector3d joint_axis = (Eigen::Vector3d)(joint_transform * current_joint.joint_axis);
                    jacobian.block<3,1>(0, joint_idx) = jacobian.block<3,1>(0, joint_idx) + joint_axis;
                    // Increment our joint count
                    joint_idx++;
                }
                else
                {
                    assert(current_joint.joint_model.IsFixed());
                    // We do nothing for fixed joints
                }
                // Check if we're done - joints after our target link aren't part of the jacobian
                if (child_link.link_name == link_name)
                {
                    break;
                }
            }
            return jacobian;
        }

        inline Eigen::Matrix<double, 3, Eigen::Dynamic> ComputeLinkPointJacobian(const std::string& link_name, const Eigen::Vector3d& link_relative_point) const
        {
            const Eigen::Matrix<double, 6, Eigen::Dynamic> full_jacobian = ComputeFullLinkPointJacobian(link_name, link_relative_point);
            Eigen::Matrix<double, 3, Eigen::Dynamic> trans_only_jacobian(3, full_jacobian.cols());
            trans_only_jacobian << full_jacobian.row(0), full_jacobian.row(1), full_jacobian.row(2);
            return trans_only_jacobian;
        }

        inline Eigen::Matrix<double, 6, Eigen::Dynamic> ComputeJacobian() const
        {
            assert(links_.size() > 0);
            const RobotLink& last_link = links_.back();
            return ComputeFullLinkPointJacobian(last_link.link_name, Eigen::Vector3d::Zero());
        }

        inline Eigen::VectorXd ProcessCorrectionAction(const Eigen::VectorXd& raw_correction_action) const
        {
            // Scale down the action
            const double action_norm = raw_correction_action.norm();
            const Eigen::VectorXd real_action = (action_norm > 0.005) ? (raw_correction_action / action_norm) * 0.005 : raw_correction_action;
            return real_action;
        }

        inline double GetMaxMotionPerStep() const
        {
            return max_motion_per_unit_step_;
        }
    };
}

#endif // SIMPLELINKED_ROBOT_HELPERS_HPP
