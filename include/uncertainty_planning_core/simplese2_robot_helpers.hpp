#include <stdio.h>
#include <vector>
#include <map>
#include <random>
#include <Eigen/Geometry>
#include <arc_utilities/eigen_helpers.hpp>
#include <uncertainty_planning_core/simple_pid_controller.hpp>
#include <uncertainty_planning_core/simple_uncertainty_models.hpp>

#ifndef SIMPLESE2_ROBOT_HELPERS_HPP
#define SIMPLESE2_ROBOT_HELPERS_HPP

namespace simplese2_robot_helpers
{
    class EigenMatrixD31Serializer
    {
    public:

        static inline std::string TypeName()
        {
            return std::string("EigenMatrixD31Serializer");
        }

        static inline uint64_t Serialize(const Eigen::Matrix<double, 3, 1>& value, std::vector<uint8_t>& buffer)
        {
            return EigenHelpers::Serialize(value, buffer);
        }

        static inline std::pair<Eigen::Matrix<double, 3, 1>, uint64_t> Deserialize(const std::vector<uint8_t>& buffer, const uint64_t current)
        {
            return EigenHelpers::Deserialize<Eigen::Matrix<double, 3, 1>>(buffer, current);
        }
    };

    class SimpleSE2BaseSampler
    {
    protected:

        std::uniform_real_distribution<double> x_distribution_;
        std::uniform_real_distribution<double> y_distribution_;
        std::uniform_real_distribution<double> zr_distribution_;

    public:

        SimpleSE2BaseSampler(const std::pair<double, double>& x_limits, const std::pair<double, double>& y_limits)
        {
            assert(x_limits.first <= x_limits.second);
            assert(y_limits.first <= y_limits.second);
            x_distribution_ = std::uniform_real_distribution<double>(x_limits.first, x_limits.second);
            y_distribution_ = std::uniform_real_distribution<double>(y_limits.first, y_limits.second);
            zr_distribution_ = std::uniform_real_distribution<double>(-M_PI, M_PI);
        }

        template<typename Generator>
        Eigen::Matrix<double, 3, 1> Sample(Generator& prng)
        {
            const double x = x_distribution_(prng);
            const double y = y_distribution_(prng);
            const double zr = zr_distribution_(prng);
            Eigen::Matrix<double, 3, 1> state;
            state << x, y, zr;
            return state;
        }

        static std::string TypeName()
        {
            return std::string("SimpleSE2BaseSampler");
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
        double r_kp;
        double r_ki;
        double r_kd;
        double r_integral_clamp;
        double r_velocity_limit;
        double r_max_sensor_noise;
        double r_max_actuator_noise;

        ROBOT_CONFIG()
        {
            kp = 0.0;
            ki = 0.0;
            kd = 0.0;
            integral_clamp = 0.0;
            velocity_limit = 0.0;
            max_sensor_noise = 0.0;
            max_actuator_noise = 0.0;
            r_kp = 0.0;
            r_ki = 0.0;
            r_kd = 0.0;
            r_integral_clamp = 0.0;
            r_velocity_limit = 0.0;
            r_max_sensor_noise = 0.0;
            r_max_actuator_noise = 0.0;
        }

        ROBOT_CONFIG(const double in_kp, const double in_ki, const double in_kd, const double in_integral_clamp, const double in_velocity_limit, const double in_max_sensor_noise, const double in_max_actuator_noise, const double in_r_kp, const double in_r_ki, const double in_r_kd, const double in_r_integral_clamp, const double in_r_velocity_limit, const double in_r_max_sensor_noise, const double in_r_max_actuator_noise)
        {
            kp = in_kp;
            ki = in_ki;
            kd = in_kd;
            integral_clamp = in_integral_clamp;
            velocity_limit = in_velocity_limit;
            max_sensor_noise = in_max_sensor_noise;
            max_actuator_noise = in_max_actuator_noise;
            r_kp = in_r_kp;
            r_ki = in_r_ki;
            r_kd = in_r_kd;
            r_integral_clamp = in_r_integral_clamp;
            r_velocity_limit = in_r_velocity_limit;
            r_max_sensor_noise = in_r_max_sensor_noise;
            r_max_actuator_noise = in_r_max_actuator_noise;
        }
    };

    class SimpleSE2Robot
    {
    protected:

        bool initialized_;
        double max_motion_per_unit_step_;
        simple_pid_controller::SimplePIDController x_axis_controller_;
        simple_pid_controller::SimplePIDController y_axis_controller_;
        simple_pid_controller::SimplePIDController zr_axis_controller_;
        simple_uncertainty_models::TruncatedNormalUncertainSensor x_axis_sensor_;
        simple_uncertainty_models::TruncatedNormalUncertainSensor y_axis_sensor_;
        simple_uncertainty_models::TruncatedNormalUncertainSensor zr_axis_sensor_;
        simple_uncertainty_models::TruncatedNormalUncertainVelocityActuator x_axis_actuator_;
        simple_uncertainty_models::TruncatedNormalUncertainVelocityActuator y_axis_actuator_;
        simple_uncertainty_models::TruncatedNormalUncertainVelocityActuator zr_axis_actuator_;
        Eigen::Affine3d pose_;
        Eigen::Matrix<double, 3, 1> config_;
        std::shared_ptr<EigenHelpers::VectorVector3d> link_points_;

        inline void SetConfig(const Eigen::Matrix<double, 3, 1>& new_config)
        {
            //std::cout << "Raw config to set: " << new_config << std::endl;
            config_(0) = new_config(0);
            config_(1) = new_config(1);
            config_(2) = EigenHelpers::EnforceContinuousRevoluteBounds(new_config(2));
            //std::cout << "Real config set: " << config_ << std::endl;
            // Update pose
            pose_ = ComputePose();
        }

    public:

        inline double ComputeMaxMotionPerStep() const
        {
            double max_motion = 0.0;
            const std::vector<std::pair<std::string, std::shared_ptr<EigenHelpers::VectorVector3d>>> robot_links_points = GetRawLinksPoints();
            // Generate motion primitives
            std::vector<Eigen::VectorXd> motion_primitives;
            motion_primitives.reserve(6);
            for (ssize_t joint_idx = 0; joint_idx < 3; joint_idx++)
            {
                Eigen::VectorXd raw_motion_plus = Eigen::VectorXd::Zero(3);
                raw_motion_plus(joint_idx) = 0.125;
                motion_primitives.push_back(raw_motion_plus);
                Eigen::VectorXd raw_motion_neg = Eigen::VectorXd::Zero(3);
                raw_motion_neg(joint_idx) = -0.125;
                motion_primitives.push_back(raw_motion_neg);
            }
            // Go through the robot model & compute how much it moves
            for (size_t link_idx = 0; link_idx < robot_links_points.size(); link_idx++)
            {
                // Grab the link name and points
                const std::string& link_name = robot_links_points[link_idx].first;
                const EigenHelpers::VectorVector3d& link_points = *(robot_links_points[link_idx].second);
                // Now, go through the points of the link
                for (size_t point_idx = 0; point_idx < link_points.size(); point_idx++)
                {
                    const Eigen::Vector3d& link_relative_point = link_points[point_idx];
                    // Get the Jacobian for the current point
                    const Eigen::Matrix<double, 3, Eigen::Dynamic> point_jacobian = ComputeLinkPointJacobian(link_name, link_relative_point);
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
            return (max_motion * 8.0);
        }

        inline SimpleSE2Robot(const std::shared_ptr<EigenHelpers::VectorVector3d>& robot_points, const Eigen::Matrix<double, 3, 1>& initial_position, const ROBOT_CONFIG& robot_config) : link_points_(robot_points)
        {
            x_axis_controller_ = simple_pid_controller::SimplePIDController(robot_config.kp, robot_config.ki, robot_config.kd, robot_config.integral_clamp);
            y_axis_controller_ = simple_pid_controller::SimplePIDController(robot_config.kp, robot_config.ki, robot_config.kd, robot_config.integral_clamp);
            zr_axis_controller_ = simple_pid_controller::SimplePIDController(robot_config.r_kp, robot_config.r_ki, robot_config.r_kd, robot_config.r_integral_clamp);
            x_axis_sensor_ = simple_uncertainty_models::TruncatedNormalUncertainSensor(-robot_config.max_sensor_noise, robot_config.max_sensor_noise);
            y_axis_sensor_ = simple_uncertainty_models::TruncatedNormalUncertainSensor(-robot_config.max_sensor_noise, robot_config.max_sensor_noise);
            zr_axis_sensor_ = simple_uncertainty_models::TruncatedNormalUncertainSensor(-robot_config.r_max_sensor_noise, robot_config.r_max_sensor_noise);
            x_axis_actuator_ = simple_uncertainty_models::TruncatedNormalUncertainVelocityActuator(robot_config.max_actuator_noise, robot_config.velocity_limit);
            y_axis_actuator_ = simple_uncertainty_models::TruncatedNormalUncertainVelocityActuator(robot_config.max_actuator_noise, robot_config.velocity_limit);
            zr_axis_actuator_ = simple_uncertainty_models::TruncatedNormalUncertainVelocityActuator(robot_config.r_max_actuator_noise, robot_config.r_velocity_limit);
            ResetPosition(initial_position);
            max_motion_per_unit_step_ = ComputeMaxMotionPerStep();
            initialized_ = true;
        }

        inline std::vector<std::pair<std::string, std::shared_ptr<EigenHelpers::VectorVector3d>>> GetRawLinksPoints() const
        {
            return std::vector<std::pair<std::string, std::shared_ptr<EigenHelpers::VectorVector3d>>>{std::make_pair("robot", link_points_)};
        }

        inline bool CheckIfSelfCollisionAllowed(const size_t link1_index, const size_t link2_index) const
        {
            UNUSED(link1_index);
            UNUSED(link2_index);
            return true;
        }

        inline void UpdatePosition(const Eigen::Matrix<double, 3, 1>& position)
        {
            SetConfig(position);
        }

        inline void ResetPosition(const Eigen::Matrix<double, 3, 1>& position)
        {
            SetConfig(position);
            x_axis_controller_.Zero();
            y_axis_controller_.Zero();
            zr_axis_controller_.Zero();
        }

        inline Eigen::Affine3d GetLinkTransform(const std::string& link_name) const
        {
            if (link_name == "robot")
            {
                return pose_;
            }
            else
            {
                throw std::invalid_argument("Invalid link_name");
            }
        }

        inline Eigen::Matrix<double, 3, 1> GetPosition() const
        {
            return config_;
        }

        template<typename PRNG>
        inline Eigen::Matrix<double, 3, 1> GetPosition(PRNG& rng) const
        {
            const Eigen::Matrix<double, 3, 1> current_config = GetPosition();
            Eigen::Matrix<double, 3, 1> noisy_config = Eigen::Matrix<double, 3, 1>::Zero();
            noisy_config(0) = x_axis_sensor_.GetSensorValue(current_config(0), rng);
            noisy_config(1) = y_axis_sensor_.GetSensorValue(current_config(1), rng);
            noisy_config(2) = zr_axis_sensor_.GetSensorValue(current_config(2), rng);
            return noisy_config;
        }

        inline double ComputeDistanceTo(const Eigen::Matrix<double, 3, 1>& target) const
        {
            return ComputeConfigurationDistance(GetPosition(), target);
        }

        template<typename PRNG>
        inline Eigen::VectorXd GenerateControlAction(const Eigen::Matrix<double, 3, 1>& target, const double controller_interval, PRNG& rng)
        {
            // Get the current position
            const Eigen::Matrix<double, 3, 1> current = GetPosition(rng);
            // Get the current error
            const Eigen::VectorXd current_error = ComputePerDimensionConfigurationRawDistance(current, target);
            // Compute feedback terms
            const double x_term = x_axis_controller_.ComputeFeedbackTerm(current_error(0), controller_interval);
            const double y_term = y_axis_controller_.ComputeFeedbackTerm(current_error(1), controller_interval);
            const double zr_term = zr_axis_controller_.ComputeFeedbackTerm(current_error(2), controller_interval);
            // Make the control action
            const double x_axis_control = x_axis_actuator_.GetControlValue(x_term);
            const double y_axis_control = y_axis_actuator_.GetControlValue(y_term);
            const double zr_axis_control = zr_axis_actuator_.GetControlValue(zr_term);
            Eigen::VectorXd control_action(3);
            control_action(0) = x_axis_control;
            control_action(1) = y_axis_control;
            control_action(2) = zr_axis_control;
            return control_action;
        }

        template<typename PRNG>
        inline void ApplyControlInput(const Eigen::VectorXd& input, PRNG& rng)
        {
            assert(input.size() == 3);
            // Sense new noisy config
            Eigen::Matrix<double, 3, 1> noisy_input = Eigen::Matrix<double, 3, 1>::Zero();
            noisy_input(0) = x_axis_actuator_.GetControlValue(input(0), rng);
            noisy_input(1) = y_axis_actuator_.GetControlValue(input(1), rng);
            noisy_input(2) = zr_axis_actuator_.GetControlValue(input(2), rng);
            // Compute new config
            const Eigen::Matrix<double, 3, 1> new_config = GetPosition() + noisy_input;
            // Update config
            SetConfig(new_config);
        }

        inline void ApplyControlInput(const Eigen::VectorXd& input)
        {
            assert(input.size() == 3);
            Eigen::Matrix<double, 3, 1> real_input = Eigen::Matrix<double, 3, 1>::Zero();
            real_input(0) = x_axis_actuator_.GetControlValue(input(0));
            real_input(1) = y_axis_actuator_.GetControlValue(input(1));
            real_input(2) = zr_axis_actuator_.GetControlValue(input(2));
            // Compute new config
            const Eigen::Matrix<double, 3, 1> new_config = GetPosition() + real_input;
            // Update config
            SetConfig(new_config);
        }

        inline Eigen::Matrix<double, 3, Eigen::Dynamic> ComputeLinkPointJacobian(const std::string& link_name, const Eigen::Vector3d& link_relative_point) const
        {
            if (link_name == "robot")
            {
                const Eigen::Matrix<double, 3, 1> current_config = GetPosition();
                // Transform the point into world frame
                const Eigen::Affine3d current_transform = GetLinkTransform("robot");
                const Eigen::Vector3d current_position(current_config(0), current_config(1), 0.0);
                const Eigen::Vector3d world_point = current_transform * link_relative_point;
                // Make the jacobian
                Eigen::Matrix<double, 3, 3> jacobian = Eigen::Matrix<double, 3, 3>::Zero();
                // Prismatic joints
                // X joint
                jacobian.block<3,1>(0, 0) = jacobian.block<3,1>(0, 0) + Eigen::Vector3d::UnitX();
                // Y joint
                jacobian.block<3,1>(0, 1) = jacobian.block<3,1>(0, 1) + Eigen::Vector3d::UnitY();
                // Rotatational joints
                // Compute Z-axis joint axis
                const Eigen::Affine3d z_joint_transform = current_transform;
                const Eigen::Vector3d z_joint_axis = (Eigen::Vector3d)(z_joint_transform.rotation() * Eigen::Vector3d::UnitZ());
                jacobian.block<3,1>(0, 2) = jacobian.block<3,1>(0, 2) + z_joint_axis.cross(world_point - current_position);
                return jacobian;
            }
            else
            {
                throw std::invalid_argument("Invalid link_name");
            }
        }

        inline Eigen::Affine3d ComputePose() const
        {
            const Eigen::Matrix<double, 3, 1> current_config = GetPosition();
            const Eigen::Translation3d current_position(current_config(0), current_config(1), 0.0);
            const double current_z_angle = current_config(2);
            const Eigen::Affine3d z_joint_transform = current_position * Eigen::Quaterniond::Identity();
            const Eigen::Affine3d body_transform = z_joint_transform * (Eigen::Translation3d::Identity() * Eigen::Quaterniond(Eigen::AngleAxisd(current_z_angle, Eigen::Vector3d::UnitZ())));
            return body_transform;
        }

        inline Eigen::VectorXd ProcessCorrectionAction(const Eigen::VectorXd& raw_correction_action) const
        {
            assert(raw_correction_action.size() == 3);
            // We treat translation and rotation separately
            const Eigen::VectorXd raw_translation_correction = raw_correction_action.head<2>();
            const double raw_rotation_correction = raw_correction_action(2);
            // Scale down the translation
            const double translation_action_norm = raw_translation_correction.norm();
            const Eigen::VectorXd real_translation_correction = (translation_action_norm > 0.005) ? (raw_translation_correction / translation_action_norm) * 0.005 : raw_translation_correction;
            // Scale down the rotation
            const double rotation_action_norm = std::abs(raw_rotation_correction);
            const double real_rotation_correction = (rotation_action_norm > 0.05) ? (raw_rotation_correction / rotation_action_norm) * 0.05 : raw_rotation_correction;
            // Put them back together
            Eigen::VectorXd real_correction(3);
            real_correction << real_translation_correction, real_rotation_correction;
            return real_correction;
        }

        inline double GetMaxMotionPerStep() const
        {
            return max_motion_per_unit_step_;
        }


        inline Eigen::Matrix<double, 3, 1> AverageConfigurations(const std::vector<Eigen::Matrix<double, 3, 1>>& configurations) const
        {
            if (configurations.size() > 0)
            {
                // Separate translation and rotation values
                std::vector<Eigen::VectorXd> translations(configurations.size());
                std::vector<double> zrs(configurations.size());
                for (size_t idx = 0; idx < configurations.size(); idx++)
                {
                    const Eigen::Matrix<double, 3, 1>& state = configurations[idx];
                    Eigen::VectorXd trans_state(2);
                    trans_state << state(0), state(1);
                    translations[idx] = trans_state;
                    zrs[idx] = state(2);
                }
                // Get the average values
                const Eigen::VectorXd average_translation = EigenHelpers::AverageEigenVectorXd(translations);
                const double average_zr = EigenHelpers::AverageContinuousRevolute(zrs);
                Eigen::Matrix<double, 3, 1> average;
                average << average_translation, average_zr;
                return average;
            }
            else
            {
                return Eigen::Matrix<double, 3, 1>::Zero();
            }
        }

        inline Eigen::Matrix<double, 3, 1> InterpolateBetweenConfigurations(const Eigen::Matrix<double, 3, 1>& start, const Eigen::Matrix<double, 3, 1>& end, const double ratio) const
        {
            Eigen::Matrix<double, 3, 1> interpolated = Eigen::Matrix<double, 3, 1>::Zero();
            interpolated(0) = EigenHelpers::Interpolate(start(0), end(0), ratio);
            interpolated(1) = EigenHelpers::Interpolate(start(1), end(1), ratio);
            interpolated(2) = EigenHelpers::InterpolateContinuousRevolute(start(2), end(2), ratio);
            return interpolated;
        }

        inline Eigen::VectorXd ComputePerDimensionConfigurationRawDistance(const Eigen::Matrix<double, 3, 1>& config1, const Eigen::Matrix<double, 3, 1>& config2) const
        {
            Eigen::VectorXd dim_distances(3);
            dim_distances(0) = config2(0) - config1(0);
            dim_distances(1) = config2(1) - config1(1);
            dim_distances(2) = EigenHelpers::ContinuousRevoluteSignedDistance(config1(2), config2(2));;
            return dim_distances;
        }

        inline Eigen::VectorXd ComputePerDimensionConfigurationDistance(const Eigen::Matrix<double, 3, 1>& config1, const Eigen::Matrix<double, 3, 1>& config2) const
        {
            return ComputePerDimensionConfigurationRawDistance(config1, config2).cwiseAbs();
        }

        inline double ComputeConfigurationDistance(const Eigen::Matrix<double, 3, 1>& config1, const Eigen::Matrix<double, 3, 1>& config2) const
        {
            const Eigen::VectorXd dim_distances = ComputePerDimensionConfigurationRawDistance(config1, config2);
            const double trans_dist = sqrt((dim_distances(0) * dim_distances(0)) + (dim_distances(1) * dim_distances(1)));
            const double rots_dist = std::abs(dim_distances(2));
            return trans_dist + rots_dist;
        }
    };
}

#endif // SIMPLESE2_ROBOT_HELPERS_HPP
