#include <stdio.h>
#include <vector>
#include <map>
#include <random>
#include <Eigen/Geometry>
#include <arc_utilities/arc_helpers.hpp>
#include <arc_utilities/eigen_helpers.hpp>
#include <arc_utilities/pretty_print.hpp>
#include <uncertainty_planning_core/simple_pid_controller.hpp>
#include <uncertainty_planning_core/simple_uncertainty_models.hpp>

#ifndef SIMPLE_ROBOT_MODELS_HPP
#define SIMPLE_ROBOT_MODELS_HPP

namespace simple_robot_models
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

    struct SE2_ROBOT_CONFIG
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

        SE2_ROBOT_CONFIG()
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

        SE2_ROBOT_CONFIG(const double in_kp, const double in_ki, const double in_kd, const double in_integral_clamp, const double in_velocity_limit, const double in_max_sensor_noise, const double in_max_actuator_noise, const double in_r_kp, const double in_r_ki, const double in_r_kd, const double in_r_integral_clamp, const double in_r_velocity_limit, const double in_r_max_sensor_noise, const double in_r_max_actuator_noise)
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

        Eigen::Isometry3d pose_;
        Eigen::Matrix<double, 3, 1> config_;
        std::shared_ptr<EigenHelpers::VectorVector4d> link_points_;
        simple_pid_controller::SimplePIDController x_axis_controller_;
        simple_pid_controller::SimplePIDController y_axis_controller_;
        simple_pid_controller::SimplePIDController zr_axis_controller_;
        simple_uncertainty_models::TruncatedNormalUncertainSensor x_axis_sensor_;
        simple_uncertainty_models::TruncatedNormalUncertainSensor y_axis_sensor_;
        simple_uncertainty_models::TruncatedNormalUncertainSensor zr_axis_sensor_;
        simple_uncertainty_models::TruncatedNormalUncertainVelocityActuator x_axis_actuator_;
        simple_uncertainty_models::TruncatedNormalUncertainVelocityActuator y_axis_actuator_;
        simple_uncertainty_models::TruncatedNormalUncertainVelocityActuator zr_axis_actuator_;
        bool initialized_;

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

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        inline SimpleSE2Robot(const std::shared_ptr<EigenHelpers::VectorVector4d>& robot_points, const Eigen::Matrix<double, 3, 1>& initial_position, const SE2_ROBOT_CONFIG& robot_config) : link_points_(robot_points)
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
            initialized_ = true;
        }

        inline std::vector<std::pair<std::string, std::shared_ptr<EigenHelpers::VectorVector4d>>> GetRawLinksPoints() const
        {
            return std::vector<std::pair<std::string, std::shared_ptr<EigenHelpers::VectorVector4d>>>{std::make_pair("robot", link_points_)};
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
            ResetControllers();
        }

        inline void ResetControllers()
        {
            x_axis_controller_.Zero();
            y_axis_controller_.Zero();
            zr_axis_controller_.Zero();
        }

        inline std::vector<simple_pid_controller::PIDParams> GetControllersParameters() const
        {
            return std::vector<simple_pid_controller::PIDParams>{x_axis_controller_.GetParams(), y_axis_controller_.GetParams(), zr_axis_controller_.GetParams()};
        }


        inline const simple_uncertainty_models::TruncatedNormalUncertainVelocityActuator& GetActuatorModel(const size_t joint_idx) const
        {
            if (joint_idx == 0)
            {
                return x_axis_actuator_;
            }
            else if (joint_idx == 1)
            {
                return y_axis_actuator_;
            }
            else if (joint_idx == 2)
            {
                return zr_axis_actuator_;
            }
            else
            {
                assert(false);
            }
        }

        inline Eigen::Isometry3d GetLinkTransform(const std::string& link_name) const
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

        static inline Eigen::VectorXd GetDeltaBetween(const Eigen::Matrix<double, 3, 1>& start, const Eigen::Matrix<double, 3, 1>& target)
        {
            return ComputePerDimensionConfigurationRawDistance(start, target);
        }

        inline Eigen::VectorXd GetDeltaTo(const Eigen::Matrix<double, 3, 1>& target) const
        {
            return ComputePerDimensionConfigurationRawDistance(GetPosition(), target);
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

        inline Eigen::VectorXd GenerateControlAction(const Eigen::Matrix<double, 3, 1>& target, const double controller_interval)
        {
            // Get the current position
            const Eigen::Matrix<double, 3, 1> current = GetPosition();
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

        inline Eigen::VectorXd GetMaxVelocities() const
        {
            Eigen::VectorXd max_velocities = Eigen::VectorXd::Zero(3);
            max_velocities(0) = x_axis_actuator_.GetMaxVelocity();
            max_velocities(1) = y_axis_actuator_.GetMaxVelocity();
            max_velocities(2) = zr_axis_actuator_.GetMaxVelocity();
            return max_velocities;
        }

        inline Eigen::VectorXd GetMaxVelocityNoise(const Eigen::VectorXd& velocities) const
        {
            assert(velocities.size() == 3);
            Eigen::VectorXd max_velocity_noise = Eigen::VectorXd::Zero(3);
            max_velocity_noise(0) = x_axis_actuator_.GetMaxVelocityNoise(velocities(0));
            max_velocity_noise(1) = y_axis_actuator_.GetMaxVelocityNoise(velocities(1));
            max_velocity_noise(2) = zr_axis_actuator_.GetMaxVelocityNoise(velocities(2));
            return max_velocity_noise;
        }

        inline Eigen::VectorXd GetMaxVelocityNoise() const
        {
            return GetMaxVelocityNoise(GetMaxVelocities());
        }

        inline Eigen::Matrix<double, 3, Eigen::Dynamic> ComputeLinkPointJacobian(const std::string& link_name, const Eigen::Vector4d& link_relative_point) const
        {
            if (link_name == "robot")
            {
                const Eigen::Matrix<double, 3, 1> current_config = GetPosition();
                // Transform the point into world frame
                const Eigen::Isometry3d current_transform = GetLinkTransform("robot");
                const Eigen::Vector3d current_position(current_config(0), current_config(1), 0.0);
                const Eigen::Vector3d world_point = (current_transform * link_relative_point).block<3, 1>(0, 0);
                // Make the jacobian
                Eigen::Matrix<double, 3, 3> jacobian = Eigen::Matrix<double, 3, 3>::Zero();
                // Prismatic joints
                // X joint
                jacobian.block<3,1>(0, 0) = jacobian.block<3,1>(0, 0) + Eigen::Vector3d::UnitX();
                // Y joint
                jacobian.block<3,1>(0, 1) = jacobian.block<3,1>(0, 1) + Eigen::Vector3d::UnitY();
                // Rotatational joints
                // Compute Z-axis joint axis
                const Eigen::Isometry3d z_joint_transform = current_transform;
                const Eigen::Vector3d z_joint_axis = (Eigen::Vector3d)(z_joint_transform.rotation() * Eigen::Vector3d::UnitZ());
                jacobian.block<3,1>(0, 2) = jacobian.block<3,1>(0, 2) + z_joint_axis.cross(world_point - current_position);
                return jacobian;
            }
            else
            {
                throw std::invalid_argument("Invalid link_name");
            }
        }

        inline Eigen::Isometry3d ComputePose() const
        {
            const Eigen::Matrix<double, 3, 1> current_config = GetPosition();
            const Eigen::Translation3d current_position(current_config(0), current_config(1), 0.0);
            const double current_z_angle = current_config(2);
            const Eigen::Isometry3d z_joint_transform = current_position * Eigen::Quaterniond::Identity();
            const Eigen::Isometry3d body_transform = z_joint_transform * (Eigen::Translation3d::Identity() * Eigen::Quaterniond(Eigen::AngleAxisd(current_z_angle, Eigen::Vector3d::UnitZ())));
            return body_transform;
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

        static inline Eigen::VectorXd ComputePerDimensionConfigurationRawDistance(const Eigen::Matrix<double, 3, 1>& config1, const Eigen::Matrix<double, 3, 1>& config2)
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

    class EigenIsometry3dSerializer
    {
    public:

        static inline std::string TypeName()
        {
            return std::string("EigenIsometry3dSerializer");
        }

        static inline uint64_t Serialize(const Eigen::Isometry3d& value, std::vector<uint8_t>& buffer)
        {
            return EigenHelpers::Serialize(value, buffer);
        }

        static inline std::pair<Eigen::Isometry3d, uint64_t> Deserialize(const std::vector<uint8_t>& buffer, const uint64_t current)
        {
            return EigenHelpers::Deserialize<Eigen::Isometry3d>(buffer, current);
        }
    };

    struct SE3_ROBOT_CONFIG
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

        SE3_ROBOT_CONFIG()
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

        SE3_ROBOT_CONFIG(const double in_kp, const double in_ki, const double in_kd, const double in_integral_clamp, const double in_velocity_limit, const double in_max_sensor_noise, const double in_max_actuator_noise, const double in_r_kp, const double in_r_ki, const double in_r_kd, const double in_r_integral_clamp, const double in_r_velocity_limit, const double in_r_max_sensor_noise, const double in_r_max_actuator_noise)
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

    class SimpleSE3Robot
    {
    protected:

        Eigen::Isometry3d config_;
        std::shared_ptr<EigenHelpers::VectorVector4d> link_points_;
        simple_pid_controller::SimplePIDController x_axis_controller_;
        simple_pid_controller::SimplePIDController y_axis_controller_;
        simple_pid_controller::SimplePIDController z_axis_controller_;
        simple_pid_controller::SimplePIDController xr_axis_controller_;
        simple_pid_controller::SimplePIDController yr_axis_controller_;
        simple_pid_controller::SimplePIDController zr_axis_controller_;
        simple_uncertainty_models::TruncatedNormalUncertainSensor x_axis_sensor_;
        simple_uncertainty_models::TruncatedNormalUncertainSensor y_axis_sensor_;
        simple_uncertainty_models::TruncatedNormalUncertainSensor z_axis_sensor_;
        simple_uncertainty_models::TruncatedNormalUncertainSensor xr_axis_sensor_;
        simple_uncertainty_models::TruncatedNormalUncertainSensor yr_axis_sensor_;
        simple_uncertainty_models::TruncatedNormalUncertainSensor zr_axis_sensor_;
        simple_uncertainty_models::TruncatedNormalUncertainVelocityActuator x_axis_actuator_;
        simple_uncertainty_models::TruncatedNormalUncertainVelocityActuator y_axis_actuator_;
        simple_uncertainty_models::TruncatedNormalUncertainVelocityActuator z_axis_actuator_;
        simple_uncertainty_models::TruncatedNormalUncertainVelocityActuator xr_axis_actuator_;
        simple_uncertainty_models::TruncatedNormalUncertainVelocityActuator yr_axis_actuator_;
        simple_uncertainty_models::TruncatedNormalUncertainVelocityActuator zr_axis_actuator_;
        bool initialized_;

        inline void SetConfig(const Eigen::Isometry3d& new_config)
        {
            //std::cout << "Raw config to set: " << new_config << std::endl;
            config_ = new_config;
        }

    public:

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        inline SimpleSE3Robot(const std::shared_ptr<EigenHelpers::VectorVector4d>& robot_points, const Eigen::Isometry3d& initial_position, const SE3_ROBOT_CONFIG& robot_config) : link_points_(robot_points)
        {
            x_axis_controller_ = simple_pid_controller::SimplePIDController(robot_config.kp, robot_config.ki, robot_config.kd, robot_config.integral_clamp);
            y_axis_controller_ = simple_pid_controller::SimplePIDController(robot_config.kp, robot_config.ki, robot_config.kd, robot_config.integral_clamp);
            z_axis_controller_ = simple_pid_controller::SimplePIDController(robot_config.kp, robot_config.ki, robot_config.kd, robot_config.integral_clamp);
            xr_axis_controller_ = simple_pid_controller::SimplePIDController(robot_config.r_kp, robot_config.r_ki, robot_config.r_kd, robot_config.r_integral_clamp);
            yr_axis_controller_ = simple_pid_controller::SimplePIDController(robot_config.r_kp, robot_config.r_ki, robot_config.r_kd, robot_config.r_integral_clamp);
            zr_axis_controller_ = simple_pid_controller::SimplePIDController(robot_config.r_kp, robot_config.r_ki, robot_config.r_kd, robot_config.r_integral_clamp);
            x_axis_sensor_ = simple_uncertainty_models::TruncatedNormalUncertainSensor(-robot_config.max_sensor_noise, robot_config.max_sensor_noise);
            y_axis_sensor_ = simple_uncertainty_models::TruncatedNormalUncertainSensor(-robot_config.max_sensor_noise, robot_config.max_sensor_noise);
            z_axis_sensor_ = simple_uncertainty_models::TruncatedNormalUncertainSensor(-robot_config.max_sensor_noise, robot_config.max_sensor_noise);
            xr_axis_sensor_ = simple_uncertainty_models::TruncatedNormalUncertainSensor(-robot_config.r_max_sensor_noise, robot_config.r_max_sensor_noise);
            yr_axis_sensor_ = simple_uncertainty_models::TruncatedNormalUncertainSensor(-robot_config.r_max_sensor_noise, robot_config.r_max_sensor_noise);
            zr_axis_sensor_ = simple_uncertainty_models::TruncatedNormalUncertainSensor(-robot_config.r_max_sensor_noise, robot_config.r_max_sensor_noise);
            x_axis_actuator_ = simple_uncertainty_models::TruncatedNormalUncertainVelocityActuator(robot_config.max_actuator_noise, robot_config.velocity_limit);
            y_axis_actuator_ = simple_uncertainty_models::TruncatedNormalUncertainVelocityActuator(robot_config.max_actuator_noise, robot_config.velocity_limit);
            z_axis_actuator_ = simple_uncertainty_models::TruncatedNormalUncertainVelocityActuator(robot_config.max_actuator_noise, robot_config.velocity_limit);
            xr_axis_actuator_ = simple_uncertainty_models::TruncatedNormalUncertainVelocityActuator(robot_config.r_max_actuator_noise, robot_config.r_velocity_limit);
            yr_axis_actuator_ = simple_uncertainty_models::TruncatedNormalUncertainVelocityActuator(robot_config.r_max_actuator_noise, robot_config.r_velocity_limit);
            zr_axis_actuator_ = simple_uncertainty_models::TruncatedNormalUncertainVelocityActuator(robot_config.r_max_actuator_noise, robot_config.r_velocity_limit);
            ResetPosition(initial_position);
            initialized_ = true;
        }

        inline std::vector<std::pair<std::string, std::shared_ptr<EigenHelpers::VectorVector4d>>> GetRawLinksPoints() const
        {
            return std::vector<std::pair<std::string, std::shared_ptr<EigenHelpers::VectorVector4d>>>{std::make_pair("robot", link_points_)};
        }

        inline bool CheckIfSelfCollisionAllowed(const size_t link1_index, const size_t link2_index) const
        {
            UNUSED(link1_index);
            UNUSED(link2_index);
            return true;
        }

        inline void UpdatePosition(const Eigen::Isometry3d& position)
        {
            SetConfig(position);
        }

        inline void ResetPosition(const Eigen::Isometry3d& position)
        {
            SetConfig(position);
            ResetControllers();
        }

        inline void ResetControllers()
        {
            x_axis_controller_.Zero();
            y_axis_controller_.Zero();
            z_axis_controller_.Zero();
            xr_axis_controller_.Zero();
            yr_axis_controller_.Zero();
            zr_axis_controller_.Zero();
        }

        inline std::vector<simple_pid_controller::PIDParams> GetControllersParameters() const
        {
            return std::vector<simple_pid_controller::PIDParams>{x_axis_controller_.GetParams(), y_axis_controller_.GetParams(), z_axis_controller_.GetParams(), xr_axis_controller_.GetParams(), yr_axis_controller_.GetParams(), zr_axis_controller_.GetParams()};
        }

        inline const simple_uncertainty_models::TruncatedNormalUncertainVelocityActuator& GetActuatorModel(const size_t joint_idx) const
        {
            if (joint_idx == 0)
            {
                return x_axis_actuator_;
            }
            else if (joint_idx == 1)
            {
                return y_axis_actuator_;
            }
            else if (joint_idx == 2)
            {
                return z_axis_actuator_;
            }
            else if (joint_idx == 3)
            {
                return xr_axis_actuator_;
            }
            else if (joint_idx == 4)
            {
                return yr_axis_actuator_;
            }
            else if (joint_idx == 5)
            {
                return zr_axis_actuator_;
            }
            else
            {
                assert(false);
            }
        }

        inline Eigen::Isometry3d GetLinkTransform(const std::string& link_name) const
        {
            if (link_name == "robot")
            {
                return GetPosition();
            }
            else
            {
                throw std::invalid_argument("Invalid link_name");
            }
        }

        inline Eigen::Isometry3d GetPosition() const
        {
            return config_;
        }

        template<typename PRNG>
        inline Eigen::Isometry3d GetPosition(PRNG& rng) const
        {
            const Eigen::Matrix<double, 6, 1> twist = EigenHelpers::TwistBetweenTransforms(Eigen::Isometry3d::Identity(), GetPosition());
            Eigen::Matrix<double, 6, 1> noisy_twist = Eigen::Matrix<double, 6, 1>::Zero();
            noisy_twist(0) = x_axis_sensor_.GetSensorValue(twist(0), rng);
            noisy_twist(1) = y_axis_sensor_.GetSensorValue(twist(1), rng);
            noisy_twist(2) = z_axis_sensor_.GetSensorValue(twist(2), rng);
            noisy_twist(3) = xr_axis_sensor_.GetSensorValue(twist(3), rng);
            noisy_twist(4) = yr_axis_sensor_.GetSensorValue(twist(4), rng);
            noisy_twist(5) = zr_axis_sensor_.GetSensorValue(twist(5), rng);
            // Compute the motion transform
            const Eigen::Isometry3d motion_transform = EigenHelpers::ExpTwist(noisy_twist, 1.0);
            return motion_transform;
        }

        inline double ComputeDistanceTo(const Eigen::Isometry3d& target) const
        {
            return ComputeConfigurationDistance(GetPosition(), target);
        }

        static inline Eigen::VectorXd GetDeltaBetween(const Eigen::Isometry3d& start, const Eigen::Isometry3d& target)
        {
            return EigenHelpers::TwistBetweenTransforms(start, target);
        }

        inline Eigen::VectorXd GetDeltaTo(const Eigen::Isometry3d& target) const
        {
            return EigenHelpers::TwistBetweenTransforms(GetPosition(), target);
        }

        template<typename PRNG>
        inline Eigen::VectorXd GenerateControlAction(const Eigen::Isometry3d& target, const double controller_interval, PRNG& rng)
        {
            // Get the current position
            const Eigen::Isometry3d current = GetPosition(rng);
            // Get the twist from the current position to the target position
            const Eigen::Matrix<double, 6, 1> twist = EigenHelpers::TwistBetweenTransforms(current, target);
            // Compute feedback terms
            const double x_term = x_axis_controller_.ComputeFeedbackTerm(twist(0), controller_interval);
            const double y_term = y_axis_controller_.ComputeFeedbackTerm(twist(1), controller_interval);
            const double z_term = z_axis_controller_.ComputeFeedbackTerm(twist(2), controller_interval);
            const double xr_term = xr_axis_controller_.ComputeFeedbackTerm(twist(3), controller_interval);
            const double yr_term = yr_axis_controller_.ComputeFeedbackTerm(twist(4), controller_interval);
            const double zr_term = zr_axis_controller_.ComputeFeedbackTerm(twist(5), controller_interval);
            // Make the control action
            const double x_axis_control = x_axis_actuator_.GetControlValue(x_term);
            const double y_axis_control = y_axis_actuator_.GetControlValue(y_term);
            const double z_axis_control = z_axis_actuator_.GetControlValue(z_term);
            const double xr_axis_control = xr_axis_actuator_.GetControlValue(xr_term);
            const double yr_axis_control = yr_axis_actuator_.GetControlValue(yr_term);
            const double zr_axis_control = zr_axis_actuator_.GetControlValue(zr_term);
            Eigen::VectorXd control_action(6);
            control_action(0) = x_axis_control;
            control_action(1) = y_axis_control;
            control_action(2) = z_axis_control;
            control_action(3) = xr_axis_control;
            control_action(4) = yr_axis_control;
            control_action(5) = zr_axis_control;
            return control_action;
        }

        inline Eigen::VectorXd GenerateControlAction(const Eigen::Isometry3d& target, const double controller_interval)
        {
            // Get the current position
            const Eigen::Isometry3d current = GetPosition();
            // Get the twist from the current position to the target position
            const Eigen::Matrix<double, 6, 1> twist = EigenHelpers::TwistBetweenTransforms(current, target);
            // Compute feedback terms
            const double x_term = x_axis_controller_.ComputeFeedbackTerm(twist(0), controller_interval);
            const double y_term = y_axis_controller_.ComputeFeedbackTerm(twist(1), controller_interval);
            const double z_term = z_axis_controller_.ComputeFeedbackTerm(twist(2), controller_interval);
            const double xr_term = xr_axis_controller_.ComputeFeedbackTerm(twist(3), controller_interval);
            const double yr_term = yr_axis_controller_.ComputeFeedbackTerm(twist(4), controller_interval);
            const double zr_term = zr_axis_controller_.ComputeFeedbackTerm(twist(5), controller_interval);
            // Make the control action
            const double x_axis_control = x_axis_actuator_.GetControlValue(x_term);
            const double y_axis_control = y_axis_actuator_.GetControlValue(y_term);
            const double z_axis_control = z_axis_actuator_.GetControlValue(z_term);
            const double xr_axis_control = xr_axis_actuator_.GetControlValue(xr_term);
            const double yr_axis_control = yr_axis_actuator_.GetControlValue(yr_term);
            const double zr_axis_control = zr_axis_actuator_.GetControlValue(zr_term);
            Eigen::VectorXd control_action(6);
            control_action(0) = x_axis_control;
            control_action(1) = y_axis_control;
            control_action(2) = z_axis_control;
            control_action(3) = xr_axis_control;
            control_action(4) = yr_axis_control;
            control_action(5) = zr_axis_control;
            return control_action;
        }

        template<typename PRNG>
        inline void ApplyControlInput(const Eigen::VectorXd& input, PRNG& rng)
        {
            assert(input.size() == 6);
            // Sense new noisy config
            Eigen::Matrix<double, 6, 1> noisy_twist = Eigen::Matrix<double, 6, 1>::Zero();
            noisy_twist(0) = x_axis_actuator_.GetControlValue(input(0), rng);
            noisy_twist(1) = y_axis_actuator_.GetControlValue(input(1), rng);
            noisy_twist(2) = z_axis_actuator_.GetControlValue(input(2), rng);
            noisy_twist(3) = xr_axis_actuator_.GetControlValue(input(3), rng);
            noisy_twist(4) = yr_axis_actuator_.GetControlValue(input(4), rng);
            noisy_twist(5) = zr_axis_actuator_.GetControlValue(input(5), rng);
            // Compute the motion transform
            const Eigen::Isometry3d motion_transform = EigenHelpers::ExpTwist(noisy_twist, 1.0);
            const Eigen::Isometry3d new_config = GetPosition() * motion_transform;
            // Update config
            SetConfig(new_config);
        }

        inline void ApplyControlInput(const Eigen::VectorXd& input)
        {
            assert(input.size() == 6);
            // Sense new noisy config
            Eigen::Matrix<double, 6, 1> twist = Eigen::Matrix<double, 6, 1>::Zero();
            twist(0) = x_axis_actuator_.GetControlValue(input(0));
            twist(1) = y_axis_actuator_.GetControlValue(input(1));
            twist(2) = z_axis_actuator_.GetControlValue(input(2));
            twist(3) = xr_axis_actuator_.GetControlValue(input(3));
            twist(4) = yr_axis_actuator_.GetControlValue(input(4));
            twist(5) = zr_axis_actuator_.GetControlValue(input(5));
            // Compute the motion transform
            const Eigen::Isometry3d motion_transform = EigenHelpers::ExpTwist(twist, 1.0);
            const Eigen::Isometry3d new_config = GetPosition() * motion_transform;
            // Update config
            SetConfig(new_config);
        }

        inline Eigen::VectorXd GetMaxVelocities() const
        {
            Eigen::VectorXd max_velocities = Eigen::VectorXd::Zero(6);
            max_velocities(0) = x_axis_actuator_.GetMaxVelocity();
            max_velocities(1) = y_axis_actuator_.GetMaxVelocity();
            max_velocities(2) = z_axis_actuator_.GetMaxVelocity();
            max_velocities(3) = xr_axis_actuator_.GetMaxVelocity();
            max_velocities(4) = yr_axis_actuator_.GetMaxVelocity();
            max_velocities(5) = zr_axis_actuator_.GetMaxVelocity();
            return max_velocities;
        }

        inline Eigen::VectorXd GetMaxVelocityNoise(const Eigen::VectorXd& velocities) const
        {
            assert(velocities.size() == 6);
            Eigen::VectorXd max_velocity_noise = Eigen::VectorXd::Zero(6);
            max_velocity_noise(0) = x_axis_actuator_.GetMaxVelocityNoise(velocities(0));
            max_velocity_noise(1) = y_axis_actuator_.GetMaxVelocityNoise(velocities(1));
            max_velocity_noise(2) = z_axis_actuator_.GetMaxVelocityNoise(velocities(2));
            max_velocity_noise(3) = xr_axis_actuator_.GetMaxVelocityNoise(velocities(3));
            max_velocity_noise(4) = yr_axis_actuator_.GetMaxVelocityNoise(velocities(4));
            max_velocity_noise(5) = zr_axis_actuator_.GetMaxVelocityNoise(velocities(5));
            return max_velocity_noise;
        }

        inline Eigen::VectorXd GetMaxVelocityNoise() const
        {
            return GetMaxVelocityNoise(GetMaxVelocities());
        }

        inline Eigen::Matrix<double, 3, Eigen::Dynamic> ComputeLinkPointJacobian(const std::string& link_name, const Eigen::Vector4d& link_relative_point) const
        {
            if (link_name == "robot")
            {
                const Eigen::Matrix3d rot_matrix = GetPosition().rotation();
                const Eigen::Matrix3d hatted_link_relative_point = EigenHelpers::Skew(link_relative_point.block<3, 1>(0, 0));
                Eigen::Matrix<double, 3, 6> body_velocity_jacobian = Eigen::Matrix<double, 3, 6>::Zero();
                body_velocity_jacobian.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
                body_velocity_jacobian.block<3, 3>(0, 3) = -hatted_link_relative_point;
                #pragma GCC diagnostic push
                #pragma GCC diagnostic ignored "-Wconversion"
                const Eigen::Matrix<double, 3, 6> jacobian = rot_matrix * body_velocity_jacobian;
                #pragma GCC diagnostic pop
                return jacobian;
            }
            else
            {
                throw std::invalid_argument("Invalid link_name");
            }
        }

        inline Eigen::Isometry3d AverageConfigurations(const EigenHelpers::VectorIsometry3d& configurations) const
        {
            if (configurations.size() > 0)
            {
                return EigenHelpers::AverageEigenIsometry3d(configurations);
            }
            else
            {
                return Eigen::Isometry3d::Identity();
            }
        }

        inline Eigen::Isometry3d InterpolateBetweenConfigurations(const Eigen::Isometry3d& start, const Eigen::Isometry3d& end, const double ratio) const
        {
            return EigenHelpers::Interpolate(start, end, ratio);
        }

        inline Eigen::VectorXd ComputePerDimensionConfigurationRawDistance(const Eigen::Isometry3d& config1, const Eigen::Isometry3d& config2) const
        {
            // This should change to use a 6dof twist instead
            const Eigen::Vector3d tc1 = config1.translation();
            const Eigen::Quaterniond qc1(config1.rotation());
            const Eigen::Vector3d tc2 = config2.translation();
            const Eigen::Quaterniond qc2(config2.rotation());
            Eigen::VectorXd dim_distances(4);
            dim_distances(0) = tc2.x() - tc1.x();
            dim_distances(1) = tc2.y() - tc1.y();
            dim_distances(2) = tc2.z() - tc1.z();
            dim_distances(3) = EigenHelpers::Distance(qc1, qc2);
            return dim_distances;
        }

        inline Eigen::VectorXd ComputePerDimensionConfigurationDistance(const Eigen::Isometry3d& config1, const Eigen::Isometry3d& config2) const
        {
            return ComputePerDimensionConfigurationRawDistance(config1, config2).cwiseAbs();
        }

        inline double ComputeConfigurationDistance(const Eigen::Isometry3d& config1, const Eigen::Isometry3d& config2) const
        {
            return (EigenHelpers::Distance(config1, config2, 0.5) * 2.0);
        }
    };

    class SimpleJointModel
    {
    public:

        enum JOINT_TYPE : uint32_t {PRISMATIC=4, REVOLUTE=1, CONTINUOUS=2, FIXED=0};

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
                limits_ = std::pair<double, double>(-std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity());
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
            arc_utilities::SerializeFixedSizePOD<double>(limits_.first, buffer);
            arc_utilities::SerializeFixedSizePOD<double>(limits_.second, buffer);
            arc_utilities::SerializeFixedSizePOD<double>(value_, buffer);
            arc_utilities::SerializeFixedSizePOD<uint32_t>((uint32_t)type_, buffer);
            // Figure out how many bytes were written
            const uint64_t end_buffer_size = buffer.size();
            const uint64_t bytes_written = end_buffer_size - start_buffer_size;
            return bytes_written;
        }

        inline uint64_t DeserializeSelf(const std::vector<uint8_t>& buffer, const uint64_t current)
        {
            assert(current < buffer.size());
            uint64_t current_position = current;
            const std::pair<double, uint64_t> deserialized_limits_first = arc_utilities::DeserializeFixedSizePOD<double>(buffer, current_position);
            limits_.first = deserialized_limits_first.first;
            current_position += deserialized_limits_first.second;
            const std::pair<double, uint64_t> deserialized_limits_second = arc_utilities::DeserializeFixedSizePOD<double>(buffer, current_position);
            limits_.second = deserialized_limits_second.first;
            current_position += deserialized_limits_second.second;
            const std::pair<double, uint64_t> deserialized_value = arc_utilities::DeserializeFixedSizePOD<double>(buffer, current_position);
            value_ = deserialized_value.first;
            current_position += deserialized_value.second;
            const std::pair<uint32_t, uint64_t> deserialized_type = arc_utilities::DeserializeFixedSizePOD<uint32_t>(buffer, current_position);
            type_ = (JOINT_TYPE)deserialized_type.first;
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

        inline JOINT_TYPE GetType() const
        {
            return type_;
        }

        inline bool IsSameType(const SimpleJointModel& other) const
        {
            if (type_ == other.GetType())
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        inline std::string GetTypeString() const
        {
            if (IsContinuous())
            {
                return std::string("Continuous");
            }
            else if (IsRevolute())
            {
                return std::string("Revolute");
            }
            else if (IsPrismatic())
            {
                return std::string("Prismatic");
            }
            else if (IsFixed())
            {
                return std::string("Fixed");
            }
            else
            {
                assert(false);
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
            assert(std::isnan(value) == false);
            assert(std::isinf(value) == false);
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

        inline double SignedDistance(const SimpleJointModel& other) const
        {
            assert(IsSameType(other));
            return SignedDistance(GetValue(), other.GetValue());
        }

        inline double Distance(const double v1, const double v2) const
        {
            return std::abs(SignedDistance(v1, v2));
        }

        inline double Distance(const double v) const
        {
            return std::abs(SignedDistance(GetValue(), v));
        }

        inline double Distance(const SimpleJointModel& other) const
        {
            assert(IsSameType(other));
            return std::abs(SignedDistance(GetValue(), other.GetValue()));
        }

        inline SimpleJointModel CopyWithNewValue(const double value) const
        {
            return SimpleJointModel(limits_, value, type_);
        }
    };

    std::ostream& operator<<(std::ostream& strm, const SimpleJointModel& joint_model)
    {
        const std::pair<double, double> limits = joint_model.GetLimits();
        strm << joint_model.GetValue() << "[" << limits.first << "," << limits.second << ")";
        return strm;
    }

    class SimpleLinkedConfigurationSerializer
    {
    public:

        static inline std::string TypeName()
        {
            return std::string("SimpleLinkedConfigurationSerializer");
        }

        static inline uint64_t Serialize(const std::vector<SimpleJointModel>& value, std::vector<uint8_t>& buffer)
        {
            return arc_utilities::SerializeVector<SimpleJointModel>(value, buffer, SimpleJointModel::Serialize);
        }

        static inline std::pair<std::vector<SimpleJointModel>, uint64_t> Deserialize(const std::vector<uint8_t>& buffer, const uint64_t current)
        {
            return arc_utilities::DeserializeVector<SimpleJointModel>(buffer, current, SimpleJointModel::Deserialize);
        }
    };

    // Typedef to make our life a bit easier
    typedef std::vector<SimpleJointModel> SimpleLinkedConfiguration;

    struct LINKED_ROBOT_CONFIG
    {
        double kp;
        double ki;
        double kd;
        double integral_clamp;
        double velocity_limit;
        double max_sensor_noise;
        double max_actuator_noise;

        LINKED_ROBOT_CONFIG()
        {
            kp = 0.0;
            ki = 0.0;
            kd = 0.0;
            integral_clamp = 0.0;
            velocity_limit = 0.0;
            max_sensor_noise = 0.0;
            max_actuator_noise = 0.0;
        }

        LINKED_ROBOT_CONFIG(const double in_kp, const double in_ki, const double in_kd, const double in_integral_clamp, const double in_velocity_limit, const double in_max_sensor_noise, const double in_max_actuator_noise)
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

    template<typename ActuatorModel>
    struct JointControllerGroup
    {
        simple_pid_controller::SimplePIDController controller;
        simple_uncertainty_models::TruncatedNormalUncertainSensor sensor;
        ActuatorModel actuator;

        JointControllerGroup(const LINKED_ROBOT_CONFIG& config, const ActuatorModel& actuator_model) : actuator(actuator_model)
        {
            controller = simple_pid_controller::SimplePIDController(config.kp, config.ki, config.kd, config.integral_clamp);
            sensor = simple_uncertainty_models::TruncatedNormalUncertainSensor(-config.max_sensor_noise, config.max_sensor_noise);
        }

        JointControllerGroup(const LINKED_ROBOT_CONFIG& config) : actuator(ActuatorModel())
        {
            controller = simple_pid_controller::SimplePIDController(config.kp, config.ki, config.kd, config.integral_clamp);
            sensor = simple_uncertainty_models::TruncatedNormalUncertainSensor(-config.max_sensor_noise, config.max_sensor_noise);
        }

        JointControllerGroup() : actuator(ActuatorModel())
        {
            controller = simple_pid_controller::SimplePIDController(0.0, 0.0, 0.0, 0.0);
            sensor = simple_uncertainty_models::TruncatedNormalUncertainSensor(0.0, 0.0);
        }
    };

    struct RobotLink
    {
        Eigen::Isometry3d link_transform;
        std::shared_ptr<EigenHelpers::VectorVector4d> link_points;
        std::string link_name;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        RobotLink() : link_points(new EigenHelpers::VectorVector4d()) {}

        RobotLink(const std::shared_ptr<EigenHelpers::VectorVector4d>& points, const std::string& name) : link_points(points), link_name(name) {}

        inline void SetLinkPoints(const std::shared_ptr<EigenHelpers::VectorVector4d>& points)
        {
            link_points = points;
        }
    };

    template<typename ActuatorModel>
    struct RobotJoint
    {
        Eigen::Isometry3d joint_transform;
        Eigen::Vector3d joint_axis;
        int64_t parent_link_index;
        int64_t child_link_index;
        SimpleJointModel joint_model;
        JointControllerGroup<ActuatorModel> joint_controller;
        std::string name;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    template<typename ActuatorModel>
    class SimpleLinkedRobot
    {
    protected:

        struct KinematicSkeletonElement
        {
            int64_t parent_link_index_;
            int64_t parent_joint_index_;
            std::vector<int64_t> child_link_indices_;
            std::vector<int64_t> child_joint_indices_;

            KinematicSkeletonElement() : parent_link_index_(-1), parent_joint_index_(-1) {}
        };

        Eigen::Isometry3d base_transform_;
        Eigen::MatrixXi self_collision_map_;
        std::vector<RobotLink> links_;
        std::vector<RobotJoint<ActuatorModel>> joints_;
        std::vector<KinematicSkeletonElement> kinematic_skeleton_;
        std::map<int64_t, int64_t> active_joint_index_map_;
        std::vector<double> joint_distance_weights_;
        size_t num_active_joints_;
        bool initialized_;

        inline void UpdateTransforms()
        {
            // Update the transform for the first link
            links_[0].link_transform = base_transform_;
            //std::cout << "Set base link " << links_[0].link_name << " transform: " << PrettyPrint::PrettyPrint(links_[0].link_transform) << std::endl;
            // Go out the kinematic chain
            for (size_t idx = 0; idx < joints_.size(); idx++)
            {
                // Get the current joint
                const RobotJoint<ActuatorModel>& current_joint = joints_[idx];
                // Get the parent link
                const RobotLink& parent_link = links_[(size_t)current_joint.parent_link_index];
                // Get the child link
                RobotLink& child_link = links_[(size_t)current_joint.child_link_index];
                // Get the parent_link transform
                const Eigen::Isometry3d parent_transform = parent_link.link_transform;
                // Get the parent_link->joint transform
                const Eigen::Isometry3d parent_to_joint_transform = current_joint.joint_transform;
                //std::cout << "Parent link origin->joint " << idx << " transform: " << PrettyPrint::PrettyPrint(parent_to_joint_transform) << std::endl;
                // Compute the base->joint_transform
                const Eigen::Isometry3d complete_transform = parent_transform * parent_to_joint_transform;
                // Compute the joint transform
                if (current_joint.joint_model.IsRevolute())
                {
                    const Eigen::Isometry3d joint_transform = Eigen::Translation3d(0.0, 0.0, 0.0) * Eigen::Quaterniond(Eigen::AngleAxisd(current_joint.joint_model.GetValue(), current_joint.joint_axis));
                    const Eigen::Isometry3d child_transform = complete_transform * joint_transform;
                    child_link.link_transform = child_transform;
                    //std::cout << "Computed joint transform for revolute joint " << idx << " with value " << current_joint.joint_model.GetValue() << " transform: " << PrettyPrint::PrettyPrint(joint_transform) << std::endl;
                }
                else if (current_joint.joint_model.IsPrismatic())
                {
                    const Eigen::Translation3d joint_translation = (Eigen::Translation3d)(current_joint.joint_axis * current_joint.joint_model.GetValue());
                    const Eigen::Isometry3d joint_transform = joint_translation * Eigen::Quaterniond::Identity();
                    const Eigen::Isometry3d child_transform = complete_transform * joint_transform;
                    child_link.link_transform = child_transform;
                    //std::cout << "Computed transform for prismatic joint " << idx << " with value " << current_joint.joint_model.GetValue() << " transform: " << PrettyPrint::PrettyPrint(joint_transform) << std::endl;
                }
                else
                {
                    // Joint is fixed
                    child_link.link_transform = complete_transform;
                }
                //std::cout << "Set link " << child_link.link_name << " transform: " << PrettyPrint::PrettyPrint(child_link.link_transform) << std::endl;
            }
        }

        inline void SetConfig(const SimpleLinkedConfiguration& new_config)
        {
            //std::cout << "Setting new config\nCurrent: " << PrettyPrint::PrettyPrint(GetPosition()) << "\nNew: " << PrettyPrint::PrettyPrint(new_config) << std::endl;
            assert(new_config.size() == num_active_joints_);
            size_t config_idx = 0u;
            for (size_t idx = 0; idx < joints_.size(); idx++)
            {
                RobotJoint<ActuatorModel>& current_joint = joints_[idx];
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
            //std::cout << "Set: " << PrettyPrint::PrettyPrint(GetPosition()) << std::endl;
            // Update forward kinematics
            UpdateTransforms();
        }

        static inline std::pair<size_t, std::map<int64_t, int64_t>> MakeActiveJointIndexMap(const std::vector<RobotJoint<ActuatorModel>>& joints)
        {
            size_t num_active_joints = 0;
            std::map<int64_t, int64_t> active_joint_index_map;
            for (size_t idx = 0; idx < joints.size(); idx++)
            {
                const RobotJoint<ActuatorModel>& current_joint = joints[idx];
                // Skip fixed joints
                if (!(current_joint.joint_model.IsFixed()))
                {
                    active_joint_index_map[(int64_t)idx] = (int64_t)num_active_joints;
                    num_active_joints++;
                }
                else
                {
                    active_joint_index_map[(int64_t)idx] = -1;
                }
            }
            return std::make_pair(num_active_joints, active_joint_index_map);
        }

        static bool ExploreChildren(const std::vector<KinematicSkeletonElement>& kinematic_skeleton, const int64_t current_index, std::map<int64_t, uint8_t>& cycle_detection_map)
        {
            // Check if we've been here before
            cycle_detection_map[current_index]++;
            const uint8_t check_val = cycle_detection_map[current_index];
            if (check_val != 0x01)
            {
                std::cerr << "Invalid kinematic structure, link " << current_index << " is part of a cycle" << std::endl;
                return true;
            }
            else
            {
                const KinematicSkeletonElement& current_element = kinematic_skeleton[current_index];
                const std::vector<int64_t>& current_child_indices = current_element.child_link_indices_;
                for (size_t idx = 0; idx < current_child_indices.size(); idx++)
                {
                    // Get the current child index
                    const int64_t current_child_index = current_child_indices[idx];
                    const bool child_cycle_detected = ExploreChildren(kinematic_skeleton, current_child_index, cycle_detection_map);
                    if (child_cycle_detected)
                    {
                        return true;
                    }
                }
                return false;
            }
        }

        static bool CheckKinematicSkeleton(const std::vector<KinematicSkeletonElement>& kinematic_skeleton)
        {
            // Step through each state in the tree. Make sure that the linkage to the parent and child states are correct
            for (size_t current_index = 0; current_index < kinematic_skeleton.size(); current_index++)
            {
                // For every element, make sure all the parent<->child linkages are valid
                const KinematicSkeletonElement& current_element = kinematic_skeleton[current_index];
                // Check the linkage to the parent state
                const int64_t parent_index = current_element.parent_link_index_;
                if ((parent_index >= 0) && (parent_index < (int64_t)kinematic_skeleton.size()))
                {
                    if (parent_index != (int64_t)current_index)
                    {
                        const KinematicSkeletonElement& parent_element = kinematic_skeleton[parent_index];
                        // Make sure the corresponding parent contains the current node in the list of child indices
                        const std::vector<int64_t>& parent_child_indices = parent_element.child_link_indices_;
                        auto index_found = std::find(parent_child_indices.begin(), parent_child_indices.end(), (int64_t)current_index);
                        if (index_found == parent_child_indices.end())
                        {
                            std::cerr << "Parent element " << parent_index << " does not contain child index for current element " << current_index << std::endl;
                            return false;
                        }
                    }
                    else
                    {
                        std::cerr << "Invalid parent index " << parent_index << " for element " << current_index << " [Indices can't be the same]" << std::endl;
                        return false;
                    }
                }
                else if (parent_index < -1)
                {
                    std::cerr << "Invalid parent index " << parent_index << " for element " << current_index << std::endl;
                    return false;
                }
                // Check the linkage to the child states
                const std::vector<int64_t>& current_child_indices = current_element.child_link_indices_;
                for (size_t idx = 0; idx < current_child_indices.size(); idx++)
                {
                    // Get the current child index
                    const int64_t current_child_index = current_child_indices[idx];
                    if ((current_child_index > 0) && (current_child_index < (int64_t)kinematic_skeleton.size()))
                    {
                        if (current_child_index != (int64_t)current_index)
                        {
                            const KinematicSkeletonElement& child_element = kinematic_skeleton[current_child_index];
                            // Make sure the child node points to us as the parent index
                            const int64_t child_parent_index = child_element.parent_link_index_;
                            if (child_parent_index != (int64_t)current_index)
                            {
                                std::cerr << "Parent index " << child_parent_index << " for current child element " << current_child_index << " does not match index " << current_index << " for current element " << std::endl;
                                return false;
                            }
                        }
                        else
                        {
                            std::cerr << "Invalid child index " << current_child_index << " for element " << current_index << " [Indices can't be the same]" << std::endl;
                            return false;
                        }
                    }
                    else
                    {
                        std::cerr << "Invalid child index " << current_child_index << " for element " << current_index << std::endl;
                        return false;
                    }
                }
            }
            // Now that we know all the linkage connections are valid, we need to make sure it is connected and acyclic
            std::map<int64_t, uint8_t> cycle_detection_map;
            const bool cycle_detected = ExploreChildren(kinematic_skeleton, 0, cycle_detection_map);
            if (cycle_detected)
            {
                return false;
            }
            // Make sure we encountered every link ONCE
            for (size_t link_idx = 0; link_idx < kinematic_skeleton.size(); link_idx++)
            {
                const uint8_t check_val = arc_helpers::RetrieveOrDefault(cycle_detection_map, (int64_t)link_idx, (uint8_t)0x00);
                if (check_val != 0x01)
                {
                    std::cerr << "Invalid kinematic structure, link " << link_idx << " not reachable from the root" << std::endl;
                    return false;
                }
            }
            return true;
        }

    public:

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        static inline std::pair<std::vector<KinematicSkeletonElement>, bool> SanityCheckRobotModel(const std::vector<RobotLink>& links, const std::vector<RobotJoint<ActuatorModel>>& joints)
        {
            // Make sure we have enough links and joints
            if (links.size() != (joints.size() + 1))
            {
                std::cerr << links.size() << " links are not enough for " << joints.size() << " joints" << std::endl;
                return std::make_pair(std::vector<KinematicSkeletonElement>(), false);
            }
            // Make sure the links all have unique names
            std::map<std::string, uint32_t> link_name_check_map;
            for (size_t idx = 0; idx < links.size(); idx++)
            {
                const std::string& link_name = links[idx].link_name;
                link_name_check_map[link_name]++;
                if (link_name_check_map[link_name] > 1)
                {
                    std::cerr << "Link " << link_name << " is not unique" << std::endl;
                    return std::make_pair(std::vector<KinematicSkeletonElement>(), false);
                }
            }
            // Make sure the joints all have unique names
            std::map<std::string, uint32_t> joint_name_check_map;
            for (size_t idx = 0; idx < joints.size(); idx++)
            {
                const std::string& joint_name = joints[idx].name;
                joint_name_check_map[joint_name]++;
                if (joint_name_check_map[joint_name] > 1)
                {
                    std::cerr << "Joint " << joint_name << " is not unique" << std::endl;
                    return std::make_pair(std::vector<KinematicSkeletonElement>(), false);
                }
            }
            // Make the kinematic skeleton structure for checking
            std::vector<KinematicSkeletonElement> kinematic_skeleton(links.size(), KinematicSkeletonElement());
            for (size_t joint_idx = 0; joint_idx < joints.size(); joint_idx++)
            {
                // Get the current joint
                const RobotJoint<ActuatorModel>& current_joint = joints[joint_idx];
                // Check the joint axes
                const double joint_axis_norm = current_joint.joint_axis.norm();
                const double error = std::abs(joint_axis_norm - 1.0);
                if (error > std::numeric_limits<double>::epsilon())
                {
                    std::cerr << "Joint axis is not a unit vector" << std::endl;
                    return std::make_pair(std::vector<KinematicSkeletonElement>(), false);
                }
                // Add the edge into the kinematic skeleton
                const int64_t parent_link_index = current_joint.parent_link_index;
                const int64_t child_link_index = current_joint.child_link_index;
                // Set the child->parent relationship
                kinematic_skeleton[(size_t)child_link_index].parent_link_index_ = parent_link_index;
                kinematic_skeleton[(size_t)child_link_index].parent_joint_index_ = joint_idx;
                // Set the parent->child relationship
                kinematic_skeleton[(size_t)parent_link_index].child_link_indices_.push_back(child_link_index);
                kinematic_skeleton[(size_t)parent_link_index].child_joint_indices_.push_back(joint_idx);
            }
            // Make sure the first link is the root
            const KinematicSkeletonElement& root_element = kinematic_skeleton[0];
            if (root_element.parent_joint_index_ != -1 || root_element.parent_link_index_ != -1)
            {
                std::cerr << "Invalid kinematic structure, first link is not the root" << std::endl;
                return std::make_pair(std::vector<KinematicSkeletonElement>(), false);
            }
            const bool valid_kinematics = CheckKinematicSkeleton(kinematic_skeleton);
            return std::make_pair(kinematic_skeleton, valid_kinematics);
        }

        static inline Eigen::MatrixXi GenerateAllowedSelfColllisionMap(const size_t num_links, const std::vector<std::pair<size_t, size_t>>& allowed_self_collisions)
        {
            Eigen::MatrixXi allowed_self_collision_map = Eigen::MatrixXi::Identity((ssize_t)(num_links), (ssize_t)(num_links));
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

        inline SimpleLinkedRobot(const Eigen::Isometry3d& base_transform, const std::vector<RobotLink>& links, const std::vector<RobotJoint<ActuatorModel>>& joints, const std::vector<std::pair<size_t, size_t>>& allowed_self_collisions, const SimpleLinkedConfiguration& initial_position, const std::vector<double>& joint_distance_weights)
        {
            base_transform_ = base_transform;
            // We take a list of robot links and a list of robot joints, but we have to sanity-check them first
            links_ = links;
            joints_ = joints;
            const auto model_validity_check = SanityCheckRobotModel(links_, joints_);
            if (!model_validity_check.second)
            {
                throw std::invalid_argument("Attempted to construct a SimpleLinkedRobot with an invalid robot model");
            }
            kinematic_skeleton_ = model_validity_check.first;
            // Get information about the active joints
            const auto active_joint_query = MakeActiveJointIndexMap(joints_);
            num_active_joints_ = active_joint_query.first;
            active_joint_index_map_ = active_joint_query.second;
            assert(joint_distance_weights.size() == num_active_joints_);
            joint_distance_weights_ = EigenHelpers::Abs(joint_distance_weights);
            // Generate the self colllision map
            self_collision_map_ = GenerateAllowedSelfColllisionMap(links_.size(), allowed_self_collisions);
            ResetPosition(initial_position);
            initialized_ = true;
        }

        inline SimpleLinkedRobot()
        {
            initialized_ = false;
        }

        inline std::vector<std::string> GetActiveJointNames() const
        {
            std::vector<std::string> active_joint_names;
            active_joint_names.reserve(num_active_joints_);
            for (size_t idx = 0; idx < joints_.size(); idx++)
            {
                const RobotJoint<ActuatorModel>& current_joint = joints_[idx];
                // Skip fixed joints
                if (!(current_joint.joint_model.IsFixed()))
                {
                    active_joint_names.push_back(current_joint.name);
                }
            }
            active_joint_names.shrink_to_fit();
            assert(active_joint_names.size() == num_active_joints_);
            return active_joint_names;
        }

        inline bool IsInitialized() const
        {
            return initialized_;
        }

        inline void UpdateBaseTransform(const Eigen::Isometry3d& base_transform)
        {
            base_transform_ = base_transform;
            UpdateTransforms();
        }

        inline Eigen::Isometry3d GetBaseTransform() const
        {
            return base_transform_;
        }

        inline bool CheckIfSelfCollisionAllowed(const size_t link1_index, const size_t link2_index) const
        {
            assert(link1_index < links_.size());
            assert(link2_index < links_.size());
            if (link1_index == link2_index)
            {
                return true;
            }
            const int32_t stored = self_collision_map_((int64_t)link1_index, (int64_t)link2_index);
            if (stored > 0)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        inline std::vector<std::pair<std::string, std::shared_ptr<EigenHelpers::VectorVector4d>>> GetRawLinksPoints() const
        {
            std::vector<std::pair<std::string, std::shared_ptr<EigenHelpers::VectorVector4d>>> links_points;
            for (size_t idx = 0; idx < links_.size(); idx++)
            {
                const RobotLink& current_link = links_[idx];
                links_points.push_back(std::make_pair(current_link.link_name, current_link.link_points));
            }
            return links_points;
        }

        inline void UpdatePosition(const SimpleLinkedConfiguration& position)
        {
            SetConfig(position);
        }

        inline void ResetPosition(const SimpleLinkedConfiguration& position)
        {
            SetConfig(position);
            ResetControllers();
        }

        inline void ResetControllers()
        {
            for (size_t idx = 0; idx < joints_.size(); idx++)
            {
                RobotJoint<ActuatorModel>& current_joint = joints_[idx];
                current_joint.joint_controller.controller.Zero();
            }
        }

        inline std::vector<simple_pid_controller::PIDParams> GetControllersParameters() const
        {
            std::vector<simple_pid_controller::PIDParams> controllers_parameters;
            controllers_parameters.reserve(num_active_joints_);
            for (size_t idx = 0; idx < joints_.size(); idx++)
            {
                const RobotJoint<ActuatorModel>& current_joint = joints_[idx];
                // Skip fixed joints
                if (current_joint.joint_model.IsFixed())
                {
                    continue;
                }
                else
                {
                    controllers_parameters.push_back(current_joint.joint_controller.controller.GetParams());
                }
            }
            controllers_parameters.shrink_to_fit();
            return controllers_parameters;
        }

        inline const ActuatorModel& GetActuatorModel(const size_t joint_idx) const
        {
            size_t joint_index = 0;
            for (size_t idx = 0; idx < joints_.size(); idx++)
            {
                const RobotJoint<ActuatorModel>& current_joint = joints_[idx];
                // Skip fixed joints
                if (current_joint.joint_model.IsFixed())
                {
                    continue;
                }
                else
                {
                    if (joint_index == joint_idx)
                    {
                        return current_joint.joint_controller.actuator;
                    }
                    joint_index++;
                }
            }
            assert(false);
        }

        inline Eigen::Isometry3d GetLinkTransform(const std::string& link_name) const
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
                const RobotJoint<ActuatorModel>& current_joint = joints_[idx];
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

        template<typename PRNG>
        inline SimpleLinkedConfiguration GetPosition(PRNG& rng) const
        {
            SimpleLinkedConfiguration configuration;
            configuration.reserve(num_active_joints_);
            for (size_t idx = 0; idx < joints_.size(); idx++)
            {
                const RobotJoint<ActuatorModel>& current_joint = joints_[idx];
                // Skip fixed joints
                if (current_joint.joint_model.IsFixed())
                {
                    continue;
                }
                else
                {
                    const double current_val = current_joint.joint_model.GetValue();
                    const double sensed_val = current_joint.joint_controller.sensor.GetSensorValue(current_val, rng);
                    configuration.push_back(current_joint.joint_model.CopyWithNewValue(sensed_val));
                }
            }
            configuration.shrink_to_fit();
            return configuration;
        }

        inline double ComputeDistanceTo(const SimpleLinkedConfiguration& target) const
        {
            return ComputeConfigurationDistance(GetPosition(), target);
        }

        static inline Eigen::VectorXd GetDeltaBetween(const SimpleLinkedConfiguration& start, const SimpleLinkedConfiguration& target)
        {
            return ComputeUnweightedPerDimensionConfigurationRawDistance(start, target);
        }

        inline Eigen::VectorXd GetDeltaTo(const SimpleLinkedConfiguration& target) const
        {
            return ComputeUnweightedPerDimensionConfigurationRawDistance(GetPosition(), target);
        }

        template<typename PRNG>
        inline Eigen::VectorXd GenerateControlAction(const SimpleLinkedConfiguration& target, const double controller_interval, PRNG& rng)
        {
            // Get the current position
            const SimpleLinkedConfiguration current = GetPosition(rng);
            // Get the current error
            const Eigen::VectorXd current_error = ComputeUnweightedPerDimensionConfigurationRawDistance(current, target);
            // Make the control action
            Eigen::VectorXd control_action = Eigen::VectorXd::Zero(current_error.size());
            int64_t control_idx = 0;
            for (size_t idx = 0; idx < joints_.size(); idx++)
            {
                RobotJoint<ActuatorModel>& current_joint = joints_[idx];
                // Skip fixed joints
                if (current_joint.joint_model.IsFixed())
                {
                    continue;
                }
                else
                {
                    const double joint_error = current_error(control_idx);
                    const double joint_term = current_joint.joint_controller.controller.ComputeFeedbackTerm(joint_error, controller_interval);
                    const double joint_control = current_joint.joint_controller.actuator.GetControlValue(joint_term);
                    control_action(control_idx) = joint_control;
                    control_idx++;
                }
            }
            return control_action;
        }

        inline Eigen::VectorXd GenerateControlAction(const SimpleLinkedConfiguration& target, const double controller_interval)
        {
            // Get the current position
            const SimpleLinkedConfiguration current = GetPosition();
            // Get the current error
            const Eigen::VectorXd current_error = ComputeUnweightedPerDimensionConfigurationRawDistance(current, target);
            // Make the control action
            Eigen::VectorXd control_action = Eigen::VectorXd::Zero(current_error.size());
            int64_t control_idx = 0;
            for (size_t idx = 0; idx < joints_.size(); idx++)
            {
                RobotJoint<ActuatorModel>& current_joint = joints_[idx];
                // Skip fixed joints
                if (current_joint.joint_model.IsFixed())
                {
                    continue;
                }
                else
                {
                    const double joint_error = current_error(control_idx);
                    const double joint_term = current_joint.joint_controller.controller.ComputeFeedbackTerm(joint_error, controller_interval);
                    const double joint_control = current_joint.joint_controller.actuator.GetControlValue(joint_term);
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
                const RobotJoint<ActuatorModel>& current_joint = joints_[idx];
                // Skip fixed joints
                if (current_joint.joint_model.IsFixed())
                {
                    continue;
                }
                else
                {
                    assert(input_idx < input.size());
                    const double input_val = input(input_idx);
                    const double noisy_input_val = current_joint.joint_controller.actuator.GetControlValue(input_val, rng);
                    const double current_val = current_joint.joint_model.GetValue();
                    const double noisy_new_val = current_val + noisy_input_val;
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
                const RobotJoint<ActuatorModel>& current_joint = joints_[idx];
                // Skip fixed joints
                if (current_joint.joint_model.IsFixed())
                {
                    continue;
                }
                else
                {
                    assert(input_idx < input.size());
                    const double input_val = input(input_idx);
                    const double real_input_val = current_joint.joint_controller.actuator.GetControlValue(input_val);
                    const double current_val = current_joint.joint_model.GetValue();
                    const double raw_new_val = current_val + real_input_val;
                    new_config.push_back(current_joint.joint_model.CopyWithNewValue(raw_new_val));
                    input_idx++;
                }
            }
            new_config.shrink_to_fit();
            // Update config
            SetConfig(new_config);
        }

        inline Eigen::VectorXd GetMaxVelocities() const
        {
            Eigen::VectorXd max_velocities = Eigen::VectorXd::Zero(num_active_joints_);
            int64_t active_joint_idx = 0u;
            for (size_t idx = 0; idx < joints_.size(); idx++)
            {
                const RobotJoint<ActuatorModel>& current_joint = joints_[idx];
                // Skip fixed joints
                if (current_joint.joint_model.IsFixed())
                {
                    continue;
                }
                else
                {
                    assert(active_joint_idx < max_velocities.size());
                    max_velocities(active_joint_idx) = current_joint.joint_controller.actuator.GetMaxVelocity();
                    active_joint_idx++;
                }
            }
            return max_velocities;
        }

        inline Eigen::VectorXd GetMaxVelocityNoise(const Eigen::VectorXd& velocities) const
        {
            assert((size_t)velocities.size() == num_active_joints_);
            Eigen::VectorXd max_velocity_noise = Eigen::VectorXd::Zero(num_active_joints_);
            int64_t active_joint_idx = 0u;
            for (size_t idx = 0; idx < joints_.size(); idx++)
            {
                const RobotJoint<ActuatorModel>& current_joint = joints_[idx];
                // Skip fixed joints
                if (current_joint.joint_model.IsFixed())
                {
                    continue;
                }
                else
                {
                    assert(active_joint_idx < max_velocity_noise.size());
                    const double velocity_command = velocities(active_joint_idx);
                    max_velocity_noise(active_joint_idx) = current_joint.joint_controller.actuator.GetMaxVelocityNoise(velocity_command);
                    active_joint_idx++;
                }
            }
            return max_velocity_noise;
        }

        inline Eigen::VectorXd GetMaxVelocityNoise() const
        {
            Eigen::VectorXd max_velocity_noise = Eigen::VectorXd::Zero(num_active_joints_);
            int64_t active_joint_idx = 0u;
            for (size_t idx = 0; idx < joints_.size(); idx++)
            {
                const RobotJoint<ActuatorModel>& current_joint = joints_[idx];
                // Skip fixed joints
                if (current_joint.joint_model.IsFixed())
                {
                    continue;
                }
                else
                {
                    assert(active_joint_idx < max_velocity_noise.size());
                    max_velocity_noise(active_joint_idx) = current_joint.joint_controller.actuator.GetMaxVelocityNoise();
                    active_joint_idx++;
                }
            }
            return max_velocity_noise;
        }

        inline Eigen::Matrix<double, 6, Eigen::Dynamic> ComputeFullLinkPointJacobian(const std::string& link_name, const Eigen::Vector4d& link_relative_point) const
        {
            // Get the link transform (by extension, this ensures we've been given a valid link)
            const Eigen::Isometry3d link_transform = GetLinkTransform(link_name);
            // Transform the point into world frame
            const Eigen::Vector3d world_point = (link_transform * link_relative_point).block<3, 1>(0, 0);
            // Make the jacobian storage
            Eigen::Matrix<double, 6, Eigen::Dynamic> jacobian = Eigen::Matrix<double, 6, Eigen::Dynamic>::Zero((ssize_t)6, (ssize_t)num_active_joints_);
            // First, check if the link name is a valid link name
            int64_t link_idx = -1;
            for (size_t idx = 0; idx < links_.size(); idx++)
            {
                const std::string& current_link_name = links_[idx].link_name;
                if (current_link_name == link_name)
                {
                    link_idx = (int64_t)idx;
                    break;
                }
            }
            if (link_idx < 0)
            {
                std::cerr << "Link " << link_name << " does not exist" << std::endl;
                return jacobian;
            }
            // Second, check if the link name is the first (root) link
            if (link_idx == 0)
            {
                // If so, return zeros for the jacobian, because the point *CANNOT MOVE*
                return jacobian;
            }
            // Walk up the kinematic tree and fill in the blocks one at a time
            int64_t current_link_index = link_idx;
            while (current_link_index > 0)
            {
                const KinematicSkeletonElement& current_element = kinematic_skeleton_[current_link_index];
                const int64_t parent_joint_index = current_element.parent_joint_index_;
                assert(parent_joint_index >= 0);
                const int64_t active_joint_index = active_joint_index_map_.at(parent_joint_index);
                // Get the current joint
                const RobotJoint<ActuatorModel>& parent_joint = joints_[parent_joint_index];
                // Get the child link
                const RobotLink& child_link = links_[(size_t)parent_joint.child_link_index];
                // Get the transform of the current joint
                const Eigen::Isometry3d joint_transform = child_link.link_transform;
                // Compute the jacobian for the current joint
                if (parent_joint.joint_model.IsRevolute())
                {
                    const Eigen::Vector3d joint_axis = (Eigen::Vector3d)(joint_transform.rotation() * parent_joint.joint_axis);
                    jacobian.block<3,1>(0, active_joint_index) = joint_axis.cross(world_point - joint_transform.translation());
                    jacobian.block<3,1>(3, active_joint_index) = joint_axis;
                }
                else if (parent_joint.joint_model.IsPrismatic())
                {
                    const Eigen::Vector3d joint_axis = (Eigen::Vector3d)(joint_transform * parent_joint.joint_axis);
                    jacobian.block<3,1>(0, active_joint_index) = joint_axis;
                }
                else
                {
                    assert(parent_joint.joint_model.IsFixed());
                    // We do nothing for fixed joints
                }
                // Update
                current_link_index = current_element.parent_link_index_;
            }
            return jacobian;
        }

        inline Eigen::Matrix<double, 3, Eigen::Dynamic> ComputeLinkPointJacobian(const std::string& link_name, const Eigen::Vector4d& link_relative_point) const
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

        inline SimpleLinkedConfiguration AverageConfigurations(const std::vector<SimpleLinkedConfiguration>& configurations) const
        {
            if (configurations.size() > 0)
            {
                // Safety checks
                const SimpleLinkedConfiguration& representative_config = configurations.front();
                for (size_t idx = 0; idx < configurations.size(); idx++)
                {
                    const SimpleLinkedConfiguration& current_config = configurations[idx];
                    assert(representative_config.size() == current_config.size());
                    for (size_t jdx = 0; jdx < representative_config.size(); jdx++)
                    {
                        const SimpleJointModel& jr = representative_config[jdx];
                        const SimpleJointModel& jc = current_config[jdx];
                        assert(jr.IsRevolute() == jc.IsRevolute());
                        assert(jr.IsContinuous() == jc.IsContinuous());
                    }
                }
                // Get number of DoF
                const size_t config_size = representative_config.size();
                // Separate the joint values
                std::vector<std::vector<double>> raw_values(config_size);
                for (size_t idx = 0; idx < configurations.size(); idx++)
                {
                    const SimpleLinkedConfiguration& q = configurations[idx];
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
                    const SimpleJointModel& representative_joint = representative_config[jdx];
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

        inline SimpleLinkedConfiguration InterpolateBetweenConfigurations(const SimpleLinkedConfiguration& start, const SimpleLinkedConfiguration& end, const double ratio) const
        {
            assert(start.size() == end.size());
            // Make the interpolated config
            SimpleLinkedConfiguration interpolated;
            interpolated.reserve(start.size());
            for (size_t idx = 0; idx < start.size(); idx++)
            {
                const SimpleJointModel& j1 = start[idx];
                const SimpleJointModel& j2 = end[idx];
                assert(j1.IsRevolute() == j2.IsRevolute());
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

        static inline Eigen::VectorXd ComputeUnweightedPerDimensionConfigurationRawDistance(const SimpleLinkedConfiguration& config1, const SimpleLinkedConfiguration& config2)
        {
            assert(config1.size() == config2.size());
            Eigen::VectorXd distances = Eigen::VectorXd::Zero((ssize_t)(config1.size()));
            for (size_t idx = 0; idx < config1.size(); idx++)
            {
                const SimpleJointModel& j1 = config1[idx];
                const SimpleJointModel& j2 = config2[idx];
                assert(j1.IsRevolute() == j2.IsRevolute());
                assert(j1.IsContinuous() == j2.IsContinuous());
                distances((int64_t)idx) = j1.SignedDistance(j1.GetValue(), j2.GetValue());
            }
            return distances;
        }

        inline Eigen::VectorXd ComputePerDimensionConfigurationRawDistance(const SimpleLinkedConfiguration& config1, const SimpleLinkedConfiguration& config2) const
        {
            assert(config1.size() == config2.size());
            assert(config1.size() == num_active_joints_);
            Eigen::VectorXd distances = Eigen::VectorXd::Zero((ssize_t)(config1.size()));
            for (size_t idx = 0; idx < config1.size(); idx++)
            {
                const SimpleJointModel& j1 = config1[idx];
                const SimpleJointModel& j2 = config2[idx];
                assert(j1.IsRevolute() == j2.IsRevolute());
                assert(j1.IsContinuous() == j2.IsContinuous());
                const double raw_distance = j1.SignedDistance(j1.GetValue(), j2.GetValue());
                const double joint_distance_weight = joint_distance_weights_[idx];
                distances((int64_t)idx) = raw_distance * joint_distance_weight;
            }
            return distances;
        }

        inline Eigen::VectorXd ComputePerDimensionConfigurationDistance(const SimpleLinkedConfiguration& config1, const SimpleLinkedConfiguration& config2) const
        {
            return ComputePerDimensionConfigurationRawDistance(config1, config2).cwiseAbs();
        }

        inline double ComputeConfigurationDistance(const SimpleLinkedConfiguration& config1, const SimpleLinkedConfiguration& config2) const
        {
            return ComputePerDimensionConfigurationRawDistance(config1, config2).norm();
        }
    };
}

#endif // SIMPLE_ROBOT_MODELS_HPP
