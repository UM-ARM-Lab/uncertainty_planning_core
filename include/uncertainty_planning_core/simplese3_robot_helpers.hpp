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

#ifndef SIMPLESE3_ROBOT_HELPERS_HPP
#define SIMPLESE3_ROBOT_HELPERS_HPP

namespace simplese3_robot_helpers
{
    class EigenAffine3dSerializer
    {
    public:

        static inline std::string TypeName()
        {
            return std::string("EigenAffine3dSerializer");
        }

        static inline uint64_t Serialize(const Eigen::Affine3d& value, std::vector<uint8_t>& buffer)
        {
            return EigenHelpers::Serialize(value, buffer);
        }

        static inline std::pair<Eigen::Affine3d, uint64_t> Deserialize(const std::vector<uint8_t>& buffer, const uint64_t current)
        {
            return EigenHelpers::Deserialize<Eigen::Affine3d>(buffer, current);
        }
    };

    class SimpleSE3BaseSampler
    {
    protected:

        std::uniform_real_distribution<double> x_distribution_;
        std::uniform_real_distribution<double> y_distribution_;
        std::uniform_real_distribution<double> z_distribution_;
        arc_helpers::RandomRotationGenerator rotation_generator_;

    public:

        SimpleSE3BaseSampler(const std::pair<double, double>& x_limits, const std::pair<double, double>& y_limits, const std::pair<double, double>& z_limits)
        {
            assert(x_limits.first <= x_limits.second);
            assert(y_limits.first <= y_limits.second);
            assert(z_limits.first <= z_limits.second);
            x_distribution_ = std::uniform_real_distribution<double>(x_limits.first, x_limits.second);
            y_distribution_ = std::uniform_real_distribution<double>(y_limits.first, y_limits.second);
            z_distribution_ = std::uniform_real_distribution<double>(z_limits.first, z_limits.second);
        }

        template<typename Generator>
        Eigen::Affine3d Sample(Generator& prng)
        {
            const double x = x_distribution_(prng);
            const double y = y_distribution_(prng);
            const double z = z_distribution_(prng);
            const Eigen::Quaterniond quat = rotation_generator_.GetQuaternion(prng);
            const Eigen::Affine3d state = Eigen::Translation3d(x, y, z) * quat;
            return state;
        }

        static std::string TypeName()
        {
            return std::string("Proper6DOFBaseSampler");
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

    class SimpleSE3Robot
    {
    protected:

        bool initialized_;
        double max_motion_per_unit_step_;
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
        Eigen::Affine3d config_;
        std::shared_ptr<EigenHelpers::VectorVector3d> link_points_;

        inline void SetConfig(const Eigen::Affine3d& new_config)
        {
            //std::cout << "Raw config to set: " << new_config << std::endl;
            config_ = new_config;
        }

    public:

        inline double ComputeMaxMotionPerStep() const
        {
            double max_motion = 0.0;
            const std::vector<std::pair<std::string, std::shared_ptr<EigenHelpers::VectorVector3d>>> robot_links_points = GetRawLinksPoints();
            // Generate motion primitives
            std::vector<Eigen::VectorXd> motion_primitives;
            motion_primitives.reserve(12);
            for (ssize_t joint_idx = 0; joint_idx < 6; joint_idx++)
            {
                Eigen::VectorXd raw_motion_plus = Eigen::VectorXd::Zero(6);
                raw_motion_plus(joint_idx) = 0.125;
                motion_primitives.push_back(raw_motion_plus);
                Eigen::VectorXd raw_motion_neg = Eigen::VectorXd::Zero(6);
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

        inline SimpleSE3Robot(const std::shared_ptr<EigenHelpers::VectorVector3d>& robot_points, const Eigen::Affine3d& initial_position, const ROBOT_CONFIG& robot_config) : link_points_(robot_points)
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

        inline void UpdatePosition(const Eigen::Affine3d& position)
        {
            SetConfig(position);
        }

        inline void ResetPosition(const Eigen::Affine3d& position)
        {
            SetConfig(position);
            x_axis_controller_.Zero();
            y_axis_controller_.Zero();
            z_axis_controller_.Zero();
            xr_axis_controller_.Zero();
            yr_axis_controller_.Zero();
            zr_axis_controller_.Zero();
        }

        inline Eigen::Affine3d GetLinkTransform(const std::string& link_name) const
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

        inline Eigen::Affine3d GetPosition() const
        {
            return config_;
        }

        template<typename PRNG>
        inline Eigen::Affine3d GetPosition(PRNG& rng) const
        {
            const Eigen::Matrix<double, 6, 1> twist = EigenHelpers::TwistBetweenTransforms(Eigen::Affine3d::Identity(), GetPosition());
            Eigen::Matrix<double, 6, 1> noisy_twist = Eigen::Matrix<double, 6, 1>::Zero();
            noisy_twist(0) = x_axis_sensor_.GetSensorValue(twist(0), rng);
            noisy_twist(1) = y_axis_sensor_.GetSensorValue(twist(1), rng);
            noisy_twist(2) = z_axis_sensor_.GetSensorValue(twist(2), rng);
            noisy_twist(3) = xr_axis_sensor_.GetSensorValue(twist(3), rng);
            noisy_twist(4) = yr_axis_sensor_.GetSensorValue(twist(4), rng);
            noisy_twist(5) = zr_axis_sensor_.GetSensorValue(twist(5), rng);
            // Compute the motion transform
            const Eigen::Affine3d motion_transform = EigenHelpers::ExpTwist(noisy_twist, 1.0);
            return motion_transform;
        }

        inline double ComputeDistanceTo(const Eigen::Affine3d& target) const
        {
            return ComputeConfigurationDistance(GetPosition(), target);
        }

        template<typename PRNG>
        inline Eigen::VectorXd GenerateControlAction(const Eigen::Affine3d& target, const double controller_interval, PRNG& rng)
        {
            // Get the current position
            const Eigen::Affine3d current = GetPosition(rng);
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
            const Eigen::Affine3d motion_transform = EigenHelpers::ExpTwist(noisy_twist, 1.0);
            const Eigen::Affine3d new_config = GetPosition() * motion_transform;
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
            const Eigen::Affine3d motion_transform = EigenHelpers::ExpTwist(twist, 1.0);
            const Eigen::Affine3d new_config = GetPosition() * motion_transform;
            // Update config
            SetConfig(new_config);
        }

        inline Eigen::Matrix<double, 3, Eigen::Dynamic> ComputeLinkPointJacobian(const std::string& link_name, const Eigen::Vector3d& link_relative_point) const
        {
            if (link_name == "robot")
            {
                const Eigen::Matrix3d rot_matrix = GetPosition().rotation();
                const Eigen::Matrix3d hatted_link_relative_point = EigenHelpers::Skew(link_relative_point);
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

        inline Eigen::VectorXd ProcessCorrectionAction(const Eigen::VectorXd& raw_correction_action) const
        {
            assert(raw_correction_action.size() == 6);
//            const double action_norm = raw_correction_action.norm();
//            const Eigen::VectorXd real_correction = (action_norm > 0.05) ? (raw_correction_action / action_norm) * 0.05 : raw_correction_action;
//            return real_correction;
            // We treat translation and rotation separately
            const Eigen::Vector3d raw_translation_correction = raw_correction_action.head<3>();
            const Eigen::Vector3d raw_rotation_correction = raw_correction_action.tail<3>();
            // Scale down the translation
            const double translation_action_norm = raw_translation_correction.norm();
            const Eigen::Vector3d real_translation_correction = (translation_action_norm > 0.005) ? (raw_translation_correction / translation_action_norm) * 0.005 : raw_translation_correction;
            // Scale down the rotation
            const double rotation_action_norm = raw_rotation_correction.norm();
            const Eigen::Vector3d real_rotation_correction = (rotation_action_norm > 0.05) ? (raw_rotation_correction / rotation_action_norm) * 0.05 : raw_rotation_correction;
            // Put them back together
            Eigen::VectorXd real_correction(6);
            real_correction << real_translation_correction, real_rotation_correction;
            return real_correction;
        }

        inline double GetMaxMotionPerStep() const
        {
            return max_motion_per_unit_step_;
        }

        inline Eigen::Affine3d AverageConfigurations(const EigenHelpers::VectorAffine3d& configurations) const
        {
            if (configurations.size() > 0)
            {
                return EigenHelpers::AverageEigenAffine3d(configurations);
            }
            else
            {
                return Eigen::Affine3d::Identity();
            }
        }

        inline Eigen::Affine3d InterpolateBetweenConfigurations(const Eigen::Affine3d& start, const Eigen::Affine3d& end, const double ratio) const
        {
            return EigenHelpers::Interpolate(start, end, ratio);
        }

        inline Eigen::VectorXd ComputePerDimensionConfigurationRawDistance(const Eigen::Affine3d& config1, const Eigen::Affine3d& config2) const
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

        inline Eigen::VectorXd ComputePerDimensionConfigurationDistance(const Eigen::Affine3d& config1, const Eigen::Affine3d& config2) const
        {
            return ComputePerDimensionConfigurationRawDistance(config1, config2).cwiseAbs();
        }

        inline double ComputeConfigurationDistance(const Eigen::Affine3d& config1, const Eigen::Affine3d& config2) const
        {
            return (EigenHelpers::Distance(config1, config2, 0.5) * 2.0);
        }
    };
}

#endif // SIMPLESE3_ROBOT_HELPERS_HPP
