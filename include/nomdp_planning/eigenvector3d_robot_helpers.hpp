#include <stdio.h>
#include <vector>
#include <map>
#include <random>
#include <Eigen/Geometry>
#include <arc_utilities/eigen_helpers.hpp>
#include <nomdp_planning/simple_pid_controller.hpp>
#include <nomdp_planning/simple_uncertainty_models.hpp>

#ifndef EIGENVECTOR3D_ROBOT_HELPERS_HPP
#define EIGENVECTOR3D_ROBOT_HELPERS_HPP

namespace eigenvector3d_robot_helpers
{
    class EigenVector3dBaseSampler
    {
    protected:

        std::uniform_real_distribution<double> x_distribution_;
        std::uniform_real_distribution<double> y_distribution_;
        std::uniform_real_distribution<double> z_distribution_;

    public:

        EigenVector3dBaseSampler(const std::pair<double, double>& x_limits, const std::pair<double, double>& y_limits, const std::pair<double, double>& z_limits)
        {
            assert(x_limits.first <= x_limits.second);
            assert(y_limits.first <= y_limits.second);
            assert(z_limits.first <= z_limits.second);
            x_distribution_ = std::uniform_real_distribution<double>(x_limits.first, x_limits.second);
            y_distribution_ = std::uniform_real_distribution<double>(y_limits.first, y_limits.second);
            z_distribution_ = std::uniform_real_distribution<double>(z_limits.first, z_limits.second);
        }

        template<typename Generator>
        Eigen::Vector3d Sample(Generator& prng)
        {
            const double x = x_distribution_(prng);
            const double y = y_distribution_(prng);
            const double z = z_distribution_(prng);
            return Eigen::Vector3d(x, y, z);
        }
    };

    class EigenVector3dInterpolator
    {
    public:

        Eigen::Vector3d operator()(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, const double ratio) const
        {
            return EigenHelpers::Interpolate(v1, v2, ratio);
        }

        static Eigen::Vector3d Interpolate(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2, const double ratio)
        {
            return EigenHelpers::Interpolate(v1, v2, ratio);
        }
    };

    class EigenVector3dAverager
    {
    public:

        Eigen::Vector3d operator()(const EigenHelpers::VectorVector3d& vec) const
        {
            if (vec.size() > 0)
            {
                return EigenHelpers::AverageEigenVector3d(vec);
            }
            else
            {
                return Eigen::Vector3d(0.0, 0.0, 0.0);
            }
        }

        static Eigen::Vector3d Average(const EigenHelpers::VectorVector3d& vec)
        {
            if (vec.size() > 0)
            {
                return EigenHelpers::AverageEigenVector3d(vec);
            }
            else
            {
                return Eigen::Vector3d(0.0, 0.0, 0.0);
            }
        }
    };

    class EigenVector3dDistancer
    {
    public:

        double operator()(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2) const
        {
            return (v1 - v2).norm();
        }

        static double Distance(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2)
        {
            return (v1 - v2).norm();
        }
    };

    class EigenVector3dDimDistancer
    {
    public:

        Eigen::VectorXd operator()(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2) const
        {
            Eigen::VectorXd dim_distances(3);
            dim_distances << fabs(v1.x() - v2.x()), fabs(v1.y() - v2.y()), fabs(v1.z() - v2.z());
            return dim_distances;
        }

        static Eigen::VectorXd Distance(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2)
        {
            Eigen::VectorXd dim_distances(3);
            dim_distances << fabs(v1.x() - v2.x()), fabs(v1.y() - v2.y()), fabs(v1.z() - v2.z());
            return dim_distances;
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

    class SimpleEigenVector3dRobot
    {
    protected:

        bool initialized_;
        simple_pid_controller::SimplePIDController x_axis_controller_;
        simple_pid_controller::SimplePIDController y_axis_controller_;
        simple_pid_controller::SimplePIDController z_axis_controller_;
        simple_uncertainty_models::SimpleUncertainSensor x_axis_sensor_;
        simple_uncertainty_models::SimpleUncertainSensor y_axis_sensor_;
        simple_uncertainty_models::SimpleUncertainSensor z_axis_sensor_;
        simple_uncertainty_models::SimpleUncertainVelocityActuator x_axis_actuator_;
        simple_uncertainty_models::SimpleUncertainVelocityActuator y_axis_actuator_;
        simple_uncertainty_models::SimpleUncertainVelocityActuator z_axis_actuator_;
        Eigen::Vector3d position_;
        EigenHelpers::VectorVector3d link_points_;

    public:

        inline SimpleEigenVector3dRobot(const EigenHelpers::VectorVector3d& robot_points, const Eigen::Vector3d& initial_position, const ROBOT_CONFIG& robot_config) : link_points_(robot_points)
        {
            x_axis_controller_ = simple_pid_controller::SimplePIDController(robot_config.kp, robot_config.ki, robot_config.kd, robot_config.integral_clamp);
            y_axis_controller_ = simple_pid_controller::SimplePIDController(robot_config.kp, robot_config.ki, robot_config.kd, robot_config.integral_clamp);
            z_axis_controller_ = simple_pid_controller::SimplePIDController(robot_config.kp, robot_config.ki, robot_config.kd, robot_config.integral_clamp);
            x_axis_sensor_ = simple_uncertainty_models::SimpleUncertainSensor(-robot_config.max_sensor_noise, robot_config.max_sensor_noise);
            y_axis_sensor_ = simple_uncertainty_models::SimpleUncertainSensor(-robot_config.max_sensor_noise, robot_config.max_sensor_noise);
            z_axis_sensor_ = simple_uncertainty_models::SimpleUncertainSensor(-robot_config.max_sensor_noise, robot_config.max_sensor_noise);
            x_axis_actuator_ = simple_uncertainty_models::SimpleUncertainVelocityActuator(-robot_config.max_actuator_noise, robot_config.max_actuator_noise, robot_config.velocity_limit);
            y_axis_actuator_ = simple_uncertainty_models::SimpleUncertainVelocityActuator(-robot_config.max_actuator_noise, robot_config.max_actuator_noise, robot_config.velocity_limit);
            z_axis_actuator_ = simple_uncertainty_models::SimpleUncertainVelocityActuator(-robot_config.max_actuator_noise, robot_config.max_actuator_noise, robot_config.velocity_limit);
            position_ = initial_position;
            initialized_ = true;
        }

        inline std::vector<std::pair<std::string, EigenHelpers::VectorVector3d>> GetRawLinksPoints() const
        {
            return std::vector<std::pair<std::string, EigenHelpers::VectorVector3d>>{std::pair<std::string, EigenHelpers::VectorVector3d>("robot", link_points_)};
        }

        inline void UpdatePosition(const Eigen::Vector3d& position)
        {
            position_ = position;
            x_axis_controller_.Zero();
            y_axis_controller_.Zero();
            z_axis_controller_.Zero();
        }

        inline Eigen::Affine3d GetLinkTransform(const std::string& link_name) const
        {
            if (link_name == "robot")
            {
                const Eigen::Affine3d transform = (Eigen::Translation3d)GetPosition() * Eigen::Quaterniond::Identity();
                return transform;
            }
            else
            {
                throw std::invalid_argument("Invalid link_name");
            }
        }

        inline Eigen::Vector3d GetPosition() const
        {
            return position_;
        }

        inline double ComputeDistanceTo(const Eigen::Vector3d& target) const
        {
            return EigenVector3dDistancer::Distance(GetPosition(), target);
        }

        template<typename PRNG>
        inline Eigen::VectorXd GenerateControlAction(const Eigen::Vector3d& target, PRNG& rng)
        {
            const Eigen::Vector3d current = GetPosition();
            double x_axis_control = x_axis_actuator_.GetControlValue(x_axis_controller_.ComputeFeedbackTerm(target.x(), current.x(), 1.0), rng);
            double y_axis_control = y_axis_actuator_.GetControlValue(y_axis_controller_.ComputeFeedbackTerm(target.y(), current.y(), 1.0), rng);
            double z_axis_control = z_axis_actuator_.GetControlValue(z_axis_controller_.ComputeFeedbackTerm(target.z(), current.z(), 1.0), rng);
            Eigen::VectorXd control_action(3);
            control_action(0) = x_axis_control;
            control_action(1) = y_axis_control;
            control_action(2) = z_axis_control;
            return control_action;
        }

        template<typename PRNG>
        inline void ApplyControlInput(const Eigen::VectorXd& input, PRNG& rng)
        {
            assert(input.size() == 3);
            const Eigen::Vector3d new_position = position_ + input;
            const double noisy_x = x_axis_sensor_.GetSensorValue(new_position.x(), rng);
            const double noisy_y = y_axis_sensor_.GetSensorValue(new_position.y(), rng);
            const double noisy_z = z_axis_sensor_.GetSensorValue(new_position.z(), rng);
            position_ = Eigen::Vector3d(noisy_x, noisy_y, noisy_z);
        }

        inline void ApplyControlInput(const Eigen::VectorXd& input)
        {
            assert(input.size() == 3);
            const Eigen::Vector3d new_position = position_ + input;
            position_ = new_position;
        }

        inline Eigen::Matrix<double, 3, Eigen::Dynamic> ComputeLinkPointJacobian(const std::string& link_name, const Eigen::Vector3d& link_relative_point) const
        {
            UNUSED(link_relative_point);
            if (link_name == "robot")
            {
                Eigen::Matrix<double, 3, 3> jacobian = Eigen::Matrix<double, 3, 3>::Identity();
                return jacobian;
            }
            else
            {
                throw std::invalid_argument("Invalid link_name");
            }
        }

        inline Eigen::Matrix<double, 6, Eigen::Dynamic> ComputeJacobian() const
        {
            Eigen::Matrix<double, 6, 3> jacobian = Eigen::Matrix<double, 6, 3>::Zero();
            jacobian(0,0) = 1.0;
            jacobian(1,1) = 1.0;
            jacobian(2,2) = 1.0;
            return jacobian;
        }

        inline Eigen::VectorXd ProcessCorrectionAction(const Eigen::VectorXd& raw_correction_action) const
        {
            assert(raw_correction_action.size() == 3);
            const double action_norm = raw_correction_action.norm();
            if (action_norm > 0.05)
            {
                return (raw_correction_action / action_norm) * 0.05;
            }
            else
            {
                return raw_correction_action;
            }
        }
    };
}

#endif // EIGENVECTOR3D_ROBOT_HELPERS_HPP
