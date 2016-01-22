#include <stdio.h>
#include <vector>
#include <map>
#include <random>
#include <Eigen/Geometry>
#include <arc_utilities/eigen_helpers.hpp>
#include <nomdp_planning/simple_pid_controller.hpp>
#include <nomdp_planning/simple_uncertainty_models.hpp>

#ifndef EIGENAFFINE3D_ROBOT_HELPERS_HPP
#define EIGENAFFINE3D_ROBOT_HELPERS_HPP

namespace eigenaffine3d_robot_helpers
{
    class EigenAffine3dBaseSampler
    {
    protected:

        std::uniform_real_distribution<double> x_distribution_;
        std::uniform_real_distribution<double> y_distribution_;
        std::uniform_real_distribution<double> z_distribution_;
        arc_helpers::RandomRotationGenerator rotation_distribution_;

    public:

        EigenAffine3dBaseSampler(const std::pair<double, double>& x_limits, const std::pair<double, double>& y_limits, const std::pair<double, double>& z_limits)
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
            const Eigen::Translation3d translation(x, y, z);
            const Eigen::Quaterniond rotation = rotation_distribution_.GetQuaternion(prng);
            const Eigen::Affine3d transform = translation * rotation;
            return transform;
        }
    };

    class EigenAffine3dInterpolator
    {
    public:

        Eigen::Affine3d operator()(const Eigen::Affine3d& t1, const Eigen::Affine3d& t2, const double ratio) const
        {
            return EigenHelpers::Interpolate(t1, t2, ratio);
        }

        static Eigen::Affine3d Interpolate(const Eigen::Affine3d& t1, const Eigen::Affine3d& t2, const double ratio)
        {
            return EigenHelpers::Interpolate(t1, t2, ratio);
        }
    };

    class EigenAffine3dAverager
    {
    public:

        Eigen::Affine3d operator()(const EigenHelpers::VectorAffine3d& vec) const
        {
            if (vec.size() > 0)
            {
                return EigenHelpers::AverageEigenAffine3d(vec);
            }
            else
            {
                return Eigen::Affine3d::Identity();
            }
        }

        static Eigen::Affine3d Average(const EigenHelpers::VectorAffine3d& vec)
        {
            if (vec.size() > 0)
            {
                return EigenHelpers::AverageEigenAffine3d(vec);
            }
            else
            {
                return Eigen::Affine3d::Identity();
            }
        }
    };

    class EigenAffine3dDistancer
    {
    public:

        double operator()(const Eigen::Affine3d& t1, const Eigen::Affine3d& t2) const
        {
            return EigenHelpers::Distance(t1, t2, 0.25);
        }

        static double Distance(const Eigen::Affine3d& t1, const Eigen::Affine3d& t2)
        {
            return EigenHelpers::Distance(t1, t2, 0.25);
        }
    };

    class EigenAffine3dDimDistancer
    {
    public:

        Eigen::VectorXd operator()(const Eigen::Affine3d& t1, const Eigen::Affine3d& t2) const
        {
            Eigen::VectorXd dim_distances(6);
            dim_distances.head<3>() = t1.translation() - t2.translation();
            dim_distances.tail<3>() = 0.5 * (t2.linear().col(0).cross(t1.linear().col(0)) + t2.linear().col(1).cross(t1.linear().col(1)) + t2.linear().col(2).cross(t1.linear().col(2)));
            return dim_distances.cwiseAbs();
        }

        static Eigen::VectorXd Distance(const Eigen::Affine3d& t1, const Eigen::Affine3d& t2)
        {
            Eigen::VectorXd dim_distances(6);
            dim_distances.head<3>() = t1.translation() - t2.translation();
            dim_distances.tail<3>() = 0.5 * (t2.linear().col(0).cross(t1.linear().col(0)) + t2.linear().col(1).cross(t1.linear().col(1)) + t2.linear().col(2).cross(t1.linear().col(2)));
            return dim_distances.cwiseAbs();
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

    class SimpleEigenAffine3dRobot
    {
    protected:

        bool initialized_;
        simple_pid_controller::SimplePIDController x_axis_controller_;
        simple_pid_controller::SimplePIDController y_axis_controller_;
        simple_pid_controller::SimplePIDController z_axis_controller_;
        simple_pid_controller::SimplePIDController xr_axis_controller_;
        simple_pid_controller::SimplePIDController yr_axis_controller_;
        simple_pid_controller::SimplePIDController zr_axis_controller_;
        simple_uncertainty_models::SimpleUncertainSensor x_axis_sensor_;
        simple_uncertainty_models::SimpleUncertainSensor y_axis_sensor_;
        simple_uncertainty_models::SimpleUncertainSensor z_axis_sensor_;
        simple_uncertainty_models::SimpleUncertainSensor xr_axis_sensor_;
        simple_uncertainty_models::SimpleUncertainSensor yr_axis_sensor_;
        simple_uncertainty_models::SimpleUncertainSensor zr_axis_sensor_;
        simple_uncertainty_models::SimpleUncertainVelocityActuator x_axis_actuator_;
        simple_uncertainty_models::SimpleUncertainVelocityActuator y_axis_actuator_;
        simple_uncertainty_models::SimpleUncertainVelocityActuator z_axis_actuator_;
        simple_uncertainty_models::SimpleUncertainVelocityActuator xr_axis_actuator_;
        simple_uncertainty_models::SimpleUncertainVelocityActuator yr_axis_actuator_;
        simple_uncertainty_models::SimpleUncertainVelocityActuator zr_axis_actuator_;
        Eigen::Affine3d pose_;
        Eigen::Matrix<double, 6, 1> config_;
        EigenHelpers::VectorVector3d link_points_;

    public:

        inline SimpleEigenAffine3dRobot(const EigenHelpers::VectorVector3d& robot_points, const Eigen::Affine3d& initial_position, const ROBOT_CONFIG& robot_config) : link_points_(robot_points)
        {
            x_axis_controller_ = simple_pid_controller::SimplePIDController(robot_config.kp, robot_config.ki, robot_config.kd, robot_config.integral_clamp);
            y_axis_controller_ = simple_pid_controller::SimplePIDController(robot_config.kp, robot_config.ki, robot_config.kd, robot_config.integral_clamp);
            z_axis_controller_ = simple_pid_controller::SimplePIDController(robot_config.kp, robot_config.ki, robot_config.kd, robot_config.integral_clamp);
            xr_axis_controller_ = simple_pid_controller::SimplePIDController(robot_config.r_kp, robot_config.r_ki, robot_config.r_kd, robot_config.r_integral_clamp);
            yr_axis_controller_ = simple_pid_controller::SimplePIDController(robot_config.r_kp, robot_config.r_ki, robot_config.r_kd, robot_config.r_integral_clamp);
            zr_axis_controller_ = simple_pid_controller::SimplePIDController(robot_config.r_kp, robot_config.r_ki, robot_config.r_kd, robot_config.r_integral_clamp);
            x_axis_sensor_ = simple_uncertainty_models::SimpleUncertainSensor(-robot_config.max_sensor_noise, robot_config.max_sensor_noise);
            y_axis_sensor_ = simple_uncertainty_models::SimpleUncertainSensor(-robot_config.max_sensor_noise, robot_config.max_sensor_noise);
            z_axis_sensor_ = simple_uncertainty_models::SimpleUncertainSensor(-robot_config.max_sensor_noise, robot_config.max_sensor_noise);
            xr_axis_sensor_ = simple_uncertainty_models::SimpleUncertainSensor(-robot_config.r_max_sensor_noise, robot_config.r_max_sensor_noise);
            yr_axis_sensor_ = simple_uncertainty_models::SimpleUncertainSensor(-robot_config.r_max_sensor_noise, robot_config.r_max_sensor_noise);
            zr_axis_sensor_ = simple_uncertainty_models::SimpleUncertainSensor(-robot_config.r_max_sensor_noise, robot_config.r_max_sensor_noise);
            x_axis_actuator_ = simple_uncertainty_models::SimpleUncertainVelocityActuator(-robot_config.max_actuator_noise, robot_config.max_actuator_noise, robot_config.velocity_limit);
            y_axis_actuator_ = simple_uncertainty_models::SimpleUncertainVelocityActuator(-robot_config.max_actuator_noise, robot_config.max_actuator_noise, robot_config.velocity_limit);
            z_axis_actuator_ = simple_uncertainty_models::SimpleUncertainVelocityActuator(-robot_config.max_actuator_noise, robot_config.max_actuator_noise, robot_config.velocity_limit);
            xr_axis_actuator_ = simple_uncertainty_models::SimpleUncertainVelocityActuator(-robot_config.r_max_actuator_noise, robot_config.r_max_actuator_noise, robot_config.r_velocity_limit);
            yr_axis_actuator_ = simple_uncertainty_models::SimpleUncertainVelocityActuator(-robot_config.r_max_actuator_noise, robot_config.r_max_actuator_noise, robot_config.r_velocity_limit);
            zr_axis_actuator_ = simple_uncertainty_models::SimpleUncertainVelocityActuator(-robot_config.r_max_actuator_noise, robot_config.r_max_actuator_noise, robot_config.r_velocity_limit);
            UpdatePosition(initial_position);
            initialized_ = true;
        }

        inline std::vector<std::pair<std::string, EigenHelpers::VectorVector3d>> GetRawLinksPoints() const
        {
            return std::vector<std::pair<std::string, EigenHelpers::VectorVector3d>>{std::pair<std::string, EigenHelpers::VectorVector3d>("robot", link_points_)};
        }

        inline void UpdatePosition(const Eigen::Affine3d& position)
        {
            const Eigen::Vector3d trans = position.translation();
            const Eigen::Vector3d rot = EigenHelpers::EulerAnglesFromAffine3d(position);
            // Update config
            config_(0) = trans.x();
            config_(1) = trans.y();
            config_(2) = trans.z();
            config_(3) = rot.x();
            config_(4) = rot.y();
            config_(5) = rot.z();
            // Update pose
            pose_ = ComputePose();
            // Safety check
            assert(EigenHelpers::Distance(position, pose_) < 0.01);
            // Zero controllers
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
                return pose_;
            }
            else
            {
                throw std::invalid_argument("Invalid link_name");
            }
        }

        inline Eigen::Affine3d GetPosition() const
        {
            return pose_;
        }

        inline double ComputeDistanceTo(const Eigen::Affine3d& target) const
        {
            return EigenAffine3dDistancer::Distance(GetPosition(), target);
        }

        template<typename PRNG>
        Eigen::VectorXd GenerateControlAction(const Eigen::Affine3d& target, PRNG& rng)
        {
            // Get the current position
            const Eigen::Affine3d current = GetPosition();
            // Get the current Jacobian
            const Eigen::Matrix<double, 6, 6> current_jacobian = ComputeJacobian();
            // Get the Jacobian inverse
            const Eigen::Matrix<double, 6, 6> current_jpinv = EigenHelpers::Pinv(current_jacobian, EigenHelpers::SuggestedRcond());
            // Get the error between the current position and target
            Eigen::Matrix<double, 6, 1> pose_error = Eigen::Matrix<double, 6, 1>::Zero();
            pose_error.head<3>() = current.translation() - target.translation();
            pose_error.tail<3>() = 0.5 * (target.linear().col(0).cross(current.linear().col(0)) + target.linear().col(1).cross(current.linear().col(1)) + target.linear().col(2).cross(current.linear().col(2)));
            pose_error = -1.0 * pose_error;
            // Do controller things
            const Eigen::MatrixXd raw_jpinv_error = current_jpinv * pose_error;
            const Eigen::VectorXd jpinv_error = raw_jpinv_error.col(0);
            // Compute feedback terms
            const double x_term = x_axis_controller_.ComputeFeedbackTerm(jpinv_error(0), 1.0);
            const double y_term = y_axis_controller_.ComputeFeedbackTerm(jpinv_error(1), 1.0);
            const double z_term = z_axis_controller_.ComputeFeedbackTerm(jpinv_error(2), 1.0);
            const double xr_term = xr_axis_controller_.ComputeFeedbackTerm(jpinv_error(3), 1.0);
            const double yr_term = yr_axis_controller_.ComputeFeedbackTerm(jpinv_error(4), 1.0);
            const double zr_term = zr_axis_controller_.ComputeFeedbackTerm(jpinv_error(5), 1.0);
            // Make the control action
            const double x_axis_control = x_axis_actuator_.GetControlValue(x_term, rng);
            const double y_axis_control = y_axis_actuator_.GetControlValue(y_term, rng);
            const double z_axis_control = z_axis_actuator_.GetControlValue(z_term, rng);
            const double xr_axis_control = xr_axis_actuator_.GetControlValue(xr_term, rng);
            const double yr_axis_control = yr_axis_actuator_.GetControlValue(yr_term, rng);
            const double zr_axis_control = zr_axis_actuator_.GetControlValue(zr_term, rng);
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
            // Compute new config
            const Eigen::Matrix<double, 6, 1> new_config = config_ + input;
            // Sense new noisy config
            Eigen::Matrix<double, 6, 1> noisy_config = Eigen::Matrix<double, 6, 1>::Zero();
            noisy_config(0) = x_axis_sensor_.GetSensorValue(new_config(0), rng);
            noisy_config(1) = y_axis_sensor_.GetSensorValue(new_config(1), rng);
            noisy_config(2) = z_axis_sensor_.GetSensorValue(new_config(2), rng);
            noisy_config(3) = xr_axis_sensor_.GetSensorValue(new_config(3), rng);
            noisy_config(4) = yr_axis_sensor_.GetSensorValue(new_config(4), rng);
            noisy_config(5) = zr_axis_sensor_.GetSensorValue(new_config(5), rng);
            // Update config
            config_ = noisy_config;
            // Update pose
            pose_ = ComputePose();
        }

        inline Eigen::Matrix<double, 3, Eigen::Dynamic> ComputeLinkPointJacobian(const std::string& link_name, const Eigen::Vector3d& link_relative_point) const
        {
            if (link_name == "robot")
            {
                // Transform the point into world frame
                const Eigen::Affine3d current_transform = GetPosition();
                const Eigen::Vector3d current_position(config_(0), config_(1), config_(2));
                const Eigen::Vector3d current_angles(config_(3), config_(4), config_(5));
                const Eigen::Vector3d world_point = current_transform * link_relative_point;
                // Make the jacobian
                Eigen::Matrix<double, 3, 6> jacobian = Eigen::Matrix<double, 3, 6>::Zero();
                // Prismatic joints
                // X joint
                jacobian.block<3,1>(0, 0) = jacobian.block<3,1>(0, 0) + Eigen::Vector3d::UnitX();
                // Y joint
                jacobian.block<3,1>(0, 1) = jacobian.block<3,1>(0, 1) + Eigen::Vector3d::UnitY();
                // Z joint
                jacobian.block<3,1>(0, 2) = jacobian.block<3,1>(0, 2) + Eigen::Vector3d::UnitZ();
                // Rotatational joints
                // X joint
                // Compute X-axis joint axis (this is basically a formality)
                const Eigen::Affine3d x_joint_transform = (Eigen::Translation3d)current_position * Eigen::Quaterniond::Identity();
                const Eigen::Vector3d x_joint_axis = (Eigen::Vector3d)(x_joint_transform.rotation() * Eigen::Vector3d::UnitX());
                jacobian.block<3,1>(0, 3) = jacobian.block<3,1>(0, 3) + x_joint_axis.cross(world_point - current_position);
                // Compute Y-axis joint axis
                const Eigen::Affine3d y_joint_transform = x_joint_transform * (Eigen::Translation3d::Identity() * Eigen::Quaterniond(Eigen::AngleAxisd(current_angles.x(), Eigen::Vector3d::UnitX())));
                const Eigen::Vector3d y_joint_axis = (Eigen::Vector3d)(y_joint_transform.rotation() * Eigen::Vector3d::UnitY());
                jacobian.block<3,1>(0, 4) = jacobian.block<3,1>(0, 4) + y_joint_axis.cross(world_point - current_position);
                // Compute Z-axis joint axis
                const Eigen::Affine3d z_joint_transform = y_joint_transform * (Eigen::Translation3d::Identity() * Eigen::Quaterniond(Eigen::AngleAxisd(current_angles.y(), Eigen::Vector3d::UnitY())));
                const Eigen::Vector3d z_joint_axis = (Eigen::Vector3d)(z_joint_transform.rotation() * Eigen::Vector3d::UnitZ());
                jacobian.block<3,1>(0, 5) = jacobian.block<3,1>(0, 5) + z_joint_axis.cross(world_point - current_position);
                return jacobian;
            }
            else
            {
                throw std::invalid_argument("Invalid link_name");
            }
        }

        inline Eigen::Matrix<double, 6, Eigen::Dynamic> ComputeJacobian() const
        {
            // Transform the point into world frame
            const Eigen::Affine3d current_transform = GetPosition();
            const Eigen::Vector3d current_position(config_(0), config_(1), config_(2));
            const Eigen::Vector3d current_angles(config_(3), config_(4), config_(5));
            const Eigen::Vector3d link_relative_point(0.0, 0.0, 0.0);
            const Eigen::Vector3d world_point = current_transform * link_relative_point;
            // Make the jacobian
            Eigen::Matrix<double, 6, 6> jacobian = Eigen::Matrix<double, 6, 6>::Zero();
            // Prismatic joints
            // X joint
            jacobian.block<3,1>(0, 0) = jacobian.block<3,1>(0, 0) + Eigen::Vector3d::UnitX();
            // Y joint
            jacobian.block<3,1>(0, 1) = jacobian.block<3,1>(0, 1) + Eigen::Vector3d::UnitY();
            // Z joint
            jacobian.block<3,1>(0, 2) = jacobian.block<3,1>(0, 2) + Eigen::Vector3d::UnitZ();
            // Rotatational joints
            // X joint
            // Compute X-axis joint axis (this is basically a formality)
            const Eigen::Affine3d x_joint_transform = (Eigen::Translation3d)current_position * Eigen::Quaterniond::Identity();
            const Eigen::Vector3d x_joint_axis = (Eigen::Vector3d)(x_joint_transform.rotation() * Eigen::Vector3d::UnitX());
            jacobian.block<3,1>(0, 3) = jacobian.block<3,1>(0, 3) + x_joint_axis.cross(world_point - current_position);
            jacobian.block<3,1>(3, 3) = jacobian.block<3,1>(3, 3) + x_joint_axis;
            // Compute Y-axis joint axis
            const Eigen::Affine3d y_joint_transform = x_joint_transform * (Eigen::Translation3d::Identity() * Eigen::Quaterniond(Eigen::AngleAxisd(current_angles.x(), Eigen::Vector3d::UnitX())));
            const Eigen::Vector3d y_joint_axis = (Eigen::Vector3d)(y_joint_transform.rotation() * Eigen::Vector3d::UnitY());
            jacobian.block<3,1>(0, 4) = jacobian.block<3,1>(0, 4) + y_joint_axis.cross(world_point - current_position);
            jacobian.block<3,1>(3, 4) = jacobian.block<3,1>(3, 4) + y_joint_axis;
            // Compute Z-axis joint axis
            const Eigen::Affine3d z_joint_transform = y_joint_transform * (Eigen::Translation3d::Identity() * Eigen::Quaterniond(Eigen::AngleAxisd(current_angles.y(), Eigen::Vector3d::UnitY())));
            const Eigen::Vector3d z_joint_axis = (Eigen::Vector3d)(z_joint_transform.rotation() * Eigen::Vector3d::UnitZ());
            jacobian.block<3,1>(0, 5) = jacobian.block<3,1>(0, 5) + z_joint_axis.cross(world_point - current_position);
            jacobian.block<3,1>(3, 5) = jacobian.block<3,1>(3, 5) + z_joint_axis;
            return jacobian;
        }

        inline Eigen::Affine3d ComputePose() const
        {
            const Eigen::Translation3d current_position(config_(0), config_(1), config_(2));
            const Eigen::Vector3d current_angles(config_(3), config_(4), config_(5));
            const Eigen::Affine3d x_joint_transform = current_position * Eigen::Quaterniond::Identity();
            const Eigen::Affine3d y_joint_transform = x_joint_transform * (Eigen::Translation3d::Identity() * Eigen::Quaterniond(Eigen::AngleAxisd(current_angles.x(), Eigen::Vector3d::UnitX())));
            const Eigen::Affine3d z_joint_transform = y_joint_transform * (Eigen::Translation3d::Identity() * Eigen::Quaterniond(Eigen::AngleAxisd(current_angles.y(), Eigen::Vector3d::UnitY())));
            return z_joint_transform;
        }
    };
}

#endif // EIGENAFFINE3D_ROBOT_HELPERS_HPP
