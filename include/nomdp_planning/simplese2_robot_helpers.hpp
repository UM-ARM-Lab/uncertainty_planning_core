#include <stdio.h>
#include <vector>
#include <map>
#include <random>
#include <Eigen/Geometry>
#include <arc_utilities/eigen_helpers.hpp>
#include <nomdp_planning/simple_pid_controller.hpp>
#include <nomdp_planning/simple_uncertainty_models.hpp>

#ifndef SIMPLESE2_ROBOT_HELPERS_HPP
#define SIMPLESE2_ROBOT_HELPERS_HPP

namespace simplese2_robot_helpers
{
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
    };

    class SimpleSE2Interpolator
    {
    public:

        Eigen::Matrix<double, 3, 1> operator()(const Eigen::Matrix<double, 3, 1>& t1, const Eigen::Matrix<double, 3, 1>& t2, const double ratio) const
        {
            return Interpolate(t1, t2, ratio);
        }

        static Eigen::Matrix<double, 3, 1> Interpolate(const Eigen::Matrix<double, 3, 1>& t1, const Eigen::Matrix<double, 3, 1>& t2, const double ratio)
        {
            Eigen::Matrix<double, 3, 1> interpolated = Eigen::Matrix<double, 3, 1>::Zero();
            interpolated(0) = EigenHelpers::Interpolate(t1(0), t2(0), ratio);
            interpolated(1) = EigenHelpers::Interpolate(t1(1), t2(1), ratio);
            interpolated(2) = EigenHelpers::InterpolateContinuousRevolute(t1(2), t2(2), ratio);
            return interpolated;
        }
    };

    class SimpleSE2Averager
    {
    public:

        Eigen::Matrix<double, 3, 1> operator()(const std::vector<Eigen::Matrix<double, 3, 1>>& vec) const
        {
            return Average(vec);
        }

        static Eigen::Matrix<double, 3, 1> Average(const std::vector<Eigen::Matrix<double, 3, 1>>& vec)
        {
            if (vec.size() > 0)
            {
                // Separate translation and rotation values
                std::vector<Eigen::VectorXd> translations(vec.size());
                std::vector<double> zrs(vec.size());
                for (size_t idx = 0; idx < vec.size(); idx++)
                {
                    const Eigen::Matrix<double, 3, 1>& state = vec[idx];
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
    };

    class SimpleSE2DimDistancer
    {
    public:

        Eigen::VectorXd operator()(const Eigen::Matrix<double, 3, 1>& t1, const Eigen::Matrix<double, 3, 1>& t2) const
        {
            return Distance(t1, t2);
        }

        static Eigen::VectorXd Distance(const Eigen::Matrix<double, 3, 1>& t1, const Eigen::Matrix<double, 3, 1>& t2)
        {
            return RawDistance(t1, t2).cwiseAbs();
        }

        static Eigen::VectorXd RawDistance(const Eigen::Matrix<double, 3, 1>& t1, const Eigen::Matrix<double, 3, 1>& t2)
        {
            Eigen::VectorXd dim_distances(3);
            dim_distances(0) = t2(0) - t1(0);
            dim_distances(1) = t2(1) - t1(1);
            dim_distances(2) = EigenHelpers::ContinuousRevoluteSignedDistance(t1(2), t2(2));;
            return dim_distances;
        }
    };

    class SimpleSE2Distancer
    {
    public:

        double operator()(const Eigen::Matrix<double, 3, 1>& t1, const Eigen::Matrix<double, 3, 1>& t2) const
        {
            return Distance(t1, t2);
        }

        static double Distance(const Eigen::Matrix<double, 3, 1>& t1, const Eigen::Matrix<double, 3, 1>& t2)
        {
            const Eigen::VectorXd dim_distances = SimpleSE2DimDistancer::Distance(t1, t2);
            const double trans_dist = sqrt((dim_distances(0) * dim_distances(0)) + (dim_distances(1) * dim_distances(1)));
            const double rots_dist = fabs(dim_distances(2));
            return trans_dist + rots_dist;
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
        simple_uncertainty_models::SimpleUncertainSensor x_axis_sensor_;
        simple_uncertainty_models::SimpleUncertainSensor y_axis_sensor_;
        simple_uncertainty_models::SimpleUncertainSensor zr_axis_sensor_;
        simple_uncertainty_models::SimpleUncertainVelocityActuator x_axis_actuator_;
        simple_uncertainty_models::SimpleUncertainVelocityActuator y_axis_actuator_;
        simple_uncertainty_models::SimpleUncertainVelocityActuator zr_axis_actuator_;
        Eigen::Affine3d pose_;
        Eigen::Matrix<double, 3, 1> config_;
        EigenHelpers::VectorVector3d link_points_;

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

        static inline double ComputeMaxMotionPerStep(SimpleSE2Robot robot)
        {
            double max_motion = 0.0;
            const std::vector<std::pair<std::string, EigenHelpers::VectorVector3d>> robot_links_points = robot.GetRawLinksPoints();
            // Generate motion primitives
            std::vector<Eigen::VectorXd> motion_primitives;
            motion_primitives.reserve(6);
            for (size_t joint_idx = 0; joint_idx < 3; joint_idx++)
            {
                Eigen::VectorXd raw_motion_plus = Eigen::VectorXd::Zero(3);
                raw_motion_plus(joint_idx) = 1.0;
                motion_primitives.push_back(raw_motion_plus);
                Eigen::VectorXd raw_motion_neg = Eigen::VectorXd::Zero(3);
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

        inline SimpleSE2Robot(const EigenHelpers::VectorVector3d& robot_points, const Eigen::Matrix<double, 3, 1>& initial_position, const ROBOT_CONFIG& robot_config) : link_points_(robot_points)
        {
            x_axis_controller_ = simple_pid_controller::SimplePIDController(robot_config.kp, robot_config.ki, robot_config.kd, robot_config.integral_clamp);
            y_axis_controller_ = simple_pid_controller::SimplePIDController(robot_config.kp, robot_config.ki, robot_config.kd, robot_config.integral_clamp);
            zr_axis_controller_ = simple_pid_controller::SimplePIDController(robot_config.r_kp, robot_config.r_ki, robot_config.r_kd, robot_config.r_integral_clamp);
            x_axis_sensor_ = simple_uncertainty_models::SimpleUncertainSensor(-robot_config.max_sensor_noise, robot_config.max_sensor_noise);
            y_axis_sensor_ = simple_uncertainty_models::SimpleUncertainSensor(-robot_config.max_sensor_noise, robot_config.max_sensor_noise);
            zr_axis_sensor_ = simple_uncertainty_models::SimpleUncertainSensor(-robot_config.r_max_sensor_noise, robot_config.r_max_sensor_noise);
            x_axis_actuator_ = simple_uncertainty_models::SimpleUncertainVelocityActuator(-robot_config.max_actuator_noise, robot_config.max_actuator_noise, robot_config.velocity_limit);
            y_axis_actuator_ = simple_uncertainty_models::SimpleUncertainVelocityActuator(-robot_config.max_actuator_noise, robot_config.max_actuator_noise, robot_config.velocity_limit);
            zr_axis_actuator_ = simple_uncertainty_models::SimpleUncertainVelocityActuator(-robot_config.r_max_actuator_noise, robot_config.r_max_actuator_noise, robot_config.r_velocity_limit);
            max_motion_per_unit_step_ = ComputeMaxMotionPerStep(*this);
            UpdatePosition(initial_position);
            initialized_ = true;
        }

        inline std::vector<std::pair<std::string, EigenHelpers::VectorVector3d>> GetRawLinksPoints() const
        {
            return std::vector<std::pair<std::string, EigenHelpers::VectorVector3d>>{std::pair<std::string, EigenHelpers::VectorVector3d>("robot", link_points_)};
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

        inline double ComputeDistanceTo(const Eigen::Matrix<double, 3, 1>& target) const
        {
            return SimpleSE2Distancer::Distance(GetPosition(), target);
        }

        template<typename PRNG>
        inline Eigen::VectorXd GenerateControlAction(const Eigen::Matrix<double, 3, 1>& target, PRNG& rng)
        {
            // Get the current position
            const Eigen::Matrix<double, 3, 1> current = GetPosition();
            // Get the current error
            const Eigen::VectorXd current_error = SimpleSE2DimDistancer::RawDistance(current, target);
            // Compute feedback terms
            const double x_term = x_axis_controller_.ComputeFeedbackTerm(current_error(0), 1.0);
            const double y_term = y_axis_controller_.ComputeFeedbackTerm(current_error(1), 1.0);
            const double zr_term = zr_axis_controller_.ComputeFeedbackTerm(current_error(2), 1.0);
            // Make the control action
            const double x_axis_control = x_axis_actuator_.GetControlValue(x_term, rng);
            const double y_axis_control = y_axis_actuator_.GetControlValue(y_term, rng);
            const double zr_axis_control = zr_axis_actuator_.GetControlValue(zr_term, rng);
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
            // Compute new config
            const Eigen::Matrix<double, 3, 1> new_config = config_ + input;
            // Sense new noisy config
            Eigen::Matrix<double, 3, 1> noisy_config = Eigen::Matrix<double, 3, 1>::Zero();
            noisy_config(0) = x_axis_sensor_.GetSensorValue(new_config(0), rng);
            noisy_config(1) = y_axis_sensor_.GetSensorValue(new_config(1), rng);
            noisy_config(2) = zr_axis_sensor_.GetSensorValue(new_config(2), rng);
            // Update config
            SetConfig(noisy_config);
        }

        inline void ApplyControlInput(const Eigen::VectorXd& input)
        {
            assert(input.size() == 3);
            // Compute new config
            const Eigen::Matrix<double, 3, 1> new_config = config_ + input;
            // Update config
            SetConfig(new_config);
        }

        inline Eigen::Matrix<double, 3, Eigen::Dynamic> ComputeLinkPointJacobian(const std::string& link_name, const Eigen::Vector3d& link_relative_point) const
        {
            if (link_name == "robot")
            {
                // Transform the point into world frame
                const Eigen::Affine3d current_transform = pose_;
                const Eigen::Vector3d current_position(config_(0), config_(1), 0.0);
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

        inline Eigen::Matrix<double, 6, Eigen::Dynamic> ComputeJacobian() const
        {
            // Transform the point into world frame
            const Eigen::Affine3d current_transform = pose_;
            const Eigen::Vector3d current_position(config_(0), config_(1), 0.0);
            const Eigen::Vector3d link_relative_point(0.0, 0.0, 0.0);
            const Eigen::Vector3d world_point = current_transform * link_relative_point;
            // Make the jacobian
            Eigen::Matrix<double, 6, 3> jacobian = Eigen::Matrix<double, 6, 3>::Zero();
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
            jacobian.block<3,1>(3, 2) = jacobian.block<3,1>(3, 2) + z_joint_axis;
            return jacobian;
        }

        inline Eigen::Affine3d ComputePose() const
        {
            const Eigen::Translation3d current_position(config_(0), config_(1), 0.0);
            const double current_z_angle = config_(2);
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
            const double rotation_action_norm = fabs(raw_rotation_correction);
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
    };
}

#endif // SIMPLESE2_ROBOT_HELPERS_HPP

