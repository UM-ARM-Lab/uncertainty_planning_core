#include <stdio.h>
#include <vector>
#include <map>
#include <random>
#include <Eigen/Geometry>
#include <arc_utilities/arc_helpers.hpp>
#include <arc_utilities/eigen_helpers.hpp>
#include <arc_utilities/pretty_print.hpp>
#include <uncertainty_planning_core/simple_robot_models.hpp>

#ifndef SIMPLE_SAMPLERS_HPP
#define SIMPLE_SAMPLERS_HPP

namespace simple_samplers
{
    template <typename Configuration, typename Generator>
    class SimpleBaseSampler
    {
    public:

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SimpleBaseSampler() {}

        virtual Configuration Sample(Generator& prng) = 0;
    };

    template <typename Generator>
    class SimpleSE2BaseSampler : public SimpleBaseSampler<Eigen::Matrix<double, 3, 1>, Generator>
    {
    protected:

        std::pair<double, double> x_limits_;
        std::pair<double, double> y_limits_;
        std::uniform_real_distribution<double> uniform_unit_distribution_;

    public:

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SimpleSE2BaseSampler(const std::pair<double, double>& x_limits, const std::pair<double, double>& y_limits) : SimpleBaseSampler<Eigen::Matrix<double, 3, 1>, Generator>(), uniform_unit_distribution_(0.0, 1.0)
        {
            assert(!(std::isnan(x_limits.first) || std::isinf(x_limits.first)));
            assert(!(std::isnan(x_limits.second) || std::isinf(x_limits.second)));
            assert(!(std::isnan(y_limits.first) || std::isinf(y_limits.first)));
            assert(!(std::isnan(y_limits.second) || std::isinf(y_limits.second)));
            assert(x_limits.first <= x_limits.second);
            assert(y_limits.first <= y_limits.second);
            x_limits_ = x_limits;
            y_limits_ = y_limits;
        }

        virtual Eigen::Matrix<double, 3, 1> Sample(Generator& prng)
        {
            const double x = EigenHelpers::Interpolate(x_limits_.first, x_limits_.second, uniform_unit_distribution_(prng));
            const double y = EigenHelpers::Interpolate(y_limits_.first, y_limits_.second, uniform_unit_distribution_(prng));
            const double zr = EigenHelpers::Interpolate(-M_PI, M_PI, uniform_unit_distribution_(prng));
            Eigen::Matrix<double, 3, 1> state;
            state << x, y, zr;
            return state;
        }
    };

    template <typename Generator>
    class SimpleSE3BaseSampler : public SimpleBaseSampler<Eigen::Affine3d, Generator>
    {
    protected:

        std::pair<double, double> x_limits_;
        std::pair<double, double> y_limits_;
        std::pair<double, double> z_limits_;
        std::uniform_real_distribution<double> uniform_unit_distribution_;
        arc_helpers::RandomRotationGenerator rotation_generator_;

    public:

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SimpleSE3BaseSampler(const std::pair<double, double>& x_limits, const std::pair<double, double>& y_limits, const std::pair<double, double>& z_limits) : SimpleBaseSampler<Eigen::Affine3d, Generator>(), uniform_unit_distribution_(0.0, 1.0)
        {
            assert(!(std::isnan(x_limits.first) || std::isinf(x_limits.first)));
            assert(!(std::isnan(x_limits.second) || std::isinf(x_limits.second)));
            assert(!(std::isnan(y_limits.first) || std::isinf(y_limits.first)));
            assert(!(std::isnan(y_limits.second) || std::isinf(y_limits.second)));
            assert(!(std::isnan(z_limits.first) || std::isinf(z_limits.first)));
            assert(!(std::isnan(z_limits.second) || std::isinf(z_limits.second)));
            assert(x_limits.first <= x_limits.second);
            assert(y_limits.first <= y_limits.second);
            assert(z_limits.first <= z_limits.second);
            x_limits_ = x_limits;
            y_limits_ = y_limits;
            z_limits_ = z_limits;
        }

        virtual Eigen::Affine3d Sample(Generator& prng)
        {
            const double x = EigenHelpers::Interpolate(x_limits_.first, x_limits_.second, uniform_unit_distribution_(prng));
            const double y = EigenHelpers::Interpolate(y_limits_.first, y_limits_.second, uniform_unit_distribution_(prng));
            const double z = EigenHelpers::Interpolate(z_limits_.first, z_limits_.second, uniform_unit_distribution_(prng));
            const Eigen::Quaterniond quat = rotation_generator_.GetQuaternion(prng);
            const Eigen::Affine3d state = Eigen::Translation3d(x, y, z) * quat;
            return state;
        }
    };

    template <typename Generator>
    class SimpleLinkedBaseSampler : public SimpleBaseSampler<simple_robot_models::SimpleLinkedConfiguration, Generator>
    {
    protected:

        std::uniform_real_distribution<double> uniform_unit_distribution_;
        simple_robot_models::SimpleLinkedConfiguration representative_configuration_;

    public:

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SimpleLinkedBaseSampler(const simple_robot_models::SimpleLinkedConfiguration& representative_configuration) : SimpleBaseSampler<simple_robot_models::SimpleLinkedConfiguration, Generator>(), uniform_unit_distribution_(0.0, 1.0), representative_configuration_(representative_configuration) {}

        virtual simple_robot_models::SimpleLinkedConfiguration Sample(Generator& prng)
        {
            simple_robot_models::SimpleLinkedConfiguration sampled;
            sampled.reserve(representative_configuration_.size());
            for (size_t idx = 0; idx < representative_configuration_.size(); idx++)
            {
                const simple_robot_models::SimpleJointModel& current_joint = representative_configuration_[idx];
                if (current_joint.IsContinuous())
                {
                    const double sampled_val = EigenHelpers::Interpolate(-M_PI, M_PI, uniform_unit_distribution_(prng));
                    sampled.push_back(current_joint.CopyWithNewValue(sampled_val));
                }
                else
                {
                    const std::pair<double, double> limits = current_joint.GetLimits();
                    const double sampled_val = EigenHelpers::Interpolate(limits.first, limits.second, uniform_unit_distribution_(prng));
                    sampled.push_back(current_joint.CopyWithNewValue(sampled_val));
                }
            }
            return sampled;
        }
    };
}

#endif // SIMPLE_SAMPLERS_HPP
