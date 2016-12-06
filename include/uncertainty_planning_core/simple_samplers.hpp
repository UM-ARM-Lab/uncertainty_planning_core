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

        SimpleBaseSampler() {}

        virtual Configuration Sample(Generator& prng) = 0;
    };

    template <typename Generator>
    class SimpleSE2BaseSampler : public SimpleBaseSampler<Eigen::Matrix<double, 3, 1>, Generator>
    {
    protected:

        std::uniform_real_distribution<double> x_distribution_;
        std::uniform_real_distribution<double> y_distribution_;
        std::uniform_real_distribution<double> zr_distribution_;

    public:

        SimpleSE2BaseSampler(const std::pair<double, double>& x_limits, const std::pair<double, double>& y_limits) : SimpleBaseSampler<Eigen::Matrix<double, 3, 1>, Generator>()
        {
            assert(x_limits.first <= x_limits.second);
            assert(y_limits.first <= y_limits.second);
            x_distribution_ = std::uniform_real_distribution<double>(x_limits.first, x_limits.second);
            y_distribution_ = std::uniform_real_distribution<double>(y_limits.first, y_limits.second);
            zr_distribution_ = std::uniform_real_distribution<double>(-M_PI, M_PI);
        }

        virtual Eigen::Matrix<double, 3, 1> Sample(Generator& prng)
        {
            const double x = x_distribution_(prng);
            const double y = y_distribution_(prng);
            const double zr = zr_distribution_(prng);
            Eigen::Matrix<double, 3, 1> state;
            state << x, y, zr;
            return state;
        }
    };

    template <typename Generator>
    class SimpleSE3BaseSampler : public SimpleBaseSampler<Eigen::Affine3d, Generator>
    {
    protected:

        std::uniform_real_distribution<double> x_distribution_;
        std::uniform_real_distribution<double> y_distribution_;
        std::uniform_real_distribution<double> z_distribution_;
        arc_helpers::RandomRotationGenerator rotation_generator_;

    public:

        SimpleSE3BaseSampler(const std::pair<double, double>& x_limits, const std::pair<double, double>& y_limits, const std::pair<double, double>& z_limits) : SimpleBaseSampler<Eigen::Affine3d, Generator>()
        {
            assert(x_limits.first <= x_limits.second);
            assert(y_limits.first <= y_limits.second);
            assert(z_limits.first <= z_limits.second);
            x_distribution_ = std::uniform_real_distribution<double>(x_limits.first, x_limits.second);
            y_distribution_ = std::uniform_real_distribution<double>(y_limits.first, y_limits.second);
            z_distribution_ = std::uniform_real_distribution<double>(z_limits.first, z_limits.second);
        }

        virtual Eigen::Affine3d Sample(Generator& prng)
        {
            const double x = x_distribution_(prng);
            const double y = y_distribution_(prng);
            const double z = z_distribution_(prng);
            const Eigen::Quaterniond quat = rotation_generator_.GetQuaternion(prng);
            const Eigen::Affine3d state = Eigen::Translation3d(x, y, z) * quat;
            return state;
        }
    };

    template <typename Generator>
    class SimpleLinkedBaseSampler : public SimpleBaseSampler<simple_robot_models::SimpleLinkedConfiguration, Generator>
    {
    protected:

        std::vector<std::uniform_real_distribution<double>> distributions_;
        simple_robot_models::SimpleLinkedConfiguration representative_configuration_;

    public:

        SimpleLinkedBaseSampler(const simple_robot_models::SimpleLinkedConfiguration& representative_configuration) : SimpleBaseSampler<simple_robot_models::SimpleLinkedConfiguration, Generator>()
        {
            representative_configuration_ = representative_configuration;
            for (size_t idx = 0; idx < representative_configuration_.size(); idx++)
            {
                const simple_robot_models::SimpleJointModel& current_joint = representative_configuration_[idx];
                const std::pair<double, double> limits = current_joint.GetLimits();
                distributions_.push_back(std::uniform_real_distribution<double>(limits.first, limits.second));
            }
        }

        virtual simple_robot_models::SimpleLinkedConfiguration Sample(Generator& prng)
        {
            simple_robot_models::SimpleLinkedConfiguration sampled;
            sampled.reserve(representative_configuration_.size());
            for (size_t idx = 0; idx < representative_configuration_.size(); idx++)
            {
                const simple_robot_models::SimpleJointModel& current_joint = representative_configuration_[idx];
                const double sampled_val = distributions_[idx](prng);
                sampled.push_back(current_joint.CopyWithNewValue(sampled_val));
            }
            return sampled;
        }
    };
}

#endif // SIMPLE_SAMPLERS_HPP
