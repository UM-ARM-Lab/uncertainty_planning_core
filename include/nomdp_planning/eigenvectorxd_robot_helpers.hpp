#include <vector>
#include <random>
#include <Eigen/Geometry>
#include <arc_utilities/eigen_helpers.hpp>

#ifndef EIGENVECTORXD_ROBOT_HELPERS_HPP
#define EIGENVECTORXD_ROBOT_HELPERS_HPP

namespace eigenvectorxd_robot_helpers
{
    class EigenVectorXdBaseSampler
    {
    protected:

        std::vector<std::uniform_real_distribution<double>> distributions_;

    public:

        EigenVectorXdBaseSampler(const std::vector<std::pair<double, double>>& limits)
        {
            distributions_.reserve(limits.size());
            for (size_t idx = 0; idx < limits.size(); idx++)
            {
                const std::pair<double, double>& limit = limits[idx];
                assert(limit.first <= limit.second);
                std::uniform_real_distribution<double> new_dist(limit.first, limit.second);
                distributions_.push_back(new_dist);
            }
        }

        template<typename Generator>
        Eigen::VectorXd Sample(Generator& prng)
        {
            Eigen::VectorXd sampled = Eigen::VectorXd::Zero(distributions_.size());
            for (size_t idx = 0; idx < distributions_.size(); idx++)
            {
                sampled(idx) = distributions_[idx](prng);
            }
            return sampled;
        }
    };

    class EigenVectorXdInterpolator
    {
    public:

        Eigen::VectorXd operator()(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2, const double ratio) const
        {
            return EigenHelpers::Interpolate(v1, v2, ratio);
        }

        static Eigen::VectorXd Interpolate(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2, const double ratio)
        {
            return EigenHelpers::Interpolate(v1, v2, ratio);
        }
    };

    class EigenVectorXdAverager
    {
    public:

        Eigen::VectorXd operator()(const std::vector<Eigen::VectorXd>& vec) const
        {
            if (vec.size() > 0)
            {
                return EigenHelpers::AverageEigenVectorXd(vec);
            }
            else
            {
                return Eigen::VectorXd();
            }
        }

        static Eigen::VectorXd Average(const std::vector<Eigen::VectorXd>& vec)
        {
            if (vec.size() > 0)
            {
                return EigenHelpers::AverageEigenVectorXd(vec);
            }
            else
            {
                return Eigen::VectorXd();
            }
        }
    };

    class EigenVectorXdDistancer
    {
    public:

        double operator()(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2) const
        {
            assert(v1.size() == v2.size());
            return (v1 - v2).norm();
        }

        static double Distance(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2)
        {
            assert(v1.size() == v2.size());
            return (v1 - v2).norm();
        }
    };

    class EigenVectorXdDimDistancer
    {
    public:

        Eigen::VectorXd operator()(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2) const
        {
            assert(v1.size() == v2.size());
            Eigen::VectorXd dim_distances(v1.size());
            for (long idx = 0; idx < v1.size(); idx++)
            {
                dim_distances(idx) = fabs(v1(idx) - v2(idx));
            }
            return dim_distances;
        }

        static Eigen::VectorXd Distance(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2)
        {
            assert(v1.size() == v2.size());
            Eigen::VectorXd dim_distances(v1.size());
            for (long idx = 0; idx < v1.size(); idx++)
            {
                dim_distances(idx) = fabs(v1(idx) - v2(idx));
            }
            return dim_distances;
        }
    };
}

#endif // EIGENVECTORXD_ROBOT_HELPERS_HPP
