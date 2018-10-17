#pragma once

namespace uncertainty_planning_core
{
template <typename Configuration, typename Generator>
class SimpleSamplerInterface
{
public:
  virtual ~SimpleSamplerInterface() {}

  virtual Configuration Sample(Generator& prng) = 0;

  virtual Configuration SampleGoal(Generator& prng) = 0;
};
}  // namespace uncertainty_planning_core
