#include <uncertainty_planning_core/task_planner_adapter.hpp>

class PutInBoxState
{

private:

  int32_t objects_available_;
  bool object_put_away_;
  bool box_open_;

public:

  PutInBoxState()
    : objects_available_(-1), object_put_away_(false), box_open_(false) {}

  PutInBoxState(const int32_t objects_available,
                const bool object_put_away,
                const bool box_open)
    : objects_available_(objects_available),
      object_put_away_(object_put_away),
      box_open_(box_open) {}

  int32_t ObjectsAvailable() const { return objects_available_; }

  bool ObjectPutAway() const { return object_put_away_; }

  bool BoxOpen() const { return box_open_; }

  uint32_t GetStateReadiness() const
  {
    return GetStateReadiness(*this);
  }

  static uint32_t GetStateReadiness(const PutInBoxState& state)
  {
    const uint32_t BOX_OPEN = 0x01;
    const uint32_t NUM_OBJECTS_KNOWN = 0x02;
    const uint32_t OBJECTS_ALL_PUT_AWAY = 0x04;
    uint32_t state_readiness = 0u;
    if (state.BoxOpen())
    {
      state_readiness |= BOX_OPEN;
    }
    if (state.ObjectsAvailable() >= 0)
    {
      state_readiness |= NUM_OBJECTS_KNOWN;
      if (state.ObjectsAvailable() == 0)
      {
        state_readiness |= OBJECTS_ALL_PUT_AWAY;
      }
    }
    return state_readiness;
  }

  std::string Print() const
  {
    std::string rep
        = "Objects available: " + std::to_string(objects_available_)
          + " Object put away: " + PrettyPrint::PrettyPrint(object_put_away_)
          + " Box open: " + PrettyPrint::PrettyPrint(box_open_);
    return rep;
  }

  static bool IsSingleExecutionComplete(const PutInBoxState& state)
  {
    if (((state.ObjectsAvailable() > 0) && (state.ObjectPutAway() == true))
        || ((state.ObjectsAvailable() == 0) && (state.BoxOpen() == false)))
    {
      return true;
    }
    else
    {
      return false;
    }
  }

  static bool IsTaskComplete(const PutInBoxState& state)
  {
    if ((state.ObjectsAvailable() == 0) && (state.BoxOpen() == false))
    {
      return true;
    }
    else
    {
      return false;
    }
  }

  static std::string TypeName()
  {
    return std::string("PutInBoxState");
  }

  static uint64_t
  Serialize(const PutInBoxState& value, std::vector<uint8_t>& buffer)
  {
    const uint64_t start_buffer_size = buffer.size();
    arc_helpers::SerializeFixedSizePOD<int32_t>(
          value.ObjectsAvailable(), buffer);
    arc_helpers::SerializeFixedSizePOD<uint8_t>(
          static_cast<uint8_t>(value.ObjectPutAway()), buffer);
    arc_helpers::SerializeFixedSizePOD<uint8_t>(
          static_cast<uint8_t>(value.BoxOpen()), buffer);
    const uint64_t end_buffer_size = buffer.size();
    const uint64_t bytes_written = end_buffer_size - start_buffer_size;
    return bytes_written;
  }

  static std::pair<PutInBoxState, uint64_t>
  Deserialize(const std::vector<uint8_t>& buffer, const uint64_t current)
  {
    uint64_t current_position = current;
    auto deser_objects_available
        = arc_helpers::DeserializeFixedSizePOD<int32_t>(buffer, current);
    const int32_t objects_available = deser_objects_available.first;
    current_position += deser_objects_available.second;
    auto deser_object_put_away
        = arc_helpers::DeserializeFixedSizePOD<uint8_t>(buffer, current);
    const bool object_put_away = static_cast<bool>(deser_object_put_away.first);
    current_position += deser_object_put_away.second;
    auto deser_box_open
        = arc_helpers::DeserializeFixedSizePOD<uint8_t>(buffer, current);
    const bool box_open = static_cast<bool>(deser_box_open.first);
    current_position += deser_box_open.second;
    // How much did we read?
    const uint64_t bytes_read = current_position - current;
    return std::make_pair(PutInBoxState(objects_available,
                                        object_put_away,
                                        box_open),
                          bytes_read);
  }
};

std::ostream& operator<<(std::ostream& strm, const PutInBoxState& state)
{
    strm << state.Print();
    return strm;
}

class OpenBoxPrimitive
    : public task_planner_adapter::ActionPrimitiveInterface<PutInBoxState>
{
public:

  OpenBoxPrimitive()
    : task_planner_adapter::ActionPrimitiveInterface<PutInBoxState>() {}

  virtual bool IsCandidate(const PutInBoxState& state) const
  {
    if (state.BoxOpen())
    {
      std::cout << "State " << state.Print() << " is not a candidate for "
                << Name() << " since box is already open" << std::endl;
      return false;
    }
    else
    {
      return true;
    }
  }

  virtual std::string Name() const
  {
    return "OpenBoxPrimitive";
  }

  virtual double Ranking() const
  {
    return 1.0;
  }

  virtual std::vector<std::pair<PutInBoxState, bool>>
  GetOutcomes(const PutInBoxState& state)
  {
    if (IsCandidate(state))
    {
      auto box_open
          = std::make_pair(PutInBoxState(state.ObjectsAvailable(),
                                         state.ObjectPutAway(), true),
                           false);
      return std::vector<std::pair<PutInBoxState, bool>>{box_open};
    }
    else
    {
      throw std::invalid_argument("State is not a candidate for primitive");
    }
  }

  virtual std::vector<PutInBoxState>
  Execute(const PutInBoxState& state)
  {
    if (IsCandidate(state))
    {
      return std::vector<PutInBoxState>{
        PutInBoxState(state.ObjectsAvailable(), state.ObjectPutAway(), true)};
    }
    else
    {
      throw std::invalid_argument("State is not a candidate for primitive");
    }
  }
};

class CloseBoxPrimitive
    : public task_planner_adapter::ActionPrimitiveInterface<PutInBoxState>
{
public:

  CloseBoxPrimitive()
    : task_planner_adapter::ActionPrimitiveInterface<PutInBoxState>() {}

  virtual bool IsCandidate(const PutInBoxState& state) const
  {
    if (state.BoxOpen())
    {
      return true;
    }
    else
    {
      std::cout << "State " << state.Print() << " is not a candidate for "
                << Name() << " since box is already closed" << std::endl;
      return false;
    }
  }

  virtual std::string Name() const
  {
    return "CloseBoxPrimitive";
  }

  virtual double Ranking() const
  {
    return 2.0;
  }

  virtual std::vector<std::pair<PutInBoxState, bool>>
  GetOutcomes(const PutInBoxState& state)
  {
    if (IsCandidate(state))
    {
      auto box_closed
          = std::make_pair(PutInBoxState(state.ObjectsAvailable(),
                                         state.ObjectPutAway(), false),
                           false);
      return std::vector<std::pair<PutInBoxState, bool>>{box_closed};
    }
    else
    {
      throw std::invalid_argument("State is not a candidate for primitive");
    }
  }

  virtual std::vector<PutInBoxState>
  Execute(const PutInBoxState& state)
  {
    if (IsCandidate(state))
    {
      return std::vector<PutInBoxState>{
        PutInBoxState(state.ObjectsAvailable(), state.ObjectPutAway(), false)};
    }
    else
    {
      throw std::invalid_argument("State is not a candidate for primitive");
    }
  }
};

class CheckIfAvailableObjectPrimitive
    : public task_planner_adapter::ActionPrimitiveInterface<PutInBoxState>
{
public:

  CheckIfAvailableObjectPrimitive()
    : task_planner_adapter::ActionPrimitiveInterface<PutInBoxState>() {}

  virtual bool IsCandidate(const PutInBoxState& state) const
  {
    if (state.ObjectsAvailable() < 0)
    {
      return true;
    }
    else
    {
      std::cout << "State " << state.Print() << " is not a candidate for "
                << Name() << " since num objects is already known" << std::endl;
      return false;
    }
  }

  virtual std::string Name() const
  {
    return "CheckIfAvailableObjectPrimitive";
  }

  virtual double Ranking() const
  {
    return 3.0;
  }

  virtual std::vector<std::pair<PutInBoxState, bool>>
  GetOutcomes(const PutInBoxState& state)
  {
    if (IsCandidate(state))
    {
      auto object_available
          = std::make_pair(PutInBoxState(1, state.ObjectPutAway(),
                                         state.BoxOpen()),
                           false);
      auto none_available
          = std::make_pair(PutInBoxState(0, state.ObjectPutAway(),
                                         state.BoxOpen()),
                           false);
      return std::vector<std::pair<PutInBoxState, bool>>{object_available,
                                                         none_available};
    }
    else
    {
      throw std::invalid_argument("State is not a candidate for primitive");
    }
  }

  virtual std::vector<PutInBoxState>
  Execute(const PutInBoxState& state)
  {
    if (IsCandidate(state))
    {
      return std::vector<PutInBoxState>{
        PutInBoxState(std::abs(state.ObjectsAvailable()),
                      state.ObjectPutAway(), state.BoxOpen())};
    }
    else
    {
      throw std::invalid_argument("State is not a candidate for primitive");
    }
  }
};

class PutObjectInBoxPrimitive
    : public task_planner_adapter::ActionPrimitiveInterface<PutInBoxState>
{
public:

  PutObjectInBoxPrimitive()
    : task_planner_adapter::ActionPrimitiveInterface<PutInBoxState>() {}

  virtual bool IsCandidate(const PutInBoxState& state) const
  {
    if ((state.ObjectsAvailable() > 0) && (state.ObjectPutAway() == false)
        && state.BoxOpen())
    {
      return true;
    }
    else
    {
      std::cout << "State " << state.Print() << " is not a candidate for "
                << Name() << std::endl;
      return false;
    }
  }

  virtual std::string Name() const
  {
    return "PutObjectInBoxPrimitive";
  }

  virtual double Ranking() const
  {
    return 4.0;
  }

  virtual std::vector<std::pair<PutInBoxState, bool>>
  GetOutcomes(const PutInBoxState& state)
  {
    if (IsCandidate(state))
    {
      auto object_remaining
          = std::make_pair(PutInBoxState(1, true, true), false);
      auto task_done
          = std::make_pair(PutInBoxState(0, true, true), false);
      return std::vector<std::pair<PutInBoxState, bool>>{object_remaining,
                                                         task_done};
    }
    else
    {
      throw std::invalid_argument("State is not a candidate for primitive");
    }
  }

  virtual std::vector<PutInBoxState>
  Execute(const PutInBoxState& state)
  {
    if (IsCandidate(state))
    {
      return std::vector<PutInBoxState>{
        PutInBoxState(state.ObjectsAvailable() - 1, true, true)};
    }
    else
    {
      throw std::invalid_argument("State is not a candidate for primitive");
    }
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "task_planner_adapter_test");
  ros::NodeHandle nh;
  ros::NodeHandle nhp;
  // Get debug level
  const int32_t debug_level = nhp.param(std::string("debug_level"), 1);
  // Make display function
  ros::Publisher display_debug_publisher =
      nh.advertise<visualization_msgs::MarkerArray>(
        "task_planner_debug_display_markers", 1, true);
  std::function<void(const visualization_msgs::MarkerArray&)> display_fn
      = [&] (const visualization_msgs::MarkerArray& markers)
  {
    display_debug_publisher.publish(markers);
  };
  // Make logging function
  std::function<void(const std::string&, const int32_t)> logging_fn
      = [&] (const std::string& msg, const int32_t level)
  {
    ROS_INFO_NAMED(ros::this_node::getName(), "[%d] %s", level, msg.c_str());
  };
  // Get seed for PRNG
  int32_t prng_seed_init
      = (int32_t)nhp.param(std::string("prng_seed_init"), -1);
  if (prng_seed_init == -1)
  {
    prng_seed_init = (int32_t)std::chrono::high_resolution_clock::now()
                     .time_since_epoch().count();
    logging_fn("No PRNG seed provided, initializing from clock to ["
               + std::to_string(prng_seed_init) + "]", 1);
  }
  const int64_t prng_seed
      = static_cast<int64_t>(
          arc_helpers::SplitMix64PRNG((uint64_t)prng_seed_init)());
  // Make the planner interface
  const std::function<uint32_t(const PutInBoxState&)> state_readiness_fn
      = [] (const PutInBoxState& state)
  {
    return PutInBoxState::GetStateReadiness(state);
  };

  const std::function<bool(const PutInBoxState&)> single_execution_complete_fn
      = [] (const PutInBoxState& state)
  {
    return PutInBoxState::IsSingleExecutionComplete(state);
  };

  const std::function<bool(const PutInBoxState&)> task_complete_fn
      = [] (const PutInBoxState& state)
  {
    return PutInBoxState::IsTaskComplete(state);
  };

  task_planner_adapter::TaskPlannerAdapter<PutInBoxState, PutInBoxState>
      planner(state_readiness_fn,
              single_execution_complete_fn,
              task_complete_fn,
              logging_fn, display_fn, prng_seed, debug_level);
  // Add primitives
  task_planner_adapter::ActionPrimitivePtr<PutInBoxState>
      open_box_primitive_ptr(new OpenBoxPrimitive());
  task_planner_adapter::ActionPrimitivePtr<PutInBoxState>
      close_box_primitive_ptr(new CloseBoxPrimitive());
  task_planner_adapter::ActionPrimitivePtr<PutInBoxState>
      check_if_available_primitive_ptr(new CheckIfAvailableObjectPrimitive());
  task_planner_adapter::ActionPrimitivePtr<PutInBoxState>
      put_object_in_box_primitive_ptr(new PutObjectInBoxPrimitive());
  planner.RegisterPrimitive(open_box_primitive_ptr);
  planner.RegisterPrimitive(close_box_primitive_ptr);
  planner.RegisterPrimitive(check_if_available_primitive_ptr);
  planner.RegisterPrimitive(put_object_in_box_primitive_ptr);
  // Plan
  auto plan_result
      = planner.PlanPolicy(PutInBoxState(), 10.0, 1.0, 50u, 50u);
  logging_fn("Task planning statistics: "
             + PrettyPrint::PrettyPrint(plan_result.second), 1);
  int32_t objects_to_put_away = 5;
  const std::function<PutInBoxState(void)> single_execution_initialization_fn
      = [&] (void)
  {
    const int32_t objects_to_put_away_old = objects_to_put_away;
    objects_to_put_away--;
    return PutInBoxState(-objects_to_put_away_old, false, false);
  };
  auto exec_result
      = planner.ExecutePolicy(plan_result.first,
                              single_execution_initialization_fn,
                              100u, 100u, true, false);
  logging_fn("Task execution statistics: "
             + PrettyPrint::PrettyPrint(exec_result.second), 1);
  return 0;
}
