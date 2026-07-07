cmake_minimum_required(VERSION 3.8)
project(nasal_endoscopy_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(action_msgs REQUIRED)

# ── Messages ──────────────────────────────────────────────
set(msg_files
  msg/NostrilDetection.msg
  msg/APFForceField.msg
  msg/MotionCommand.msg
  msg/NavigationState.msg
  msg/MotionResult.msg
)

# ── Services ──────────────────────────────────────────────
set(srv_files
  srv/SetMode.srv
)

# ── Actions ───────────────────────────────────────────────
set(action_files
  action/NavigateOutside.action
  action/NavigateInside.action
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  ${action_files}
  DEPENDENCIES std_msgs geometry_msgs action_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
