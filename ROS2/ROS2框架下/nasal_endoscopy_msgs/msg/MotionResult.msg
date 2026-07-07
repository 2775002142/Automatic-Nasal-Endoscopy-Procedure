# Motion result — published by motion_executor after each robot move.
# Navigation nodes subscribe to this to update their internal state.

std_msgs/Header header

# Whether the motion succeeded (RobotController returned success)
bool success

# The actual deltas that were requested (echo of the command)
float32 executed_dx_mm
float32 executed_dy_mm
float32 executed_dz_mm
float32 executed_rx_deg
float32 executed_ry_deg
float32 executed_rz_deg

# Error description when success == false
string error_msg
