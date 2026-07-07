# Navigation state published for monitoring / diagnostics
std_msgs/Header header

# State name (matching SystemState enum)
string state

# Cumulative depth (mm)
float32 current_depth_mm

# Max allowed depth (mm)
float32 max_depth_mm

# Pixel distance to target
float32 target_dist_px

# Force magnitude (APF, inside only)
float32 force_magnitude

# Auto-run engaged?
bool auto_run

# Task finished?
bool finished
