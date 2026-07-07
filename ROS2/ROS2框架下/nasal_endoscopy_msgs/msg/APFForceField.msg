# APF force field output from inside vision
std_msgs/Header header

# Net force vector in pixel units (2D)
float32 force_x
float32 force_y
float32 force_magnitude

# Goal pixel coordinates (-1 if none detected)
int32 goal_x
int32 goal_y
bool goal_valid

# Image centre (for reference)
int32 center_x
int32 center_y
