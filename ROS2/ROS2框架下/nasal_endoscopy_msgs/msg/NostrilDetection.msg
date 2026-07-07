# Nostril detection result from outside vision (MediaPipe)
std_msgs/Header header

# Pixel coordinates of the target nostril centre (-1 if not detected)
int32 nose_x
int32 nose_y

# Feature width (nostril wing-to-wing) in pixels
float32 feature_width

# Nostril reference distance (landmark 279→49) in pixels
float32 nostril_distance_px

# Whether this frame contains a valid detection
bool valid
