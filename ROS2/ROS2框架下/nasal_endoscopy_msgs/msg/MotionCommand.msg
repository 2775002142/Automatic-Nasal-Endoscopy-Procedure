# Incremental motion command in tool-frame coordinates
std_msgs/Header header

# Rotation about tool axes (degrees)
float32 rx_deg
float32 ry_deg
float32 rz_deg

# Translation along tool axes (mm)
float32 dx_mm
float32 dy_mm
float32 dz_mm

# Source identifier: "outside" | "inside" | "manual" | "none"
string source
