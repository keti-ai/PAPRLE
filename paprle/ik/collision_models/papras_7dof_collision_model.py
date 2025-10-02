"""Creating a link collision model for the Franka with a series of spheres of various radii"""

link_1_pos = (
    (0, 0, 0),
)
link_1_radii = (
    0.05,
)

link_2_pos = (
    (0.0000000e+00,  0.0000000e+00,  6.3000001e-02),
    (0.0000000e+00, 0.0000000e+00, -6.3000001e-02),
    (0.05000000e+00, 0.0000000e+00, 0),
)
link_2_radii = (
    0.03679674,
    0.03679674,
    0.08,
)

link_3_pos = (
    (0,  0,  0),
    (0, 0.05, 0.),
)
link_3_radii = (
    0.05,
    0.05,
)

link_4_pos = (
    (0.0,-0.02,0),
    (0.0, -0.08, 0),
)
link_4_radii = (
    0.03,
    0.03,
)
link_5_pos = (
    (0.0, 0.0, 0.0),
    (-0.08, 0.0, 0.0),
)
link_5_radii = (
    0.045,
    0.045,
)
link_6_pos = (
    (0.0,-0.02,0),
    (0.0, -0.08, 0),
    (0.02, -0.02, 0),
    (0.02, -0.08, 0),
    (0.04, -0.045, 0),
    (0.08, -0.045, 0),
    (0.12, -0.045, 0),
)
link_6_radii = (
    0.03,
    0.03,
    0.03,
    0.03,
    0.03,
    0.03,
    0.03,
)
link_7_pos = (
    (5.0216153e-02, -8.9019688e-04, -2.5000001e-04),
)
link_7_radii = (
    0.045,
)


positions = {
    "link_1": link_1_pos,
    "link_2": link_2_pos,
    "link_3": link_3_pos,
    "link_4": link_4_pos,
    "link_5": link_5_pos,
    "link_6": link_6_pos,
    "link_7": link_7_pos,
}
radii = {
    "link_1": link_1_radii,
    "link_2": link_2_radii,
    "link_3": link_3_radii,
    "link_4": link_4_radii,
    "link_5": link_5_radii,
    "link_6": link_6_radii,
    "link_7": link_7_radii,
}

positions_list = (
    link_1_pos,
    link_2_pos,
    link_3_pos,
    link_4_pos,
    link_5_pos,
    link_6_pos,
    link_7_pos,
)
radii_list = (
    link_1_radii,
    link_2_radii,
    link_3_radii,
    link_4_radii,
    link_5_radii,
    link_6_radii,
    link_7_radii,
)

papras_collision_data = {"positions": positions_list, "radii": radii_list}