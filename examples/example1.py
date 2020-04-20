import modesolverpy.mode_solver_semi_vectorial as ms
import modesolverpy.structure as st

# All units are relative.  [um] were chosen in this case.
x_step = 0.02
y_step = 0.02
wg_height = 0.4
wg_width = 0.5
sub_height = 0.5
sub_width = 2.0
clad_height = 0.5
n_sub = 1.4
n_wg = 3.0
n_clad = 1.0
film_thickness = 0.5
wavelength = 1.55
angle = 75.0

wg = st.waveguide(
    x_step=0.02,
    y_step=0.02,
    wg_height=0.4,
    wg_width=0.5,
    sub_height=0.5,
    sub_width=2.0,
    clad_height=0.5,
    n_sub=1.4,
    n_wg=3.0,
    n_clad=1.0,
    film_thickness=0.5,
    wavelength=1.55,
    angle=75.0,
)

st.write_material_index(wg)

mode_solver = ms.ModeSolverSemiVectorial(2, semi_vectorial_method="Ey")
mode_solver.solve(wg)
mode_solver.write_modes_to_file("example_modes_1.dat")
