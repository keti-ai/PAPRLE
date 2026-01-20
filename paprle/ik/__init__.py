import importlib
import os
import sys

# Get Python version
py = sys.version_info

# Set up /opt/openrobots environment variables
openrobots_base = "/opt/openrobots"
if os.path.exists(openrobots_base):
    # Set PATH
    openrobots_bin = f"{openrobots_base}/bin"
    if openrobots_bin not in os.environ.get("PATH", ""):
        current_path = os.environ.get("PATH", "")
        os.environ["PATH"] = f"{openrobots_bin}:{current_path}"
    
    # Set PKG_CONFIG_PATH
    pkg_config_path = f"{openrobots_base}/lib/pkgconfig"
    if pkg_config_path not in os.environ.get("PKG_CONFIG_PATH", ""):
        current_pkg_config = os.environ.get("PKG_CONFIG_PATH", "")
        os.environ["PKG_CONFIG_PATH"] = f"{pkg_config_path}:{current_pkg_config}" if current_pkg_config else pkg_config_path
    
    # Set LD_LIBRARY_PATH
    lib_path = f"{openrobots_base}/lib"
    if lib_path not in os.environ.get("LD_LIBRARY_PATH", ""):
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{lib_path}:{current_ld_path}" if current_ld_path else lib_path
    
    # Set PYTHONPATH (adapt to Python version)
    python_site_packages = f"{openrobots_base}/lib/python{py.major}.{py.minor}/site-packages"
    if os.path.exists(python_site_packages):
        if python_site_packages not in sys.path:
            sys.path.insert(0, python_site_packages)
        # Also set as environment variable
        current_pythonpath = os.environ.get("PYTHONPATH", "")
        if python_site_packages not in current_pythonpath:
            os.environ["PYTHONPATH"] = f"{python_site_packages}:{current_pythonpath}" if current_pythonpath else python_site_packages
    
    # Set CMAKE_PREFIX_PATH
    if openrobots_base not in os.environ.get("CMAKE_PREFIX_PATH", ""):
        current_cmake = os.environ.get("CMAKE_PREFIX_PATH", "")
        os.environ["CMAKE_PREFIX_PATH"] = f"{openrobots_base}:{current_cmake}" if current_cmake else openrobots_base


class LazySolverDict(dict):
    def __init__(self, mapping):
        super().__init__(mapping)
        self._cache = {}

    def __getitem__(self, key):
        if key in self._cache:
            return self._cache[key]

        if not key in self:
            raise KeyError(f"Solver '{key}' not found in the IK solver dictionary. Available solvers are: {list(self.keys())}")

        module_name, attr_name = super().__getitem__(key)
        try:
            module = importlib.import_module(module_name)
            solver_cls = getattr(module, attr_name)
            self._cache[key] = solver_cls
            return solver_cls
        except ImportError as e:
            raise ImportError(
                f"[paprle.ik] The solver '{attr_name}' cannot be imported, please make sure all the dependencies are installed."
            ) from e

IK_SOLVER_DICT = LazySolverDict({
    'mujoco': ('paprle.ik.mujoco_solver', 'MujocoIKSolverMultiWrapper'),
    'oscbf': ('paprle.ik.oscbf_solver', 'OSCBFIKSolverMultiWrapper'),
    'pinocchio': ('paprle.ik.pinocchio_solver', 'PinocchioIKSolverMultiWrapper'),
    'pyroki': ('paprle.ik.pyroki_solver', 'PyrokiIKSolver'),
    'pincasadi': ('paprle.ik.pinocchio_casadi_multi_solver', 'PinocchioCasadiMultiIKSolver'),
    'pincasadi_single': ('paprle.ik.pinocchio_casadi_single_solver', 'PinocchioCasadiIKSolverMultiWrapper'),
})

