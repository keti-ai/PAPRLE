import importlib

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

