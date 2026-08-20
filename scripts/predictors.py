"""Property predictors/measurers for generated structures.

Each predictor maps a pymatgen Structure -> a scalar property value. Geometric
properties (density_atomic, volume, density) are read directly from the structure
with no model; model-based ones (band_gap) lazily load their model in setup() so
importing this module stays cheap for the geometric path. compute_predictions.py
selects one by name via REGISTRY and drives the shared load/validate/checkpoint loop.

To add a property: implement a PropertyPredictor subclass (or reuse GeometricPredictor)
and register a factory in REGISTRY.
"""
from abc import ABC, abstractmethod


class PropertyPredictor(ABC):
    # Column stem for the output. The driver writes <output_base>_raw (computed from
    # cif_steered) and <output_base> (computed from cif_relaxed).
    output_base: str = "value"

    def setup(self) -> None:
        """Load any heavy resources once (e.g. a model). No-op by default."""

    @abstractmethod
    def predict(self, structure) -> float:
        """Return the property value for a single pymatgen Structure."""


class GeometricPredictor(PropertyPredictor):
    """Direct geometric reads from the structure — no model."""

    def __init__(self, prop: str):
        self.prop = prop
        self.output_base = prop

    def predict(self, s) -> float:
        if self.prop == "density_atomic":
            return s.volume / len(s)      # Å^3 / atom (intensive -> cell-invariant)
        if self.prop == "volume":
            return float(s.volume)        # Å^3 (extensive -> cell-dependent)
        if self.prop == "density":
            return float(s.density)       # g/cm^3
        raise ValueError(f"GeometricPredictor: unknown property '{self.prop}'")


class BandGapPredictor(PropertyPredictor):
    """MEGNet multi-fidelity band gap; fidelity 0 = PBE/GGA (matches our data)."""

    output_base = "predicted_bandgap_ev"
    MODEL_NAME = "MEGNet-BandGap-mfi-MP-2019.4.1"

    def __init__(self, fidelity: int = 0):
        self.fidelity = fidelity
        self.model = None
        self.state_attr = None

    def setup(self) -> None:
        import matgl
        import torch
        self.model = matgl.load_model(self.MODEL_NAME)
        self.state_attr = torch.tensor([self.fidelity])

    def predict(self, s) -> float:
        return float(self.model.predict_structure(structure=s, state_attr=self.state_attr))


# property name -> factory(args) -> PropertyPredictor
REGISTRY = {
    "density_atomic": lambda args: GeometricPredictor("density_atomic"),
    "density":        lambda args: GeometricPredictor("density"),
    "volume":         lambda args: GeometricPredictor("volume"),
    "band_gap":       lambda args: BandGapPredictor(fidelity=getattr(args, "fidelity", 0)),
}
