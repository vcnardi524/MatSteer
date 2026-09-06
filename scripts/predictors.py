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


class HullLookup:
    """Materials Project convex hulls, in FORMATION-energy space, cached per system.

    MP entries carry total energies with MP2020 corrections; a formation-energy model
    predicts something else entirely, and subtracting one from the other is meaningless.
    So each system's diagram is rebuilt: take MP's corrected entries, read their formation
    energies off a first PhaseDiagram, and construct a second one from those plus
    elemental entries pinned at 0 -- which is what "formation energy" means. A query then
    only needs a composition and an E_form per atom.

    Verified against ground truth: rebuilt this way, all 176 Fe-Li-O materials in
    metadata_mp reproduce MP's own energy_above_hull to 4e-15 eV/atom.

    One system is one API call, so diagrams are cached; a sweep over 1,000 prompts spans
    ~989 systems but hits far fewer distinct ones after dedup.
    """

    def __init__(self, api_key: str = None, key_file: str = "api_keys.json",
                 thermo_types=("GGA_GGA+U",)):
        self.api_key = api_key or __import__("json").load(open(key_file))["mp_api_key"]
        self.thermo_types = list(thermo_types)
        self._cache = {}          # frozenset(elements) -> PhaseDiagram or None
        self._mpr = None

    def setup(self) -> None:
        # MP serialises energy adjustments under pymatgen.analysis.compatibility, which
        # moved to pymatgen.entries.compatibility. Without the alias every entry fails
        # to decode with ModuleNotFoundError.
        import sys
        import pymatgen.entries.compatibility as _compat
        sys.modules.setdefault("pymatgen.analysis.compatibility", _compat)
        from mp_api.client import MPRester
        self._mpr = MPRester(self.api_key)

    def diagram(self, elements):
        """PhaseDiagram in formation-energy space for one system, or None if MP has none."""
        from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
        from pymatgen.core import Composition
        key = frozenset(str(e) for e in elements)
        if key in self._cache:
            return self._cache[key]
        try:
            entries = self._mpr.get_entries_in_chemsys(
                elements=sorted(key),
                additional_criteria={"thermo_types": self.thermo_types})
        except Exception:
            entries = []
        if not entries:
            self._cache[key] = None
            return None
        try:
            pd_total = PhaseDiagram(entries)
            form = [PDEntry(e.composition, pd_total.get_form_energy(e)) for e in entries]
            form += [PDEntry(Composition(el), 0.0) for el in key]
            self._cache[key] = PhaseDiagram(form)
        except Exception:
            self._cache[key] = None
        return self._cache[key]

    def e_above_hull(self, composition, e_form_per_atom: float) -> float:
        """eV/atom above the hull, or NaN when MP has no diagram for the system."""
        from pymatgen.analysis.phase_diagram import PDEntry
        pd_form = self.diagram(composition.elements)
        if pd_form is None:
            return float("nan")
        try:
            return float(pd_form.get_e_above_hull(
                PDEntry(composition, e_form_per_atom * composition.num_atoms)))
        except Exception:
            return float("nan")


class FormationEnergyPredictor(PropertyPredictor):
    """MEGNet formation energy per atom, trained on MP."""

    output_base = "formation_energy_per_atom"
    MODEL_NAME = "MEGNet-Eform-MP-2018.6.1"

    def __init__(self):
        self.model = None

    def setup(self) -> None:
        import matgl
        self.model = matgl.load_model(self.MODEL_NAME)

    def predict(self, s) -> float:
        return float(self.model.predict_structure(structure=s))


class EnergyAboveHullPredictor(PropertyPredictor):
    """Formation energy from MEGNet, placed on the Materials Project hull.

    The hull arithmetic is exact (see HullLookup); the error is MEGNet's own, roughly
    0.03-0.05 eV/atom MAE. e_above_hull is a small difference of two such numbers, so
    absolute values are indicative -- the paired control-vs-steered shift is what carries
    signal, since both share the model and the prompts.
    """

    output_base = "energy_above_hull"

    def __init__(self, key_file: str = "api_keys.json"):
        self.eform = FormationEnergyPredictor()
        self.hull = HullLookup(key_file=key_file)

    def setup(self) -> None:
        self.eform.setup()
        self.hull.setup()

    def predict(self, s) -> float:
        return self.hull.e_above_hull(s.composition, self.eform.predict(s))


# property name -> factory(args) -> PropertyPredictor
REGISTRY = {
    "density_atomic": lambda args: GeometricPredictor("density_atomic"),
    "density":        lambda args: GeometricPredictor("density"),
    "volume":         lambda args: GeometricPredictor("volume"),
    "band_gap":       lambda args: BandGapPredictor(fidelity=getattr(args, "fidelity", 0)),
    "formation_energy":  lambda args: FormationEnergyPredictor(),
    "energy_above_hull": lambda args: EnergyAboveHullPredictor(),
}
