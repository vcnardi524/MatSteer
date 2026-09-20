#!/usr/bin/env python3
"""The LLaMat-CIF unconditional generation prompt, and the crystal-string encoder.

VENDORED, ON PURPOSE. These functions are copied verbatim from
`llamat/cif_prompts/cif_prompts.py`, which is itself a verified verbatim copy out of
M3RG-IITD/llamat ("copied verbatim from the repo so they can be reused on a new
dataset"). `llamat/` is an UNTRACKED clone, so without this file the prompt work is lost
on a fresh checkout and extraction cannot be reproduced. Provenance is noted per block.
`scripts/embeddings/extract_cif_embeddings.py` asserts this file still matches the clone
whenever the clone is present.

WHY THIS TEXT AND NOT A CIF
---------------------------
LLaMat-2-CIF reads CIFs but never writes one. Every generation task it was tuned on
emits a compact "crystal string" -- lengths, angles, then one element line and one
coordinate line per site -- which is turned into a CIF in Python afterwards. A direction
fitted on CIF-reading activations need not transfer to generation, which is what we
steer. So the embedding is taken from the model doing the thing we care about: the
unconditional generation prompt, with the structure's crystal string sitting exactly
where the model's own answer would go.

"Unconditional" means the prompt names no property -- no formula, no formation energy,
no energy above hull, no space group. In the source that is `generation_task_nate`
drawing k = 0 conditions, which is the same builder with an empty dict.

See README.md, "LLaMat-2 was trained on a different CIF layout than the one we feed it".
"""
import re

# ---------------------------------------------------------------------------
# Crystal string: what the model is trained to output.
#   llamat/src/cifs/notebooks/cif_tasks.ipynb cell 4 (get_crystal_string_nate)
# ---------------------------------------------------------------------------

def crystal_string(structure) -> str:
    """
    a b c              lengths, 1 decimal
    alpha beta gamma   angles, int() TRUNCATION (119.99999 -> 119), not rounding
    El                 one line per site, pymatgen species string
    x y z              fractional coordinates, 2 decimals

    Sites come out in pymatgen's parse order, which is what the authors' own pipeline
    fed the model -- do not sort them.

    NOTE the training-time random translation (cell 16) is deliberately NOT applied
    here. Extraction has to be deterministic, and since training randomised the offset
    on every example, any translation is equally in-distribution.
    """
    lengths = structure.lattice.parameters[:3]
    angles = structure.lattice.parameters[3:]
    return (
        " ".join("{0:.1f}".format(x) for x in lengths) + "\n"
        + " ".join(str(int(x)) for x in angles) + "\n"
        + "\n".join(
            str(species) + "\n" + " ".join("{0:.2f}".format(x) for x in coords)
            for species, coords in zip(structure.species, structure.frac_coords)
        )
    )


# ---------------------------------------------------------------------------
# System messages. Training picked one at RANDOM per example (random.choice).
#   llamat/src/cifs/notebooks/cif_tasks.ipynb cell 10.
# Extraction fixes one instead, so the only thing varying across rows is the structure.
# ---------------------------------------------------------------------------

GENERATION_SYSTEMS = [
    "You are a Material Science expert who works with crystallographic files (CIF files). Use your expertise to answer the following question related to generation of stable material when some information about it is described.",
    "Employ your expertise in Material Science, particularly in working with CIF files, to address the question concerning the creation of stable materials with partial descriptive information.",
    "Utilize your proficiency in Material Science and handling CIF files to provide insights into generating stable materials with limited descriptive data.",
    "Apply your knowledge as a Material Science specialist, specifically in manipulating CIF files, to respond to queries regarding the production of stable materials given incomplete information.",
    "Utilize your skills as a Material Science expert, with a focus on CIF files, to tackle the question concerning the development of stable materials based on partial descriptions.",
    "Employ your expertise in Material Science, particularly in the realm of CIF files, to address inquiries related to the creation of stable materials despite incomplete data.",
    "Utilize your proficiency in working with CIF files, as well as your background in Material Science, to answer questions regarding the generation of stable materials with limited descriptive details.",
    "Apply your knowledge and experience in Material Science, including your familiarity with CIF files, to provide solutions for generating stable materials when only partial information is available.",
    "Employ your specialized knowledge in Material Science, specifically your experience with CIF files, to tackle questions related to creating stable materials with partial information.",
    "Apply your skills as a Material Science expert, particularly in managing CIF files, to provide insights into generating stable materials despite incomplete descriptive data.",
]


# ---------------------------------------------------------------------------
# The input text. llamat/src/cifs/notebooks/cif_tasks.ipynb cell 10.
# ---------------------------------------------------------------------------

GENERATE_INSTRUCTION = (
    "Generate a description of the lengths and angles of the lattice vectors "
    "and then the element type and coordinates for each atom within the lattice"
)

# The spec asks for commas ("l1, l2, l3") but the target output uses spaces. That
# inconsistency is in the source; keep it.
OUTPUT_FORMAT_SPEC = (
    "The output should be of the following format ONLY:\n"
    "l1, l2, l3\n"
    "a1, a2, a3\n"
    "atom1\nx, y, z\natom2\nx, y, z\n ...\n\n"
    "l1, l2, l3 should be the predicted cell lengths.\n"
    "a1, a2, a3 should be the predicted cell angles.\n"
    "atom1, atom2, atom3, and so on, should be replaced with atom names and corresponding x, y, z with their coordinates in the lattice.\n"
)


def conditional_generation_input(conditions) -> str:
    """Cell 10. An EMPTY dict gives the unconditional prompt, which is what we use."""
    text = "Below is a description of a bulk material. "
    for key, value in conditions.items():
        raise NotImplementedError(
            "only the unconditional form is vendored -- see llamat/cif_prompts for the "
            "conditioned builders")
    return text + GENERATE_INSTRUCTION + ".\n" + OUTPUT_FORMAT_SPEC


# ---------------------------------------------------------------------------
# Wrappers. The repo is INCONSISTENT here and it is not resolvable from the source:
# the committed training pipeline (Megatron-LLM/tools/preprocess_instruct_data.py:84)
# uses ChatML, while every inference script the authors wrote uses `notebook`.
# Default to `notebook` and settle it empirically once the weights are in hand.
# ---------------------------------------------------------------------------

def notebook_prompt(system: str, input_text: str) -> str:
    """llamat/src/cifs/notebooks/cuda_*.ipynb and crystal-text-llm notebooks.
    The input already ends in a newline, so this reads "...lattice.\\noutput-"."""
    return f"{system} input-{input_text}output-"


def chatml_training_text(system: str, input_text: str, output=None) -> str:
    """Megatron-LLM/tools/preprocess_instruct_data.py:84, via src/preprocess_ift.sh.
    output=None returns the inference prompt, ending at the answer header."""
    def message(text, role):
        return f"<|im_start|>{role}\n{text}<|im_end|>\n"
    text = message(system, "system") + message(input_text, "question")
    if output is None:
        return text + "<|im_start|>answer\n"
    return text + message(output, "answer")


WRAPPERS = {"notebook": notebook_prompt, "chatml": chatml_training_text}


def unconditional_prompt(system_index: int = 0, wrapper: str = "notebook") -> str:
    """The constant prefix every structure's crystal string is appended to.

    Ours, not the repo's -- it just composes the two pieces above. Constant across the
    whole corpus, which is exactly why the extractor tokenises it once and pools only
    over the tokens that come AFTER it.
    """
    if wrapper not in WRAPPERS:
        raise ValueError(f"wrapper must be one of {sorted(WRAPPERS)}, got {wrapper!r}")
    return WRAPPERS[wrapper](GENERATION_SYSTEMS[system_index],
                             conditional_generation_input({}))


# ---------------------------------------------------------------------------
# Decoding: crystal string -> CIF. The inverse of crystal_string() above, kept
# beside it so the format is defined in one place.
#
# clean_first_line and parse_fn are VERBATIM from
#   llamat/src/cifs/crystal-text-llm/parsing_fn.py:4 and :61
# the ROBUST parser, not the strict one at cif_prompts.py:53. llamat-2 commonly emits
# a leading fragment -- ' is 4.2 4.2 6.7' -- and the strict parser dies on it.
# Vendored because that clone is untracked and would vanish on a fresh checkout.
# ---------------------------------------------------------------------------

def clean_first_line(line):
    """Pull three numbers out of a line that may carry leading text. Theirs, verbatim."""
    decimal_pattern = r"(\d*\.\d+|\d+)"
    numbers = re.findall(decimal_pattern, line)
    if len(numbers) >= 3:
        try:
            return [float(num) for num in numbers[:3]]
        except ValueError:
            pass
    text_decimal_pattern = r"[a-zA-Z]+\.(\d+)"
    matches = re.findall(text_decimal_pattern, line)
    if matches:
        fixed_line = line
        for match in matches:
            fixed_line = re.sub(r"[a-zA-Z]+\." + re.escape(match), "0." + match, fixed_line)
        numbers = re.findall(decimal_pattern, fixed_line)
        if len(numbers) >= 3:
            try:
                return [float(num) for num in numbers[:3]]
            except ValueError:
                pass
    parts = line.split()
    numeric_parts = []
    for part in parts:
        try:
            numeric_parts.append(float(part))
            if len(numeric_parts) == 3:
                break
        except ValueError:
            continue
    return numeric_parts if len(numeric_parts) >= 3 else []


def parse_fn(gen_str):
    """(lengths, angles, species, coords) from a generated crystal string. Theirs, verbatim.

    Two behaviours the caller must handle rather than trust:
      * total failure returns ([], [], [], []) instead of raising
      * a malformed coordinate line becomes [0.0, 0.0, 0.0], silently placing an atom at
        the origin. A structure that parses but is wrong is worse than one that fails, so
        crystal_string_to_cif counts these and reports them.
    """
    gen_str = gen_str.strip().strip('"')
    lines = [x.strip() for x in gen_str.split("\n") if len(x.strip()) > 0]
    start_idx = -1
    for i, line in enumerate(lines):
        numeric_parts = clean_first_line(line)
        if len(numeric_parts) == 3 and all(x > 0 for x in numeric_parts):
            start_idx = i
            break
    if start_idx == -1 or start_idx >= len(lines) - 1:
        return [], [], [], []
    try:
        lengths = clean_first_line(lines[start_idx])
        angles = clean_first_line(lines[start_idx + 1]) if start_idx + 1 < len(lines) else []
        species, coords = [], []
        for i in range(start_idx + 2, len(lines), 2):
            element = re.sub(r"[^A-Za-z]", "", lines[i].strip())
            if not element:
                continue
            species.append(element)
            if i + 1 < len(lines):
                parts = lines[i + 1].strip().split()
                try:
                    coords.append([float(x) for x in parts[:3]] if len(parts) >= 3
                                  else [0.0, 0.0, 0.0])
                except ValueError:
                    coords.append([0.0, 0.0, 0.0])
            else:
                coords.append([0.0, 0.0, 0.0])
        return lengths, angles, species, coords
    except (ValueError, IndexError):
        return [], [], [], []


DEFAULT_SYMPREC = 0.1        # matches crystallm _metrics.is_space_group_consistent


def crystal_string_to_cif(text, symprec=DEFAULT_SYMPREC):
    """(cif_text, reason) for one generated crystal string; (None, reason) on failure.

    WHY symprec IS NOT OPTIONAL HERE. The decoded Structure carries no symmetry -- every
    atom is listed and pymatgen writes `P 1`. CrystaLLM's is_space_group_consistent
    (_metrics.py:70) compares a CIF's STATED space group against what SpacegroupAnalyzer
    detects from its coordinates, and 87% of these structures do have real symmetry, so a
    P1-written CIF fails by construction: measured 0/40 valid. Passing symprec makes
    Structure.to() hand it to pymatgen.io.cif.CifWriter, which runs SpacegroupAnalyzer and
    writes the DETECTED symbol -- 27/40.

    That is the right source of truth rather than a workaround: llamat2-cif never emits a
    space group, so the only meaningful one is whatever its coordinates imply.

    LEAVE CifWriter's refine_struct AT ITS DEFAULT (True). It rewrites the cell in the
    conventional setting, which on 3.5% of structures returns twice the atoms in twice the
    volume -- the same crystal, re-expressed, so composition and every intensive property
    (density per atom, energy per atom) are untouched. Setting refine_struct=False to
    "preserve" the cell is worse, not better: the symmetry operators are only valid in the
    standard setting, so writing them against an unrefined cell breaks the round trip.
    Measured over 200 structures: atom count survives 96.5% with the default against 90.5%
    without, and is_valid 29/40 against 25/40.

    Round-tripping is lossy on lengths regardless, because the ENCODER rounds lengths to 1
    decimal and truncates angles with int(). Volume per atom moves by a median 0.64%
    (p95 2.3%, max 3.9%). That is the format's precision, not a decode error.
    """
    from pymatgen.core.lattice import Lattice
    from pymatgen.core.structure import Structure

    lengths, angles, species, coords = parse_fn(text or "")
    if len(lengths) != 3:
        return None, "no line with three positive numbers (lattice) found"
    if len(angles) != 3:
        return None, "no angle line after the lattice line"
    if not species:
        return None, "no element lines"
    if len(species) != len(coords):
        return None, f"{len(species)} elements against {len(coords)} coordinate lines"
    # Their parser substitutes [0,0,0] for an unparseable coordinate line, which reads as
    # a real atom at the origin. Report it; the caller decides whether to keep the row.
    n_origin = sum(1 for c in coords if c == [0.0, 0.0, 0.0])
    try:
        struct = Structure(Lattice.from_parameters(*lengths, *angles), species, coords,
                           coords_are_cartesian=False)
        cif = struct.to(fmt="cif", symprec=symprec)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    reason = f"{n_origin} coordinate line(s) defaulted to the origin" if n_origin else ""
    return cif, reason


if __name__ == "__main__":
    p = unconditional_prompt()
    print(p)
    print(f"\n[{len(p)} chars, constant for every structure]")
