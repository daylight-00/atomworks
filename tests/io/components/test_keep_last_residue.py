import biotite.structure as struc
import numpy as np

from atomworks.io.transforms.atom_array import keep_last_residue


def _atom_array(residues: list[tuple[int, str, str]]) -> struc.AtomArray:
    """Two atoms per ``(res_id, ins_code, res_name)`` residue, all on chain A."""
    atoms = [
        struc.Atom(np.zeros(3), chain_id="A", res_id=r, ins_code=i, res_name=n, atom_name=a, element=a[0])
        for r, i, n in residues
        for a in ("N", "CA")
    ]
    return struc.array(atoms)


def test_insertion_codes_are_distinct_positions():
    # PDB numbering as in 1PPB chain L: 1H, 1G, ... 1
    residues = [(1, "H", "THR"), (1, "G", "PHE"), (1, "F", "GLY"), (1, "", "CYS"), (2, "", "GLY")]
    kept = keep_last_residue(_atom_array(residues))
    assert len(kept) == 2 * len(residues)
    assert list(kept.ins_code[::2]) == ["H", "G", "F", "", ""]


def test_sequence_heterogeneity_keeps_last_residue():
    residues = [(1, "", "MET"), (2, "", "ALA"), (2, "", "GLY"), (3, "A", "SER"), (3, "A", "THR"), (4, "", "LYS")]
    kept = keep_last_residue(_atom_array(residues))
    assert list(kept.res_name[::2]) == ["MET", "GLY", "THR", "LYS"]


def test_without_insertion_codes_annotation():
    array = _atom_array([(1, "", "MET"), (2, "", "ALA"), (2, "", "GLY")])
    array.del_annotation("ins_code")
    kept = keep_last_residue(array)
    assert list(kept.res_name[::2]) == ["MET", "GLY"]
