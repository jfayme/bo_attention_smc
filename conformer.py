"""Conformer ensembles from RDKit embedding and MLIP relaxation (MACE-MH1).

Conformers are produced in four stages, each one cutting down the number of
structures handed to the next:

1. embedding with RDKit ETKDGv3, scaled to the rotatable bond count;
2. optimisation with a classical force field (MMFF94, falling back to UFF for
   elements MMFF has no parameters for);
3. clustering of the low-energy survivors on heavy-atom RMSD, keeping one
   representative per cluster;
4. relaxation of those representatives with a MACE machine-learned
   interatomic potential (MLIP), followed by duplicate minima removal.

Every tunable parameter is read from the `[conformer]` table of the settings
module.

The ensemble this module returns holds every unique minimum the search found,
up to the number of MLIP relaxations the settings allow. It can be saved and
read back, so that the search, by far the slowest step of the pipeline, runs
once per molecule.

"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.optimize import BFGS
from mace.calculators import MACECalculator, mace_mp
from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    Descriptors,
    rdMolAlign,
    rdMolDescriptors,
)
from rdkit.ML.Cluster import Butina

from logging_config import get_logger
from settings import SETTINGS

logger = get_logger(__name__)

CONFORMER_SETTINGS = SETTINGS["conformer"]

EV_TO_KCAL = units.mol / units.kcal  # 23.0605 kcal/mol per eV


@dataclass(frozen=True)
class ConformerEnsemble:
    """Relaxed conformers of a single molecule, sorted by MLIP energy.

    The ensemble is ordered lowest energy first, so `coords[0]` is always the
    best minimum the search found.

    Attributes
    ----------
    numbers
        Atomic numbers of the molecule, shape (N,), dtype int64. Shared by
        every conformer, which all describe the same atoms in the same order.
    coords
        Cartesian coordinates (in Angstrom), shape (K, N, 3), dtype float32.
    energies
        Potential energy of each conformer (in eV), shape (K,), dtype
        float64, in the same order as the coordinates.
    charge
        Total formal charge of the molecule.
    multiplicity
        Spin multiplicity of the molecule, derived from its radical electron
        count.

    """

    numbers: np.ndarray  # (N,)      int64
    coords: np.ndarray  # (K, N, 3) float32, Angstrom
    energies: np.ndarray  # (K,)      float64, eV
    charge: int
    multiplicity: int

    def __len__(self) -> int:
        return len(self.energies)

    @property
    def lowest(self) -> np.ndarray:
        """Coordinates of the lowest-energy conformer, shape (N, 3)."""
        return self.coords[0]

    @property
    def as_atoms(self) -> tuple[np.ndarray, np.ndarray, int]:
        """Numbers, coordinates and charge of the lowest-energy conformer.

        Packaged in the order that the downstream calculators expect their
        input in.

        """
        return self.numbers, self.lowest, self.charge

    @property
    def relative_energies_kcal(self) -> np.ndarray:
        """Energies relative to the lowest conformer, shape (K,), kcal/mol."""
        return (self.energies - self.energies.min()) * EV_TO_KCAL

    def boltzmann_weights(self, temperature: float) -> np.ndarray:
        """Normalised Boltzmann populations of the conformers.

        Parameters
        ----------
        temperature
            Temperature at which the populations are evaluated (in K).

        Returns
        -------
        weights
            Populations summing to one, shape (K,), in the same order as the
            energies.

        Notes
        -----
        The populations are taken over the conformers that the search
        happened to find, which is not an exhaustive sample, and are
        therefore a smooth weighting rather than a thermodynamic quantity.

        """
        exponent = -(self.energies - self.energies.min()) / (
            units.kB * temperature
        )
        weights = np.exp(exponent)
        return weights / weights.sum()

    def save(self, path: Path, smiles: str) -> None:
        """Write the ensemble to an .npz file.

        Parameters
        ----------
        path
            File to write.
        smiles
            SMILES of the molecule, stored with the ensemble so that a
            reader can tell another molecule's file from its own.

        Notes
        -----
        The file is written under a temporary name and then moved into
        place, so an interrupted write never leaves a torn file behind. Every
        array is numeric or text, so reading it back never needs pickle.

        """
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as handle:
            np.savez(
                handle,
                smiles=np.array(smiles),
                numbers=self.numbers,
                coords=self.coords,
                energies=self.energies,
                charge=np.int64(self.charge),
                multiplicity=np.int64(self.multiplicity),
            )
        temporary.replace(path)

    @classmethod
    def load(
        cls: type["ConformerEnsemble"], path: Path, smiles: str
    ) -> "ConformerEnsemble | None":
        """Read an ensemble written by `save`.

        Parameters
        ----------
        path
            File to read.
        smiles
            SMILES of the molecule the ensemble must belong to.

        Returns
        -------
        ensemble
            The ensemble, or None if the file is missing or holds another
            molecule.

        """
        if not path.is_file():
            return None
        with np.load(path) as saved:
            if str(saved["smiles"]) != smiles:
                logger.warning(
                    "%s holds another molecule than %r", path, smiles
                )
                return None
            return cls(
                numbers=saved["numbers"],
                coords=saved["coords"],
                energies=saved["energies"],
                charge=int(saved["charge"]),
                multiplicity=int(saved["multiplicity"]),
            )


class ConformerGenerator:
    """Conformer search combining RDKit embedding with MLIP relaxation.

    Calling an instance on a SMILES string runs the whole pipeline and
    returns the selected minima as a ConformerEnsemble dataclass.

    Parameters
    ----------
    confs_per_rot_bond
        Conformers embedded per rotatable bond, before the cap below is
        applied; flexible molecules therefore get a larger starting pool.
    max_conformers
        Upper bound on the number of embedded conformers, whatever the
        flexibility of the molecule.
    seed
        Random seed handed to ETKDG.
    small_ring_torsions
        If True, ETKDG uses the torsion potentials fitted for small rings.
    prune_rms_thresh
        Embeddings closer than this RMSD (in Angstrom) are treated as
        duplicates and dropped as they are generated.
    n_threads
        Threads used by the RDKit routines; 0 uses every available core.
    max_iters
        Force-field optimisation steps allowed per conformer.
    mmff_window_kcal
        Only conformers within this energy of the force-field minimum (in
        kcal/mol) are considered for clustering.
    rmsd_cluster_thresh
        Butina cut-off (in Angstrom) on the heavy-atom RMSD, deciding which
        conformers count as the same shape.
    max_automorphisms
        Number of symmetry-equivalent atom mappings above which the
        clustering RMSD is measured on the molecule with stripped
        terminal atoms.
    pre_filter_conf_max
        Cap on the number of cluster representatives to be relaxed with MLIP.
    model_paths
        Path to the MACE weights to load. If None, cached mh-1 weights are
        reused, and downloaded first if the cache does not hold them.
    device
        Torch device the potential is evaluated on.
    head
        MACE output head selecting which trained target is predicted.
    default_dtype
        Floating-point precision the potential is evaluated at.
    enable_cueq
        If True, the cuEquivariance acceleration path is used.
    fmax
        Force convergence criterion for the BFGS relaxation, in
        eV/Angstrom.
    opt_steps
        Step budget for the BFGS relaxation of each conformer.
    log_every
        Relaxation progress is logged every this many conformers.
    dedup_energy_kcal
        Two relaxed minima closer than this in energy (in kcal/mol) are
        candidates for being the same structure.
    dedup_rmsd
        Two such candidates closer than this heavy-atom RMSD (in Angstrom)
        are taken to be the same structure, and one is discarded.
    max_final
        Number of lowest-energy unique minima kept in the returned ensemble.

    Notes
    -----
    Every default is read from the `[conformer]` table of the settings file.

    """

    def __init__(
        self,
        confs_per_rot_bond: int = CONFORMER_SETTINGS["confs_per_rot_bond"],
        max_conformers: int = CONFORMER_SETTINGS["max_conformers"],
        seed: int = CONFORMER_SETTINGS["seed"],
        small_ring_torsions: bool = CONFORMER_SETTINGS["small_ring_torsions"],
        prune_rms_thresh: float = CONFORMER_SETTINGS["prune_rms_thresh"],
        n_threads: int = CONFORMER_SETTINGS["n_threads"],
        max_iters: int = CONFORMER_SETTINGS["max_iters"],
        mmff_window_kcal: float = CONFORMER_SETTINGS["mmff_window_kcal"],
        rmsd_cluster_thresh: float = CONFORMER_SETTINGS["rmsd_cluster_thresh"],
        max_automorphisms: int = CONFORMER_SETTINGS["max_automorphisms"],
        pre_filter_conf_max: int = CONFORMER_SETTINGS["pre_filter_conf_max"],
        model_paths: str | None = CONFORMER_SETTINGS["model_paths"] or None,
        device: str = CONFORMER_SETTINGS["device"],
        head: str = CONFORMER_SETTINGS["head"],
        default_dtype: str = CONFORMER_SETTINGS["default_dtype"],
        enable_cueq: bool = CONFORMER_SETTINGS["enable_cueq"],
        fmax: float = CONFORMER_SETTINGS["fmax"],
        opt_steps: int = CONFORMER_SETTINGS["opt_steps"],
        log_every: int = CONFORMER_SETTINGS["log_every"],
        dedup_energy_kcal: float = CONFORMER_SETTINGS["dedup_energy_kcal"],
        dedup_rmsd: float = CONFORMER_SETTINGS["dedup_rmsd"],
        max_final: int = CONFORMER_SETTINGS["max_final"],
    ) -> None:
        self.confs_per_rot_bond = confs_per_rot_bond
        self.max_conformers = max_conformers
        self.seed = seed
        self.small_ring_torsions = small_ring_torsions
        self.prune_rms_thresh = prune_rms_thresh
        self.n_threads = n_threads
        self.max_iters = max_iters
        self.mmff_window_kcal = mmff_window_kcal
        self.rmsd_cluster_thresh = rmsd_cluster_thresh
        self.max_automorphisms = max_automorphisms
        self.pre_filter_conf_max = pre_filter_conf_max
        self.model_paths = model_paths
        self.device = device
        self.head = head
        self.default_dtype = default_dtype
        self.enable_cueq = enable_cueq
        self.fmax = fmax
        self.opt_steps = opt_steps
        self.log_every = log_every
        self.dedup_energy_kcal = dedup_energy_kcal
        self.dedup_rmsd = dedup_rmsd
        self.max_final = max_final
        self.mace_calc = None

    def __call__(self, smiles: str) -> ConformerEnsemble:
        """Run the full conformer search on a molecule.

        Parameters
        ----------
        smiles
            SMILES string of the molecule to search.

        Returns
        -------
        ensemble
            The unique relaxed minima, lowest energy first.

        """
        mol, conf_ids, results = self.generate_rdkit_conformers(smiles)
        return self.conformer_search(mol, conf_ids, results)

    @property
    def calc(self) -> MACECalculator:
        """MACE calculator used for the relaxations."""
        if self.mace_calc is None:
            model = self.model_paths or self._cached_mh1()
            if model is None:
                # nothing cached: mace_mp downloads mh-1 and caches it
                logger.info(
                    "Loading MACE 'mh-1' on %s (downloads if needed)",
                    self.device,
                )
                self.mace_calc = mace_mp(
                    model="mh-1",
                    device=self.device,
                    default_dtype=self.default_dtype,
                    head=self.head,
                    enable_cueq=self.enable_cueq,
                )
            else:
                logger.info("Loading MACE model %r on %s", model, self.device)
                self.mace_calc = MACECalculator(
                    model_paths=model,
                    device=self.device,
                    default_dtype=self.default_dtype,
                    head=self.head,
                    enable_cueq=self.enable_cueq,
                )
        return self.mace_calc

    @staticmethod
    def _cached_mh1() -> str | None:
        """Find already-downloaded mh-1 weights in the MACE cache.

        The match is made on the normalised name rather than an exact one.

        Returns
        -------
        path
            Path to the cached weights, or None if the cache does not hold
            them.

        """
        from mace.calculators.foundations_models import get_cache_dir

        cache = Path(get_cache_dir())
        if not cache.is_dir():
            return None
        for path in cache.iterdir():
            normalised = "".join(filter(str.isalnum, path.name.lower()))
            if path.is_file() and normalised == "macemh1model":
                return str(path)
        return None

    def generate_rdkit_conformers(
        self, smiles: str
    ) -> tuple[Chem.Mol, list[int], list[tuple[int, float]]]:
        """Embed and force-field optimise a pool of conformers.

        The size of the pool is scaled with the rotatable bond count, so that
        flexible molecules are sampled more heavily, up to a cap. MMFF94
        is used where it has parameters for every atom, and UFF otherwise.

        Parameters
        ----------
        smiles
            SMILES string of the molecule to embed.

        Returns
        -------
        mol
            The molecule, with explicit hydrogens and the optimised
            conformers attached.
        conf_ids
            Identifiers of the embedded conformers.
        results
            One status and energy pair per conformer; a non-zero status marks
            an optimisation that did not converge.

        Raises
        ------
        ValueError
            If RDKit cannot parse the SMILES string.
        RuntimeError
            If ETKDG embeds no conformer at all.

        """
        # Parse SMILES and add hydrogens to the molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
        mol = Chem.AddHs(mol)

        # Count rotatable bonds to estimate conformational complexity
        n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
        logger.info("Rotatable bonds: %d for %r", n_rot, smiles)

        # Generate a 3D conformer of the molecule using the ETKDGv3 method
        params = AllChem.ETKDGv3()
        params.randomSeed = self.seed
        params.useSmallRingTorsions = self.small_ring_torsions
        params.numThreads = self.n_threads
        params.pruneRmsThresh = self.prune_rms_thresh

        # Scale the number of conformers with flexibility, then cap it
        n_confs = min(
            self.confs_per_rot_bond * (n_rot + 1), self.max_conformers
        )
        conf_ids = list(
            AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
        )
        if not conf_ids:
            logger.warning("ETKDG embedded no conformer for %r", smiles)
            raise RuntimeError(
                f"ETKDG failed to embed a conformer for {smiles!r}"
            )

        logger.info("Generated %d conformers for %r", len(conf_ids), smiles)

        # MMFF94 has no parameters for some elements (B, As, Se, Cs, K, etc.),
        # which UFF covers

        if AllChem.MMFFHasAllMoleculeParams(mol):
            results = AllChem.MMFFOptimizeMoleculeConfs(
                mol, maxIters=self.max_iters, numThreads=self.n_threads
            )
            logger.info("MMFF optimisation used for %r", smiles)
        else:
            results = AllChem.UFFOptimizeMoleculeConfs(
                mol, maxIters=self.max_iters, numThreads=self.n_threads
            )
            logger.info("UFF optimisation used for %r", smiles)

        return mol, conf_ids, results

    def conformer_search(
        self,
        mol: Chem.Mol,
        conf_ids: list[int],
        results: list[tuple[int, float]],
    ) -> ConformerEnsemble:
        """Relax the selected conformers and assemble the conformer ensemble.

        The cluster representatives (conformer with lowest force-field energy
        of the cluster) are relaxed one at a time with the MLIP, the ones that
        do not converge are dropped. The resulting conformers are sorted by
        energy and deduplicated, and the lowest minima are returned.

        Parameters
        ----------
        mol
            The molecule carrying the force-field optimised conformers.
        conf_ids
            Identifiers of the conformers to consider.
        results
            One status and energy pair per conformer, as returned by the
            embedding step.

        Returns
        -------
        ensemble
            The unique relaxed minima, lowest energy first.

        Raises
        ------
        RuntimeError
            If every selected conformer failed the MLIP relaxation.

        """
        # Extract info common to all conformers
        smiles = Chem.MolToSmiles(mol)
        charge = Chem.GetFormalCharge(mol)
        atomic_nbrs = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

        ff_selected, energy_by_id = self._cluster_conf_select(
            mol, conf_ids, results
        )

        # MLIP relaxation of the selected conformers
        relaxed = []

        for i, conf_id in enumerate(ff_selected, 1):
            # create an ASE Atoms object for the conformer
            conf = mol.GetConformer(conf_id)
            positions = conf.GetPositions()
            atoms = Atoms(
                numbers=atomic_nbrs,
                positions=positions,
            )

            # Run the optimisation
            atoms.calc = self.calc

            optimizer = BFGS(atoms, logfile=None)
            converged = optimizer.run(fmax=self.fmax, steps=self.opt_steps)
            energy = atoms.get_potential_energy()

            relaxed.append(
                {
                    "conf_id": conf_id,
                    "atoms": atoms.copy(),
                    "ff_energy_kcal": energy_by_id[conf_id],
                    "energy_eV": energy,
                    "converged": converged,
                }
            )

            # optimisation progress logging
            if i % self.log_every == 0:
                logger.info(
                    "relaxed %d/%d conformers for %r",
                    i,
                    len(ff_selected),
                    smiles,
                )

        # Remove failed optimisations
        optimized = [conf for conf in relaxed if conf["converged"]]

        if not optimized:
            raise RuntimeError(
                f"every conformer failed MLIP relaxation for {smiles!r}"
            )

        # Sort by energy
        optimized.sort(key=lambda conf: conf["energy_eV"])

        # Energies relative to the lowest conformer, in kcal/mol: the unit
        # the dedup gate and the span logged below are both expressed in.
        e_min = optimized[0]["energy_eV"]
        for conf in optimized:
            conf["rel_energy_kcal"] = (conf["energy_eV"] - e_min) * EV_TO_KCAL

        # Remove duplicates based on energy and RMSD
        unique = self._remove_duplicates(mol, optimized)
        logger.info(
            "%s: %d relaxed -> %d unique minima, spanning %.2f kcal/mol",
            smiles,
            len(optimized),
            len(unique),
            unique[-1]["rel_energy_kcal"],
        )

        # cut down to the max_final number of conformers to return
        unique = unique[: self.max_final]

        numbers = np.asarray(atomic_nbrs, dtype=np.int64)
        coords = np.stack(
            [conf["atoms"].get_positions() for conf in unique]
        ).astype(np.float32)
        energies = np.asarray(
            [conf["energy_eV"] for conf in unique], dtype=np.float64
        )

        multiplicity = Descriptors.NumRadicalElectrons(mol) + 1

        return ConformerEnsemble(
            numbers=numbers,
            coords=coords,
            energies=energies,
            charge=charge,
            multiplicity=multiplicity,
        )

    @staticmethod
    def _strip_terminal_atoms(mol_no_h: Chem.Mol) -> Chem.Mol:
        """Drop heavy atoms bonded to only one other heavy atom.

        Such atoms are the fluorines of a CF3 group, the methyls of an
        isopropyl, or the methyl of a methoxy (i.e., groups whose rotation
        is not a conformational degree of freedom). Their atoms are
        interchangeable, which is what makes the symmetry-aware RMSD
        expensive to evaluate. The skeleton that defines the conformer shape
        is left untouched.

        Parameters
        ----------
        mol_no_h
            Heavy-atom molecule, carrying the embedded conformers.

        Returns
        -------
        core
            The same molecule without its terminal heavy atoms. Conformers
            are carried over, minus the dropped coordinates.

        """
        core = Chem.RWMol(mol_no_h)
        terminal = [
            atom.GetIdx()
            for atom in mol_no_h.GetAtoms()
            if atom.GetDegree() == 1
        ]
        for idx in sorted(terminal, reverse=True):
            core.RemoveAtom(idx)
        return core.GetMol()

    def _rmsd_reference(self, mol_no_h: Chem.Mol, smiles: str) -> Chem.Mol:
        """Choose the molecule that the clustering RMSD is measured on.

        A molecule carrying many interchangeable terminal groups (e.g., CF3,
        isopropyl, OMe, etc.) can make the distance matrix of GetBestRMS
        prohibitively expensive, thus above max_automorphisms the terminal
        atoms are dropped.

        Parameters
        ----------
        mol_no_h
            Heavy-atom molecule, carrying the embedded conformers.
        smiles
            SMILES of the molecule, used to identify it in the warning.

        Returns
        -------
        reference
            Either the molecule unchanged, or its stripped core.

        """
        n_automorphisms = len(
            mol_no_h.GetSubstructMatches(
                mol_no_h,
                uniquify=False,
                useChirality=False,
                maxMatches=self.max_automorphisms,
            )
        )
        if n_automorphisms < self.max_automorphisms:
            return mol_no_h

        core = self._strip_terminal_atoms(mol_no_h)

        logger.warning(
            "%s: at least %d symmetry mappings; clustering on the %d-atom "
            "core instead of all %d heavy atoms (%d terminal atoms dropped)",
            smiles,
            self.max_automorphisms,
            core.GetNumAtoms(),
            mol_no_h.GetNumAtoms(),
            mol_no_h.GetNumAtoms() - core.GetNumAtoms(),
        )
        return core

    def _cluster_conf_select(
        self,
        mol: Chem.Mol,
        conf_ids: list[int],
        results: list[tuple[int, float]],
    ) -> tuple[list[int], dict[int, float]]:
        """Choose which force-field conformers are worth relaxing.

        Conformers that failed to converge or those outside the energy window
        are dropped. The rest are clustered on heavy-atom RMSD so that only one
        representative (the lowest energy one) per shape / cluster survives.
        The representatives are then truncated to a cap, to limit the MLIP
        calculations.

        Parameters
        ----------
        mol
            The molecule carrying the embedded conformers.
        conf_ids
            Identifiers of the conformers to consider.
        results
            One status and energy pair per conformer, as returned by the
            embedding step.

        Returns
        -------
        selected
            Identifiers of the conformers to hand to the MLIP.
        energy_by_id
            Force-field energy of every converged conformer, keyed by
            identifier.

        Raises
        ------
        RuntimeError
            If no conformer survived the force-field optimisation.

        """
        smiles = Chem.MolToSmiles(mol)

        # Remove failed optimisation and map conf_id to energy
        energy_by_id = {
            conf_id: energy
            for conf_id, (status, energy) in zip(
                conf_ids, results, strict=True
            )
            if status == 0  # 1: the force field did not converge
        }
        if not energy_by_id:
            raise RuntimeError(
                "no conformer survived force-field optimisation for "
                f"{smiles!r}"
            )

        # pre-filtering keeping only conformers under the mmff_window_kcal
        ff_ranked = sorted(energy_by_id, key=energy_by_id.__getitem__)

        energy_cutoff = energy_by_id[ff_ranked[0]] + self.mmff_window_kcal
        ff_in_window = [
            cid for cid in ff_ranked if energy_by_id[cid] <= energy_cutoff
        ]

        # Cluster the surviving conformers on heavy-atom RMSD and energy
        mol_no_h = Chem.RemoveHs(mol)

        # Check if the mol has too many automorphisms, strip terminal atoms if
        # necessary
        rmsd_mol = self._rmsd_reference(mol_no_h, smiles)

        # Create a new molecule with only the conformers in the energy window
        cluster_mol = Chem.Mol(rmsd_mol)
        cluster_mol.RemoveAllConformers()
        kept_ids: list[int] = []
        for conf_id in ff_in_window:
            cluster_mol.AddConformer(
                Chem.Conformer(rmsd_mol.GetConformer(conf_id)), assignId=True
            )
            kept_ids.append(conf_id)

        if len(kept_ids) == 1:
            # GetAllConformerBestRMS returns an empty matrix for a single
            # conformer, which Butina cannot be handed.
            representatives = list(kept_ids)
        else:
            # Compute the RMSD matrix for the conformers in the window
            rmsd_matrix = rdMolAlign.GetAllConformerBestRMS(
                cluster_mol, numThreads=self.n_threads
            )

            # Perform Butina clustering on the RMSD matrix
            clusters = Butina.ClusterData(
                data=rmsd_matrix,
                nPts=len(kept_ids),
                distThresh=self.rmsd_cluster_thresh,
                isDistData=True,
                reordering=True,
            )
            # Select the lowest energy member for each cluster
            representatives = [
                kept_ids[
                    min(cluster, key=lambda pos: energy_by_id[kept_ids[pos]])
                ]
                for cluster in clusters
            ]

        # Sort the representatives by energy and select the lowest-energy
        # ones up to the pre_filter_conf_max limit
        representatives.sort(key=energy_by_id.__getitem__)
        selected = representatives[: self.pre_filter_conf_max]

        logger.info(
            "%s: %d embedded -> %d within %.1f kcal/mol -> %d clusters "
            "-> %d to the MLIP",
            smiles,
            len(conf_ids),
            len(ff_in_window),
            self.mmff_window_kcal,
            len(representatives),
            len(selected),
        )
        return selected, energy_by_id

    def _remove_duplicates(
        self, mol: Chem.Mol, optimized: list[dict]
    ) -> list[dict]:
        """Discard relaxed conformers that describe the same minimum.

        Two conformers are judged duplicates only if they agree on both
        counts: their energies are within the deduplication window and their
        heavy-atom RMSD is below the deduplication threshold.

        Parameters
        ----------
        mol
            The molecule the conformers belong to, used as the topology
            template for the RMSD comparison.
        optimized
            Relaxed conformer records, sorted by energy, each carrying an
            ASE atoms object and a relative energy in kcal/mol.

        Returns
        -------
        uniques
            The subset of records kept, in the order they were given in.

        """
        relaxed_mol = Chem.Mol(mol)
        relaxed_mol.RemoveAllConformers()

        # Add the relaxed conformers to the molecule
        for entry in optimized:
            conformer = Chem.Conformer(relaxed_mol.GetNumAtoms())
            positions = entry["atoms"].get_positions()  # float64 (N, 3)
            conformer.SetPositions(positions)
            entry["rms_id"] = relaxed_mol.AddConformer(
                conformer, assignId=True
            )

        # RMSD on heavy atoms only
        rms_mol = Chem.RemoveHs(relaxed_mol)

        uniques: list[dict] = []
        for entry in optimized:
            duplicate = False
            for kept in uniques:
                if (
                    abs(entry["rel_energy_kcal"] - kept["rel_energy_kcal"])
                    >= self.dedup_energy_kcal
                ):
                    continue  # too far apart in energy to be the same
                rmsd = rdMolAlign.GetBestRMS(
                    # a copy: GetBestRMS aligns the probe in place
                    Chem.Mol(rms_mol),
                    rms_mol,
                    entry["rms_id"],
                    kept["rms_id"],
                    numThreads=self.n_threads,
                )
                if rmsd < self.dedup_rmsd:
                    duplicate = True
                    break
            if not duplicate:
                uniques.append(entry)
        return uniques
