"""Featurisers turning molecules into the vectors a GP is given.

Three descriptor families feed the length-scale study: Morgan fingerprints
and T5 text embeddings read the SMILES alone, while MLIP descriptors are
read per atom and per conformer and so carry two axes that have to be
collapsed.

The atom axis is collapsed first, by the `aggregation`, and the conformer
axis second, by the `pooling`. Under the `attention` aggregation the
weighting is fitted through the surrogate instead, so both axes survive the
featuriser and the same two collapses run inside the GP, in that order.

"""

import numpy as np
import torch
from aimnet.calculators import AIMNet2Calculator
from ase import Atoms
from mace.calculators import MACECalculator, mace_mp
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from transformers import AutoTokenizer, T5Config, T5EncoderModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from conformer import ConformerEnsemble
from logging_config import get_logger
from settings import SETTINGS

logger = get_logger(__name__)

AIMNET2_SETTINGS = SETTINGS["aimnet2"]
MACE_SETTINGS = SETTINGS["mace"]
MORGAN_SETTINGS = SETTINGS["morgan"]
FEATURIZER_SETTINGS = SETTINGS["featurizer"]

# How a conformer ensemble descriptor matrix is collapsed to a single vector.
POOLINGS = ("lowest", "boltzmann", "ensemble_stats")

# Order of the blocks ensemble_stats concatenates.
STAT_NAMES = ("min", "max", "mean", "std")

# MLIPS checkpoints.
MLIPS = ("mace", "aimnet2")

# T5 checkpoints.
T5_VARIANTS = ("vanilla", "augm")

# Atom aggregations fitted through the GP.
LEARNED_AGGREGATIONS = ("attention",)

# Atom aggregations that are a fixed function of the descriptor matrix.
FIXED_AGGREGATIONS = ("mean", "mean_std_max")

# Order of the blocks mean_std_max concatenates.
POOL_STAT_NAMES = ("mean", "std", "max")


class MLIPFeaturizer:
    """Per-atom MLIP descriptors, collapsed to one vector per molecule.

    Both potentials live here; `mlip` selects which one is used, and the one
    not selected is never loaded.

    Parameters
    ----------
    mlip
        Potential the descriptors are read from, one of MLIPS.
    model_paths
        MACE weights to load. If None, `mace_model_name` is resolved and
        cached.
    mace_model_name
        MACE checkpoint the calculator resolves when no weights are given.
    head
        MACE output head selecting the trained target.
    default_dtype
        Precision the MACE potential is evaluated at.
    enable_cueq
        Whether to use the cuEquivariance acceleration path.
    mace_layers
        MACE interaction layers read, one of `last` or `all`; each layer
        gives the same number of invariant channels.
    calculator
        Already-built MACE calculator. If None, one is built on first use.
    aimnet_model_name
        AIMNet2 checkpoint the calculator resolves.
    aimnet_layers
        AIMNet2 MLP passes read, one of `last` or `all`.
    device
        Torch device the potential runs on. If None, taken from the settings
        table of the selected potential.
    log_every
        Molecules between progress log records. If None, taken from the
        settings table of the selected potential.
    aggregation
        Collapse applied to the atom axis, one of FIXED_AGGREGATIONS or
        LEARNED_AGGREGATIONS.
    pooling
        Collapse applied to the conformer axis, one of POOLINGS.
    temperature
        Temperature (in K) used by the `boltzmann` pooling.

    Attributes
    ----------
    name
        Name of the potential, used to label the featuriser in results.

    Raises
    ------
    ValueError
        If the potential, the aggregation or the pooling is not one of the
        names its tuple holds.

    Notes
    -----
    `device` and `log_every` appear in both the `[mace]` and `[aimnet2]`
    tables, so they cannot be defaulted in the signature and are resolved
    once the potential is known.

    """

    def __init__(
        self,
        mlip: str = FEATURIZER_SETTINGS["mlip"],
        # MACE #
        model_paths: str | None = MACE_SETTINGS["model_paths"] or None,
        mace_model_name: str = MACE_SETTINGS["mace_model_name"],
        head: str = MACE_SETTINGS["head"],
        default_dtype: str = MACE_SETTINGS["default_dtype"],
        enable_cueq: bool = MACE_SETTINGS["enable_cueq"],
        mace_layers: str = MACE_SETTINGS["mace_layers"],
        calculator: MACECalculator | None = None,
        # AIMNET2 #
        aimnet_model_name: str = AIMNET2_SETTINGS["aimnet_model_name"],
        aimnet_layers: str = AIMNET2_SETTINGS["aimnet_layers"],
        # SHARED #
        device: str | None = None,
        log_every: int | None = None,
        aggregation: str = FEATURIZER_SETTINGS["aggregation"],
        pooling: str = FEATURIZER_SETTINGS["pooling"],
        temperature: float = FEATURIZER_SETTINGS["temperature_k"],
    ) -> None:
        # Sanity checks
        if mlip not in MLIPS:
            raise ValueError(f"unknown mlip {mlip!r}, expected one of {MLIPS}")
        if aggregation not in FIXED_AGGREGATIONS + LEARNED_AGGREGATIONS:
            raise ValueError(
                f"unknown aggregation {aggregation!r}, expected one of "
                f"{FIXED_AGGREGATIONS + LEARNED_AGGREGATIONS}"
            )
        if pooling not in POOLINGS:
            raise ValueError(
                f"unknown pooling {pooling!r}, expected one of {POOLINGS}"
            )

        self.mlip = mlip
        self.name = mlip
        # MACE #
        self.model_paths = model_paths
        self.mace_model_name = mace_model_name
        self.head = head
        self.default_dtype = default_dtype
        self.enable_cueq = enable_cueq
        self.mace_layers = mace_layers
        self.mace_calc = calculator
        # AIMNET2 #
        self.aimnet_model_name = aimnet_model_name
        self.aimnet_layers = aimnet_layers
        self.aimnet_model: torch.nn.Module | None = None
        # SHARED #
        if mlip == "mace":
            table = MACE_SETTINGS
        if mlip == "aimnet2":
            table = AIMNET2_SETTINGS
        self.device = table["device"] if device is None else device
        self.log_every = table["log_every"] if log_every is None else log_every
        self.aggregation = aggregation
        self.pooling = pooling
        self.temperature = temperature

    @property
    def calc(self) -> MACECalculator:
        """MACE calculator."""
        if self.mace_calc is None:
            model = self.model_paths
            if model is None:
                logger.info(
                    "Loading MACE %r on %s (downloads if needed)",
                    self.mace_model_name,
                    self.device,
                )
                self.mace_calc = mace_mp(
                    model=self.mace_model_name,
                    device=self.device,
                    default_dtype=self.default_dtype,
                    head=self.head,
                    enable_cueq=self.enable_cueq,
                )
            else:
                logger.info("Loading MACE model %r on %s", model, self.device)
                # set MACE calculator
                self.mace_calc = MACECalculator(
                    model_paths=model,
                    device=self.device,
                    default_dtype=self.default_dtype,
                    head=self.head,
                    enable_cueq=self.enable_cueq,
                )
        return self.mace_calc

    @property
    def model(self) -> torch.nn.Module:
        """AIMNet2 calculator."""
        if self.aimnet_model is None:
            logger.info(
                "Loading AIMNet2 %r on %s (downloads if needed)",
                self.aimnet_model_name,
                self.device,
            )
            # Set AIMNET calculator
            calculator = AIMNet2Calculator(
                self.aimnet_model_name, device=self.device
            )
            self.aimnet_model = calculator.model
        return self.aimnet_model

    def per_atom(
        self,
        numbers: np.ndarray,
        coords: np.ndarray,
        charge: int = 0,
    ) -> np.ndarray:
        """Describe every atom of one conformer.

        Parameters
        ----------
        numbers
            Atomic numbers of the molecule, shape (N,).
        coords
            Cartesian coordinates of the conformer, shape (N, 3), in
            Angstrom.
        charge
            Total formal charge. Conditions the AIMNet2 charge
            equilibration; ignored by the MACE invariants.

        Returns
        -------
        features
            Per-atom descriptors, shape (N, C).

        """
        if self.mlip == "mace":
            # The invariant descriptors of every interaction layer, in layer
            # order; `last` keeps the last layer's block alone.
            atoms = Atoms(numbers=numbers, positions=coords)
            descriptors = self.calc.get_descriptors(
                atoms, invariants_only=True, num_layers=-1
            )
            if self.mace_layers == "last":
                n_layers = int(self.calc.models[0].num_interactions)
                width = descriptors.shape[1] // n_layers
                descriptors = descriptors[:, -width:]
            return descriptors

        # Calculate AIMNet2 atom descriptors, AIMNET2 does not have a
        # direct function to get atom descriptors, so we use forward hooks
        # to capture the outputs of the MLPs

        # Select which MLP passes to capture based on the `layers` setting
        mlps = list(self.model.mlps)

        if self.aimnet_layers == "all":
            passes = mlps
        elif self.aimnet_layers == "last":
            passes = mlps[-1:]

        # Prepare the input dict for AIMNet2
        data = {
            "coord": torch.as_tensor(
                coords, dtype=torch.float32, device=self.device
            ).unsqueeze(0),
            "numbers": torch.as_tensor(
                numbers, dtype=torch.long, device=self.device
            ).unsqueeze(0),
            "charge": torch.as_tensor(
                [float(charge)], dtype=torch.float32, device=self.device
            ),
        }

        # Storage for selected MLP output
        captured: dict[int, torch.Tensor] = {}

        # Attach a hook to each MLP
        handles = []
        for index, mlp in enumerate(passes):

            def _save_output(
                module: torch.nn.Module,
                inputs: tuple[torch.Tensor, ...],
                output: torch.Tensor,
                index: int = index,
            ) -> None:
                captured[index] = output.detach()

            handle = mlp.register_forward_hook(_save_output)
            handles.append(handle)
        try:
            # Run calculation
            self.model(data)
        finally:
            # Remove the hook
            for handle in handles:
                handle.remove()

        # Concatenate the representations of the selected MLPs, earlier MLPs
        # first, so that the columns follow the network's depth.
        return (
            torch.cat(
                [captured[index] for index in range(len(passes))], dim=-1
            )
            .squeeze(0)
            .cpu()
            .numpy()
        )

    def __call__(
        self, ensembles: list[ConformerEnsemble]
    ) -> np.ndarray | list[np.ndarray]:
        """Featurise a list of molecules.

        Parameters
        ----------
        ensembles
            One relaxed conformer ensemble per molecule.

        Returns
        -------
        features
            Under a fixed aggregation, a matrix of shape
            (len(ensembles), width), dtype float32. Under a learned one,
            a list of (K, N, C) tensors, ragged in K and N.

        Raises
        ------
        ValueError
            If no ensemble is given, since the width belongs to the model
            and cannot be recovered from an empty result.

        """
        if not ensembles:
            raise ValueError("no ensemble to featurise")

        # Check if attention aggregation is used
        learned = self.aggregation in LEARNED_AGGREGATIONS
        rows = []
        for count, ensemble in enumerate(ensembles, 1):
            # Create descriptor tensor of shape (K, N, C) in the conformer and
            # atom order the ensemble holds.
            tensor = np.stack(
                [
                    self.per_atom(ensemble.numbers, coords, ensemble.charge)
                    for coords in ensemble.coords
                ]
            )

            if learned:
                # If attention aggregation return (K, N, C) tensor
                rows.append(tensor)
            else:
                # If fixed aggregation return a single vector D for the
                # ensemble
                rows.append(
                    self.conformers_pooling(
                        np.stack(
                            [
                                self.fixed_aggregation(atoms, self.aggregation)
                                for atoms in tensor
                            ]
                        ),
                        ensemble,
                        self.pooling,
                        self.temperature,
                    )
                )

            # Progress counter
            if count % self.log_every == 0:
                logger.info(
                    "featurised %d/%d molecules", count, len(ensembles)
                )

        if learned:
            return rows
        return np.asarray(rows, dtype=np.float32)

    @staticmethod
    def conformers_pooling(
        features: np.ndarray,
        ensemble: ConformerEnsemble,
        mode: str = FEATURIZER_SETTINGS["pooling"],
        temperature: float = FEATURIZER_SETTINGS["temperature_k"],
    ) -> np.ndarray:
        """Collapse a per-conformer descriptor matrix to one vector.

        Parameters
        ----------
        features
            Descriptors of each conformer, shape (K, D), in the conformer
            order of the ensemble.
        ensemble
            Ensemble the descriptors were computed from.
        mode
            Collapse applied, one of POOLINGS. `lowest` takes the
            lowest-energy conformer, `boltzmann` a population-weighted
            average, `ensemble_stats` the blocks named in STAT_NAMES.
        temperature
            Temperature in Kelvin, used by `boltzmann` only.

        Returns
        -------
        vector
            Shape (D,) for `lowest` and `boltzmann`, or (4 * D,) for
            `ensemble_stats`.

        Raises
        ------
        ValueError
            If the mode is unknown, the matrix is not two-dimensional, or
            the matrix and the ensemble disagree on the conformer count.

        """
        # Check the mode and the shapes of input
        if mode not in POOLINGS:
            raise ValueError(
                f"unknown pooling {mode!r}, expected one of {POOLINGS}"
            )
        if features.ndim != 2:
            raise ValueError(
                f"features must be (K, D), got shape {features.shape}"
            )
        if len(features) != len(ensemble):
            raise ValueError(
                f"{len(features)} descriptor rows for "
                f"{len(ensemble)} conformers"
            )

        # Collapse the per-conformer matrix into one vector by the mode
        if mode == "lowest":
            return features[0]

        if mode == "boltzmann":
            weights = ensemble.boltzmann_weights(temperature)
            return weights @ features

        return np.concatenate(
            [
                features.min(axis=0),
                features.max(axis=0),
                features.mean(axis=0),
                features.std(axis=0),
            ]
        )

    @staticmethod
    def conformers_pooling_width(n_descriptors: int, mode: str) -> int:
        """Width `conformers_pooling` returns for a given mode.

        Parameters
        ----------
        n_descriptors
            Number of columns in the per-conformer descriptor matrix.
        mode
            Collapse applied, one of POOLINGS.

        Returns
        -------
        width
            Length of the pooled vector.

        Raises
        ------
        ValueError
            If the mode is unknown.

        """
        if mode not in POOLINGS:
            raise ValueError(
                f"Unknown pooling {mode!r}, expected one of {POOLINGS}"
            )
        if mode == "ensemble_stats":
            return n_descriptors * len(STAT_NAMES)
        return n_descriptors

    @staticmethod
    def fixed_aggregation(
        per_atom: np.ndarray,
        mode: str = FEATURIZER_SETTINGS["aggregation"],
    ) -> np.ndarray:
        """Collapse a descriptor matrix to one vector.

        Parameters
        ----------
        per_atom
            Descriptors of each atom, shape (N, C), in the atom order of
            the ensemble.
        mode
            Collapse applied, one of FIXED_AGGREGATIONS. `mean` is the
            average atom, `mean_std_max` the blocks named in
            POOL_STAT_NAMES.

        Returns
        -------
        vector
            Shape (C,) for `mean`, or (3 * C,) for `mean_std_max`.

        Raises
        ------
        ValueError
            If the mode is not one of FIXED_AGGREGATIONS, or the matrix is
            not two-dimensional.

        Notes
        -----
        A learned aggregation is fitted through the surrogate and is not a
        function of this matrix alone, so it is refused here.

        """
        if mode not in FIXED_AGGREGATIONS:
            raise ValueError(
                f"Invalid aggregation {mode!r}, expected one "
                f"of {FIXED_AGGREGATIONS}"
            )
        if per_atom.ndim != 2:
            raise ValueError(
                f"per_atom must be (N, C), got shape {per_atom.shape}"
            )

        mean = per_atom.mean(axis=0, dtype=np.float64)
        if mode == "mean":
            return mean

        return np.concatenate(
            [
                mean,
                per_atom.std(axis=0, dtype=np.float64),
                per_atom.max(axis=0),
            ]
        )

    @staticmethod
    def fixed_aggregation_width(n_channels: int, mode: str) -> int:
        """Width the atom axis leaves behind for a given mode.

        Parameters
        ----------
        n_channels
            Number of columns in the per-atom descriptor matrix.
        mode
            Collapse applied, fixed or learned.

        Returns
        -------
        width
            Length of the aggregated vector.

        Raises
        ------
        ValueError
            If the mode is unknown.

        Notes
        -----
        A learned aggregation is answered with the channel width rather
        than refused, since the fitted head returns a convex combination of
        the atom descriptors. The fitted arm can therefore be sized before
        anything is fitted.

        """
        if mode not in FIXED_AGGREGATIONS + LEARNED_AGGREGATIONS:
            raise ValueError(
                f"Invalid aggregation {mode!r}, expected one of "
                f"{FIXED_AGGREGATIONS + LEARNED_AGGREGATIONS}"
            )

        if mode == "mean_std_max":
            return n_channels * len(POOL_STAT_NAMES)
        return n_channels


class MorganFeaturizer:
    """Morgan (ECFP) fingerprints of a list of molecules.

    Parameters
    ----------
    radius
        Bonds the circular substructures are grown over; 2 gives ECFP4.
    n_bits
        Length of the folded bit vector, and so the feature width.

    Attributes
    ----------
    name
        Short identifier used to label the featuriser in results.

    Notes
    -----
    Topological, so there is no conformer axis and nothing is collapsed.
    Both defaults are read from the `[morgan]` settings table.

    """

    name = "morgan"

    def __init__(
        self,
        radius: int = MORGAN_SETTINGS["radius"],
        n_bits: int = MORGAN_SETTINGS["n_bits"],
    ) -> None:
        self.n_bits = n_bits
        self.generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits
        )

    def __call__(self, smiles: list[str]) -> np.ndarray:
        """Featurise a list of molecules.

        Parameters
        ----------
        smiles
            SMILES strings of the molecules to featurise.

        Returns
        -------
        features
            Fingerprint matrix, shape (len(smiles), n_bits), dtype float32.

        Raises
        ------
        ValueError
            If RDKit cannot parse one of the SMILES strings.

        """
        features = np.zeros((len(smiles), self.n_bits), dtype=np.float32)
        for row, s in enumerate(smiles):
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                raise ValueError(f"RDKit could not parse SMILES: {s!r}")
            features[row] = self.generator.GetFingerprintAsNumPy(mol)
        return features


class T5Featurizer:
    """Encoder embeddings of the SMILES read as text.

    A chemistry T5 encoder is given the SMILES as a sentence and its last
    hidden state is pooled over the tokens.

    Parameters
    ----------
    t5
        Checkpoint variant, one of T5_VARIANTS, naming the `[t5_<variant>]`
        settings table the remaining defaults are read from.
    model_name
        HuggingFace checkpoint the encoder and tokeniser are loaded from.
    device
        Torch device the encoder runs on.
    batch_size
        Number of SMILES encoded per forward pass.
    max_length
        Tokens kept per SMILES; a longer string is truncated.

    Attributes
    ----------
    name
        Short identifier used to label the featuriser in results.

    Raises
    ------
    ValueError
        If the variant is not one of T5_VARIANTS.
    """

    def __init__(
        self,
        t5: str = FEATURIZER_SETTINGS["t5"],
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
    ) -> None:
        if t5 not in T5_VARIANTS:
            raise ValueError(
                f"unknown t5 variant {t5!r}, expected one of {T5_VARIANTS}"
            )
        table = SETTINGS[f"t5_{t5}"]

        self.t5 = t5
        self.model_name = (
            table["model_name"] if model_name is None else model_name
        )
        self.device = table["device"] if device is None else device
        self.batch_size = (
            table["batch_size"] if batch_size is None else batch_size
        )
        self.max_length = (
            table["max_length"] if max_length is None else max_length
        )

        self.t5_tokeniser: PreTrainedTokenizerBase | None = None
        self.t5_encoder: T5EncoderModel | None = None

    @property
    def tokeniser(self) -> PreTrainedTokenizerBase:
        """Tokeniser."""
        if self.t5_tokeniser is None:
            self.t5_tokeniser = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
        return self.t5_tokeniser

    @property
    def encoder(self) -> T5EncoderModel:
        """Encoder."""
        if self.t5_encoder is None:
            logger.info(
                "Loading T5 %r on %s (downloads if needed)",
                self.model_name,
                self.device,
            )

            # set drop_out to 0 to get more deterministic embeddings
            config = T5Config.from_pretrained(self.model_name)
            config.dropout_rate = 0

            encoder = T5EncoderModel.from_pretrained(
                self.model_name, config=config
            )

            self.t5_encoder = encoder.to(self.device)
        return self.t5_encoder

    def __call__(
        self,
        smiles: list[str],
        aggregation: str = FEATURIZER_SETTINGS["aggregation"],
    ) -> np.ndarray | list[np.ndarray]:
        """Featurise a list of molecules.

        Parameters
        ----------
        smiles
            SMILES strings of the molecules to featurise.
        aggregation
            Collapse applied to the token axis, one of FIXED_AGGREGATIONS or
            LEARNED_AGGREGATIONS, with the same meaning as for the atom
            axis of an MLIP.

        Returns
        -------
        features
            Under a fixed aggregation, a matrix of shape
            (len(smiles), width), dtype float32, with one row per input
            molecule. The width is the hidden size of the checkpoint (768
            for the T5 base encoder), tripled by `mean_std_max`. Under a
            learned one, a list of (L, C) arrays, one per molecule and
            ragged in L, holding its real tokens with the padding removed.

        Raises
        ------
        ValueError
            If no SMILES is given, since the width belongs to the
            checkpoint and cannot be recovered from an empty result, or if
            the aggregation is unknown.

        """
        # Sanity check
        if not smiles:
            raise ValueError("no SMILES to featurise")
        if aggregation not in FIXED_AGGREGATIONS + LEARNED_AGGREGATIONS:
            raise ValueError(
                f"unknown aggregation {aggregation!r}, expected one of "
                f"{FIXED_AGGREGATIONS + LEARNED_AGGREGATIONS}"
            )
        # Check if attention aggregation is used
        learned = aggregation in LEARNED_AGGREGATIONS

        rows = []
        for start in range(0, len(smiles), self.batch_size):
            end = start + self.batch_size
            # Tokenize the batch of smiles
            batch = self.tokeniser(
                smiles[start:end],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                hidden = self.encoder(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ).last_hidden_state

            hidden = hidden.float().cpu().numpy()
            real = batch["attention_mask"].bool().cpu().numpy()

            # Drop the T5 padding
            for states, keep in zip(hidden, real, strict=True):
                tokens = states[keep]
                rows.append(
                    tokens
                    if learned
                    else MLIPFeaturizer.fixed_aggregation(tokens, aggregation)
                )

        if learned:
            return rows
        return np.asarray(rows, dtype=np.float32)
