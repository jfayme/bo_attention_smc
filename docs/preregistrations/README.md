# Pre-registrations

The study's pre-registered tests of the per-molecule term and of SMC, moved
here unchanged on 2026-10-01, when `bo_per_molecule/` and `bo_smc/` were
consolidated into `bo_attention_smc/`. Each was written, and its SHA-256
logged in the run folder's `preregistration.sha256`, before any of its
campaigns ran; its result was appended after they all had. The hash logs
name the old paths. The commands inside the files name the old packages,
whose code is in git history (the commit before the consolidation).
Every commit named here is one of
[jfayme/lengthscale_bo](https://github.com/jfayme/lengthscale_bo), where
these files were written; this repository holds them from 2026-10-02, byte
for byte (`.gitattributes` keeps git from changing their line endings).

| file | old path | logged hash | that text |
|---|---|---|---|
| `bo_per_molecule/PREREGISTRATION.md` | `bo_per_molecule/PREREGISTRATION.md` | `c2ec7477…` (runs/bo_per_molecule_confirm) | this file, as it is |
| `bo_per_molecule/PREREGISTRATION_CONFIG.md` | `bo_per_molecule/PREREGISTRATION_CONFIG.md` | `bdf8d1ef…` (runs/bo_per_molecule_config) | commit `ebb4264`; the result followed in `e462013` |
| `bo_smc/PREREGISTRATION.md` | `bo_smc/PREREGISTRATION.md` | `ee60b272…`, amendment 1 `2ca68b3b…` (runs/bo_smc_confirm) | commits `9bfd375`, `46d5220`; the result followed in `da97510` |
| `bo_smc/PREREGISTRATION_CHEN.md` | `bo_smc/PREREGISTRATION_CHEN.md` | `aa9148fd…` (runs/bo_smc_chen) | commit `fd5cfc0`; the result followed in `f502638` |
| `bo_smc/PREREGISTRATION_TERM.md` | `bo_smc/PREREGISTRATION_TERM.md` | `b39674de…` (runs/bo_smc_term) | never committed before its result; its first 57 lines hash to the logged value (`head -n 57 … \| sha256sum`) |

Each hash above was checked on 2026-10-01 against the commit or lines named.
The pre-registration of C3 and of the earlier bo_fixes tests stays in
`bo_fixes/PREREGISTRATION.md`.
