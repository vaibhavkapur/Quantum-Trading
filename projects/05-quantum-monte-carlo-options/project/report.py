"""Alias for ``qmc_options.report`` so the README CLI contract works verbatim."""
from qmc_options.report import main

if __name__ == "__main__":
    raise SystemExit(main())
