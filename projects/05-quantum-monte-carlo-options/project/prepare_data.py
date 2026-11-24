"""Alias for ``qmc_options.prepare_data`` so the README CLI contract works verbatim."""
from qmc_options.prepare_data import main

if __name__ == "__main__":
    raise SystemExit(main())
