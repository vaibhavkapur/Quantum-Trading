"""Alias for ``qmc_options.run`` so the README CLI contract works verbatim."""
from qmc_options.run import main

if __name__ == "__main__":
    raise SystemExit(main())
