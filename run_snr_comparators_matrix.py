"""Compatibility alias for :mod:`run_strict_comparators`.

The recovered historical script depended on missing modules. This alias runs
the seven maintained Python comparators; run ``run_strict_oaster.py``
separately for OASTER.
"""

from run_strict_comparators import main


if __name__ == "__main__":
    main()
