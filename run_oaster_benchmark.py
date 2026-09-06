"""Compatibility alias for :mod:`run_oaster_dev_matrix`.

The recovered historical script depended on modules that were not preserved.
Use this name only for old command lines; new work should call the canonical
development runner directly.
"""

from run_oaster_dev_matrix import main


if __name__ == "__main__":
    main()
