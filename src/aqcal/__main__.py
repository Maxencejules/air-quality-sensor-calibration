"""Allow ``python -m aqcal ...``."""

from aqcal.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
