=========
Changelog
=========

Version 1.1.2 (unreleased)
=============

- Env vars ITR_SCHEMA and/or ITR_PREFIX can override code defaults (demo_dv, itr_) to prevent collisions between users and CI/CD
- Implement support for CH4 emissions disclosures (i.e., fugitive S1 emissions)
- Use concurrent Data Vault operations to optimize queries when possible
- Support automobile and other production equivalences (e.g., veh=150k kpm), with new sample data file `20231031 ITR V2 Sample Data.xlsx`
- Add `ITR/test/inputs/20230106 ITR V2 Sample Data.xlsx` for automobiles
- Sample Data fixes
- Various bug fixes and dependency version updates

Version 1.1.1
=============

- Update __hash__ functions used by uncertainties
- Quiet vault warnings due to Pint #1897
- Optimize validation of Data Vault initialization for ITR
- Update `ITR_DV.py` to demonstrate connections to Data Mesh for ITR
- Created unit tests `test_vault_providers` for Data Vault
- Update Data Vault to Pydantic 2.x, other related
- Rewrite list-map-zip for mypy
- Migrate flake8 to use pyproject.toml configurationx
- Sample Data fixes
- Various bug fixes and dependency version updates

Version 1.1.0
=============

- Refactor code to resolve mypy errors
- Bump sphinx-autoapi from 2.0.1 to 3.0.0
- Various bug fixes and dependency version updates

Version 1.0.12
==============

- Prune examples and ITR-examples
- Create dependabot.yml
- reorder imports to ensure primacy of our Quantity (feeding pint.Quantity, not the other way around)
- Add tomllint.sh script
- Update git submodule for ITR-examples
- Update docs to reflect ITR/ITR-examples packaging
- Workaround for Pandas #55824
- Various bug fixes and dependency version updates

Version 1.0.11
==============

- Yanked

Version 1.0.10
==============

- Automate backfilling of data to BASE_YEAR so that more companies can be analyzed
- Various bug fixes and dependency version updates


Version 1.0.9
=============

- Update `pyproject.toml` and `pdm.lock` files.  Otherwise this is more of a test of pushing releases than any new released functionality.

Version 1.0.8
=============

- Remove notebook workflow; now in ITR-examples
- Move benchmark data to a subdirectory of src/ITR/data
- Update osc-ingest-tools so that we can use SqlAlchemy 2.0.
- Add flake8 and mypy linting tools
- Migrate `examples` to new repository `ITR-examples`
- Minor documentation updates
- Various bug fixes and dependency version updates

Version 1.0.7
=============

- Update license strings in pyproject.toml
- Various bug fixes and dependency version updates

Version 1.0.6
=============

- Swap build system to tox/pdm/pyproject.toml
- Address Pandas 2.1.0 deprecations
- Implement eslint TOML file linting
- Add code linting
- Swap build system to tox/pdm/pyproject.toml
- Migrate from Pydantic V1 -> V2
- Various bug fixes and dependency version updates


Version 1.0.5
=============

- Re-enable Python 3.11
- Various bug fixes and dependency version updates

Version 1.0.4
=============

- Speed up template-based test cases (reduce time to run tests by 5 minutes)
- Implemented PyScaffold to simplify release process
- Various bug fixes and dependency version updates
- Load new Oil&Gas benchmark synthesized from itr-data-pipline

Version 1.0.3
=============

- Clean up docstring documentation
- Rewrite internals to use PintArrays much more effectively (transposed EI tables etc).
- Better align units handling between benchmarks, disclosures, and targets.  For example, benchmark defines `t CO2e/GJ` intensity, disclosure defines `bcm CH4` gas distributed and target defines absolute `t CO2e` target.  Intentional unit conversion leads to greater consistency and fewer failed conversions than waiting to see what Pint will do.
- Support for `target_probabilities`
- Support new synthetic OECM `Oil&Gas` sector (combining `Oil` and `Gas` budgets and production values)
- Implement SBTi budget scaling methodology (`cumulative_scaled_budget`)
- Switch unit testing framework from `unittest` to `pytest`
- Plot uncertainties in ITR_UI.py
- Initial prototype of Data Vault (aka Data Mesh) functionality
- Support TPI and OECM benchmarks in unified way
- Support intensity metrics (and infer emissions from those)
- Implement benchmark-aligned inferencing of S3 data (with uncertainties if available)
- Calculate and display Activity-level budgets based on sector/region/scope selections
- Infer S2 metrics to better harmonize comparability of S1 and S1S2 metrics
- Rewrite ITR_UI.py to avoid excess dependencies on global variables
- Remove support for Python 3.8
- Re-establish PyPi publication
- Automate release publication via git tags and github actions
- Sample Data fixes
- Various bug fixes and dependency version updates
