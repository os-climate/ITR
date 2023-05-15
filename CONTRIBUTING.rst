
Contributing
============

Any contribution to this project is highly appreciated. The most common way to contribute to the project is through coding, however contributions to the documentation are also very welcome.

Submitting a bug report or a feature request
==============================================
To keep track of open issues and feature requests, we will use `Github's issue tracker <https://github.com/OFBDABV/ITR/issues/>`_.

If you encounter any bugs or missing features, please do not hesitate to open a ticket. Before submitting a report, please check that the issue has not already been reported.
For ease of comprehension, please ensure your report includes the following characteristics:

* Reproducible: It should be possible for others to reproduce the issue, ideally through a small code snippet in the description of the issue
* Labelled: Add a label that describes the contents of the ticket, e.g. "bug", "feature request" or "documentation"

Contributing code
====================
Our preferred way for contributing code is to fork the repository, make changes on your personal fork and then create a pull request to merge your changes back into the main repository.
Before a pull request can be approved it needs to be reviewed by two core contributors, then these automated checks need to be passed (see "Coding Guidelines" section below).

.. note:: Please assign the issue you are working on to yourself so we can avoid duplicate efforts.

Getting started
-----------------
Before getting started it is important to have a clean Python environment. The recommended Python version is 3.6 or higher.
An anaconda environment file is provided in the root of the project, as is a PIP requirements.txt file.
The easiest way to work on the Python module is by installing it in development mode. This can be done using the following command::

    pip install -e .[dev]

This will install the module as a reference to your current directory.
The Jupyter notebooks in the "examples" directory have been setup to use auto-reload. Thus, if you now make any changes to your local code, they will be automatically reflected in the notebook.

Coding guidelines
-------------------
In general the code follows three principals, `OOP <https://en.wikipedia.org/wiki/Object-oriented_programming>`_, `PEP8 (code style) <https://www.python.org/dev/peps/pep-0008/>`_ and `PEP 484 (type hinting) <https://www.python.org/dev/peps/pep-0484/>`_.
In addition, we use Flake8 to lint the code, MyPy to check the type hints and Nose2 to do unit testing. These checks are done automatically when attempting to merge a pull request into master.

Releasing
===============
To release a new version of the module you need to take two steps, create a new PyPi release and generate the documentation.


Generate documentation
------------------------
To generate the documentation, you should follow these steps in the "ITR" repository (the one containing the Python module).

1. `cd docs`

2. `make html`

3. Copy the contents of the `docs/_build/` folder into the `gh_pages` branch

4. Start the FastAPI server from the `ITR_api` branch

5. Go to `the docs <http://127.0.0.1/openapi.json>`_ and copy this file to the "swagger" directory in the `gh_pages` branch.

6. Commit and push the gh_pages branch


Code of conduct
===============
Everyone's goals here are aligned: to help asset owners and asset managers reduce their impact on climate change.
The only way we can achieve this goal is by fostering an inclusive and welcome environment. Therefore, this project is governed by a `code of conduct <https://github.com/OFBDABV/ITR/blob/master/CODE_OF_CONDUCT.md>`_. By participating, you are expected to uphold this code. If you encounter any violations, please report them.

.. toctree::
   :maxdepth: 4


********************
Contributing
********************
Any contribution is highly appreciated. The most common way to contribute to the project is through coding, however contributions to
the documentation are also very welcome.

Submitting a bug report or a feature request
==============================================
To keep track of open issues and feature requests, we will use `Github's issue tracker <https://github.com/OFBDABV/ITR/issues/>`_.

If you encounter any bugs or missing features, please do not hesitate to open a ticket. Before submitting a report, please check that the issue has not already been reported.
For ease of comprehension, please ensure your report includes the following characteristics:

* Reproducible: It should be possible for others to reproduce the issue, ideally through a small code snippet in the description of the issue
* Labelled: Add a label that describes the contents of the ticket, e.g. "bug", "feature request" or "documentation"

Contributing code
====================
Our preferred way for contributing code is to fork the repository, make changes on your personal fork and then create a pull request to merge your changes back into the main repository.
Before a pull request can be approved it needs to be reviewed by two core contributors, then these automated checks need to be passed (see "Coding Guidelines" section below).

.. note:: Please assign the issue you are working on to yourself so we can avoid duplicate efforts.

Getting started
-----------------
Before getting started it is important to have a clean Python environment. The recommended Python version is 3.6 or higher.
An anaconda environment file is provided in the root of the project, as is a PIP requirements.txt file.
The easiest way to work on the Python module is by installing it in development mode. This can be done using the following command::

    pip install -e .[dev]

This will install the module as a reference to your current directory.
The Jupyter notebooks in the "examples" directory have been setup to use auto-reload. Thus, if you now make any changes to your local code, they will be automatically reflected in the notebook.

Coding guidelines
-------------------
In general the code follows three principals, `OOP <https://en.wikipedia.org/wiki/Object-oriented_programming>`_, `PEP8 (code style) <https://www.python.org/dev/peps/pep-0008/>`_ and `PEP 484 (type hinting) <https://www.python.org/dev/peps/pep-0484/>`_.
In addition, we use Flake8 to lint the code, MyPy to check the type hints and Nose2 to do unit testing. These checks are done automatically when attempting to merge a pull request into master.

Releasing
===============
To release a new version of the module you need to take two steps, create a new PyPi release and generate the documentation.


Generate documentation
------------------------
To generate the documentation, you should follow these steps in the "ITR" repository (the one containing the Python module).

1. `cd docs`

2. `make html`

3. Copy the contents of the `docs/_build/` folder into the `gh_pages` branch

4. Start the FastAPI server from the `ITR_api` branch

5. Go to `the docs <http://127.0.0.1/openapi.json>`_ and copy this file to the "swagger" directory in the `gh_pages` branch.

6. Commit and push the gh_pages branch


Code of conduct
===============
Everyone's goals here are aligned: to help asset owners and asset managers reduce their impact on climate change.
The only way we can achieve this goal is by fostering an inclusive and welcome environment. Therefore, this project is governed by a `code of conduct <https://github.com/OFBDABV/ITR/blob/master/CODE_OF_CONDUCT.md>`_. By participating, you are expected to uphold this code. If you encounter any violations, please report them.

.. toctree::
   :maxdepth: 4


Contributing
============

Welcome to ``ITR`` contributor's guide.

This document focuses on getting any potential contributor familiarized
with the development processes, but `other kinds of contributions`_ are also
appreciated.

If you are new to using git_ or have never collaborated in a project previously,
please have a look at `contribution-guide.org`_. Other resources are also
listed in the excellent `guide created by FreeCodeCamp`_ [#contrib1]_.

Please notice, all users and contributors are expected to be **open,
considerate, reasonable, and respectful**. When in doubt, `Python Software
Foundation's Code of Conduct`_ is a good reference in terms of behavior
guidelines.


Issue Reports
=============

If you experience bugs or general issues with ``ITR``, please have a look
on the `issue tracker`_. If you don't see anything useful there, please feel
free to fire an issue report.

.. tip::
   Please don't forget to include the closed issues in your search.
   Sometimes a solution was already reported, and the problem is considered
   **solved**.

New issue reports should include information about your programming environment
(e.g., operating system, Python version) and steps to reproduce the problem.
Please try also to simplify the reproduction steps to a very minimal example
that still illustrates the problem you are facing. By removing other factors,
you help us to identify the root cause of the issue.


Documentation Improvements
==========================

You can help improve ``ITR`` docs by making them more readable and coherent, or
by adding missing information and correcting mistakes.

``ITR`` documentation uses Sphinx_ as its main documentation compiler.
This means that the docs are kept in the same repository as the project code, and
that any documentation update is done in the same way was a code contribution.

.. todo:: Don't forget to mention which markup language you are using.

    e.g.,  reStructuredText_ or CommonMark_ with MyST_ extensions.

.. todo:: If your project is hosted on GitHub, you can also mention the following tip:

   .. tip::
      Please notice that the `GitHub web interface`_ provides a quick way of
      propose changes in ``ITR``'s files. While this mechanism can
      be tricky for normal code contributions, it works perfectly fine for
      contributing to the docs, and can be quite handy.

      If you are interested in trying this method out, please navigate to
      the ``docs`` folder in the source repository_, find which file you
      would like to propose changes and click in the little pencil icon at the
      top, to open `GitHub's code editor`_. Once you finish editing the file,
      please write a message in the form at the bottom of the page describing
      which changes have you made and what are the motivations behind them and
      submit your proposal.

When working on documentation changes in your local machine, you can
compile them using |tox|_::

    tox -e docs

and use Python's built-in web server for a preview in your web browser
(``http://localhost:8000``)::

    python3 -m http.server --directory 'docs/_build/html'


Code Contributions
==================

.. todo:: Please include a reference or explanation about the internals of the project.

   An architecture description, design principles or at least a summary of the
   main concepts will make it easy for potential contributors to get started
   quickly.

Submit an issue
---------------

Before you work on any non-trivial code contribution it's best to first create
a report in the `issue tracker`_ to start a discussion on the subject.
This often provides additional considerations and avoids unnecessary work.

Create an environment
---------------------

Before you start coding, we recommend creating an isolated `virtual
environment`_ to avoid any problems with your installed Python packages.
This can easily be done via either |virtualenv|_::

    virtualenv <PATH TO VENV>
    source <PATH TO VENV>/bin/activate

or Miniconda_::

    conda create -n ITR python=3 six virtualenv pytest pytest-cov
    conda activate ITR

Clone the repository
--------------------

#. Create an user account on |the repository service| if you do not already have one.
#. Fork the project repository_: click on the *Fork* button near the top of the
   page. This creates a copy of the code under your account on |the repository service|.
#. Clone this copy to your local disk::

    git clone git@github.com:YourLogin/ITR.git
    cd ITR

#. You should run::

    pip install -U pip setuptools -e .

   to be able to import the package under development in the Python REPL.

   .. todo:: if you are not using pre-commit, please remove the following item:

#. Install |pre-commit|_::

    pip install pre-commit
    pre-commit install

   ``ITR`` comes with a lot of hooks configured to automatically help the
   developer to check the code being written.

Implement your changes
----------------------

#. Create a branch to hold your changes::

    git checkout -b my-feature

   and start making changes. Never work on the main branch!

#. Start your work on this branch. Don't forget to add docstrings_ to new
   functions, modules and classes, especially if they are part of public APIs.

#. Add yourself to the list of contributors in ``AUTHORS.rst``.

#. When you’re done editing, do::

    git add <MODIFIED FILES>
    git commit

   to record your changes in git_.

   .. todo:: if you are not using pre-commit, please remove the following item:

   Please make sure to see the validation messages from |pre-commit|_ and fix
   any eventual issues.
   This should automatically use flake8_/black_ to check/fix the code style
   in a way that is compatible with the project.

   .. important:: Don't forget to add unit tests and documentation in case your
      contribution adds an additional feature and is not just a bugfix.

      Moreover, writing a `descriptive commit message`_ is highly recommended.
      In case of doubt, you can check the commit history with::

         git log --graph --decorate --pretty=oneline --abbrev-commit --all

      to look for recurring communication patterns.

#. Please check that your changes don't break any unit tests with::

    tox

   (after having installed |tox|_ with ``pip install tox`` or ``pipx``).

   You can also use |tox|_ to run several other pre-configured tasks in the
   repository. Try ``tox -av`` to see a list of the available checks.

Submit your contribution
------------------------

#. If everything works fine, push your local branch to |the repository service| with::

    git push -u origin my-feature

#. Go to the web page of your fork and click |contribute button|
   to send your changes for review.

   .. todo:: if you are using GitHub, you can uncomment the following paragraph

      Find more detailed information in `creating a PR`_. You might also want to open
      the PR as a draft first and mark it as ready for review after the feedbacks
      from the continuous integration (CI) system or any required fixes.


Troubleshooting
---------------

The following tips can be used when facing problems to build or test the
package:

#. Make sure to fetch all the tags from the upstream repository_.
   The command ``git describe --abbrev=0 --tags`` should return the version you
   are expecting. If you are trying to run CI scripts in a fork repository,
   make sure to push all the tags.
   You can also try to remove all the egg files or the complete egg folder, i.e.,
   ``.eggs``, as well as the ``*.egg-info`` folders in the ``src`` folder or
   potentially in the root of your project.

#. Sometimes |tox|_ misses out when new dependencies are added, especially to
   ``setup.cfg`` and ``docs/requirements.txt``. If you find any problems with
   missing dependencies when running a command with |tox|_, try to recreate the
   ``tox`` environment using the ``-r`` flag. For example, instead of::

    tox -e docs

   Try running::

    tox -r -e docs

#. Make sure to have a reliable |tox|_ installation that uses the correct
   Python version (e.g., 3.7+). When in doubt you can run::

    tox --version
    # OR
    which tox

   If you have trouble and are seeing weird errors upon running |tox|_, you can
   also try to create a dedicated `virtual environment`_ with a |tox|_ binary
   freshly installed. For example::

    virtualenv .venv
    source .venv/bin/activate
    .venv/bin/pip install tox
    .venv/bin/tox -e all

#. `Pytest can drop you`_ in an interactive session in the case an error occurs.
   In order to do that you need to pass a ``--pdb`` option (for example by
   running ``tox -- -k <NAME OF THE FALLING TEST> --pdb``).
   You can also setup breakpoints manually instead of using the ``--pdb`` option.


Maintainer tasks
================

Releases
--------

.. todo:: This section assumes you are using PyPI to publicly release your package.

   If instead you are using a different/private package index, please update
   the instructions accordingly.

If you are part of the group of maintainers and have correct user permissions
on PyPI_, the following steps can be used to release a new version for
``ITR``:

#. Make sure all unit tests are successful.
#. Tag the current commit on the main branch with a release tag, e.g., ``v1.2.3``.
#. Push the new tag to the upstream repository_, e.g., ``git push upstream v1.2.3``
#. Clean up the ``dist`` and ``build`` folders with ``tox -e clean``
   (or ``rm -rf dist build``)
   to avoid confusion with old builds and Sphinx docs.
#. Run ``tox -e build`` and check that the files in ``dist`` have
   the correct version (no ``.dirty`` or git_ hash) according to the git_ tag.
   Also check the sizes of the distributions, if they are too big (e.g., >
   500KB), unwanted clutter may have been accidentally included.
#. Run ``tox -e publish -- --repository pypi`` and check that everything was
   uploaded to PyPI_ correctly.



.. [#contrib1] Even though, these resources focus on open source projects and
   communities, the general ideas behind collaborating with other developers
   to collectively create software are general and can be applied to all sorts
   of environments, including private companies and proprietary code bases.


.. <-- start -->
.. todo:: Please review and change the following definitions:

.. |the repository service| replace:: GitHub
.. |contribute button| replace:: "Create pull request"

.. _repository: https://github.com/<USERNAME>/ITR
.. _issue tracker: https://github.com/<USERNAME>/ITR/issues
.. <-- end -->


.. |virtualenv| replace:: ``virtualenv``
.. |pre-commit| replace:: ``pre-commit``
.. |tox| replace:: ``tox``


.. _black: https://pypi.org/project/black/
.. _CommonMark: https://commonmark.org/
.. _contribution-guide.org: https://www.contribution-guide.org/
.. _creating a PR: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request
.. _descriptive commit message: https://chris.beams.io/posts/git-commit
.. _docstrings: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
.. _first-contributions tutorial: https://github.com/firstcontributions/first-contributions
.. _flake8: https://flake8.pycqa.org/en/stable/
.. _git: https://git-scm.com
.. _GitHub's fork and pull request workflow: https://guides.github.com/activities/forking/
.. _guide created by FreeCodeCamp: https://github.com/FreeCodeCamp/how-to-contribute-to-open-source
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _MyST: https://myst-parser.readthedocs.io/en/latest/syntax/syntax.html
.. _other kinds of contributions: https://opensource.guide/how-to-contribute
.. _pre-commit: https://pre-commit.com/
.. _PyPI: https://pypi.org/
.. _PyScaffold's contributor's guide: https://pyscaffold.org/en/stable/contributing.html
.. _Pytest can drop you: https://docs.pytest.org/en/stable/how-to/failures.html#using-python-library-pdb-with-pytest
.. _Python Software Foundation's Code of Conduct: https://www.python.org/psf/conduct/
.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/
.. _Sphinx: https://www.sphinx-doc.org/en/master/
.. _tox: https://tox.wiki/en/stable/
.. _virtual environment: https://realpython.com/python-virtual-environments-a-primer/
.. _virtualenv: https://virtualenv.pypa.io/en/stable/

.. _GitHub web interface: https://docs.github.com/en/repositories/working-with-files/managing-files/editing-files
.. _GitHub's code editor: https://docs.github.com/en/repositories/working-with-files/managing-files/editing-files
