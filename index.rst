ITR Temperature Scoring
==============================================================

*Do you want to understand what drives the temperature score of your
portfolio to make better engagement and investment decisions?*

|image1|

+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Quickstart**                                                                                                                                                                                                                                    |
|                                                                                                                                                                                                                                                   |
| If you prefer to get up and running quickly, we’ve got a no-code and a Python option:                                                                                                                                                             |
|                                                                                                                                                                                                                                                   |
| -  **No-code**: Run the project locally as a \ `web application using Docker <https://os-c.github.io/ITR/rest_api.html#locally>`__                                                                                                            |
|                                                                                                                                                                                                                                                   |
| -  **Python**: Run a Jupyter notebook, without any installation in \ `Google Colab <https://os-c.github.io/ITR/getting_started.html#google-colab>`__ or `locally <https://os-c.github.io/ITR/getting_started.html#jupyter-notebooks>`__.  |
|                                                                                                                                                                                                                                                   |
| .. rubric::                                                                                                                                                                                                                                       |
|    :name: section                                                                                                                                                                                                                                 |
+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

If you are unsure whether the tool will be useful for your application
and workflow, or you would first like to run some examples to get a
better idea of how the tool works and what types of outputs it
generates, the @TODO: add link
offers a quick and no-code opportunity for such testing. The notebook
combines text and code to provide a testing environment for your
research, to give you an understanding for how the tool can help you
analyze companies’ and portfolios’ temperature scores, to aid your
engagement and investment decisions.

The notebook is loaded with example data, but you can also use your own
data. For your first test, you can simply run the code cells one by one
in the current sequence, to get an understanding of how it works. If you
are not familiar with Notebooks, please refer to `this
introduction <https://colab.research.google.com/notebooks/basic_features_overview.ipynb>`__.

The following diagram provides an overview of the different parts of the
full toolkit and their dependencies: 

|image2|

As shown above, the Python code forms the core codebase of the
ITR tool. It is recommended to use the Python package if the
user would like to integrate the tool in their own codebase. In turn,
the second option is running the tool via the API if the user’s
preference is to include the tool as a Microservice in their existing IT
infrastructure in the cloud or on premise. The development project also
included the creation of a simple user interface (UI), which can be used
for easier user interaction in combination with the API.

The ITR tool enables two main ways of installing and/or running the
tool:  

1. Users can integrate the **Python package** in their codebase. For
   more detailed and up-to-date information on how to run the tool via
   the Python package, please consult the ‘Getting Started Using Python’
   section.

2. The tool can be included as a Microservice (**containerized REST
   API**) in any IT infrastructure (in the cloud or on premise). For
   more detailed and up-to-date information on how to run the tool via
   the API, please consult the ‘Getting Started Using REST API’ section.
   Optionally, the API can be run with a frontend (UI). This simple user
   interface makes testing by non-technical users easier. For more
   detailed and up-to-date information on how to use the UI as a
   frontend to the API, please consult the ‘Getting Started Using REST
   API’ section.


Given the open source nature of the tool, the community is encouraged to
make contributions (refer to `Contributing <https://os-c.github.io/ITR/contributing.html>`__ section to further develop
and/or update the codebase. Contributions can range from submitting a
bug report, to submitting a new feature request, all the way to further
enhancing the tool’s functionalities by contributing code.


.. |image1| image:: image1.png
   :width: 6.47222in
   :height: 4.97682in
.. |image2| image:: image2.png
   :width: 3.71642in
   :height: 3.58652in

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   intro
   getting_started
   rest_api
   FunctionalOverview
   DataRequirements
   Legends
   contributing
   links
   terms
   

Contents
========

.. toctree::
   :maxdepth: 2

   Overview <readme>
   Contributions & Help <contributing>
   License <license>
   Authors <authors>
   Changelog <changelog>
   Module Reference <api/modules>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _toctree: https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html
.. _reStructuredText: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _references: https://www.sphinx-doc.org/en/stable/markup/inline.html
.. _Python domain syntax: https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#the-python-domain
.. _Sphinx: https://www.sphinx-doc.org/
.. _Python: https://docs.python.org/
.. _Numpy: https://numpy.org/doc/stable
.. _SciPy: https://docs.scipy.org/doc/scipy/reference/
.. _matplotlib: https://matplotlib.org/contents.html#
.. _Pandas: https://pandas.pydata.org/pandas-docs/stable
.. _Scikit-Learn: https://scikit-learn.org/stable
.. _autodoc: https://www.sphinx-doc.org/en/master/ext/autodoc.html
.. _Google style: https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
.. _NumPy style: https://numpydoc.readthedocs.io/en/latest/format.html
.. _classical style: https://www.sphinx-doc.org/en/master/domains.html#info-field-lists
