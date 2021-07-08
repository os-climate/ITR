ITR Temperature Scoring
==============================================================

*Do you want to understand what drives the temperature score of your
portfolio to make better engagement and investment decisions?*

|image1|

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Morbi eleifend, augue ac fringilla mattis, velit libero sagittis nibh, iaculis tempus justo ex at erat. In eu nisl pulvinar, pellentesque felis quis, laoreet dui. Maecenas ornare sollicitudin purus, aliquam commodo justo cursus lobortis. Integer at dui sollicitudin, luctus velit in, pellentesque felis. Morbi sed diam nec neque tempor fringilla. Morbi et porttitor ligula. Ut imperdiet mattis nisi eget malesuada. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Proin non ipsum vel felis vehicula venenatis volutpat non massa. Mauris at dolor orci. Suspendisse potenti. In non dapibus.

An introduction to the technical documentation
----------------------------------------------

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Morbi eleifend, augue ac fringilla mattis, velit libero sagittis nibh, iaculis tempus justo ex at erat. In eu nisl pulvinar, pellentesque felis quis, laoreet dui. Maecenas ornare sollicitudin purus, aliquam commodo justo cursus lobortis. Integer at dui sollicitudin, luctus velit in, pellentesque felis. Morbi sed diam nec neque tempor fringilla. Morbi et porttitor ligula. Ut imperdiet mattis nisi eget malesuada. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Proin non ipsum vel felis vehicula venenatis volutpat non massa. Mauris at dolor orci. Suspendisse potenti. In non dapibus.

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
   