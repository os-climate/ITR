Functional Overview
================================================================

Temperature Score
-----------------

Portfolio temperature scores are calculated through an aggregation of
all companies’ temperature scores within the portfolio. For each
company, the score is calculated based on the publicly announced
targets, which are mapped to regression models based on IPCC climate
scenarios. The company is then awarded a score for each period and scope
based on the ambition and coverage of the relevant targets. For more
information, refer to the methodology `here <https://sciencebasedtargets.org/resources/files/SBTi-TCFD-reporting-guidance.pdf>`__\ .

Time Frames
~~~~~~~~~~~

*This documentation and/or functionality needs to be tested and updated.*

The original intention of the ``TIME_FRAME`` parameter presumed that
targets were expressed without quantifying actual year dates.

The original documentation stated:
> By default, the ITR temperature scoring tool reports temperature scores on the mid-term time frames (i.e. based on emissions reduction targets aimed at 5-15 years into the future). However, it is also possible to inspect short (less than 5 years) and long-term time frames (15 to 30 years).

What has been implemented is the following:

The tool collects all the interim targets that each company provides
(which almost always include a 2030 date or a "50% reduction by X"
date and/or a 2050 date or a "100% reduction by Y" date). A minority
of companies also provide a short-term term (2-3 years out). The tool
uses all the stated targets for shaping the totality of target
projections from the year after the last reported emissions data
through 2050. Interpolation from reported data through successive
targets is based on a hybrid CAGR/Linear model (explained below).

A company that supplies short, medium, and long-term targets would
have a target reduction curve that interpolates between the three
points. A company that provides only a single long-term target would
have a target reduction curve that intercepts that single target.

Then, instead of selecting SHORT, MEDIUM or LONG, we allow the user to
set the end-date of the analysis (to one of 2025, 2030, 2035, 2040,
2045, or 2050). We then score the temperature based on the
overshoot/undershoot of the benchmark's temperature as of that
date. Thus, if the benchmark is presumed 1.2˚C in the current era and
1.5˚C in 2050 and the end-date selected is 2030, the temperature will
be scored against 1.35˚C (if that's what the carbon budget/ITR of the
benchmark is at 2030).

The target interpolation method is based on a hybrid CAGR/Linear
model.  Because no CAGR model can ever reach zero (Xeno's paradox), a
pure CAGR-based model is inappropriate for describing an emissions
reduction curve that reaches zero.  To solve this problem, when the
target reduction is > 90% we create a weighed average of a CAGR-based
reduction and a linear interpolation.  Thus, if the present analysis
begins 2025 with a 50% reduction target for 2030, calculate the CAGR
for a 50% reduction across the 6 years (2025, 2026, 2027, 2028, 2029,
2030).  If that company has a 100% reduction target for 2050, we
create a CAGR curve targeting a 90% reduction over that 20 year period
(2031-2050) and a linear curve targeting a 100% reduction from
2031-2050, and then compute a date-weighted average of the two, with
all the reduction coming from the CAGR curve in 2031 and all the
reduction coming from the linear curve in 2050.

Scopes
~~~~~~

ITR temperature scoring tool reports on a selected scope or set of
scopes (S1, S2, S3, S1+S2, S1+S2+S3).  If companies disclose data for
the selected scope, the copmany's ITR score (and other analyses) will
be displayed by the tool.  If the company does not set a target for
that particular scope, then the tool will use only trajectory
information in its calculations.

Scope reporting by companies is not perfectly consistent, and so the
following should be kept in mind.  When there is only a single number
reported for scope data, that is the number the tool presumes to be
correct.  When companies report scope data based on a particular
submetric, the tool uses a dictionary to prioritize and interpret the reported value.

The first dictionary sorts reporting based on sectors:

.. code-block:: python

    {
	"cement": "Cement",
	"clinker": "Cement",
	"chemicals": "Chemicals",
	"electricity": "Electricity Utilities",
	"generation": "Electricity Utilities",
	"gas": "Gas Utilities",
	"distribution": "Gas Utilities",
	"coal": "Coal",
	"lng": "Gas",
	"ng": "Gas",
	"oil": "Oil",
    }

While the GHG protocol has always defined 15 categories for Scope 3
emissions, the tool treats Scope 3 datapoints as monolithic.  The OECM
benchmark defines S3 emissions based on Activities (Primary Energy,
Secondary Energy, and End Use), which solves the problem of Scope 3
double-counting.  The TPI benchmark defines on a sector-by-sector what
are the "material" scope 3 values, which has a similar effect.  An Oil
& Gas company that reports all 15 categories (including business
travel and employee commuting) will only be bechmarked against Scope 3
category 11 (use of sold products).  Same with an Automotive company
(business travel is counted against Aviation, and employee commuting
is already counted in the use of sold products of the very vehciles
that employees use for commuting).  The sector-based dictionary
controls how Scope 3 data is selected when reported with submetrics.
When Scope 3 is reported as a monolithic number (violating the GHG
reporting standards), it is taken to be the correct, material Scope 3
(even if in fact it is the sum total of all Scope 3 categories).

For ``Electricity Utilities``, Scope 3 category 3 data is interpreted as
emissions from purchased power **and is converted to Scope 1 to align
with benchmarks**.  For Oil, Gas, and Gas Utilities, Scope 3 category
11 data is interpreted as _the_ material Scope 3 data to align with
benchmarks.

Scope 1 emissions submetrics stand in for boundary definitions, and can be:

.. code-block:: python

    {
	"operated",
	"own",
	"revenue",
	"equity",
	"",  # An (unspecified) empty boundary definition
	"total",
	"gross",
	"net",
	"full",
	"*unrecognized*",
    }

If an unknown boundary is defined, the tool will silently treat it as ``*uncategorized*`` (but it will choose it).

Scope 2 emissions can be reported as:

.. code-block:: python

    {
	"location",
	"location-based",
	"market",
	"market-based",
    }

The tool will choose the S2 value reported according to that priority
order, which means that when data contains both, location-based data
will be preferred to market-based data.  But if some companies report
only one or the other, the reported data will be treated as S2 data
(with no lineage to whether it is location- or market-based data).

The *All Scopes* option causes the tool to display any temperature,
trajectory, target, or other calculations it can for any scope.  If a
company discloses S1, S2, S1+S2, and S3 data, the tool will display
calculations for each of those four scopes (it will not syntehsize
S1+S2+S3).  This option creates confusing graphical information, but
it can be helpful when outputing a spreadsheet that allows users to
subsequently filter on scopes as part of a subsequent analytic step
outside of the tool.

Scope Estimation
~~~~~~~~~~~~~~~~

Some companies report combined S1+S2 emissions without reporting
individual S1 and S2 data.  Some companies report no S3 data but set
overall netzero targets for combined S1+S2+S3 scopes.  When data is
missing, companies cannot be scored.  The tool allows scope emissions
to be estimated (or not) in various ways.

When there are multiple datapoints for a single metric (such as S1
emissions for the year 2020), the tool will average the reported
values and, if the ``uncertainties`` package is installed, associate an
uncertainty value with the average.  A programmer could change this to
preferring only the most recent of the several datapoints.

The tool implements a function
``DataWarehouse.estimate_missing_s3_data`` to provide an estimate of S3
data based on benchmark-aligned values.  By default, the tool
constructs a ``DataWarehouse`` with this estimation enabled, but the
tool can be modified to disable the estimation by passing the value
``None`` as the ``estimate_missing_data`` parameter to the ``DataWarehouse``
constructor.  The estimation heuristics are as follows:

* If the company reports S1+S2+S3 data (or targets) but no granular scoppe data, the tool assumes that the sum total is proportional to the sum of the scope data for the benchmark and prorates the S1+S2+S3 data across S1+S2 and S3 scopes.
* If the company reports only S1+S2 emissions, it will penalize the company by inferring that S3 emissions are 2x the benchmark-aligned S3 emissions, plus or minus the S3 emissions, creating an uncertainty range from 1x benchmark-aligned S3 emissions to 3x benchmark-aligned emissions.  When looking at the graphical temperature scores, the lower boundary are the benchmark-aligned S3 emissions and the upper boundary are 3x benchmark-aligned emissions.  This allows the company to be compared with companies that properly report, and allows the data analyst to see the uncertainty so as to give the non-reporting companies either the benefit of the doubt or a penalty for not reporting.

``Utilities`` are an example of a multi-sectoral company (they include both ``Electricity Utilities`` and ``Gas Utilities`` sub-sectors, which have different benchmarks).  Ideally, Scope emissions for a Utility are given sub-sector metrics (such as ``generation`` and ``gas``) so they can be properly allocated to each respective sub-sector.  But sometimes emissions reports are monolithic.  To make these sub-sector companies more comparable in their own sectors, the tool will prorate the emissions according to benchmark data.  There are four cases:

- Case 1: emissions need to be prorated across sectors using benchmark alignment method
- Case 2: emissions are good as is; no benchmark alignment needed
- Case 3: there is an ambiguous overlap of emissions (i.e., Scope 3 general to Utilities (it's really just gas) and Scope 3 gas specific to Gas Utilities
- Case 4: case_1 scopes containing case_2 scopes that need to be removed before remaining scopes can be allocated

All this complexity can be avoided by using data that correlates precise Scope emissions to single-sector lines of business of a company.

Many Gas Utilities report Scope 1 and Scope 2 data as well as gas delivery, but they do not report Scope 3 category 11 data(!).  The tool will estimate missing Scope 3 category 11 data based on delivered $CH_4$ using AR5GWP100 conversion statistics.

Aggregation methods
~~~~~~~~~~~~~~~~~~~

The portfolio temperature score can be calculated using different
aggregation methods based on emission and financial data of the
individual companies. The available options are:

- Weighted average temperature score (WATS)

- Total emissions weighted temperature score (TETS)

- Market Owned emissions weighted temperature score (MOTS)

- Enterprise Owned emissions weighted temperature score (EOTS).

- EV + Cash emissions weighted temperature score (ECOTS)

- Total Assets emissions weighted temperature score (AOTS)

- Revenue emissions weighted temperature score (ROTS)

It is also possible to calculate scores of the individual companies
without aggregating to a portfolio score.

Grouping data
~~~~~~~~~~~~~

This functionality enables the user to analyze (for examples see Jupyter
notebook
`analysis_example <https://github.com/OFBDABV/ITR/blob/master/examples/1_analysis_example.ipynb>`__\ )
the temperature score of the portfolio in depth by slicing and dicing
through the portfolio. By choosing to “group by” a certain field (for
example region or sector), the user receives output of temperature
scores per category in the chosen field (so per region or sector). It is
possible to group over region, country, sector, and industry level 1-4.
Furthermore, it is also possible to add your own fields to group the
score over (e.g. investments strategies, market cap buckets) via the
portfolio data.

Choose fields to show
~~~~~~~~~~~~~~~~~~~~~

By default, the ITR temperature scoring tool reports Company name,
Company ID, Scope, Time frame and Temperature score for each individual
combination. However, using this option allows the user to add
additional columns to the output. It is possible to add all fields
imported via either the portfolio data or the company data (fundamental
and target).

What-If Analyses
~~~~~~~~~~~~~~~~

*This documentation and/or functionality needs to be tested and updated.*

To analyze the effect of engagement on your portfolio temperature score
it is possible to run “what-if” analyses. In these scenarios, the
temperature score is recalculated with the presumption that based on
various engagements some or all companies decided to set different (more
ambitious) targets.

The possible scenarios are:

-  Scenario 1: In this scenario, all companies in the portfolio that did
   not yet set a valid target have been persuaded to set 2.0\ :sup:`o`
   Celsius (C) targets. This is simulated by changing all scores that
   used the default score to a score of 2.0\ :sup:`o` C.

-  Scenario 2: In this scenario, all companies that already set targets
   are persuaded to set “Well Below 2.0\:sup:`o` C (WB2C) targets. This
   is simulated by setting all scores of the companies that have valid
   targets to at most 1.75\ :sup:`o` C.

-  Scenario 3: In these scenarios, the top 10 contributors to the
   portfolio temperature score are persuaded to set 2.0\ :sup:`o` C
   targets.

   -  Scenario 3a: All top 10 contributors set 2.0\ :sup:`o` C targets.

   -  Scenario 3b: All top 10 contributors set WB2C, i.e. 1.75\ :sup:`o` C targets.

-  Scenario 4: In this scenario, the user can specify which companies it
   wants to engage with to influence to set 2.0\ :sup:`o` C or WB2C
   targets. The user selects companies to engage with in the portfolio
   input file by settings the *engagement_target* field to TRUE for
   these companies.

   -  Scenario 4a: All companies that are marked as engagement targets
      set 2.0\ :sup:`o` C targets

   -  Scenario 4b: All companies that are marked as engagement targets
      set WB2C targets.


Output options
--------------

The temperature score can be requested for all time frames and scope
combinations on the following levels.

-  Portfolio temperature score: the aggregated score over all companies
   in the portfolio

-  Company temperature score: the temperature score of an individual
   company

-  Grouped temperature score: using the “group by” option, the user can
   get the aggregated temperature score per category in a chosen field
   (for example per region or per sector).

For the portfolio temperature score and the temperature score grouped by
some category, the following additional information is reported for the
composition of the score

-  Contributions: the level to which each company contributes to the
   total score based on the chosen aggregation method. This value is
   split into company temperature score and relative contribution.

-  The percentage of the score that is based on reported targets vs. the
   percentage based on the default score

-  For the grouped temperature scores: the percentage each group
   contributes to the portfolio temperature score. For example: how much
   each region or sector contributes to the total score.

For the company temperature scores it is possible to request all
underlying data.

-  Portfolio data

-  Financial data

-  GHG emissions

-  Used target and all its parameters

-  Values used during calculation such as the Linear annual reduction
   (LAR), compound annual growth/reduction (CAGR), mapped regression
   scenario, and parameters for the formula to calculate the
   temperature score.
