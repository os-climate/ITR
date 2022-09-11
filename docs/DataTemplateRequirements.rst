**************************
Data Template Requirements
**************************

The ITR Data Template comes with two sheets dedicated to documentation
(Read me, Definitions) and three to be filled by data by users (ITR
input data, ITR target input data, Portfolio).  The documentation
should be self-explanatory, but some additional words are provided
concerning the data requirements.

During this first release of the ITR Tool to an audience of testers,
we want to first thank you for your time and interest in our work, and
to provide some guidance.  The tool does do some error checking, but
at the moment it relies heavily on data being both somewhat
constrained and well-formatted.  We will talk more about what that
means in the following sections.  At the end we will reiterate how to set up your environment to test out the Jupyter Notebook that implements the tool.

ITR Input Data
--------------

The :code:`ITR Input Data` sheet is effectively the Universe of all
instruments the tool can analyze.  We are currently limiting our
analysis to stock issues, but expect to support bonds in the near
future.  We also presently make the assumption that there is a 1:1
relationship between stock instruments (ISINs) and copmanies
(company_id).  If the future we expect to support corporate
hierarchies (aggregating by LEI), but today it's based on a single
ISIN which is used as the company_id.

The tool's benchmarks are based on sectors and regions, whereas most
companies are domiciled in countries.  The tool automatically
translates ISO-3166 2- and 3-character country codes into regions, as
well as common names of countries as well.  If the tool throws an
error for a country name you are using, please replace that name with
an ISO-3166 abbreviation it can understand.

Currently the tool analyzes only two sectors: Electricity Utilities
and Steel.  When we have benchmarks with greater sector coverage, we
will release a version that supports those additional sectors.

The tool uses a currency field to value all financial data for a given
row.  However, the tool has no FX data, so cannot convert from one
financial unit of measurement to another.  It is therefore best to
present all financial information in a single currency.

The report_date field is not used by the tool, but it helps to ensure
that financial information is rooted to a date that can be fed onward
to other BI analysis.

The fundamental financial data includes:

- market_cap (public float)
- revenue (could be FY, CY, TTM, or any period that's consistent across all rows)
- ev (enterprise value = public float + debt - cash equivalents)
- evic (enterprise value including cash = public float + debt)
- assets (the sum total of valorized assets on the balance sheet)

Units, Scopes, Emissions, and Production
----------------------------------------

Unlike many tools which treat numbers as dimensionless objects (in
which case the `1` of `1 dollar`, `1 fish`, and `1 kg of fish` are all
the same value--one), the ITR Tool works with *quantities*, which have
both a magnitude (how much/how many) and units (of what).  To make
this work, emissions and production values are assigned units on a
row-by-row basis.  The emissions_metric can be `t CO2` (metric tonnes of
CO2), `Mt CO2` (million metric tonnes of CO2), or any other imperial
or metric measure of weight.

The production metric depends on the sector.  Electricity Utilities
deliver power measured in MWh, TWh, GJ, PJ, etc.  Again, any imperial
or SI unit of power can be accepted.  Steel production is based on
tons (or megatons) of steel produced.  We have created the unit Fe_ton
(Fe is the symbol for Iron, the principle element in Steel).
1000000 Fe_ton = 1 megaFe_ton.

As previously mentioned, the tool accepts any imperial or SI unit for
these metrics, and there is no trouble if one row of data reports
:code:`3.6 tCO2/MWh` and the next row reports :code:`1 t CO2/GJ` (which happen to
be the same intensity value).  All of it will be converted as
necessary--and can be converted to some final standard for output if
desired.  It is quite OK to see :code:`t CO2/MWh` and :code:`Mt CO2/TWh` in
different rows, but whatever are the metrics for a given row, that's
how the numbers will be interpreted for that row.

Columns of the form YYYY_ghg_sX are emissions data.  If the value of
such a column is 10, it means 10 emissions units, which may be tons or
megatons, depending on the emissions_metric for that row.

Currently the template accepts data from 2016-2022 in the `ITR input
data` sheet.  Not all companies have yet reported 2021 emissions data,
so there may be some rows that have only 2016-2020 data.  Some
companies--for whatever reason--may have also skipped a year in
between their 2016 disclosure and their latest (2020, 2021, or 2022)
disclosures.  The tool deals with three types of missing data:

- If the data is missing from the left (ie., there's no data for 2016 or years until a certain date), the tool ignores the missing data.  As long as there is data present for the base year of the temperature score (typically 2019 or 2020), it will work.
- If the data is missing between two points, the tool fills the data with a linear interpolation.  So if data from 2017 is missing, it would average data from 2016 and 2018 if those years are available.
- If the data is missing to the right, it will extrapolate the data until it has filled in all cells up to the latest reported data.  If all but a few companies report 2020 data and none report 2021 data, tool will extrapolation 2019 data for those companies missing 2020 data.  If there are also some companies with 2021 data, the tool will extrapolate missing data for 2021 and, if needed, also 2020 data.

The tool handles data reports for all scopes defined by the GHG
Protocol: :math:`Scope 1` (own emissions), :math:`Scope 2`(emissinos caused by
utilities supplying electric power), :math:`Scope 3` (upstream and
downstream emissions caused by transportation, use, and disposal of
products).  The tool also handles :math:`S1+S2` as a combined emission and
:math:`S1+S2+S3` as a combined emission.  *HOWEVER*, at the present time the
tool does not do anything with :math:`S3` emissions.  Also, it interprets
the benchmark data as applying to :math:`S1+S2` emissions, upon which all
temperature scoring depeneds.  If data is given as separate :math:`S1` and
:math:`S2` data, the tool will combine them to create :math:`S1+S2` data.  If :math:`S1`,
:math:`S2`, and :math:`S1+S2` data is given, the tool will collect them all, but
will not check the math that :math:`S1 + S2 == S1+S2`.

Over time we expect the tool will be more useful with the more
granular reporting of S1 and S2 data, more accurate in its
interpretation of how these should combine or remain separated
according to sectors and benchmarks, but for the present time we
strongly encourage that all data either have both :math:`S1` and :math:`S2`data or
combined :math:`S1+S2` data.

Finally, columns of the form YYYY_production are for production
metrics.  As with the _ghg_ columns, the numbers in the production
columns are interpreted to denote amounts based on the
production_metrics column.  Missing production data is filled in the
same way as missing emissions data.

ITR target input data
---------------------

The same identifiers--*company_name*, *ompany_lei*, and *company_id*--are
used to connect a row of :code:`ITR input data` to rows of :code:`ITR target input
data`.  Most companies have set a short-term reduction ambition target
(such as reduce absolute emissions by 50% compared with a base year
by 2030) and a long-term net-zero target (the tool does not presently
distinguish between true zero-emissions and positive emissions with
some kind of offset), a single row of data suffices:

- *netzero_year* is the year at which the netzero ambition should be realized.  If multiple netzero_year values are given for the same company, the tool chooses the most recently communicated (latest target_start_year).  If there are multiple such, it chooses the earliest netzero attainment date (target_end_year).
- *target_type* defines whether the short-term ambition is based on absolute emissions or intensity.  Note that when it comes to a long-term netzero ambition, zero is zero, whether emissions or intensity.
- *target_scope* defines the scope(s) of the target.  While it is possible to define :math:`S1, S2, S1+S2, S3, S1+S2+S3`, at present the most reliable choice is :math:`S1+S2` (because we don't have a complete theory yet for interpreting the benchmarks upon which the tools is based for other than :math:`S1+S2`).
- *target_start_year* is the year the target was set.  *target_end_year* is the year the target is to be attained.  In the event that multiple targets aim for a reduction ambition to be attained at the *target_end_year*, the latest *target_start_year* will be the one the tool uses and all other targets for that year will be dropped.  If there are both intensity and absolute targets with the same *target_start_year*, the tool will silently choose the intensity target over the absolute target.  If there are multiple targets with that prioritization, the tool will warn that it is going to pick just one.
- *target_base_year* and *target_base_year_qty* define the "when" and the "from how much" that the target_ambition_reduction applies to (and hopefully is achieved by the *target_year*).  Because all computations require units, the *target_base_year_unit* is needed so that target quantities can be compared with other emissions, production, and intensity data.

Some companies have set more than just one short-term target.  In that
case, additional rows of target data can be set, one for each
additional short-term target.  In those cases it's best to duplicate
the netzero target year date (though ultimately the tool should work
correctly only having seen such information once per company).

If a company has only one target, which happens to be a netzero
ambition, it is OK to specify it as just a short-term 100% reduction
ambition (without a netzero year) or as both a netzero target and a
100% reduction goal.

A note about reducing to zero: at one time the tool implemented a
linear annual reduction (LAR) model, which means that if the goal was
to reduce 100 Mt CO2 to zero over 10 years, the rate of reduction
would be 10 Mt per year for 10 years.  The first year this reduction
would be 10%, but by the 5th yerar the reduction rate would be 20%,
and the last year it would be infinite (as 10 Mt goes to zero).  We
presently implement a CAGR model (constant percent reduction per
year).  This works well for everything except reducing to zero (which
cannot be done, per Xeno's paradox).  Indeed, the closer one aims to a
zero target, the more extreme the per-year percent reduction needs to
be.  (And even with 90% reduction per year for 10 years, there's still
that 0.0000000001 to go...)  To make the math square with reality, we
interpret reducing emissions to less than half-a-percent of the
initial amount as rounding down to zero.

Installation Notes
------------------

The first step is to request an invitation to join the OS-Climate GitHub team.  This is required to access repositories that are not yet public.  (They will be published soon, but not yet.)  You will need a Personal Access Token, which you can get by following these instructions: <https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token>.  Because the ITR repository is currently Private, you will need to select :code:`repo` privileges (the first option box) when requesting the token.  GitHub will magically select all the boxes indented under the `repo` option.

**Getting Started with conda**

If you don't already have a conda environment, you'll need to download one from `<https://docs.conda.io/en/latest/miniconda.html>` (Python 3.9 preferred).

If you are installing conda on a Windows system, follow these instructions: https://conda.io/projects/conda/en/latest/user-guide/install/windows.html
You will want to open the Anaconda PowerShell after installation, which you can do from the Start menu.

If you are on OSX, you will need to install parts of the (utterly massive) Xcode system.  The subset you'll need can be installed by typing :code:`xcode-select --install` into a Terminal window (which you can open from Applications>Utilities>Terminal).  Thought it is tempting to install the :code:`.pkg` version of miniconda, there's nothing user-friendly about how OSX tries to manage its own concepts of system security.  It is easier to start from the :code:`bash` version and follow those instructions.  For other installation instructions, please read https://conda.io/projects/conda/en/latest/user-guide/install/macos.html

For Linux: https://conda.io/projects/conda/en/latest/user-guide/install/linux.html.  And note that you don't have to use the fish shell.  You can use bash, csh, sh, zsh, or whatever is your favorite shell.

You will know you have succeeded in the installation and initialization of conda when you can type :code:`conda info -e` and see an environent listed as base.  If your shell cannot find a conda to run, it likely means you have not yet run :code:`conda init --all`

**Getting Started with Git**

You will use :code:`git` to access the ITR source code.  You can install git from conda thusly: :code:`conda install -c conda-forge git`.  But you can also get it other ways: https://github.com/git-guides/install-git

**Installing the ITR environment and running the Notebook**

With your conda shell and environment running,  with git installed, and starting from the directory in which you want to do the testing:

1. Set GITHUB_TOKEN to your GitHub access token (windows :code`$Env:GITHUB_TOKEN = "your_github_token"`) (OSX/Linux: :code:`export GITHUB_TOKEN=your_github_token`)
2. Clone the ITR repository: :code:`git clone https://github.com/os-climate/ITR.git`
3. Change your directory to the top-level ITR directory: :code:`cd ITR`
4. Optionally switch to the development branch: :code:`git checkout develop` (if you don't, you'll be using the branch :code:`origin/main`)
5. create the conda itr_env: :code:`conda env create -f environment.yml`
6. Activate that environment: :code:`conda activate itr_env`
7. Install the ITR libraries to your local environment: :code:`pip install -e .` (you may need :code:`--no-cache-dir` on windows to avoid permissions errors; please also note that the `.` character is part of the :code:`pip install -e .` command)
8. Change to the *examples* directory: :code:`cd examples`
9. Start your notebook: :code:`jupyter-lab`.  This should cause your default browser to pop to the front and open a page with a Jupyter Notebook.
10. Make the file browser to the left of the notebook wide enough to expose the full names of the files in the *examples* directory.  You should see a file named :code:`quick_template_score_calc.ipynb`.  Double click on that file to open it.
11. Run the notebook with a fresh kernel by pressing the :code:`>>` button.  Accept the option to Restart Kernel and clear all previous variables.

The brackets listed near the top left corner of each executable cell will change from :code:`[ ]` (before running the notebook) to :code:`[*]` while the cell's computation is pending, to a number (such as :code:`[5]` for the 5th cell) when computation is complete.  If everything is working, you will see text output, graphical output, and a newly created `data_dump.xlsx` file representing the input porfolio, enhanced with temperature score data.

**Loading your own data**

1. Place your portfolio data file under the subdirectory named *data* (found under the *examples* directory).
2. Start your notebook: :code:`jupyter-lab`
3. Open the file :code:`quick_template_score_calc.ipynb`
4. Scroll down to the section 'Download/load the sample template data'
5. Change the filename of the .xlsx in the line: :code:`for filename in ['data/<your_filename.xlsx>',`
6. Change the filename of the .xlsx in the line: :code:`template_data_path = "data/<your_filename.xlsx>"`
7. Run the notebook with a fresh kernel by pressing the :code:`>>` button.  Accept the option to Restart Kernel and clear all previous variables.

**Running the ITR Notebook Post Install**

1. Open GitHub Desktop
2. Open the Anaconda PowerShell
3. Set GITHUB_TOKEN to your GitHub access token (windows :code:`$Env:GITHUB_TOKEN = "your_github_token"`) (OSX/Linux: :code:`export GITHUB_TOKEN=your_github_token`)
4. Activate the ITR environment by typing the following command: :code:`conda activate itr_env`
5. Navigate to the *examples* subdirectory under your GitHub ITR directory
6. Start your notebook: :code:`jupyter-lab`
7. Open the file :code:`quick_template_score_calc.ipynb`
8. Run the notebook with a fresh kernel by pressing the :code:`>>` button.  Accept the option to Restart Kernel and clear all previous variables.

Filing Issues and Updating the ITR Repository
---------------------------------------------

Once you are able to run the `quick_template_score_calc.ipynb` sample notebook with the provided sample data (:code:`examples/data/20220720 ITR Tool Sample Data.xlsx`), you are ready to start trying things with your own data.  The notebook explains how to do this at the heading labeled :code:`Download/load the sample template data` before Cell 6.  As you try loading your own data, you will inevitably find errors--sometimes with the data you receive, sometimes with the data you present to the tool, sometimes with the way the tool loads or does not load your data, sometimes with the way the tool interprets or presents your data.  It is the goal of the Data Commons to streamline and simplify access to data so as to reduce the first to cases of errors, and it is the goal of the ITR project team to continuously improve the ITR tool to reduce the other cases of errors.  In all cases, the correction of errors begins with an error reporting process and ends with an effective update process.

To report errors, please use the GitHub Issues interface for the ITR tool: https://github.com/os-climate/ITR/issues

Immediately you will see all open issues filed against the tool, and you may find that a problem you are having has already been reported.  You can search for keywords, and usually in the process of solving issues, commentary on a specific issue may provide insights into work-arounds.  If you do not see an existing issue (you don't need to search exhaustively; just enough to save yourself time writing up an issue that's already been filed), then by all means open an issue describing the problem, ideally with a reproducible test case (such as an excel file containing the minimum amount of anonymozed data required to reproduce the problem).  The team can then assign the problem and you will see progress as the issue is worked.

The collective actions of many people reporting issues and many people working collaboratively to resolve issues is one of the great advantages of open source software development, and a great opportunity to see its magic at work.

At some point you will receive notice that your issue has been addressed with a new release.  There are two ways you can update to the new release.  The first (and least efficient) way is to run the installation process from top to bottom, using a new directory for the installation.  For most of us, this takes about 10 minutes, but it can take longer for various reasons.  The second way takes less than a minute:

1. Close your jupyter-lab browser tab and shut down the jupyter-lab server (typing Ctrl-C or some such in the shell)
2. Change your directory to the top of your ITR tree: :code:`cd ~/os-climate/ITR` (or some such)
3. Pull changes from upstream: git pull
4. If git complains that you have modified some files (such as your notebook, which is "modified" every time you run it), you can
   1. remove the notebook file: :code:`rm examples/data/20220720\ ITR\ Tool\ Sample\ Data.xlsx`
   2. restore it from the updated repository: :code:`git restore examples/data/20220720\ ITR\ Tool\ Sample\ Data.xlsx`
5. Restart your jupyter-lab server

Over time you may do other things to your local repository that makes it difficult to sync with git.  You can file an issue for help, you can do your own research (many of us find answers on github community forums or StackOverflow), or you can go with Option #1: run the installation process from top to bottom in a new directory.

At the same time, with your feedback we will also be working on making the tool and the environment easier to download, install, and manage.
