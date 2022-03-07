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

The `ITR Input Data` sheet is effectively the Universe of all
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
`3.6 tCO2/MWh` and the next row reports `1 t CO2/GJ` (which happen to
be the same intensity value).  All of it will be converted as
necessary--and can be converted to some final standard for output if
desired.  It is quite OK to see 't CO2/MWh' and 'Mt CO2/TWh' in
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
Protocol: Scope 1 (own emissions), Scope 2 (emissinos caused by
utilities supplying electric power), Scope 3 (upstream and
downstream emissions caused by transportation, use, and disposal of
products).  The tool also handles S1+S2 as a combined emission and
S1+S2+S3 as a combined emission.  *HOWEVER*, at the present time the
tool does not do anything with S3 emissions.  Also, it interprets
the benchmark data as applying to S1+S2 emissions, upon which all
temperature scoring depeneds.  If data is given as separate S1 and
S2 data, the tool will combine them to create S1+S2 data.  If S1,
S2, and S1+S2 data is given, the tool will collect them all, but
will not check the math that S1 + S2 == S1+S2.

Over time we expect the tool will be more useful with the more
granular reporting of S1 and S2 data, more accurate in its
interpretation of how these should combine or remain separated
according to sectors and benchmarks, but for the present time we
strongly encourage that all data either have both S1 and S2 data or
combined S1+S2 data.

Finally, columns of the form YYYY_production are for production
metrics.  As with the _ghg_ columns, the numbers in the production
columns are interpreted to denote amounts based on the
production_metrics column.  Missing production data is filled in the
same way as missing emissions data.

ITR target input data
---------------------

The same identifiers--company_name, company_lei, and company_id--are
used to connect a row of `ITR input data` to rows of `ITR target input
data`.  Most companies have set a short-term reduction ambition target
(such as reduce absolute emissions by 50% compared with a base year
by 2030) and a long-term net-zero target (the tool does not presently
distinguish between true zero-emissions and positive emissions with
some kind of offset), a single row of data suffices:

- netzero_year is the year at which the netzero ambition should be realized
- target_type defines whether the short-term ambition is based on absolute emissions or intensity.  Note that when it comes to a long-term netzero ambition, zero is zero, whether emissions or intensity.
- target_scope defines the scope(s) of the target.  While it is possible to define S1, S2, S1+S2, S3, S1+S2+S3, at present the most reliable choice is S1+S2 (because we don't have a complete theory yet for interpreting the benchmarks upon which the tools is based for other than S1+S2).
- target_start_year is the year the target was set.  In the event that multiple targets aim for a reduction ambition at the same year, the latest start_year will be the one the tool uses and all other targets for that year will be dropped.
- target_base_year and target_base_year_qty define the "when" and the "from how much" that the target_ambition_reduction applies to (and hopefully is achieved by the target_year).  Because all computations require units, the target_base_year_unit is needed so that target quantities can be compared with other emissions, production, and intensity data.

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

The first step is to request an invitation to join the OS-Climate GitHub team.  This is required to access repositories that are not yet public.  (They will be published soon, but not yet.)  You will also need a Personal Access Token, which you can get by [following these instructions](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token).

- Clone the [ITR repository](https://github.com/os-climate/ITR.git)
- In the top-level ITR directory, create the `conda` itr_env: `conda env create -f environment.yml`
- Activate that environment: `conda activate itr_env` (you may need to initialize conda by executing `conda init` for your shell first)
- Change to the `examples` directory
- Start your notebook: `jupyter-lab`
- Open the file `quick_template_score_calc.ipynb`
- Run the notebook with a fresh kernel by pressing the @icon-forward button 
