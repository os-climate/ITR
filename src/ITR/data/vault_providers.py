import logging
import os
import pathlib
import warnings
from typing import Callable, List, Optional, Type

import numpy as np
import osc_ingest_trino as osc
import pandas as pd
import sqlalchemy
from dotenv import load_dotenv
from mypy_boto3_s3.service_resource import Bucket

from ..configs import ColumnsConfig, LoggingConfig, ProjectionControls
from ..data import PintArray, ureg

# Rather than duplicating a few methods from BaseCompanyDataProvider, we just call them to delegate to them
from ..data.base_providers import (
    BaseCompanyDataProvider,
    BaseProviderIntensityBenchmark,
    BaseProviderProductionBenchmark,
)
from ..data.data_providers import (
    CompanyDataProvider,
    IntensityBenchmarkDataProvider,
    ProductionBenchmarkDataProvider,
)
from ..data.data_warehouse import DataWarehouse
from ..data.osc_units import Quantity
from ..data.template import TemplateProviderCompany
from ..interfaces import (
    EScope,
    IBenchmark,
    ICompanyAggregates,
    ICompanyData,
    IEIBenchmarkScopes,
    IProductionBenchmarkScopes,
)

# TODO handle ways to append information (from other providers, other benchmarks, new scope info, new corp data updates, etc)


logger = logging.getLogger(__name__)
LoggingConfig.add_config_to_logger(logger)


# If DF_COL contains Pint quantities (because it is a PintArray or an array of Pint Quantities),
# return a two-column dataframe of magnitudes and units.
# If DF_COL contains no Pint quanities, return it unchanged.
def dequantify_column(df_col: pd.Series) -> pd.DataFrame:
    if isinstance(df_col.values, PintArray):
        return pd.DataFrame(
            {
                df_col.name: df_col.values.quantity.m,
                df_col.name + "_units": str(df_col.values.dtype.units),
            },
            index=df_col.index,
        )
    elif df_col.size == 0:
        return df_col
    elif isinstance(df_col.iloc[0], Quantity):  # type: ignore
        m, u = list(zip(*df_col.map(lambda x: (np.nan, "dimensionless") if pd.isna(x) else (x.m, str(x.u)))))
        return pd.DataFrame({df_col.name: m, df_col.name + "_units": u}, index=df_col.index).convert_dtypes()
    else:
        return df_col


# Rewrite dataframe DF so that columns containing Pint quantities are represented by a column for the Magnitude and column for the Units.
# The magnitude column retains the original column name and the units column is renamed with a _units suffix.
def dequantify_df(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([dequantify_column(df[col]) for col in df.columns], axis=1)


# Because this DF comes from reading a Trino table, and because columns must be unqiue, we don't have to enumerate to ensure we properly handle columns with duplicated names
def requantify_df(df: pd.DataFrame, typemap={}) -> pd.DataFrame:
    units_col = None
    columns_not_found = [k for k in typemap.keys() if k not in df.columns]
    if columns_not_found:
        logger.error(f"columns {columns_not_found} not found in DataFrame")
        raise ValueError
    columns_reversed = reversed(df.columns)
    for col in columns_reversed:
        if col.endswith("_units"):
            if units_col:
                logger.error(f"Column {units_col} follows {col} without intervening value column")
                # We expect _units column to follow a non-units column
                raise ValueError
            units_col = col
            continue
        if units_col:
            if col + "_units" != units_col:
                logger.error(f"Excpecting column name {col}_units but saw {units_col} instead")
                raise ValueError
            if (df[units_col] == df[units_col].iloc[0]).all():
                # We can make a PintArray since column is of homogeneous type
                # ...and if the first valid index matches all, we can take first row as good
                new_col = PintArray(df[col], dtype=f"pint[{ureg(df[units_col].iloc[0]).u}]")
            else:
                # Make a pd.Series of Quantity in a way that does not throw UnitStrippedWarning
                new_col = pd.Series(data=df[col], name=col) * pd.Series(
                    data=df[units_col].map(lambda x: typemap.get(col, "dimensionless") if pd.isna(x) else ureg(x).u),
                    name=col,
                )
            if col in typemap.keys():
                new_col = new_col.astype(f"pint[{typemap[col]}]")
            df = df.drop(columns=units_col)
            df[col] = new_col
            units_col = None
        elif col in typemap.keys():
            df[col] = df[col].astype(f"pint[{typemap[col]}]")
    return df


def create_table_from_df(
    df: pd.DataFrame,
    schemaname: str,
    tablename: str,
    engine: sqlalchemy.Engine,
    hive_bucket: Optional[Bucket] = None,
    hive_catalog: Optional[str] = None,
    hive_schema: Optional[str] = None,
    verbose=False,
):
    """
    Create a table in the Data Vault

    :param df: The DataFrame to be written as a table in the Data Vault
    :param schemaname: The schema where the table should be written
    :param tablename: The name of the table in the Data Vault
    :param engine: The SqlAlchemy connection to the Data Vault
    :param hive_bucket: :param hive_catalog: :param hive_schema: Optional paramters.  If given we attempt to use a fast Hive ingestion process.  Otherwise use default (and slow) Trino ingestion.
    :param verbose: If True, log information about actions of the Data Vault as they happen
    """
    drop_table = f"drop table if exists {schemaname}.{tablename}"
    qres = osc._do_sql(drop_table, engine, verbose)
    logger.debug("dtypes, columns, and index of create_table_from_df(df...)")
    logger.debug(df.dtypes)
    logger.debug(df.columns)
    logger.debug(df.index)
    new_df = dequantify_df(df).convert_dtypes()
    if hive_bucket is not None:
        osc.fast_pandas_ingest_via_hive(
            new_df,
            engine,
            None,
            schemaname,
            tablename,
            hive_bucket,
            hive_catalog,
            hive_schema,
            partition_columns=["year"] if "year" in new_df.columns else None,
            overwrite=True,
            typemap={
                "datetime64[ns]": "timestamp(6)",
                "datetime64[ns, UTC]": "timestamp(6)",
                # "Int16":"integer", "int16":"integer"
            },
            verbose=verbose,
        )
    else:
        new_df.to_sql(
            tablename,
            con=engine,
            schema=schemaname,
            if_exists="append",
            index=False,
            method=osc.TrinoBatchInsert(batch_size=5000, verbose=verbose),
        )


# When reading SQL tables to import into DataFrames, it is up to the user to preserve {COL}, {COL}_units pairings so they can be reconstructed.
# If the user does a naive "select * from ..." this happens naturally.
# We can give a warning when we see a resulting dataframe that could have, but does not have, unit information properly integrated.  But
# fixing the query on the fly becomes difficult when we consider the fully complexity of parsing and rewriting SQL queries to put the units columns in the correct locations.
# (i.e., properly in the principal SELECT clause (which can have arbitrarily complex terms), not confused by FROM, WHERE, GROUP BY, ORDER BY, etc.)


def read_quantified_sql(
    sql: str,
    tablename,
    schemaname,
    engine: sqlalchemy.Engine,
    index_col=None,
) -> pd.DataFrame:
    qres = osc._do_sql(f"describe {schemaname}.{tablename}", engine, verbose=False)
    # tabledesc will be a list of tuples (column, type, extra, comment)
    colnames = [x[0] for x in qres]
    # read columns normally...this will be missing any unit-related information
    sql_df = pd.read_sql(sql, engine, index_col)
    # if the query requests columns that don't otherwise bring unit information along with them, get that information too
    extra_unit_columns = [
        (i, f"{col}_units")
        for i, col in enumerate(sql_df.columns)
        if f"{col}_units" not in sql_df.columns and f"{col}_units" in colnames
    ]
    if extra_unit_columns:
        extra_unit_columns_positions = [
            (i, extra_unit_columns[i][0], extra_unit_columns[i][1]) for i in range(len(extra_unit_columns))
        ]
        for col_tuple in extra_unit_columns_positions:
            logger.error(
                f"Missing units column '{col_tuple[2]}' after original column '{sql_df.columns[col_tuple[1]]}' (should be column #{col_tuple[0]+col_tuple[1]+1} in new query)"
            )
        raise ValueError
    else:
        return requantify_df(sql_df).convert_dtypes()


# Basic Corp Data Asumptions
#   5 year historical EI (else we presume single year is constant backward and forward)
#   5 year historical Production (else we presume single year is constant backward and forward)
#   5 year historical Emissions (else we presume single year is constant backward and forward)
#   We can infer one of the above from the other two (simple maths)
#   The above tables identify the scope(s) to which they apply (S1, S2, S12, S3, S123) and data source (e.g. 'rmi_20211120')

# Basic Benchmark Data Assumptions
#   EI for a given scope
#   Production defined in terms of growth (or negative growth) on a rolling basis (so 0.05, -0.04) would mean 5% growth followed by 4% negative growth for a total of 0.8%
#   Benchmarks are named (e.g., 'OECM')


class VaultProviderProductionBenchmark(BaseProviderProductionBenchmark):
    def __init__(
        self,
        engine: sqlalchemy.Engine,
        benchmark_name: str,
        production_benchmarks: IProductionBenchmarkScopes,
        ingest_schema: str = "",
        hive_bucket: Optional[Bucket] = None,
        hive_catalog: Optional[str] = None,
        hive_schema: Optional[str] = None,
        column_config: Type[ColumnsConfig] = ColumnsConfig,
    ):
        """
        As an alternative to using FastAPI interfaces, this creates an interface allowing access to Production benchmark data via the Data Vault.
        :param engine: the Sqlalchemy connect to the Data Vault
        :param benchmark_name: the table name of the benchmark (in Trino)
        :param production_benchmarks: List of IBenchmarkScopes
        :param ingest_schema: The database schema where the Data Vault lives
        :param hive_bucket, hive_catalog, hive_schema: Optional parameters to enable fast ingestion via Hive; otherwise uses Trino batch insertion (which is slow)
        :param column_config: An optional ColumnsConfig object containing relevant variable names
        """
        super().__init__(production_benchmarks=production_benchmarks, column_config=column_config)
        self._engine = engine
        self._schema = ingest_schema or engine.dialect.default_schema_name or "demo_dv"
        self._benchmark_name = benchmark_name
        df = pd.DataFrame()
        for scope in ["AnyScope"]:
            if production_benchmarks.model_dump()[scope] is None:
                continue
            for benchmark in production_benchmarks.model_dump()[scope]["benchmarks"]:
                bdf = pd.DataFrame.from_dict(
                    {
                        r["year"]: [
                            benchmark["sector"],
                            benchmark["region"],
                            scope,
                            r["value"],
                        ]
                        for r in benchmark["projections"]
                    },
                    columns=[
                        "sector",
                        "region",
                        "scope",
                        "production",
                    ],
                    orient="index",
                )
                df = pd.concat([df, bdf])
        df.reset_index(inplace=True)
        df.rename(columns={"index": "year"}, inplace=True)
        create_table_from_df(df, self._schema, benchmark_name, engine, hive_bucket, hive_catalog, hive_schema)

    def benchmark_changed(self, new_projected_production: ProductionBenchmarkDataProvider) -> bool:
        # The Data Vault does not keep its own copies of benchmarks
        return False

    def get_company_projected_production(self, *args, **kwargs) -> pd.DataFrame:
        return super(BaseProviderProductionBenchmark, self).__thisclass__.get_company_projected_production(  # type: ignore
            self, *args, **kwargs
        )

    def get_benchmark_projections(self, *args, **kwargs) -> pd.DataFrame:
        return super(BaseProviderProductionBenchmark, self).__thisclass__.get_benchmark_projections(  # type: ignore
            self, *args, **kwargs
        )


class VaultProviderIntensityBenchmark(BaseProviderIntensityBenchmark):
    def __init__(
        self,
        engine: sqlalchemy.Engine,
        benchmark_name: str,
        EI_benchmarks: IEIBenchmarkScopes,
        ingest_schema: str = "",
        hive_bucket: Optional[Bucket] = None,
        hive_catalog: Optional[str] = None,
        hive_schema: Optional[str] = None,
        column_config: Type[ColumnsConfig] = ColumnsConfig,
        projection_controls: ProjectionControls = ProjectionControls(),
    ):
        """
        As an alternative to using FastAPI interfaces, this creates an interface allowing access to Emission Intensity benchmark data via the Data Vault.
        :param engine: the Sqlalchemy connect to the Data Vault
        :param benchmark_name: the table name of the benchmark (in Trino)
        :param EI_benchmarks: List of IEIBenchmarkScopes
        :param ingest_schema: A prefix for all tables so that different users can use the same schema without conflicts
        :param hive_bucket, hive_catalog, hive_schema: Optional parameters to enable fast ingestion via Hive; otherwise uses Trino batch insertion (which is slow)
        :param column_config: An optional ColumnsConfig object containing relevant variable names
        :param projection_controls: Projection Controls set the target BASE_YEAR, START_YEAR, and END_YEAR parameters of the model
        """
        super().__init__(
            EI_benchmarks=EI_benchmarks,
            column_config=column_config,
            projection_controls=projection_controls,
        )
        self._engine = engine
        self._schema = ingest_schema or engine.dialect.default_schema_name or "demo_dv"
        self._benchmark_name = benchmark_name
        df = pd.DataFrame()
        for scope in EScope.get_scopes():
            if EI_benchmarks.model_dump()[scope] is None:
                continue
            for benchmark in EI_benchmarks.model_dump()[scope]["benchmarks"]:
                benchmark_df = pd.DataFrame.from_dict(
                    {
                        r["year"]: [
                            benchmark["sector"],
                            benchmark["region"],
                            scope,
                            r["value"],
                            EI_benchmarks.benchmark_global_budget,
                            EI_benchmarks.benchmark_temperature,
                        ]
                        for r in benchmark["projections"]
                    },
                    columns=[
                        "sector",
                        "region",
                        "scope",
                        "intensity",
                        "global_budget",
                        "benchmark_temp",
                    ],
                    orient="index",
                )
                # TODO: AFOLU correction
                df = pd.concat([df, benchmark_df])
        df.reset_index(inplace=True)
        df.rename(columns={"index": "year"}, inplace=True)
        create_table_from_df(df, self._schema, benchmark_name, engine, hive_bucket, hive_catalog, hive_schema)


class VaultCompanyDataProvider(BaseCompanyDataProvider):
    def __init__(
        self,
        engine: sqlalchemy.Engine,
        company_table: str,
        target_table: str = "",
        trajectory_table: str = "",
        company_schema: str = "",
        hive_bucket: Optional[Bucket] = None,
        hive_catalog: Optional[str] = None,
        hive_schema: Optional[str] = None,
        template_company_data: Optional[TemplateProviderCompany] = None,
        column_config: Type[ColumnsConfig] = ColumnsConfig,
    ):
        """
        This class serves primarily for connecting to the ITR tool to the Data Vault via Trino.

        :param company_table: the name of the Trino table that contains fundamental data for companies
        :param target_table: the name of the Trino table that contains company (emissions intensity) target data (and possibly historical data)
        :param trajectory_table: the name of the Trino table that contains company (emissions intensity) historical data (and possibly trajectory data)
        :param company_schema: the name of the schema where the company_table is found
        :param hive_bucket, hive_catalog, hive_schema: Optional parameters to enable fast ingestion via Hive; otherwise uses Trino batch insertion (which is slow)
        :param template_company_data: if not None, company data to ingest into company, target, and trajectory tables
        :param column_config: An optional ColumnsConfig object containing relevant variable names
        """
        super().__init__(
            companies=[] if template_company_data is None else template_company_data._companies,
            column_config=column_config,
        )
        self._engine = engine
        self._schema = company_schema or engine.dialect.default_schema_name or "demo_dv"
        self._company_table = company_table
        # Validate and complete the projected trajectories
        self._target_table = target_table or company_table.replace("company_", "target_")  # target_data
        self._trajectory_table = trajectory_table or company_table.replace("company_", "trajectory_")  # trajectory_data
        self._production_table = company_table.replace("company_", "production_")  # production_data
        self._emissions_table = company_table.replace("company_", "emissions_")  # emissions_data

        if template_company_data is None:
            return

        # Here we fill in the company data underlying the CompanyDataProvider

        df = (
            template_company_data.df_fundamentals[
                [
                    "company_name",
                    "company_lei",
                    "company_id",
                    "sector",
                    "country",
                    "region",
                    "exposure",
                    "currency",
                    "report_date",
                    "company_market_cap",
                    "company_revenue",
                    "company_enterprise_value",
                    "company_ev_plus_cash",
                    "company_total_assets",
                    "cash",
                    "debt",
                ]
            ]
            .copy()
            .rename(
                columns={
                    "company_enterprise_value": "company_ev",
                    "company_ev_plus_cash": "company_evic",
                    "cash": "company_cash_equivalents",
                    "debt": "company_debt",
                },
            )
        )
        df["year"] = df.report_date.dt.year
        df.drop(columns="report_date", inplace=True)
        df = dequantify_df(df).convert_dtypes()

        # ingest company data
        create_table_from_df(
            df, self._schema, self._company_table, engine, hive_bucket, hive_catalog, hive_schema, verbose=True
        )

        # We don't have any target nor trajectory projections until we connect benchmark data via DataWarehouse

    # The factors one would want to sum over companies for weighting purposes are:
    #   * market_cap_usd
    #   * enterprise_value_usd
    #   * assets_usd
    #   * revenue_usd
    #   * emissions

    # TODO: make return value a Quantity (USD or CO2)
    def sum_over_companies(
        self,
        company_ids: List[str],
        year: int,
        factor: str,
        scope: EScope = EScope.S1S2,
    ) -> float:
        if factor == "enterprise_value_usd":
            factor_sum = f"select sum(market_cap_usd + debt_usd - cash_usd)"
        elif factor == "emissions":
            if scope in [EScope.S1, EScope.S2, EScope.S3]:
                factor_sum = f"select sum(co2_{scope.name.lower()}_by_year)"
            elif scope == EScope.S1S2:
                factor_sum = f"select sum(co2_s1_by_year+if(is_nan(co2_s2_by_year),0.0,co2_s2_by_year))"
            elif scope == EScope.S1S2S3:
                factor_sum = f"select sum(co2_s1_by_year+if(is_nan(co2_s2_by_year),0.0,co2_s2_by_year)+if(is_nan(co2_s3_by_year),0.0,co2_s3_by_year))"
            else:
                raise ValueError(f"scope {scope} not supported")
        else:
            factor_sum = f"select sum({factor})"
        sql = f"{factor_sum} as {factor}_sum from {self._schema}.{self._company_table}"
        if year is not None:
            sql = f"{sql} where year={year}"
        qres = osc._do_sql(sql, self._engine, verbose=False)

        # qres[0] is the first row of the returned data; qres[0][0] is the first (and only) column of the row returned
        return qres[0][0]

    def compute_portfolio_weights(
        self,
        pa_temp_scores: pd.Series,
        year: int,
        factor: str,
        scope: EScope = EScope.S1S2,
    ) -> pd.Series:
        """
        Portfolio values could be position size, temperature scores, anything that can be multiplied by a factor.

        :param company_ids: A pd.Series of company IDs (ISINs)
        :return: A pd.Series weighted by the factor
        """
        from_sql = f"from {self._schema}.{self._company_table}"
        group_sql = "group by company_id"
        if factor == "company_evic":
            where_sql = ""
            factor_sql = f"select company_id, sum(company_market_cap + company_cash_equivalents)"
        elif factor == "emissions":
            where_sql = f"where year = {year}"
            if scope in [EScope.S1, EScope.S2, EScope.S3]:
                factor_sql = f"select company_id, sum(co2_{scope.name.lower()}_by_year)"
            elif scope == EScope.S1S2:
                factor_sql = f"select company_id, sum(co2_s1_by_year+if(is_nan(co2_s2_by_year),0.0,co2_s2_by_year))"
            elif scope == EScope.S1S2:
                factor_sql = f"select company_id, sum(co2_s1_by_year+if(is_nan(co2_s2_by_year),0.0,co2_s2_by_year)+if(is_nan(co2_s3_by_year),0.0,co2_s3_by_year))"
            else:
                raise ValueError(f"scope {scope} not supported")
        else:
            sql = f"select company_id, sum({factor})"
        qres = osc._do_sql(f"{factor_sql} as {factor} {from_sql} {where_sql} {group_sql}", self._engine, verbose=False)
        weights = pd.Series(data=[s[1] for s in qres], index=[s[0] for s in qres], dtype=float)
        weights = weights.loc[pa_temp_scores.index.intersection(weights.index)]
        weight_sum = weights.sum()
        return pa_temp_scores * weights / weight_sum

    # Commented out because this doesn't include necessary base_year_production nor ghg_s1s2 nor ghg_s3
    # def get_company_fundamentals(self, company_ids: List[str]) -> pd.DataFrame:
    #     """
    #     :param company_ids: A list of company IDs
    #     :return: A pandas DataFrame with company fundamental info per company
    #     """
    #     or_clause = " or ".join([f"company_id = '{c}'" for c in company_ids])
    #     sql = f"select * from {self._schema}.{self._company_table} where {or_clause}"
    #     df = read_quantified_sql(sql, self._company_table, self._schema, self._engine)
    #     # df = df.drop(columns=['projected_targets', 'projected_intensities'])
    #     return df.set_index(self.column_config.COMPANY_ID)


# FIXME: Need to reshape the tables TARGET_DATA and TRAJECTORY_DATA so scope is a column and the EI data relates only to that scope (wide to long)
class DataVaultWarehouse(DataWarehouse):
    def __init__(
        self,
        engine: sqlalchemy.Engine,
        company_data: VaultCompanyDataProvider,
        benchmark_projected_production: VaultProviderProductionBenchmark,
        benchmarks_projected_ei: VaultProviderIntensityBenchmark,
        estimate_missing_data: Optional[Callable[["DataWarehouse", ICompanyData], None]] = None,
        ingest_schema: str = "",
        itr_prefix: str = "",
        hive_bucket: Optional[Bucket] = None,
        hive_catalog: Optional[str] = None,
        hive_schema: Optional[str] = None,
        column_config: Type[ColumnsConfig] = ColumnsConfig,
    ):
        """
        Construct Data Vault tables for cumulative emissions budgets, trajectories, and targets,
        which rely on trajectory and target projections from benchmark production and SDA pathways.

        Fundamentally: DataWarehouse(benchmark_ei, benchmark_prod, company_data)
            -> { production_data, trajectory_data,  target_data }
            -> { cumulative_budgets, cumulative_emissions }

        :param engine: The Sqlalchemy connector to the Data Vault
        :param company_data: as a VaultCompanyDataProvider, this provides both a reference to a fundamental company data table and data structures containing historic ESG data.  Trajectory and Target projections also get filled in here.
        :param benchmark_projected_production: A reference to the benchmark production table as well as data structures used by the Data Vault for projections
        :param benchmark_projected_ei: A reference to the benchmark emissions intensity table as well as data structures used by the Data Vault for projections
        :param estimate_missing_data: If provided, a function that can fill in missing S3 data (possibly by aligning to benchmark statistics)
        :param ingest_schema: The database schema where the Data Vault lives
        :param itr_prefix: A prefix for all tables so that different users can use the same schema without conflicts
        :param hive_bucket: :param hive_catalog: :param hive_schema: Optional paramters.  If given we attempt to use a fast Hive ingestion process.  Otherwise use default (and slow) Trino ingestion.
        :param column_config: An optional ColumnsConfig object containing relevant variable names
        """
        # This initialization step adds trajectory and target projections to `company_data`
        super().__init__(
            company_data=company_data,  # type: ignore
            benchmark_projected_production=benchmark_projected_production,
            benchmarks_projected_ei=benchmarks_projected_ei,
            estimate_missing_data=estimate_missing_data,
            column_config=column_config,
        )
        self._engine = engine
        self._schema = ingest_schema or engine.dialect.default_schema_name or "demo_dv"
        self._tempscore_table = f"{itr_prefix}temperature_scores"

        # If there's no company data, we are just using the vault, not initializing it
        if company_data is None:
            return
        if benchmark_projected_production is None and benchmarks_projected_ei is None:
            return

        # FIXME: should we use VaultCompanyDataProvider instead?
        projection_tablename = [self.company_data._target_table, self.company_data._trajectory_table]  # type: ignore

        target_dfs: List[pd.DataFrame] = []
        trajectory_dfs: List[pd.DataFrame] = []

        # Ingest target and trajectory projections into the Data Vault
        for i, projection in enumerate(["projected_targets", "projected_intensities"]):
            projection_dfs = []
            for company in company_data._companies:
                ei_dict = {}
                for scope in EScope.get_scopes():
                    if getattr(company, projection)[scope]:
                        ei_dict[scope] = getattr(company, projection)[scope].projections
                    else:
                        ei_dict[scope] = pd.Series(dtype="object")
                ei_data = pd.concat([ei_dict[scope] for scope in EScope.get_scopes()], axis=1).reset_index()
                ei_data.columns = ["year"] + [f"ei_{scope.lower()}_by_year" for scope in EScope.get_scopes()]
                df = pd.DataFrame(
                    data=[[company.company_name, "", company.company_id, company.sector] for i in ei_data.index],
                    columns=["company_name", "company_lei", "company_id", "sector"],
                )
                projection_dfs.append(pd.concat([df, ei_data], axis=1))
            df2 = pd.concat(projection_dfs).reset_index(drop=True)
            if projection_tablename[i] == self.company_data._target_table:  # type: ignore
                target_df = df2
            df3 = dequantify_df(df2).convert_dtypes()
            create_table_from_df(
                df3, self._schema, projection_tablename[i], engine, hive_bucket, hive_catalog, hive_schema, verbose=True
            )

        # Calcuate production projections
        company_info_at_base_year = company_data.get_company_intensity_and_production_at_base_year(
            company_data.get_company_ids()
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # See https://github.com/hgrecco/pint-pandas/issues/128
            projected_production = benchmark_projected_production.get_company_projected_production(
                company_sector_region_scope=company_info_at_base_year
            )

        df = projected_production.droplevel("scope").drop_duplicates()
        df.columns.set_names("year", inplace=True)
        df2 = df.unstack(level=0).to_frame("production_by_year").reset_index("year")
        df3 = pd.read_sql(
            f"select distinct company_id, company_name, company_lei, sector from {company_data._company_table}", engine
        )
        df4 = df2.merge(df3, on="company_id")
        df5 = dequantify_df(df4).reset_index()
        df5 = df5[
            # Reorder columns
            [
                "company_name",
                "company_lei",
                "company_id",
                "sector",
                "year",
                "production_by_year",
                "production_by_year_units",
            ]
        ].convert_dtypes()

        # Ingest productions into Data Vault
        create_table_from_df(
            df5,
            self._schema,
            company_data._production_table,
            engine,
            hive_bucket,
            hive_catalog,
            hive_schema,
            verbose=True,
        )

        # The DataVaultWarehouse provides three calculations per company (using SQL code rather than Python):
        #    * Cumulative trajectory of emissions
        #    * Cumulative target of emissions
        #    * Cumulative budget of emissions (separately for each benchmark)
        # Because we have scope emissions in a wide format, we do our own wide-to-long conversion

        qres = osc._do_sql(
            f"drop table if exists {self._schema}.{itr_prefix}cumulative_emissions",
            self._engine,
            verbose=False,
        )
        # FIXME: we could compute all scopes by partitioning on scope as well
        emissions_from_tables = f"""
{company_data._schema}.{company_data._company_table} C
     join {company_data._schema}.{company_data._production_table} P on P.company_id=C.company_id
     join {company_data._schema}.{company_data._trajectory_table} EI on EI.company_id=C.company_id and EI.year=P.year and EI.ei_SCOPE_by_year is not NULL
     join {company_data._schema}.{company_data._target_table} ET on ET.company_id=C.company_id and ET.year=P.year and ET.ei_SCOPE_by_year is not NULL
"""

        create_emissions_sql = f"create table {self._schema}.{itr_prefix}cumulative_emissions with (format = 'ORC', partitioning = array['scope']) as"
        emissions_scope_sql = "UNION ALL".join(
            [
                f"""
select C.company_name, C.company_id, '{company_data._schema}' as source, P.year,
       sum(EI.ei_{scope}_by_year * P.production_by_year) over (partition by C.company_id order by P.year) as cumulative_trajectory,
       concat(EI.ei_{scope}_by_year_units, ' * ', P.production_by_year_units) as cumulative_trajectory_units,
       sum(ET.ei_{scope}_by_year * P.production_by_year) over (partition by C.company_id order by P.year) as cumulative_target,
       concat(ET.ei_{scope}_by_year_units, ' * ', P.production_by_year_units) as cumulative_target_units,
       '{scope.upper()}' as scope
from {emissions_from_tables.replace('SCOPE', scope)}
"""
                for scope in map(str.lower, EScope.get_scopes())
            ]
        )
        qres = osc._do_sql(f"{create_emissions_sql} {emissions_scope_sql}", self._engine, verbose=True)

        qres = osc._do_sql(
            f"drop table if exists {self._schema}.{itr_prefix}cumulative_budgets",
            self._engine,
            verbose=False,
        )
        # base_year_scale = trajectory / budget at base year (a scalar)
        # scaled cumulative budget = base_year_scale * cumulative budget (a time series)

        qres = osc._do_sql(
            f"""
create table {self._schema}.{itr_prefix}cumulative_budgets with (
    format = 'ORC',
    partitioning = array['scope']
) as
with P_BY as (select distinct company_id,
                     first_value(year) over (partition by company_id order by year) as base_year,
                     first_value(production_by_year) over (partition by company_id order by year) as production_by_year
              from {company_data._schema}.{company_data._production_table})
select C.company_name, C.company_id, '{company_data._schema}' as source, P.year,  -- FIXME: should have scenario_name and year released
       B.global_budget, B.benchmark_temp,
       sum(B.intensity * P.production_by_year) over (partition by C.company_id, B.scope order by P.year) as cumulative_budget,
       concat(B.intensity_units, ' * ', P.production_by_year_units) as cumulative_budget_units,
       CE_BY.cumulative_trajectory/(B_BY.intensity * P_BY.production_by_year) * sum(B.intensity * P.production_by_year) over (partition by C.company_id, B.scope order by P.year) as cumulative_scaled_budget,
       CE_BY.cumulative_trajectory_units as cumulative_scaled_budget_units,
       B.scope
from {company_data._schema}.{company_data._company_table} C
     join P_BY on P_BY.company_id=C.company_id
     join {company_data._schema}.{company_data._production_table} P on P.company_id=C.company_id
     join {self._schema}.{benchmarks_projected_ei._benchmark_name} B on P.year=B.year and C.region=B.region and C.sector=B.sector
     join {self._schema}.{itr_prefix}cumulative_emissions CE on CE.company_id=C.company_id and B.scope=CE.scope and CE.year=P.year
     join {self._schema}.{itr_prefix}cumulative_emissions CE_BY on CE_BY.company_id=C.company_id and CE_BY.scope=B.scope and CE_BY.year=P_BY.base_year
     join {self._schema}.{benchmarks_projected_ei._benchmark_name} B_BY on B.scope=B_BY.scope and B.region=B_BY.region and B.sector=B_BY.sector and B_BY.year=P_BY.base_year
""",
            self._engine,
            verbose=True,
        )

    def quant_init(
        self,
        engine: sqlalchemy.Engine,
        company_data: VaultCompanyDataProvider,
        ingest_schema: str = "",
        itr_prefix: str = "",
    ):
        # The Quant users of the DataVaultWarehouse produces two calculations per company:
        #    * Target and Trajectory overshoot ratios
        #    * Temperature Scores

        qres = osc._do_sql(
            f"drop table if exists {self._schema}.{itr_prefix}overshoot_ratios",
            self._engine,
            verbose=False,
        )
        qres = osc._do_sql(
            f"""
create table {self._schema}.{itr_prefix}overshoot_ratios with (
    format = 'ORC',
    partitioning = array['scope']
) as
select E.company_name, E.company_id, '{company_data._schema}' as source, B.year, -- FIXME: should have scenario_name and year released
       B.global_budget, B.benchmark_temp,
       E.cumulative_trajectory/B.cumulative_budget as trajectory_overshoot_ratio,
       concat(E.cumulative_trajectory_units, ' / (', B.cumulative_budget_units, ')') as trajectory_overshoot_ratio_units,
       E.cumulative_target/B.cumulative_budget as target_overshoot_ratio,
       concat(E.cumulative_target_units, ' / (', B.cumulative_budget_units, ')') as target_overshoot_ratio_units,
       B.scope
from {self._schema}.{itr_prefix}cumulative_emissions E
     join {self._schema}.{itr_prefix}cumulative_budgets B on E.company_id=B.company_id and E.scope=B.scope and E.year=B.year
""",
            self._engine,
            verbose=True,
        )

        qres = osc._do_sql(
            f"drop table if exists {self._schema}.{self._tempscore_table}",
            self._engine,
            verbose=False,
        )
        qres = osc._do_sql(
            f"""
create table {self._schema}.{self._tempscore_table} with (
    format = 'ORC',
    partitioning = array['scope']
) as
select R.company_name, R.company_id, '{company_data._schema}' as source,  -- FIXME: should have scenario_name and year released
       R.benchmark_temp + R.global_budget * (R.trajectory_overshoot_ratio-1) * 2.2/3664.0 as trajectory_temperature_score,
       'delta_degC' as trajectory_temperature_score_units,
       R.benchmark_temp + R.global_budget * (R.target_overshoot_ratio-1) * 2.2/3664.0 as target_temperature_score,
       'delta_degC' as target_temperature_score_units,
       R.scope
from {self._schema}.{itr_prefix}overshoot_ratios R
""",
            self._engine,
            verbose=True,
        )

    def get_preprocessed_company_data(self, company_ids: List[str]) -> List[ICompanyAggregates]:
        raise NotImplementedError

    def get_pa_temp_scores(self, probability: float, company_ids: List[str]) -> pd.Series:
        if probability < 0 or probability > 1:
            raise ValueError(f"probability value {probability} outside range [0.0, 1.0]")
        temp_scores = read_quantified_sql(
            f"select company_id, scope, target_temperature_score, target_temperature_score_units, trajectory_temperature_score, trajectory_temperature_score_units from {self._schema}.{self._tempscore_table}",
            self._tempscore_table,
            self._schema,
            self._engine,
            index_col=["company_id", "scope"],
        )
        # We may have company_ids in our portfolio not in our database, and vice-versa.
        # Return proper pa_temp_scores for what we can find, and np.nan for those we cannot
        retval = pd.Series(data=None, index=company_ids, dtype="float64")
        retval.loc[
            retval.index.intersection(temp_scores.index)
        ] = temp_scores.target_temperature_score * probability + temp_scores.trajectory_temperature_score * (
            1 - probability
        )
        return retval
