import concurrent.futures
import logging
import os
import warnings
from abc import ABC
from concurrent.futures import Future
from typing import Callable, Dict, List, Optional, Type, Union, cast

import numpy as np  # noqa F401
import osc_ingest_trino as osc
import pandas as pd
import sqlalchemy
from mypy_boto3_s3.service_resource import Bucket

import ITR

from ..configs import ColumnsConfig, LoggingConfig, ProjectionControls
from ..data import PintArray, PintType, ureg

# Rather than duplicating a few methods from BaseCompanyDataProvider, we just call them to delegate to them
from ..data.base_providers import BaseCompanyDataProvider
from ..data.data_providers import (
    IntensityBenchmarkDataProvider,
    ProductionBenchmarkDataProvider,
)
from ..data.data_warehouse import DataWarehouse
from ..data.osc_units import Q_, EmissionsQuantity, Quantity, delta_degC_Quantity
from ..data.template import TemplateProviderCompany
from ..interfaces import EScope, ICompanyAggregates, ICompanyData

# re_simplify_units = r" \/ (\w+)( \/ (\w+))? \* \1(?(3) \* \3|)"
re_simplify_units_both = r" \/ (\w+) \/ (\w+) \* \1 \* \2"
re_simplify_units_one = r" \/ (\w+) \* \1"

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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Quieting warnings due to https://github.com/hgrecco/pint/issues/1897
            m, u = list(
                zip(
                    *df_col.map(
                        lambda x: (np.nan, "dimensionless")
                        if pd.isna(x)
                        else (x.m, str(x.u))
                    )
                )
            )
            return pd.DataFrame(
                {df_col.name: m, df_col.name + "_units": u}, index=df_col.index
            ).convert_dtypes()
    else:
        return df_col


# Rewrite dataframe DF so that columns containing Pint quantities are represented by a column for the Magnitude and column for the Units.
# The magnitude column retains the original column name and the units column is renamed with a _units suffix.
def dequantify_df(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([dequantify_column(df[col]) for col in df.columns], axis=1)


# Because this DF comes from reading a Trino table, and because columns must be unique,
# we don't have to enumerate to ensure we properly handle columns with duplicated names
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
                logger.error(
                    f"Column {units_col} follows {col} without intervening value column"
                )
                # We expect _units column to follow a non-units column
                raise ValueError
            units_col = col
            continue
        if units_col:
            if col + "_units" != units_col:
                logger.error(
                    f"Excpecting column name {col}_units but saw {units_col} instead"
                )
                raise ValueError
            if (df[units_col] == df[units_col].iloc[0]).all():
                # We can make a PintArray since column is of homogeneous type
                # ...and if the first valid index matches all, we can take first row as good
                new_col = PintArray(
                    df[col], dtype=f"pint[{ureg(df[units_col].iloc[0]).u}]"
                )
            else:
                # Make a pd.Series of Quantity in a way that does not throw UnitStrippedWarning
                if df[col].map(lambda x: x is None).any():
                    # breakpoint()
                    raise
                new_col = pd.Series(data=df[col], name=col) * pd.Series(
                    data=df[units_col].map(
                        lambda x: typemap.get(col, ureg("dimensionless").u)
                        if pd.isna(x)
                        else ureg(x).u
                    ),
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


class VaultInstance(ABC):
    def __init__(
        self,
        engine: sqlalchemy.Engine,
        catalog: Optional[
            str
        ] = "",  # FIXME: this should go away when osc-ingest-tools 0.5.3 is released
        schema: Optional[str] = "",
        hive_bucket: Optional[Bucket] = None,
        hive_catalog: Optional[str] = None,
        hive_schema: Optional[str] = None,
    ):
        """As an alternative to using FastAPI interfaces, this creates an interface allowing access to Production
        benchmark data via the Data Vault. :param engine: the Sqlalchemy connect to the Data Vault
        :param schema: The database schema where the Data Vault lives
        :param hive_bucket, hive_catalog, hive_schema: Optional parameters to enable fast ingestion via Hive;
        otherwise uses Trino batch insertion (which is slow)
        """
        super().__init__()
        self.engine = engine
        self.catalog = catalog or os.environ.get(
            "ITR_CATALOG", "osc_datacommons_dev"
        )  # FIXME: needed for osc-ingest-tools < 0.5.3
        self.schema = (
            schema
            or engine.dialect.default_schema_name
            or os.environ.get("ITR_SCHEMA", "demo_dv")
        )
        self.hive_bucket = hive_bucket
        self.hive_catalog = hive_catalog
        self.hive_schema = hive_schema


def create_vault_table_from_df(
    df: pd.DataFrame,
    tablename: str,
    vault: VaultInstance,
    verbose=False,
):
    """Create a table in the Data Vault

    :param df: The DataFrame to be written as a table in the Data Vault
    :param schemaname: The schema where the table should be written
    :param tablename: The name of the table in the Data Vault
    :param engine: The SqlAlchemy connection to the Data Vault
    :param hive_bucket: :param hive_catalog: :param hive_schema: Optional paramters.
    # If given we attempt to use a fast Hive ingestion process.
    # Otherwise use default (and slow) Trino ingestion.
    :param verbose: If True, log information about actions of the Data Vault as they happen
    """
    drop_table = f"drop table if exists {vault.schema}.{tablename}"
    qres = osc._do_sql(drop_table, vault.engine, verbose)  # noqa F841
    logger.debug("dtypes, columns, and index of create_vault_table_from_df(df...)")
    logger.debug(df.dtypes)
    logger.debug(df.columns)
    logger.debug(df.index)
    new_df = dequantify_df(df).convert_dtypes()
    if vault.hive_bucket is not None:
        osc.fast_pandas_ingest_via_hive(
            new_df,
            vault.engine,
            vault.catalog,  # FIXME: this can be `None` when osc-ingest-tools 0.5.3 is released
            vault.schema,
            tablename,
            vault.hive_bucket,
            vault.hive_catalog,
            vault.hive_schema,
            partition_columns=["year"] if "year" in new_df.columns else [],
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
            con=vault.engine,
            schema=vault.schema,
            if_exists="append",
            index=False,
            method=osc.TrinoBatchInsert(batch_size=5000, verbose=verbose),
        )


# When reading SQL tables to import into DataFrames, it is up to the user to preserve {COL}, {COL}_units pairings so they can be reconstructed.
# If the user does a naive "select * from ..." this happens naturally.
# We can give a warning when we see a resulting dataframe that could have, but does not have, unit information properly integrated.  But
# fixing the query on the fly becomes difficult when we consider the fully complexity of parsing and
# rewriting SQL queries to put the units columns in the correct locations.
# (i.e., properly in the principal SELECT clause (which can have arbitrarily complex terms), not confused by FROM, WHERE, GROUP BY, ORDER BY, etc.)


def read_quantified_sql(
    sql: str,
    tablename: Union[str, None],
    engine: sqlalchemy.Engine,
    schemaname: Optional[str] = "",
    index_col: Optional[Union[List[str], str, None]] = None,
) -> pd.DataFrame:
    # read columns normally...this will be missing any unit-related information
    sql_df = pd.read_sql(sql, engine, index_col)
    if tablename:
        qres = osc._do_sql(f"describe {schemaname}.{tablename}", engine, verbose=False)
        # tabledesc will be a list of tuples (column, type, extra, comment)
        colnames = [x[0] for x in qres]
        # if the query requests columns that don't otherwise bring unit information along with them, get that information too
        extra_unit_columns = [
            (i, f"{col}_units")
            for i, col in enumerate(sql_df.columns)
            if f"{col}_units" not in sql_df.columns and f"{col}_units" in colnames
        ]
        if extra_unit_columns:
            extra_unit_columns_positions = [
                (i, extra_unit_columns[i][0], extra_unit_columns[i][1])
                for i in range(len(extra_unit_columns))
            ]
            for col_tuple in extra_unit_columns_positions:
                logger.error(
                    f"Missing units column '{col_tuple[2]}' after original column '{sql_df.columns[col_tuple[1]]}' (should be column #{col_tuple[0]+col_tuple[1]+1} in new query)"  # noqa: E501
                )
            raise ValueError
    return requantify_df(sql_df).convert_dtypes()


# Basic Corp Data Asumptions
#   5 year historical EI (else we presume single year is constant backward and forward)
#   5 year historical Production (else we presume single year is constant backward and forward)
#   5 year historical Emissions (else we presume single year is constant backward and forward)
#   We can infer one of the above from the other two (simple maths)
#   The above tables identify the scope(s) to which they apply (S1, S2, S12, S3, S123) and data source (e.g. 'rmi_20211120')

# Basic Benchmark Data Assumptions
#   EI for a given scope
#   Production defined in terms of growth (or negative growth) on a rolling basis
# (so 0.05, -0.04) would mean 5% growth followed by 4% negative growth for a total of 0.8%
#   Benchmarks are named (e.g., 'OECM')


class VaultProviderProductionBenchmark(ProductionBenchmarkDataProvider):
    def __init__(
        self,
        vault: VaultInstance,
        benchmark_name: str,
        prod_df: pd.DataFrame = pd.DataFrame(),
        column_config: Type[ColumnsConfig] = ColumnsConfig,
    ):
        """As an alternative to using FastAPI interfaces, this creates an interface allowing access to Production benchmark data via the Data Vault.
        :param vault: the Data Vault instance
        :param benchmark_name: the table name of the benchmark (in Trino)
        :param production_benchmarks: List of IBenchmarkScopes
        :param column_config: An optional ColumnsConfig object containing relevant variable names
        """
        super().__init__()
        self._v = vault
        self._benchmark_name = benchmark_name
        self.column_config = column_config
        if prod_df.empty:
            self._own_data = False
            # unstack and reshape what we read from SQL
            prod_df = read_quantified_sql(
                f"select sector, region, year, production, production_units from {self._benchmark_name}",
                None,
                self._v.engine,
                index_col=["sector", "region", "year"],
            )
            prod_df["scope"] = EScope.AnyScope
            self._prod_df = prod_df.set_index("scope", append=True).unstack(level=2)
        else:
            self._own_data = True
            self._prod_df = prod_df
            df = prod_df.stack(level=0).to_frame("production").reset_index()
            df.scope = df.scope.map(lambda x: x.name)
            create_vault_table_from_df(df, benchmark_name, self._v)

    def benchmark_changed(
        self, new_projected_production: ProductionBenchmarkDataProvider
    ) -> bool:
        # The Data Vault does not keep its own copies of benchmarks
        return False

    # Production benchmarks are dimensionless, relevant for AnyScope
    def _get_projected_production(
        self, scope: EScope = EScope.AnyScope
    ) -> pd.DataFrame:
        """Converts IProductionBenchmarkScopes into dataframe for a scope
        :param scope: a scope
        :return: a pint[dimensionless] pd.DataFrame
        """
        return self._prod_df

    def get_company_projected_production(
        self, company_sector_region_scope: pd.DataFrame
    ) -> pd.DataFrame:
        """Get the projected productions for list of companies
        :param company_sector_region_scope: DataFrame with at least the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.SECTOR, ColumnsConfig.REGION, ColumnsConfig.SCOPE
        :return: DataFrame of projected productions for [base_year through 2050]
        """
        if self._prod_df.empty:
            # breakpoint()
            raise
            # select company_id, year, production_by_year, production_by_year_units from itr_production_data where company_id='US00130H1059' order by year;
        else:
            from ..utils import get_benchmark_projections

            company_benchmark_projections = get_benchmark_projections(
                self._prod_df, company_sector_region_scope
            )

        company_production = company_sector_region_scope.set_index(
            self.column_config.SCOPE, append=True
        )[self.column_config.BASE_YEAR_PRODUCTION]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # We have to use lambda function here because company_production is heterogeneous, not a PintArray
            nan_production = company_production.map(lambda x: ITR.isna(x))
            if nan_production.any():
                # If we don't have valid production data for base year, we get back a nan result that's a pain to debug, so nag here
                logger.error(
                    f"these companies are missing production data: {nan_production[nan_production].index.get_level_values(0).to_list()}"
                )
            # We transpose the operation so that Pandas is happy to preserve the dtype integrity of the column
            company_projected_productions_t = company_benchmark_projections.T.mul(
                company_production, axis=1
            )
            return company_projected_productions_t.T


class VaultProviderIntensityBenchmark(IntensityBenchmarkDataProvider):
    def __init__(
        self,
        vault: VaultInstance,
        benchmark_name: str,
        ei_df_t: pd.DataFrame = pd.DataFrame(),
        benchmark_temperature: delta_degC_Quantity = Q_(1.5, "delta_degC"),
        benchmark_global_budget: EmissionsQuantity = Q_(396, "Gt CO2e"),
        is_AFOLU_included: bool = False,
        production_centric: bool = False,
        column_config: Type[ColumnsConfig] = ColumnsConfig,
        projection_controls: ProjectionControls = ProjectionControls(),
        # What to do about **kwargs?
    ):
        """As an alternative to using FastAPI interfaces, this creates an interface allowing access to Emission Intensity benchmark data via the Data Vault.
        :param vault: the Data Vault instance
        :param benchmark_name: the table name of the benchmark (in Trino)
        :param production_centric: FIXME
        :param ei_df_t: FIXME
        :param column_config: An optional ColumnsConfig object containing relevant variable names
        :param projection_controls: Projection Controls set the target BASE_YEAR, START_YEAR, and END_YEAR parameters of the model
        """
        self._v = vault
        self._benchmark_name = benchmark_name
        self.projection_controls = projection_controls
        if ei_df_t.empty:
            self._own_data = False
            # unstack and reshape what we read from SQL
            self._EI_df_t = (
                read_quantified_sql(
                    f"select sector, region, scope, year, intensity, intensity_units from {self._benchmark_name}",
                    None,
                    self._v.engine,
                    index_col=["sector", "region", "scope", "year"],
                )
                .unstack(level="year")
                .T
            )
            self._EI_df_t = ITR.data.osc_units.asPintDataFrame(self._EI_df_t)
            ei_bm_parameters = read_quantified_sql(
                "select benchmark_temp, benchmark_temp_units, global_budget, global_budget_units, is_AFOLU_included, production_centric"
                f" from {self._benchmark_name} limit 1",
                None,
                self._v.engine,
            )
            super().__init__(
                ei_bm_parameters["benchmark_temp"].squeeze(),
                ei_bm_parameters["global_budget"].squeeze(),
                ei_bm_parameters["is_AFOLU_included"].squeeze(),
            )
            self.production_centric = ei_bm_parameters["production_centric"].squeeze()
        else:
            super().__init__(
                benchmark_temperature,
                benchmark_global_budget,
                is_AFOLU_included,
            )  # type: ignore
            self._own_data = True
            self._EI_df_t = ei_df_t
            self.production_centric = production_centric
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Quieting warnings due to https://github.com/hgrecco/pint/issues/1897
                df = ei_df_t.T.stack(level=0).to_frame("intensity").reset_index()
            df.scope = df.scope.map(lambda x: x.name)
            df["global_budget"] = benchmark_global_budget
            df["benchmark_temp"] = benchmark_temperature
            df["is_AFOLU_included"] = is_AFOLU_included
            df["production_centric"] = production_centric
            create_vault_table_from_df(df, benchmark_name, self._v)

    def get_scopes(self) -> List[EScope]:
        scopes = self._EI_df_t.columns.get_level_values("scope").unique()
        return scopes.tolist()

    def benchmarks_changed(
        self, new_projected_ei: IntensityBenchmarkDataProvider
    ) -> bool:
        assert hasattr(new_projected_ei, "_EI_df_t")
        return self._EI_df_t.compare(new_projected_ei._EI_df_t).empty

    def prod_centric_changed(
        self, new_projected_ei: IntensityBenchmarkDataProvider
    ) -> bool:
        prev_prod_centric = self.production_centric
        next_prod_centric = False
        assert hasattr(new_projected_ei, "_EI_benchmarks")
        if getattr(new_projected_ei._EI_benchmarks, "S1S2", None):
            next_prod_centric = new_projected_ei._EI_benchmarks[
                "S1S2"
            ].production_centric
        return prev_prod_centric != next_prod_centric

    def is_production_centric(self) -> bool:
        """Returns True if benchmark is "production_centric" (as defined by OECM)"""
        return self.production_centric

    def _get_intensity_benchmarks(
        self,
        company_sector_region_scope: Optional[pd.DataFrame] = None,
        scope_to_calc: Optional[EScope] = None,
    ) -> pd.DataFrame:
        """Overrides subclass method
        returns dataframe of all EI benchmarks if COMPANY_SECTOR_REGION_SCOPE is None.  Otherwise
        returns a Dataframe with intensity benchmarks per company_id given a region and sector.
        :param company_sector_region_scope: DataFrame indexed by ColumnsConfig.COMPANY_ID
        with at least the following columns: ColumnsConfig.SECTOR, ColumnsConfig.REGION, and ColumnsConfig.SCOPE
        :return: A DataFrame with company and intensity benchmarks; rows are calendar years, columns are company data
        """
        if company_sector_region_scope is None:
            return self._EI_df_t
        sec_reg_scopes = company_sector_region_scope[["sector", "region", "scope"]]
        if scope_to_calc is not None:
            sec_reg_scopes = sec_reg_scopes[sec_reg_scopes.scope.eq(scope_to_calc)]
        sec_reg_scopes_mi = pd.MultiIndex.from_frame(sec_reg_scopes).unique()
        bm_proj_t = self._EI_df_t.loc[
            range(
                self.projection_controls.BASE_YEAR,
                self.projection_controls.TARGET_YEAR + 1,
            ),
            # Here we gather all requested combos as well as ensuring we have 'Global' regional coverage
            # for sector/scope combinations that arrive with unknown region values
            [
                col
                for col in sec_reg_scopes_mi.append(
                    pd.MultiIndex.from_frame(sec_reg_scopes.assign(region="Global"))
                ).unique()
                if col in self._EI_df_t.columns
            ],
        ]
        # This piece of work essentially does a column-based join (to avoid extra transpositions)
        result = pd.concat(
            [
                (
                    bm_proj_t[tuple(ser)].rename((idx, ser.iloc[2]))
                    if tuple(ser) in bm_proj_t
                    else (
                        bm_proj_t[ser_global].rename((idx, ser.iloc[2]))
                        if (
                            ser_global := (
                                ser.iloc[0],
                                "Global",
                                ser.iloc[2],
                            )
                        )
                        in bm_proj_t
                        else pd.Series()
                    )
                )
                for idx, ser in sec_reg_scopes.iterrows()
            ],
            axis=1,
        ).dropna(axis=1, how="all")
        result.columns = pd.MultiIndex.from_tuples(
            result.columns, names=["company_id", "scope"]
        )
        return result

    # SDA stands for Sectoral Decarbonization Approach; see https://sciencebasedtargets.org/resources/files/SBTi-Power-Sector-15C-guide-FINAL.pdf
    def get_SDA_intensity_benchmarks(
        self,
        company_info_at_base_year: pd.DataFrame,
        scope_to_calc: Optional[EScope] = None,
    ) -> pd.DataFrame:
        """Overrides subclass method
        returns a Dataframe with intensity benchmarks per company_id given a region and sector.
        :param company_info_at_base_year: DataFrame with at least the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.BASE_EI, ColumnsConfig.SECTOR, ColumnsConfig.REGION, ColumnsConfig.SCOPE
        :return: A DataFrame with company and SDA intensity benchmarks per calendar year per row
        """
        # To make pint happier, we do our math in columns that can be represented by PintArrays
        intensity_benchmarks_t = self._get_intensity_benchmarks(
            company_info_at_base_year, scope_to_calc
        )
        raise NotImplementedError
        decarbonization_paths_t = self._get_decarbonizations_paths(
            intensity_benchmarks_t
        )
        last_ei = intensity_benchmarks_t.loc[self.projection_controls.TARGET_YEAR]
        ei_base = intensity_benchmarks_t.loc[self.projection_controls.BASE_YEAR]
        df_t = decarbonization_paths_t.mul((ei_base - last_ei), axis=1)
        df_t = df_t.add(last_ei, axis=1)
        df_t.index.name = "year"
        idx = pd.Index.intersection(
            df_t.columns,
            pd.MultiIndex.from_arrays(
                [company_info_at_base_year.index, company_info_at_base_year.scope]
            ),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # pint units don't like being twisted from columns to rows, but it's ok
            df = df_t[idx].T
        return df


class VaultCompanyDataProvider(BaseCompanyDataProvider):
    def __init__(
        self,
        vault: VaultInstance,
        company_table: str,
        template_company_data: Union[TemplateProviderCompany, None],
        column_config: Type[ColumnsConfig] = ColumnsConfig,
        projection_controls: ProjectionControls = ProjectionControls(),
    ):
        """This class serves primarily for connecting to the ITR tool to the Data Vault via Trino.

        :param vault: the Data Vault instance
        :param company_table: the name of the Trino table that contains fundamental data for companies
        :param template_company_data: if not None, company data to ingest into company, target, and trajectory tables
        :param column_config: An optional ColumnsConfig object containing relevant variable names
        """
        super().__init__(
            companies=[]
            if template_company_data is None
            else template_company_data._companies,
            column_config=column_config,
        )
        self._v = vault
        self._company_table = company_table
        self._production_table = "! uninitialized table !"
        self._trajectory_table = "! uninitialized table !"

        if not template_company_data:
            self._own_data = False
            return
        if not template_company_data.own_data:
            # With our DataProvider object initialized, we'll use existing SQL table data for actual company data
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
                    "cash": "company_cash_equivalents",
                    "debt": "company_debt",
                },
            )
        )
        df["year"] = df.report_date.dt.year
        df.drop(columns="report_date", inplace=True)

        # ingest company data; no need to reset index because df_fundamentals also has "company_id" column
        create_vault_table_from_df(
            df,
            self._company_table,
            self._v,
            verbose=True,
        )

        # We don't have any target nor trajectory projections until we connect benchmark data via DataWarehouse

    def get_company_fundamentals(self, company_ids: List[str]) -> pd.DataFrame:
        """:param company_ids: A list of company IDs
        :return: A pandas DataFrame with company fundamental info per company (company_id is a column)
        """
        if self.own_data:
            return super().get_company_fundamentals(company_ids)

        company_ids_sql = ",".join([f"'{cid}'" for cid in company_ids])
        # FIXME: doesn't work with heterogeneous currencies as written

        df_dict: Dict[str, Future] = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Quieting warnings due to https://github.com/hgrecco/pint/issues/1897

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_df_var = {
                    executor.submit(
                        lambda: read_quantified_sql(
                            f"select * from {self._company_table} where company_id in ({company_ids_sql})",
                            None,
                            self._v.engine,
                            index_col=self.column_config.COMPANY_ID,
                        )
                    ): "df_fundamentals",
                    executor.submit(
                        lambda: pd.read_sql(
                            f"select company_id, production_by_year, production_by_year_units from {self._production_table}"
                            f" where year={self.projection_controls.BASE_YEAR} and company_id in ({company_ids_sql})",
                            self._v.engine,
                            index_col=self.column_config.COMPANY_ID,
                        )
                    ): "df_prod",
                    executor.submit(
                        lambda: pd.read_sql(
                            f"select company_id, ei_s1s2_by_year, ei_s1s2_by_year_units, ei_s3_by_year, ei_s3_by_year_units from {self._trajectory_table}"
                            f" where year={self.projection_controls.BASE_YEAR} and company_id in ({company_ids_sql})",
                            self._v.engine,
                            index_col=self.column_config.COMPANY_ID,
                        )
                    ): "df_ei",
                }
                for future in concurrent.futures.as_completed(future_to_df_var):
                    df_var = future_to_df_var[future]
                    try:
                        df_dict[df_var] = future.result()
                    except Exception as exc:
                        print("%r generated an exception: %s" % (df_var, exc))

            df_prod = cast(pd.DataFrame, df_dict["df_prod"])
            df_prod = df_prod.apply(
                lambda x: Q_(x.production_by_year, x.production_by_year_units), axis=1
            )
            df_prod.name = self.column_config.BASE_YEAR_PRODUCTION
            df_ei = cast(pd.DataFrame, df_dict["df_ei"])
            df_ei = df_ei.apply(
                lambda x: [
                    Q_(x.ei_s1s2_by_year, x.ei_s1s2_by_year_units),
                    Q_(x.ei_s3_by_year, x.ei_s3_by_year_units),
                ],
                axis=1,
                result_type="expand",
            )
        df_em = df_ei.mul(df_prod, axis=0).rename(
            columns={
                0: self.column_config.GHG_SCOPE12,
                1: self.column_config.GHG_SCOPE3,
            }
        )
        df = pd.concat([df_dict["df_fundamentals"], df_prod, df_em], axis=1)
        return df

    # The factors one would want to sum over companies for weighting purposes are:
    #   * market_cap_usd
    #   * enterprise_value_usd
    #   * assets_usd
    #   * revenue_usd
    #   * emissions

    def get_company_projected_trajectories(
        self, company_ids: List[str], year=None
    ) -> pd.DataFrame:
        """:param company_ids: A list of company IDs
        :param year: values for a specific year, or all years if None
        :return: A pandas DataFrame with projected intensity trajectories per company, indexed by company_id and scope
        """
        company_ids_sql = ",".join([f"'{cid}'" for cid in company_ids])
        if year is not None:
            sql = f"select * from {self._trajectory_table} where year={self.projection_controls.BASE_YEAR} and company_id in ({company_ids_sql})"
        else:
            sql = f"select * from {self._trajectory_table} where company_id in ({company_ids_sql})"
        df_ei = read_quantified_sql(
            sql, None, self._v.engine, index_col=self.column_config.COMPANY_ID
        )
        if year:
            df_ei.drop(columns="year", inplace=True)
        for col in df_ei.columns:
            if col.startswith("ei_") and col.endswith("_by_year"):
                df_ei.rename(columns={col: EScope[col[3:-8].upper()]}, inplace=True)
            elif col == "year":
                pass
            else:
                df_ei.drop(columns=col, inplace=True)
        df_ei.columns.name = "scope"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Quieting warnings due to https://github.com/hgrecco/pint/issues/1897
            if year is not None:
                df_ei = df_ei.unstack(level=0)
            else:
                df_ei = (
                    df_ei.set_index("year", append=True).stack(level=0).unstack(level=1)
                )
        return df_ei.reorder_levels(["company_id", "scope"])

    # TODO: make return value a Quantity (USD or CO2)
    def sum_over_companies(
        self,
        company_ids: List[str],
        year: int,
        factor: str,
        scope: EScope = EScope.S1S2,
    ) -> float:
        if factor == "enterprise_value_usd":
            factor_sum = "select sum(market_cap_usd + debt_usd - cash_usd)"
        elif factor == "emissions":
            if scope in [EScope.S1, EScope.S2, EScope.S3]:
                factor_sum = f"select sum(co2_{scope.name.lower()}_by_year)"
            elif scope == EScope.S1S2:
                factor_sum = "select sum(co2_s1_by_year+if(is_nan(co2_s2_by_year),0.0,co2_s2_by_year))"
            elif scope == EScope.S1S2S3:
                factor_sum = "select sum(co2_s1_by_year+if(is_nan(co2_s2_by_year),0.0,co2_s2_by_year)+if(is_nan(co2_s3_by_year),0.0,co2_s3_by_year))"
            else:
                raise ValueError(f"scope {scope} not supported")
        else:
            factor_sum = f"select sum({factor})"
        sql = (
            f"{factor_sum} as {factor}_sum from {self._v.schema}.{self._company_table}"
        )
        if year is not None:
            sql = f"{sql} where year={year}"
        qres = osc._do_sql(sql, self._v.engine, verbose=False)

        # qres[0] is the first row of the returned data; qres[0][0] is the first (and only) column of the row returned
        return qres[0][0]

    def compute_portfolio_weights(
        self,
        pa_temp_scores: pd.Series,
        year: int,
        factor: str,
        scope: EScope = EScope.S1S2,
    ) -> pd.Series:
        """Portfolio values could be position size, temperature scores, anything that can be multiplied by a factor.

        :param company_ids: A pd.Series of company IDs (ISINs)
        :return: A pd.Series weighted by the factor
        """
        from_sql = f"from {self._v.schema}.{self._company_table}"
        group_sql = "group by company_id"
        if factor == "company_evic":
            where_sql = ""
            factor_sql = (
                "select company_id, sum(company_market_cap + company_cash_equivalents)"
            )
        elif factor == "emissions":
            where_sql = f"where year = {year}"
            if scope in [EScope.S1, EScope.S2, EScope.S3]:
                factor_sql = f"select company_id, sum(co2_{scope.name.lower()}_by_year)"
            elif scope == EScope.S1S2:
                factor_sql = "select company_id, sum(co2_s1_by_year+if(is_nan(co2_s2_by_year),0.0,co2_s2_by_year))"
            elif scope == EScope.S1S2:
                factor_sql = "select company_id, sum(co2_s1_by_year+if(is_nan(co2_s2_by_year),0.0,co2_s2_by_year)+if(is_nan(co2_s3_by_year),0.0,co2_s3_by_year))"  # noqa: E501
            else:
                raise ValueError(f"scope {scope} not supported")
        else:
            factor_sql = f"select company_id, sum({factor})"
        qres = osc._do_sql(
            f"{factor_sql} as {factor} {from_sql} {where_sql} {group_sql}",
            self._v.engine,
            verbose=False,
        )
        weights = pd.Series(
            data=[s[1] for s in qres], index=[s[0] for s in qres], dtype=float
        )
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
    #     df = read_quantified_sql(sql, self._company_table, self._engine, self._schema)
    #     # df = df.drop(columns=['projected_targets', 'projected_intensities'])
    #     return df.set_index(self.column_config.COMPANY_ID)


# FIXME: Need to reshape the tables TARGET_DATA and TRAJECTORY_DATA so scope is a column and the EI data relates only to that scope (wide to long)
class DataVaultWarehouse(DataWarehouse):
    def __init__(
        self,
        vault: VaultInstance,
        company_data: VaultCompanyDataProvider,
        benchmark_projected_production: VaultProviderProductionBenchmark,
        benchmarks_projected_ei: VaultProviderIntensityBenchmark,
        estimate_missing_data: Optional[
            Callable[["DataWarehouse", ICompanyData], None]
        ] = None,
        itr_prefix: Optional[str] = os.environ.get("ITR_PREFIX", ""),
    ):
        """Construct Data Vault tables for cumulative emissions budgets, trajectories, and targets,
        which rely on trajectory and target projections from benchmark production and SDA pathways.

        Fundamentally: DataWarehouse(benchmark_ei, benchmark_prod, company_data)
            -> { production_data, trajectory_data,  target_data }
            -> { cumulative_budgets, cumulative_emissions }

        :param engine: The Sqlalchemy connector to the Data Vault
        :param company_data: as a VaultCompanyDataProvider, this provides both a reference to a fundamental
        company data table and data structures containing historic ESG data.  Trajectory and Target projections also get filled in here.
        :param benchmark_projected_production: A reference to the benchmark production table as well as data structures used by the Data Vault for projections
        :param benchmark_projected_ei: A reference to the benchmark emissions intensity table as well as data structures used by the Data Vault for projections
        :param estimate_missing_data: If provided, a function that can fill in missing S3 data (possibly by aligning to benchmark statistics)
        :param ingest_schema: The database schema where the Data Vault lives
        :param itr_prefix: A prefix for all tables so that different users can use the same schema without conflicts
        :param hive_bucket: :param hive_catalog: :param hive_schema: Optional paramters.  If given we attempt to use a
        fast Hive ingestion process.  Otherwise use default (and slow) Trino ingestion.
        """
        # This initialization step adds trajectory and target projections to `company_data`
        super().__init__(
            company_data=company_data,  # type: ignore
            benchmark_projected_production=benchmark_projected_production,
            benchmarks_projected_ei=benchmarks_projected_ei,
            estimate_missing_data=estimate_missing_data,
        )
        self._v = vault
        self._benchmark_prod_name = benchmark_projected_production._benchmark_name
        self._benchmarks_ei_name = benchmarks_projected_ei._benchmark_name
        self._company_table = company_data._company_table
        self._target_table = self._company_table.replace(
            "company_", "target_"
        )  # target_data
        self._trajectory_table = self._company_table.replace(
            "company_", "trajectory_"
        )  # trajectory_data
        self._production_table = self._company_table.replace(
            "company_", "production_"
        )  # production_data
        self._emissions_table = (
            f"{itr_prefix}cumulative_emissions"  # cumulative_emissions
        )
        self._budgets_table = f"{itr_prefix}cumulative_budgets"  # cumulative_budgets
        self._overshoot_table = f"{itr_prefix}overshoot_ratios"  # overshoot_ratios
        self._tempscore_table = f"{itr_prefix}temperature_scores"  # temperature_scores

        if not company_data.own_data:
            for slot in ["_production_table", "_target_table", "_trajectory_table"]:
                setattr(self.company_data, slot, getattr(self, slot))
            return

        assert benchmark_projected_production.own_data

        # Calculate base production data (and base emissions)
        company_idx, sector_data, region_data, prod_data = zip(
            *[
                (c.company_id, c.sector, c.region, c.base_year_production)
                for c in company_data._companies
                if c.company_id in company_data.get_company_ids()
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Quieting warnings due to https://github.com/hgrecco/pint/issues/1897
            df = pd.DataFrame(
                data={
                    "sector": sector_data,
                    "region": region_data,
                    "scope": [EScope.AnyScope] * len(company_idx),
                    "base_year_production": prod_data,
                },
                index=pd.Index(company_idx, name="company_id"),
            ).drop_duplicates()
            company_info_at_base_year = df[
                ~df["base_year_production"].map(lambda x: pd.isna(x))
            ]
        projected_production = (
            benchmark_projected_production.get_company_projected_production(
                company_info_at_base_year
            ).droplevel("scope")
        )
        projected_production.columns.name = "year"

        productions_and_projections = {
            self._production_table: projected_production.stack(level="year")
            .to_frame(name="production_by_year")
            .reset_index()
        }

        # If we have company data, we need to compute trajectories and targets
        projection_slots = ["_target_table", "_trajectory_table"]

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
                ei_data = pd.concat(
                    [ei_dict[scope] for scope in EScope.get_scopes()], axis=1
                ).reset_index()
                ei_data.columns = ["year"] + [
                    f"ei_{scope.lower()}_by_year" for scope in EScope.get_scopes()
                ]
                df = pd.DataFrame(
                    data=[
                        [
                            company.company_name,
                            "",
                            company.company_id,
                            company.sector,
                            company.region,
                        ]
                    ]
                    * len(ei_data.index),
                    columns=[
                        "company_name",
                        "company_lei",
                        "company_id",
                        "sector",
                        "region",
                    ],
                )
                projection_dfs.append(pd.concat([df, ei_data], axis=1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Quieting warnings due to https://github.com/hgrecco/pint/issues/1897
                df2 = pd.concat(projection_dfs).reset_index(drop=True)
            productions_and_projections[getattr(self, projection_slots[i])] = df2
            # Inject projection tablename into company data (needed for `get_company_projected_trajectories`
            setattr(
                self.company_data,
                projection_slots[i],
                getattr(self, projection_slots[i]),
            )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Ingest productions, trajectories, and targets
            future_to_drop_or_ingest = {
                executor.submit(
                    create_vault_table_from_df,
                    df=df,
                    tablename=tablename,
                    vault=self._v,
                    verbose=True,
                ): tablename
                for tablename, df in productions_and_projections.items()
            }
            # Drop cumulative emissions tables so they get recalculated
            future_to_drop_or_ingest[
                executor.submit(
                    lambda: osc._do_sql(
                        f"drop table if exists {self._v.schema}.{self._emissions_table}",
                        self._v.engine,
                        verbose=False,
                    )
                )
            ] = "emissions table"
            future_to_drop_or_ingest[
                executor.submit(
                    lambda: osc._do_sql(
                        f"drop table if exists {self._v.schema}.{self._budgets_table}",
                        self._v.engine,
                        verbose=False,
                    )
                )
            ] = "budgets table"
            for future in concurrent.futures.as_completed(future_to_drop_or_ingest):
                tablename = future_to_drop_or_ingest[future]
                try:
                    _ = future.result()
                except Exception as exc:
                    print("%r generated an exception: %s" % (tablename, exc))
        assert isinstance(self.company_data, VaultCompanyDataProvider)
        self.company_data._production_table = self._production_table

        # The DataVaultWarehouse provides three calculations per company (using SQL code rather than Python):
        #    * Cumulative trajectory of emissions
        #    * Cumulative target of emissions
        #    * Cumulative budget of emissions (separately for each benchmark)

        emissions_from_tables = f"""
    {self._v.schema}.{self._company_table} C
         join {self._v.schema}.{self._production_table} P on P.company_id=C.company_id
         left join {self._v.schema}.{self._trajectory_table} EI on EI.company_id=C.company_id and EI.year=P.year and EI.ei_SCOPE_by_year is not NULL
         left join {self._v.schema}.{self._target_table} ET on ET.company_id=C.company_id and ET.year=P.year and ET.ei_SCOPE_by_year is not NULL
"""

        create_emissions_sql = f"create table {self._v.schema}.{self._emissions_table} with (format = 'ORC', partitioning = array['scope']) as"
        emissions_scope_sql = "UNION ALL".join(
            [
                f"""
select C.company_name, C.company_id, '{self._v.schema}' as source, P.year,
       sum(EI.ei_{scope}_by_year * P.production_by_year) over (partition by C.company_id order by P.year) as cumulative_trajectory,
       if (EI.ei_{scope}_by_year_units is NULL, 't CO2e',
           regexp_replace(regexp_replace(concat(EI.ei_{scope}_by_year_units, ' * ', P.production_by_year_units),
                                         '{re_simplify_units_both}', ''), '{re_simplify_units_one}', '')) as cumulative_trajectory_units,
       sum(ET.ei_{scope}_by_year * P.production_by_year) over (partition by C.company_id order by P.year) as cumulative_target,
       if (ET.ei_{scope}_by_year_units is NULL, 't CO2e',
           regexp_replace(regexp_replace(concat(ET.ei_{scope}_by_year_units, ' * ', P.production_by_year_units),
                          '{re_simplify_units_both}', ''), '{re_simplify_units_one}', '')) as cumulative_target_units,
       '{scope.upper()}' as scope
from {emissions_from_tables.replace('SCOPE', scope)}
"""
                for scope in map(str.lower, EScope.get_scopes())
            ]
        )
        qres = osc._do_sql(
            f"{create_emissions_sql} {emissions_scope_sql}",
            self._v.engine,
            verbose=True,
        )
        assert len(qres) and len(qres[0]) and qres[0][0] > 0

        # base_year_scale = trajectory / budget at base year (a scalar)
        # scaled cumulative budget = base_year_scale * cumulative budget (a time series)

        budgets_from_productions = f"""
create table {self._v.schema}.{self._budgets_table} with (
    format = 'ORC',
    partitioning = array['scope']
) as
with P_BY as (select distinct company_id,
                     first_value(year) over (partition by company_id order by year) as base_year,
                     first_value(production_by_year) over (partition by company_id order by year) as production_by_year
              from {self._v.schema}.{self._production_table})
select C.company_name, C.company_id, '{self._v.schema}' as source, P.year,  -- FIXME: should have scenario_name and year released
       B.global_budget, B.global_budget_units, B.benchmark_temp, B.benchmark_temp_units,
       sum(B.intensity * P.production_by_year) over (partition by C.company_id, B.scope order by P.year) as cumulative_budget,
       regexp_replace(regexp_replace(concat(B.intensity_units, ' * ', P.production_by_year_units),
                                            '{re_simplify_units_both}', ''), '{re_simplify_units_one}', '') as cumulative_budget_units,
       CE_BY.cumulative_trajectory/(B_BY.intensity * P_BY.production_by_year)
             * sum(B.intensity * P.production_by_year) over (partition by C.company_id, B.scope order by P.year) as cumulative_scaled_budget,
       CE_BY.cumulative_trajectory_units as cumulative_scaled_budget_units,
       B.scope
from {self._v.schema}.{self._company_table} C
     join P_BY on P_BY.company_id=C.company_id
     join {self._v.schema}.{self._production_table} P on P.company_id=C.company_id
     join {self._v.schema}.{self._benchmarks_ei_name} B on P.year=B.year and C.sector=B.sector and B.region=if(C.region in ('North America', 'Europe'), C.region, 'Global')  # noqa: E501
     join {self._v.schema}.{self._emissions_table} CE on CE.company_id=C.company_id and B.scope=CE.scope and CE.year=P.year
     join {self._v.schema}.{self._emissions_table} CE_BY on CE_BY.company_id=C.company_id and CE_BY.scope=B.scope and CE_BY.year=P_BY.base_year
     join {self._v.schema}.{self._benchmarks_ei_name} B_BY on B.scope=B_BY.scope and B.region=B_BY.region and B.sector=B_BY.sector and B_BY.year=P_BY.base_year
"""

        qres = osc._do_sql(budgets_from_productions, self._v.engine, verbose=True)
        assert len(qres) and len(qres[0]) and qres[0][0] > 0

    def quant_init(
        self,
        vault: VaultInstance,
        company_data: Union[VaultCompanyDataProvider, None],
        itr_prefix: str = os.environ.get("ITR_PREFIX", ""),
    ):
        # The Quant users of the DataVaultWarehouse produces two calculations per company:
        #    * Target and Trajectory overshoot ratios
        #    * Temperature Scores
        self._v = vault

        qres = osc._do_sql(
            f"drop table if exists {self._v.schema}.{self._overshoot_table}",
            self._v.engine,
            verbose=False,
        )
        df_ratios = read_quantified_sql(
            f"""
select E.company_name, E.company_id, '{self._v.schema}' as source, B.year, -- FIXME: should have scenario_name and year released
       B.global_budget, B.global_budget_units, B.benchmark_temp, B.benchmark_temp_units,
       E.cumulative_trajectory/B.cumulative_budget as trajectory_overshoot_ratio,
       concat(E.cumulative_trajectory_units, ' / (', B.cumulative_budget_units, ')') as trajectory_overshoot_ratio_units,
       E.cumulative_target/B.cumulative_budget as target_overshoot_ratio,
       concat(E.cumulative_target_units, ' / (', B.cumulative_budget_units, ')') as target_overshoot_ratio_units,
       B.scope
from {self._v.schema}.{self._emissions_table} E
     join {self._v.schema}.{self._budgets_table} B on E.company_id=B.company_id and E.scope=B.scope and E.year=B.year
""",
            None,
            self._v.engine,
            index_col=(["company_id", "scope", "year"]),
        )
        assert isinstance(df_ratios["global_budget"].dtype, PintType)
        assert isinstance(df_ratios["benchmark_temp"].dtype, PintType)
        df_ratios["trajectory_overshoot_ratio"] = df_ratios[
            "trajectory_overshoot_ratio"
        ].astype("pint[dimensionless]")
        df_ratios["target_overshoot_ratio"] = df_ratios[
            "target_overshoot_ratio"
        ].astype("pint[dimensionless]")
        create_vault_table_from_df(
            df_ratios.reset_index()[df_ratios.index.names + df_ratios.columns.tolist()],
            self._overshoot_table,
            self._v,
            verbose=True,
        )

        qres = osc._do_sql(  # noqa F841
            f"drop table if exists {self._v.schema}.{self._tempscore_table}",
            self._v.engine,
            verbose=False,
        )
        qres = osc._do_sql(  # noqa F841
            f"""
create table {self._v.schema}.{self._tempscore_table} with (
    format = 'ORC',
    partitioning = array['scope']
) as
select R.company_name, R.company_id, '{self._v.schema}' as source, R.year, -- FIXME: should have scenario_name and year released
       R.benchmark_temp + R.global_budget * (R.trajectory_overshoot_ratio-1) * 2.2/3664.0 as trajectory_temperature_score,
       R.benchmark_temp_units as trajectory_temperature_score_units,
       R.benchmark_temp + R.global_budget * (R.target_overshoot_ratio-1) * 2.2/3664.0 as target_temperature_score,
       R.benchmark_temp_units as target_temperature_score_units,
       R.scope
from {self._v.schema}.{itr_prefix}overshoot_ratios R
""",
            self._v.engine,
            verbose=True,
        )

    def get_preprocessed_company_data(
        self, company_ids: List[str]
    ) -> List[ICompanyAggregates]:
        raise NotImplementedError

    def get_pa_temp_scores(
        self,
        probability: float,
        company_ids: List[str],
        scope: EScope = EScope.S1S2,
        year: int = 2050,
    ) -> pd.Series:
        if probability < 0 or probability > 1:
            raise ValueError(
                f"probability value {probability} outside range [0.0, 1.0]"
            )
        temp_scores = read_quantified_sql(
            "select company_id, scope, target_temperature_score, target_temperature_score_units, trajectory_temperature_score, trajectory_temperature_score_units, year"  # noqa: E501
            f" from {self._tempscore_table}  where scope='{scope.name}' and year={year}",
            None,
            self._v.engine,
            index_col=["company_id", "scope"],
        )
        # We may have company_ids in our portfolio not in our database, and vice-versa.
        # Return proper pa_temp_scores for what we can find, and np.nan for those we cannot
        retval = pd.Series(
            data=None,
            index=pd.MultiIndex.from_tuples(
                [(company_id, scope.name) for company_id in company_ids],
                names=["company_id", "scope"],
            ),
            name="temp_score",
            dtype="pint[delta_degC]",
        )
        retval.loc[retval.index.intersection(temp_scores.index)] = (
            temp_scores.target_temperature_score * probability
            + temp_scores.trajectory_temperature_score * (1 - probability)
        )
        return retval
