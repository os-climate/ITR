import concurrent.futures
import json
import os
import pathlib
import re
from typing import Tuple

import numpy as np
import osc_ingest_trino as osc
import pandas as pd
import pytest
import trino
from sqlalchemy.engine import create_engine
from sqlalchemy.exc import ProgrammingError

import ITR  # noqa F401
from ITR import data_dir as json_data_dir
from ITR.configs import ColumnsConfig, ProjectionControls, TemperatureScoreConfig
from ITR.data.base_providers import (
    BaseProviderIntensityBenchmark,
    BaseProviderProductionBenchmark,
)
from ITR.data.data_warehouse import DataWarehouse
from ITR.data.osc_units import Q_, requantify_df_from_columns
from ITR.data.template import TemplateProviderCompany
from ITR.data.vault_providers import (
    DataVaultWarehouse,
    VaultCompanyDataProvider,
    VaultInstance,
    VaultProviderIntensityBenchmark,
    VaultProviderProductionBenchmark,
    read_quantified_sql,
    requantify_df,
)
from ITR.interfaces import (
    EScope,
    ETimeFrames,
    ICompanyData,
    IEIBenchmarkScopes,
    IProductionBenchmarkScopes,
    PortfolioCompany,
)
from ITR.portfolio_aggregation import PortfolioAggregationMethod
from ITR.temperature_score import TemperatureScore

xlsx_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs")

# If there's no credientials file, this fails silently without raising
osc.load_credentials_dotenv()

ingest_catalog = "osc_datacommons_dev"
ingest_schema = "demo_dv"
itr_prefix = "itr_"

try:
    # sqlstring = "trino://{user}@{host}:{port}/".format(
    #     user=os.environ["TRINO_USER_USER1"],
    #     host=os.environ["TRINO_HOST"],
    #     port=os.environ["TRINO_PORT"],
    # )
    # sqlargs = {
    #     "auth": trino.auth.JWTAuthentication(os.environ["TRINO_PASSWD_USER1"]),
    #     "http_scheme": "https",
    #     "catalog": ingest_catalog,
    #     "schema": demo_schema,
    # }
    # engine_init = create_engine(sqlstring, connect_args=sqlargs)
    # print("connecting with engine " + str(engine_init))
    # connection_init = engine_init.connect()

    engine_init = osc.attach_trino_engine(verbose=True, catalog=ingest_catalog, schema=ingest_schema)
except KeyError:
    if pytest.__version__ < "3.0.0":
        pytest.skip()
    else:
        pytestmark = pytest.mark.skip
        pytest.skip("skipping vault because Trino auth breaks CI/CD", allow_module_level=True)

# bucket must be configured with credentials for trino, and accessible to the hive catalog
# You may need to use a different prefix here depending on how you name your credentials.env variables
try:
    hive_bucket = osc.attach_s3_bucket("S3_OSCCL2")
    hive_catalog = "osc_datacommons_hive_ingest"
    hive_schema = "ingest"
except KeyError:
    hive_bucket = None
    hive_catalog = None
    hive_schema = None

company_data_path = os.path.join(xlsx_data_dir, "20230106 ITR V2 Sample Data.xlsx")


def _get_base_prod(filename: str) -> BaseProviderProductionBenchmark:
    # load production benchmarks
    with open(os.path.join(json_data_dir, filename)) as json_file:
        parsed_json = json.load(json_file)
    prod_bms = IProductionBenchmarkScopes.model_validate(parsed_json)
    return BaseProviderProductionBenchmark(production_benchmarks=prod_bms)


def _get_base_ei(filename: str) -> BaseProviderIntensityBenchmark:
    # load intensity benchmarks
    with open(os.path.join(json_data_dir, filename)) as json_file:
        parsed_json = json.load(json_file)
    ei_bms = IEIBenchmarkScopes.model_validate(parsed_json)
    return BaseProviderIntensityBenchmark(EI_benchmarks=ei_bms)


@pytest.fixture(scope="session")
def base_benchmarks() -> Tuple[BaseProviderProductionBenchmark, BaseProviderIntensityBenchmark]:
    benchmark_dict = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_benchmark = {
            executor.submit(_get_base_prod, filename="benchmark_production_OECM.json"): "base_production_bm",
            executor.submit(_get_base_ei, filename="benchmark_EI_OECM_S3.json"): "base_EI_bm",
        }
        for future in concurrent.futures.as_completed(future_to_benchmark):
            benchmark_name = future_to_benchmark[future]
            try:
                benchmark_dict[benchmark_name] = future.result()
            except Exception as exc:
                print("%r generated an exception: %s" % (benchmark_name, exc))
    return (benchmark_dict["base_production_bm"], benchmark_dict["base_EI_bm"])


@pytest.fixture(scope="session")
def base_company_data() -> TemplateProviderCompany:
    company_data = TemplateProviderCompany(company_data_path, projection_controls=ProjectionControls())
    return company_data


@pytest.fixture(scope="session")
def base_warehouse(base_company_data, base_benchmarks) -> DataWarehouse:
    prod_bm, EI_bm = base_benchmarks
    warehouse = DataWarehouse(
        base_company_data,
        prod_bm,
        EI_bm,
        estimate_missing_data=DataWarehouse.estimate_missing_s3_data,
    )
    return warehouse


@pytest.fixture(scope="session")
def vault() -> VaultInstance:
    instance = VaultInstance(
        engine=engine_init,
        schema=ingest_schema,
        hive_bucket=hive_bucket,
        hive_catalog=hive_catalog,
        hive_schema=hive_schema,
    )
    return instance


@pytest.fixture(scope="session")
def vault_benchmarks_from_base(
    vault, base_benchmarks
) -> Tuple[VaultProviderProductionBenchmark, VaultProviderIntensityBenchmark]:
    base_prod_bm, base_EI_bm = base_benchmarks

    vault_dict = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_vault = {
            executor.submit(
                VaultProviderProductionBenchmark,
                vault=vault,
                benchmark_name=f"{itr_prefix}benchmark_prod",
                prod_df=base_prod_bm._prod_df,
            ): "vault_prod_bm",
            executor.submit(
                VaultProviderIntensityBenchmark,
                vault=vault,
                benchmark_name=f"{itr_prefix}benchmark_ei",
                ei_df_t=base_EI_bm._EI_df_t,
                benchmark_temperature=base_EI_bm.benchmark_temperature,
                benchmark_global_budget=base_EI_bm.benchmark_global_budget,
                is_AFOLU_included=base_EI_bm.is_AFOLU_included,
                production_centric=base_EI_bm.is_production_centric(),
            ): "vault_EI_bm",
        }
        for future in concurrent.futures.as_completed(future_to_vault):
            vault_name = future_to_vault[future]
            try:
                vault_dict[vault_name] = future.result()
            except Exception as exc:
                print("%r generated an exception: %s" % (vault_name, exc))

    assert vault_dict["vault_prod_bm"].own_data
    assert vault_dict["vault_EI_bm"].own_data
    return (vault_dict["vault_prod_bm"], vault_dict["vault_EI_bm"])


@pytest.fixture(scope="session")
def vault_warehouse_from_base(vault, vault_benchmarks_from_base, base_warehouse) -> DataVaultWarehouse:
    vault_company_data = VaultCompanyDataProvider(
        vault,
        company_table=f"{itr_prefix}company_data",
        # We don't use `base_company_data` because we need projections created by `base_warehouse`
        template_company_data=base_warehouse.company_data,
    )
    vault_prod_bm_from_base, vault_EI_bm_from_base = vault_benchmarks_from_base
    vault_warehouse = DataVaultWarehouse(
        vault,
        vault_company_data,
        vault_prod_bm_from_base,
        vault_EI_bm_from_base,
        estimate_missing_data=DataWarehouse.estimate_missing_s3_data,
        itr_prefix=itr_prefix,
    )
    return vault_warehouse


@pytest.fixture(scope="session")
def vault_benchmarks(vault, request) -> Tuple[VaultProviderProductionBenchmark, VaultProviderIntensityBenchmark]:
    try:
        vault_prod_bm = VaultProviderProductionBenchmark(
            vault,
            benchmark_name=f"{itr_prefix}benchmark_prod",
            prod_df=pd.DataFrame(),
        )

        vault_EI_bm = VaultProviderIntensityBenchmark(
            vault,
            benchmark_name=f"{itr_prefix}benchmark_ei",
        )
    except ProgrammingError:
        vault_prod_bm_from_base, vault_ei_bm_from_base = request.getfixturevalue("vault_benchmarks_from_base")

        vault_prod_bm = VaultProviderProductionBenchmark(
            vault,
            benchmark_name=f"{itr_prefix}benchmark_prod",
        )

        vault_EI_bm = VaultProviderIntensityBenchmark(
            vault,
            benchmark_name=f"{itr_prefix}benchmark_ei",
        )

    assert not vault_prod_bm.own_data
    assert not vault_EI_bm.own_data
    return (vault_prod_bm, vault_EI_bm)


@pytest.fixture(scope="session")
def vault_warehouse(vault, vault_benchmarks) -> DataVaultWarehouse:
    # This creates a wrapper around what should be an existing data in the Data Vault.
    # If no such data exists, it will fail

    vault_company_data = VaultCompanyDataProvider(
        vault,
        company_table=f"{itr_prefix}company_data",
        # We don't use `base_company_data` because base_warehouse creates projections we need
        template_company_data=None,
    )
    vault_production_bm, vault_ei_bm = vault_benchmarks

    # Verify that we have all the tables we need
    tablenames = [
        "company_data",
        "benchmark_prod",
        "benchmark_ei",
        "production_data",
        "trajectory_data",
        "target_data",
        "cumulative_emissions",
        "cumulative_budgets",
    ]
    sql_counts = ",".join(
        [f"{tablename}_cnt as (select count (*) as cnt from {itr_prefix}{tablename})" for tablename in tablenames]
    )
    sql_sums = "+".join([f"{tablename}_cnt.cnt" for tablename in tablenames])
    sql_joins = ",".join([f"{tablename}_cnt" for tablename in tablenames])
    # One N-clause statement executes about N times faster than N individual checks
    qres = osc._do_sql(f"with {sql_counts} select {sql_sums} from {sql_joins}", engine=vault.engine, verbose=True)
    warehouse = DataVaultWarehouse(
        vault,
        company_data=vault_company_data,
        benchmark_projected_production=vault_production_bm,
        benchmarks_projected_ei=vault_ei_bm,
        itr_prefix=itr_prefix,
    )
    return warehouse


@pytest.mark.parametrize(
    "base_warehouse_x,vault_warehouse_x",
    [
        ("base_warehouse", "vault_warehouse_from_base"),
        ("base_warehouse", "vault_warehouse"),
    ],
)
def test_warehouse(base_warehouse_x: DataWarehouse, vault_warehouse_x: DataVaultWarehouse, request) -> None:
    base_warehouse_x = request.getfixturevalue(base_warehouse_x)
    vault_warehouse_x = request.getfixturevalue(vault_warehouse_x)
    vault = vault_warehouse_x._v
    base_company = next(iter(base_warehouse_x.company_data.get_company_data(["US00130H1059"])))
    vault_company_data = vault_warehouse_x.company_data
    assert base_company.projected_targets.S1S2 is not None
    company_0_id, company_0_s1s2_ser = (
        base_company.company_id,
        base_company.projected_targets.S1S2.projections,
    )

    ser_from_vault = read_quantified_sql(
        f"select year, ei_s1s2_by_year, ei_s1s2_by_year_units from {vault_warehouse_x._target_table} where company_id='{company_0_id}' order by year",
        vault_warehouse_x._target_table,
        vault.engine,
        vault.schema,
        index_col="year",
    ).squeeze()
    assert company_0_s1s2_ser.compare(ser_from_vault).empty

    company_info_at_base_year = vault_company_data.get_company_intensity_and_production_at_base_year(
        [company_0_id],
    )
    assert base_warehouse_x.benchmark_projected_production is not None
    projected_production = base_warehouse_x.benchmark_projected_production.get_company_projected_production(
        company_info_at_base_year
    )
    company_proj_production = projected_production.loc[:, EScope.S1S2, :].stack(level=0)
    company_proj_production.index.set_names(["company_id", "year"], inplace=True)
    company_proj_production.name = "production_by_year"
    ser_from_vault = read_quantified_sql(
        f"select company_id, year, production_by_year, production_by_year_units from {vault_warehouse_x._production_table} where company_id='{company_0_id}' order by year",
        vault_warehouse_x._production_table,
        vault.engine,
        vault.schema,
        index_col=["company_id", "year"],
    ).squeeze()
    assert company_proj_production.compare(ser_from_vault).empty

    company_0_cumulative_em = DataWarehouse._get_cumulative_emissions(
        base_warehouse_x.company_data.get_company_projected_targets([company_0_id]),
        projected_production,
    ).stack(level=0)
    company_0_cumulative_em.index.set_names(["company_id", "scope", "year"], inplace=True)
    company_0_cumulative_em.name = "cumulative_target"

    df_from_vault = read_quantified_sql(
        f"select company_id, scope, year, cumulative_target, cumulative_target_units from {vault_warehouse_x._emissions_table} where company_id='{company_0_id}' order by year",
        f"{itr_prefix}cumulative_emissions",
        vault.engine,
        vault.schema,
        index_col=["company_id", "scope", "year"],
    ).astype("pint[t CO2e]")

    assert (
        company_0_cumulative_em.loc[:, EScope.S1S2, :]
        .pint.m.round(2)
        .compare(df_from_vault.loc[:, "S1S2", :].squeeze().pint.m.round(2))
        .empty
    )

    qres = osc._do_sql("show tables", engine=engine_init)
    assert len(qres) >= 8
    qres = osc._do_sql(f"select count (*) from {itr_prefix}benchmark_prod", engine=engine_init, verbose=True)
    assert len(qres) > 0 and qres[0] == (2208,)
    qres = osc._do_sql(f"select count (*) from {itr_prefix}benchmark_ei", engine=engine_init, verbose=True)
    assert len(qres) > 0 and qres[0] == (11040,)
    qres = osc._do_sql(f"select count (*) from {itr_prefix}company_data", engine=engine_init, verbose=True)
    assert len(qres) > 0 and len(qres[0]) > 0 and qres[0][0] > 0


def test_tempscore_from_base(base_warehouse) -> None:
    df_portfolio = pd.read_excel(company_data_path, sheet_name="Portfolio").iloc[[0]]

    for i, col in enumerate(df_portfolio.columns):
        if col.startswith("investment_value"):
            if match := re.match(r".*\[([A-Z]{3})\]", col, re.I):
                df_portfolio.rename(columns={col: "investment_value"}, inplace=True)
                df_portfolio["investment_value"] = df_portfolio["investment_value"].astype(f"pint[{match.group(1)}]")
    companies = ITR.utils.dataframe_to_portfolio(df_portfolio)
    temperature_score = TemperatureScore(time_frames=[ETimeFrames.LONG], scopes=EScope.get_result_scopes())
    df = temperature_score.calculate(
        data_warehouse=base_warehouse,
        portfolio=companies,
        target_probability=0.5,
    )
    assert df[df.scope == EScope.S1S2].temperature_score.pint.m.round(2).item() == 2.41


def test_temp_scores(vault_warehouse) -> None:
    engine_quant = osc.attach_trino_engine(verbose=True, catalog=ingest_catalog, schema=ingest_schema)

    quant_vault = VaultInstance(
        engine=engine_quant,
        schema=ingest_schema,
        hive_bucket=hive_bucket,
        hive_catalog=hive_catalog,
        hive_schema=hive_schema,
    )
    vault_warehouse.quant_init(quant_vault, company_data=None, itr_prefix=itr_prefix)
    df_portfolio = pd.read_excel(company_data_path, sheet_name="Portfolio", index_col="company_id")

    for i, col in enumerate(df_portfolio.columns):
        if col.startswith("investment_value"):
            if match := re.match(r".*\[([A-Z]{3})\]", col, re.I):
                df_portfolio.rename(columns={col: "investment_value"}, inplace=True)
                df_portfolio["investment_value"] = df_portfolio["investment_value"].astype(f"pint[{match.group(1)}]")
    df_portfolio["pa_score"] = (
        vault_warehouse.get_pa_temp_scores(
            probability=0.5, company_ids=df_portfolio.index.values, scope=EScope.S1S2, year=2050
        )
        .droplevel("scope")
        .astype("pint[delta_degC]")
    )
    assert df_portfolio.loc["US00130H1059"].pa_score.m.round(2).item() == 2.41
