import json
import os
import pathlib
import unittest

import osc_ingest_trino as osc
import pandas as pd
import pytest
import trino
from ITR_examples import data_dir as xlsx_data_dir

import ITR  # noqa F401
from ITR import data_dir as json_data_dir
from ITR.configs import ColumnsConfig, ProjectionControls, TemperatureScoreConfig
from ITR.data.base_providers import (
    BaseProviderIntensityBenchmark,
    BaseProviderProductionBenchmark,
)
from ITR.data.data_warehouse import DataWarehouse
from ITR.data.template import TemplateProviderCompany
from ITR.data.vault_providers import (
    DataVaultWarehouse,
    VaultCompanyDataProvider,
    VaultProviderIntensityBenchmark,
    VaultProviderProductionBenchmark,
    read_quantified_sql,
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

ingest_catalog = "osc_datacommons_dev"
ingest_schema = "demo_dv"
itr_prefix = "itr_"

osc.load_credentials_dotenv()

# bucket must be configured with credentials for trino, and accessible to the hive catalog
# You may need to use a different prefix here depending on how you name your credentials.env variables
hive_bucket = osc.attach_s3_bucket("S3_OSCCL2")

hive_catalog = "osc_datacommons_hive_ingest"
hive_schema = "ingest"

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


class Vault_and_Python_Init:
    def __init__(self) -> None:
        # load production benchmarks
        with open(os.path.join(json_data_dir, "benchmark_production_OECM.json")) as json_file:
            parsed_json = json.load(json_file)
        prod_bms = IProductionBenchmarkScopes.model_validate(parsed_json)
        self.python_production_bm = BaseProviderProductionBenchmark(
            production_benchmarks=prod_bms,
        )
        self.vault_production_bm = VaultProviderProductionBenchmark(
            engine_init,
            benchmark_name=f"{itr_prefix}benchmark_prod",
            production_benchmarks=prod_bms,
            hive_bucket=hive_bucket,
            hive_catalog=hive_catalog,
            hive_schema=hive_schema,
        )

        # load intensity benchmarks
        with open(os.path.join(json_data_dir, "benchmark_EI_OECM_S3.json")) as json_file:
            parsed_json = json.load(json_file)
        ei_bms = IEIBenchmarkScopes.model_validate(parsed_json)
        self.python_EI_bm = BaseProviderIntensityBenchmark(
            EI_benchmarks=ei_bms,
        )
        self.vault_EI_bm = VaultProviderIntensityBenchmark(
            engine_init,
            benchmark_name=f"{itr_prefix}benchmark_ei",
            EI_benchmarks=ei_bms,
            hive_bucket=hive_bucket,
            hive_catalog=hive_catalog,
            hive_schema=hive_schema,
        )

        # Use template data for initial corp data.  We will pass that to the DataVaultWarehouse constructor.
        company_data_path = os.path.join(xlsx_data_dir, "20230106 ITR V2 Sample Data.xlsx")
        template_company_data = TemplateProviderCompany(company_data_path, projection_controls=ProjectionControls())

        self.python_company_data = template_company_data

        # Ingest `template_company_data` table into company_data table.
        # We cannot calculate target nor trajectory projections,
        # nor production nor emissions projections until the DataVaultWarehouse step
        self.vault_company_data = VaultCompanyDataProvider(
            engine=engine_init,
            company_table=f"{itr_prefix}company_data",
            hive_bucket=hive_bucket,
            hive_catalog=hive_catalog,
            hive_schema=hive_schema,
            template_company_data=template_company_data,
        )


vault_and_python = Vault_and_Python_Init()


class TestVaultProvider(unittest.TestCase):
    """
    Test the Value provider
    """

    def setUp(self) -> None:
        self.python_production_bm = vault_and_python.python_production_bm
        self.python_EI_bm = vault_and_python.python_EI_bm
        self.python_company_data = vault_and_python.python_company_data
        self.vault_production_bm = vault_and_python.vault_production_bm
        self.vault_EI_bm = vault_and_python.vault_EI_bm
        self.vault_company_data = vault_and_python.vault_company_data

    def test_init(self) -> None:
        self.python_warehouse = DataWarehouse(
            self.python_company_data,
            self.python_production_bm,
            self.python_EI_bm,
            estimate_missing_data=DataWarehouse.estimate_missing_s3_data,
        )

        self.vault_warehouse = DataVaultWarehouse(
            engine_init,
            self.vault_company_data,
            self.vault_production_bm,
            self.vault_EI_bm,
            estimate_missing_data=DataWarehouse.estimate_missing_s3_data,
            itr_prefix=itr_prefix,
            hive_bucket=hive_bucket,
            hive_catalog=hive_catalog,
            hive_schema=hive_schema,
        )

        assert self.vault_company_data._companies[0].projected_targets.S1S2 is not None
        company_0_id, company_0_s1s2_ser = (
            self.vault_company_data._companies[0].company_id,
            self.vault_company_data._companies[0].projected_targets.S1S2.projections,
        )
        ser_from_vault = read_quantified_sql(
            f"select year, ei_s1s2_by_year, ei_s1s2_by_year_units from {self.vault_company_data._target_table} where company_id='{company_0_id}' order by year",
            self.vault_company_data._target_table,
            self.vault_company_data._schema,
            engine_init,
            index_col="year",
        ).squeeze()
        assert company_0_s1s2_ser.compare(ser_from_vault).empty

        company_info_at_base_year = self.python_company_data.get_company_intensity_and_production_at_base_year(
            [company_0_id],
        )
        assert self.python_warehouse.benchmark_projected_production is not None
        projected_production = self.python_warehouse.benchmark_projected_production.get_company_projected_production(
            company_info_at_base_year
        )
        company_proj_production = projected_production.loc[:, EScope.S1S2, :].stack(level=0)
        company_proj_production.index.set_names(["company_id", "year"], inplace=True)
        company_proj_production.name = "production_by_year"
        ser_from_vault = read_quantified_sql(
            f"select company_id, year, production_by_year, production_by_year_units from {self.vault_company_data._production_table} where company_id='{company_0_id}' order by year",
            self.vault_company_data._production_table,
            self.vault_company_data._schema,
            engine_init,
            index_col=["company_id", "year"],
        ).squeeze()
        assert company_proj_production.compare(ser_from_vault).empty

        company_0_cumulative_em = DataWarehouse._get_cumulative_emissions(
            self.python_warehouse.company_data.get_company_projected_targets([company_0_id]),
            projected_production,
        ).stack(level=0)
        company_0_cumulative_em.index.set_names(["company_id", "scope", "year"], inplace=True)
        company_0_cumulative_em.name = "cumulative_target"

        df_from_vault = read_quantified_sql(
            f"select company_id, scope, year, cumulative_target, cumulative_target_units from {itr_prefix}cumulative_emissions where company_id='{company_0_id}' order by year",
            f"{itr_prefix}cumulative_emissions",
            self.vault_company_data._schema,
            engine_init,
            index_col=["company_id", "scope", "year"],
        ).astype("pint[t CO2e]")

        assert (
            company_0_cumulative_em.loc[:, EScope.S1S2, :]
            .pint.m.round(2)
            .compare(df_from_vault.loc[:, "S1S2", :].squeeze().pint.m.round(2))
            .empty
        )

        qres = osc._do_sql("show tables", engine=engine_init)
        assert len(qres) == 8
        qres = osc._do_sql(f"select count (*) from {itr_prefix}benchmark_prod", engine=engine_init, verbose=True)
        assert len(qres) > 0 and qres[0] == (2208,)
        qres = osc._do_sql(f"select count (*) from {itr_prefix}benchmark_ei", engine=engine_init, verbose=True)
        assert len(qres) > 0 and qres[0] == (11040,)
        qres = osc._do_sql(f"select count (*) from {itr_prefix}company_data", engine=engine_init, verbose=True)
        assert len(qres) > 0 and len(qres[0]) > 0 and qres[0][0] > 0

    def test_N0_projections(self):
        sqlstring = "trino://{user}@{host}:{port}/".format(
            user=os.environ["TRINO_USER_USER1"],
            host=os.environ["TRINO_HOST"],
            port=os.environ["TRINO_PORT"],
        )
        sqlargs = {
            "auth": trino.auth.JWTAuthentication(os.environ["TRINO_PASSWD_USER1"]),
            "http_scheme": "https",
        }
        engine_dev = create_engine(sqlstring, connect_args=sqlargs)
        print("connecting with engine " + str(engine_dev))
        connection_dev = engine_dev.connect()
        # Show projections for emissions trajectories, production, and emission targets (N0 only)
        # Show cumulative emissions (trajectory, target) and budget (N1 can also see)
        pass

    def test_N1_temp_scores(self):
        sqlstring = "trino://{user}@{host}:{port}/".format(
            user=os.environ["TRINO_USER_USER2"],
            host=os.environ["TRINO_HOST"],
            port=os.environ["TRINO_PORT"],
        )
        sqlargs = {
            "auth": trino.auth.JWTAuthentication(os.environ["TRINO_PASSWD_USER2"]),
            "http_scheme": "https",
        }
        engine_quant = create_engine(sqlstring, connect_args=sqlargs)
        print("connecting with engine " + str(engine_quant))
        connection_quant = engine_quant.connect()
        # Show cumulative emissions (trajectory, target) and budget (N1 can see)
        # Show overshoot ratios (trajectory, target) (N1 can see)
        # Show trajectory and target temp scores (N2 can also see)
        pass

    def test_N2_portfolio(self):
        sqlstring = "trino://{user}@{host}:{port}/".format(
            user=os.environ["TRINO_USER_USER3"],
            host=os.environ["TRINO_HOST"],
            port=os.environ["TRINO_PORT"],
        )
        sqlargs = {
            "auth": trino.auth.JWTAuthentication(os.environ["TRINO_PASSWD_USER3"]),
            "http_scheme": "https",
        }
        engine_user = create_engine(sqlstring, connect_args=sqlargs)
        print("connecting with engine " + str(engine_user))
        connection_user = engine_user.connect()
        # Show weighted temp score over portfolio (N2 can see)
        # Different weighting types give different coefficients
        pass


if __name__ == "__main__":
    test = TestVaultProvider()
    test.setUp()
    test.test_init()
