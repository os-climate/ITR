import json
import unittest
import os
import pathlib
import pandas as pd
from numpy.testing import assert_array_equal
import ITR

from ITR.portfolio_aggregation import PortfolioAggregationMethod
from ITR.temperature_score import TemperatureScore
from ITR.configs import ColumnsConfig, TemperatureScoreConfig
from ITR.data.data_warehouse import DataWarehouse
from ITR.data.vault_providers import VaultCompanyDataProvider, VaultProviderProductionBenchmark, \
    VaultProviderIntensityBenchmark, DataVaultWarehouse
from ITR.interfaces import ICompanyData, EScope, ETimeFrames, PortfolioCompany, IEmissionIntensityBenchmarkScopes, \
    IProductionBenchmarkScopes

import trino
import osc_ingest_trino as osc
from sqlalchemy.engine import create_engine

ingest_schema = 'demo'

dotenv_dir = os.environ.get('CREDENTIAL_DOTENV_DIR', os.environ.get('PWD', '/opt/app-root/src'))
dotenv_path = pathlib.Path(dotenv_dir) / 'credentials.env'
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path,override=True)

sqlstring = 'trino://{user}@{host}:{port}/'.format(
    user = os.environ['TRINO_USER'],
    host = os.environ['TRINO_HOST'],
    port = os.environ['TRINO_PORT']
)
sqlargs = {
    'auth': trino.auth.JWTAuthentication(os.environ['TRINO_PASSWD']),
    'http_scheme': 'https',
    'catalog': 'osc_datacommons_dev',
    'schema': ingest_schema,
}
engine_init = create_engine(sqlstring, connect_args = sqlargs)
print("connecting with engine " + str(engine_init))
connection_init = engine_init.connect()
qres = engine_init.execute(f"create schema if not exists {ingest_schema}")
print(qres.fetchall())

class TestVaultProvider(unittest.TestCase):
    """
    Test the Value provider
    """

    def setUp(self) -> None:
        self.root = os.path.dirname(os.path.abspath(__file__))
        self.benchmark_prod_json = os.path.join(self.root, "inputs", "json", "benchmark_production_OECM.json")
        self.benchmark_EI_json = os.path.join(self.root, "inputs", "json", "benchmark_EI_OECM.json")

        # load production benchmarks
        with open(self.benchmark_prod_json) as json_file:
            parsed_json = json.load(json_file)
        prod_bms = IProductionBenchmarkScopes.parse_obj(parsed_json)
        self.vault_production_bm = VaultProviderProductionBenchmark(engine_init, ingest_schema, benchmark_name="benchmark_prod", production_benchmarks=prod_bms)

        # load intensity benchmarks
        with open(self.benchmark_EI_json) as json_file:
            parsed_json = json.load(json_file)
        ei_bms = IEmissionIntensityBenchmarkScopes.parse_obj(parsed_json)
        self.vault_EI_bm = VaultProviderIntensityBenchmark(engine_init, ingest_schema, benchmark_name="benchmark_ei", EI_benchmarks=ei_bms)

        # load company data
        # TODO: ISIC code should read as int, not float
        self.vault_company_data = VaultCompanyDataProvider(engine_init, ingest_schema, "rmi_company_data")

        self.vault_warehouse = DataVaultWarehouse(engine_init, ingest_schema, self.vault_company_data, self.vault_production_bm, self.vault_EI_bm)

    def test_N0_projections(self):
        sqlstring = 'trino://{user}@{host}:{port}/'.format(
            user = os.environ['TRINO_USER_USER1'],
            host = os.environ['TRINO_HOST'],
            port = os.environ['TRINO_PORT']
        )
        sqlargs = {
            'auth': trino.auth.JWTAuthentication(os.environ['TRINO_PASSWD_USER1']),
            'http_scheme': 'https'
        }
        engine_dev = create_engine(sqlstring, connect_args = sqlargs)
        print("connecting with engine " + str(engine_dev))
        connection_dev = engine_dev.connect()
        # Show projections for emissions trajectories, production, and emission targets (N0 only)
        # Show cumulative emissions (trajectory, target) and budget (N1 can also see)
        pass

        def test_N1_temp_scores(self):
            sqlstring = 'trino://{user}@{host}:{port}/'.format(
                user = os.environ['TRINO_USER_USER2'],
                host = os.environ['TRINO_HOST'],
                port = os.environ['TRINO_PORT']
            )
            sqlargs = {
                'auth': trino.auth.JWTAuthentication(os.environ['TRINO_PASSWD_USER2']),
                'http_scheme': 'https'
            }
            engine_quant = create_engine(sqlstring, connect_args = sqlargs)
            print("connecting with engine " + str(engine_quant))
            connection_quant = engine_quant.connect()
            # Show cumulative emissions (trajectory, target) and budget (N1 can see)
            # Show overshoot ratios (trajectory, target) (N1 can see)
            # Show trajectory and target temp scores (N2 can also see)
            pass

        def test_N2_portfolio(self):
            sqlstring = 'trino://{user}@{host}:{port}/'.format(
                user = os.environ['TRINO_USER_USER3'],
                host = os.environ['TRINO_HOST'],
                port = os.environ['TRINO_PORT']
            )
            sqlargs = {
                'auth': trino.auth.JWTAuthentication(os.environ['TRINO_PASSWD_USER3']),
                'http_scheme': 'https'
            }
            engine_user = create_engine(sqlstring, connect_args = sqlargs)
            print("connecting with engine " + str(engine_user))
            connection_user = engine_user.connect()
            # Show weighted temp score over portfolio (N2 can see)
            # Different weighting types give different coefficients
            pass
