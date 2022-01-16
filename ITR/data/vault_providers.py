import os
import pathlib
from dotenv import load_dotenv

# Load some standard environment variables from a dot-env file, if it exists.
# If no such file can be found, does not fail, and so allows these environment vars to
# be populated in some other way
dotenv_dir = os.environ.get('CREDENTIAL_DOTENV_DIR', os.environ.get('HOME', '/opt/app-root/src'))
dotenv_path = pathlib.Path(dotenv_dir) / 'credentials.env'
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path,override=True)

import trino
import osc_ingest_trino as osc
import sqlalchemy

ingest_catalog = 'osc_datacommons_dev'

import pandas as pd
from typing import List, Type
from ITR.configs import ColumnsConfig, TemperatureScoreConfig
from ITR.data.data_providers import CompanyDataProvider, ProductionBenchmarkDataProvider, \
    IntensityBenchmarkDataProvider, EmissionIntensityProjector
from ITR.data.data_warehouse import DataWarehouse
from ITR.interfaces import ICompanyData, EScope, IProductionBenchmarkScopes, IEmissionIntensityBenchmarkScopes, \
    IBenchmark, ICompanyAggregates

import boto3
s3 = boto3.resource(
    service_name="s3",
    endpoint_url=os.environ["S3_DEV_ENDPOINT"],
    aws_access_key_id=os.environ["S3_DEV_ACCESS_KEY"],
    aws_secret_access_key=os.environ["S3_DEV_SECRET_KEY"],
)
trino_bucket = s3.Bucket(os.environ["S3_DEV_BUCKET"])

# TODO handling of scopes in benchmarks

# TODO handle ways to append information (from other providers, other benchmarks, new scope info, new corp data updates, etc)

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

class VaultCompanyDataProvider(CompanyDataProvider):
    """
    This class serves primarily for connecting to the ITR tool to the Data Vault via Trino.

    :param company_table: the name of the Trino table that contains fundamental data for companies
    :param target_table: the name of the Trino table that contains company (emissions intensity) target data (and possibly historical data)
    :param trajectory_table: the name of the Trino table that contains company (emissions intensity) historical data (and possibly trajectory data)
    :param company_schema: the name of the schema where the company_table is found
    :param column_config: An optional ColumnsConfig object containing relevant variable names
    :param tempscore_config: An optional TemperatureScoreConfig object containing temperature scoring settings
    """

    def __init__(self,
                 engine: sqlalchemy.engine.base.Engine,
                 company_table: str,
                 target_table: str = None,
                 trajectory_table: str = None,
                 company_schema: str = None,
                 column_config: Type[ColumnsConfig] = ColumnsConfig,
                 tempscore_config: Type[TemperatureScoreConfig] = TemperatureScoreConfig):
        super().__init__()
        self._engine = engine
        self._schema = company_schema or engine.dialect.default_schema_name or 'demo'
        self._company_table = company_table
        self.column_config = column_config
        self.temp_config = tempscore_config
        # Validate and complete the projected trajectories
        self._intensity_table = company_table.replace('_company_', '_intensity_')
        self._trajectory_table = company_table.replace('_company_', '_trajectory_')
        self._production_table = company_table.replace('_company_', '_production_')
        self._emissions_table = company_table.replace('_company_', '_emissions_')
        companies_without_projections = self._engine.execute(f"""
select C.company_name, C.company_id from {self._schema}.{self._company_table} C left join {self._schema}.{self._intensity_table} EI on EI.company_name=C.company_name
where co2_intensity_target_by_year is NULL
""").fetchall()
        assert len(companies_without_projections)==0, f"Provide either historic emissions data or projections for companies with IDs {companies_without_projections.company_id}"

    # The factors one would want to sum over companies for weighting purposes are:
    #   * market_cap_usd
    #   * enterprise_value_usd
    #   * assets_usd
    #   * revenue_usd
    #   * emissions
    
    # TODO: make return value a Quantity (USD or CO2)
    def sum_over_companies(self, company_ids: List[str], year: int, factor: str, scope: EScope = EScope.S1S2) -> float:
        if factor=='enterprise_value_usd':
            qres = self._engine.execute(f"select sum(market_cap_usd + debt_usd - cash_usd) as {factor}_sum from {self._schema}.{self._company_table} where year={year}")
        elif factor=='emissions':
            # TODO: properly interpret SCOPE parameter
            assert scope==EScope.S1S2
            qres = self._engine.execute(f"select sum(co2_target_by_year) as {factor}_sum from {self._schema}.{self._emissions_table} where year={year}")
        else:
            qres = self._engine.execute(f"select sum({factor}) as {factor}_sum from {self._schema}.{self._company_table} where year={year}")
        sres = qres.fetchall()
        # sres[0] is the first row of the returned data; sres[0][0] is the first (and only) column of the row returned
        return sres[0][0]

    def compute_portfolio_weights(self, pa_temp_scores: pd.Series, year: int, factor: str, scope: EScope = EScope.S1S2) -> pd.Series:
        """
        Portfolio values could be position size, temperature scores, anything that can be multiplied by a factor.

        :param company_ids: A pd.Series of company IDs (ISINs)
        :return: A pd.Series weighted by the factor
        """
        if factor=='company_evic':
            qres = self._engine.execute(f"select company_id, sum(company_market_cap + company_cash_equivalents) as {factor} from {self._schema}.{self._company_table} group by company_id")
        elif factor=='emissions':
            # TODO: properly interpret SCOPE parameter
            assert scope==EScope.S1S2
            qres = self._engine.execute(f"select company_id, sum(co2_target_by_year) as {factor} from {self._schema}.{self._emissions_table} where year={year} group by company_id")
        else:
            qres = self._engine.execute(f"select company_id, sum({factor}) as {factor} from {self._schema}.{self._company_table} group by company_id")
        sres = qres.fetchall()
        weights = pd.Series(data=[s[1] for s in sres], index=[s[0] for s in sres], dtype=float)
        weights = weights.loc[pa_temp_scores.index.intersection(weights.index)]
        weight_sum = weights.sum()
        return pa_temp_scores * weights / weight_sum


    def get_company_data(self, company_ids: List[str]) -> List[ICompanyData]:
        """
        Get all relevant data for a list of company ids. This method should return a list of ICompanyData
        instances.

        :param company_ids: A list of company IDs (ISINs)
        :return: A list containing the company data
        """
        raise NotImplementedError

    def get_value(self, company_ids: List[str], variable_name: str) -> pd.Series:
        """
        Gets the value of a variable for a list of companies ids
        :param company_ids: list of company ids
        :param variable_name: variable name of the projected feature
        :return: series of values
        """
        raise NotImplementedError

    def get_company_intensity_and_production_at_base_year(self, company_ids: List[str]) -> pd.DataFrame:
        """
        overrides subclass method
        :param: company_ids: list of company ids
        :return: DataFrame the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.GHG_S1S2, ColumnsConfig.BASE_EI, ColumnsConfig.SECTOR and
        ColumnsConfig.REGION
        """
        raise NotImplementedError

    def get_company_fundamentals(self, company_ids: List[str]) -> pd.DataFrame:
        """
        :param company_ids: A list of company IDs
        :return: A pandas DataFrame with company fundamental info per company
        """
        or_clause = ' or '.join([f"company_id = '{c}'" for c in company_ids])
        sql = f"select * from {self._schema}.{self._company_table} where {or_clause}"
        df = pd.read_sql(sql, self._engine)
        # df = df.drop(columns=['projected_targets', 'projected_intensities'])
        return df

    def get_company_projected_intensities(self, company_ids: List[str]) -> pd.DataFrame:
        """
        :param company_ids: A list of company IDs
        :return: A pandas DataFrame with projected intensities per company
        """
        raise NotImplementedError

    def get_company_projected_targets(self, company_ids: List[str]) -> pd.DataFrame:
        """
        :param company_ids: A list of company IDs
        :return: A pandas DataFrame with projected targets per company
        """
        raise NotImplementedError


benchmark_scopes = ['S1S2', 'S3', 'S1S2S3']

class VaultProviderProductionBenchmark(ProductionBenchmarkDataProvider):

    def __init__(self,
                 engine: sqlalchemy.engine.base.Engine,
                 benchmark_name: str,
                 production_benchmarks: IProductionBenchmarkScopes,
                 ingest_schema: str = None,
                 column_config: Type[ColumnsConfig] = ColumnsConfig,
                 tempscore_config: Type[TemperatureScoreConfig] = TemperatureScoreConfig):
        """
        Base provider that relies on pydantic interfaces. Default for FastAPI usage
        :param benchmark_name: the table name of the benchmark (in Trino)
        :param production_benchmarks: List of IBenchmarkScopes
        :param column_config: An optional ColumnsConfig object containing relevant variable names
        :param tempscore_config: An optional TemperatureScoreConfig object containing temperature scoring settings
        """
        super().__init__(production_benchmarks=production_benchmarks,
                         column_config=column_config,
                         tempscore_config=tempscore_config)
        self._engine=engine
        self._schema = ingest_schema or engine.dialect.default_schema_name or 'demo'
        self.benchmark_name=benchmark_name
        qres = self._engine.execute(f"drop table if exists itr_mdt.{benchmark_name}")
        qres = self._engine.execute(f"drop table if exists {self._schema}.{benchmark_name}")
        qres.fetchall()
        df = pd.DataFrame()
        for scope in benchmark_scopes:
            if production_benchmarks.dict()[scope] is None:
                continue
            for benchmark in production_benchmarks.dict()[scope]['benchmarks']:
                # ??? I don't understand why I cannot use benchmark.projections and must use benchmark['projections']
                bdf = pd.DataFrame.from_dict({r['year']: [r['value'], benchmark['region'], benchmark['sector'], scope] for r in benchmark['projections']},
                                             columns=['production', 'region', 'sector', 'scope'],
                                             orient='index')
                df = pd.concat([df, bdf])
        df.reset_index(inplace=True)
        df.rename(columns={'index':'year'}, inplace=True)
        df = df.convert_dtypes()
        osc.ingest_unmanaged_parquet(df, self._schema, benchmark_name, trino_bucket)
        qres = engine.execute(osc.unmanaged_parquet_tabledef(df, ingest_catalog, self._schema, benchmark_name, trino_bucket))
        print(qres.fetchall())

    def get_company_projected_production(self, ghg_scope12: pd.DataFrame) -> pd.DataFrame:
        """
        get the projected productions for list of companies in ghg_scope12
        :param ghg_scope12: DataFrame with at least the following columns :
        ColumnsConfig.COMPANY_ID,ColumnsConfig.GHG_SCOPE12, ColumnsConfig.SECTOR and ColumnsConfig.REGION
        :return: DataFrame of projected productions for [base_year - base_year + 50]
        """
        benchmark_production_projections = self.get_benchmark_projections(ghg_scope12)
        return benchmark_production_projections.add(1).cumprod(axis=1).mul(
            ghg_scope12[self.column_config.GHG_SCOPE12].values, axis=0)

    def get_benchmark_projections(self, company_sector_region_info: pd.DataFrame,
                                  scope: EScope = EScope.S1S2) -> pd.DataFrame:
        """
        Overrides subclass method
        returns a Dataframe with production benchmarks per company_id given a region and sector.
        :param company_sector_region_info: DataFrame with at least the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.SECTOR and ColumnsConfig.REGION
        :param scope: a scope
        :return: A DataFrame with company and intensity benchmarks per calendar year per row
        """
        benchmark_projection = self._get_projected_production(scope)  # TODO optimize performance
        sectors = company_sector_region_info[self.column_config.SECTOR]
        regions = company_sector_region_info[self.column_config.REGION]
        benchmark_regions = regions.copy()
        mask = benchmark_regions.isin(benchmark_projection.reset_index()[self.column_config.REGION])
        benchmark_regions.loc[~mask] = "Global"

        benchmark_projection = benchmark_projection.loc[list(zip(benchmark_regions, sectors)),
                                                        range(self.temp_config.CONTROLS_CONFIG.base_year,
                                                              self.temp_config.CONTROLS_CONFIG.target_end_year + 1)]
        benchmark_projection.index = sectors.index

        return benchmark_projection


class VaultProviderIntensityBenchmark(IntensityBenchmarkDataProvider):
    def __init__(self,
                 engine: sqlalchemy.engine.base.Engine,
                 benchmark_name: str,
                 EI_benchmarks: IEmissionIntensityBenchmarkScopes,
                 ingest_schema: str = None,
                 column_config: Type[ColumnsConfig] = ColumnsConfig,
                 tempscore_config: Type[TemperatureScoreConfig] = TemperatureScoreConfig):
        super().__init__(EI_benchmarks.benchmark_temperature, EI_benchmarks.benchmark_global_budget,
                         EI_benchmarks.is_AFOLU_included)
        self._engine=engine
        self._schema = ingest_schema or engine.dialect.default_schema_name or 'demo'
        self.benchmark_name = benchmark_name
        osc.drop_unmanaged_table(ingest_catalog, self._schema, benchmark_name, engine, trino_bucket)
        osc.drop_unmanaged_data(self._schema, benchmark_name, trino_bucket)
        df = pd.DataFrame()
        for scope in benchmark_scopes:
            if EI_benchmarks.dict()[scope] is None:
                continue
            for benchmark in EI_benchmarks.dict()[scope]['benchmarks']:
                bdf = pd.DataFrame.from_dict({r['year']: [r['value'], benchmark['region'], benchmark['sector'], scope, EI_benchmarks.benchmark_global_budget, EI_benchmarks.benchmark_temperature] for r in benchmark['projections']},
                                   columns=['intensity', 'region', 'sector', 'scope', 'global_budget', 'benchmark_temp'],
                                            orient='index')
                # TODO: AFOLU correction
                df = pd.concat([df, bdf])
        df.reset_index(inplace=True)
        df.rename(columns={'index':'year'}, inplace=True)
        df = df.convert_dtypes()
        osc.ingest_unmanaged_parquet(df, self._schema, benchmark_name, trino_bucket)
        qres = engine.execute(osc.unmanaged_parquet_tabledef(df, ingest_catalog, self._schema, benchmark_name, trino_bucket))
        print(qres.fetchall())


    def get_SDA_intensity_benchmarks(self, company_info_at_base_year: pd.DataFrame) -> pd.DataFrame:
        """
        Overrides subclass method
        returns a Dataframe with intensity benchmarks per company_id given a region and sector.
        :param benchmark_name: the table name of the benchmark (in Trino)
        :param company_info_at_base_year: DataFrame with at least the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.BASE_EI ColumnsConfig.SECTOR and ColumnsConfig.REGION
        :return: A DataFrame with company and SDA intensity benchmarks per calendar year per row
        """
        intensity_benchmarks = self._get_intensity_benchmarks(company_info_at_base_year)
        decarbonization_paths = self._get_decarbonizations_paths(intensity_benchmarks)
        last_ei = intensity_benchmarks[self.temp_config.CONTROLS_CONFIG.target_end_year]
        ei_base = company_info_at_base_year[self.column_config.BASE_EI]

        return decarbonization_paths.mul((ei_base - last_ei), axis=0).add(last_ei, axis=0)

    def _get_decarbonizations_paths(self, intensity_benchmarks: pd.DataFrame) -> pd.DataFrame:
        """
        Overrides subclass method
        Returns a DataFrame with the projected decarbonization paths for the supplied companies in intensity_benchmarks.
        :param: A DataFrame with company and intensity benchmarks per calendar year per row
        :return: A pd.DataFrame with company and decarbonisation path s per calendar year per row
        """
        return intensity_benchmarks.apply(lambda row: self._get_decarbonization(row), axis=1)

    def _get_decarbonization(self, intensity_benchmark_row: pd.Series) -> pd.Series:
        """
        Overrides subclass method
        returns a Series with the decarbonization path for a benchmark.
        :param: A Series with company and intensity benchmarks per calendar year per row
        :return: A pd.Series with company and decarbonisation path s per calendar year per row
        """
        first_ei = intensity_benchmark_row[self.temp_config.CONTROLS_CONFIG.base_year]
        last_ei = intensity_benchmark_row[self.temp_config.CONTROLS_CONFIG.target_end_year]
        return intensity_benchmark_row.apply(lambda x: (x - last_ei) / (first_ei - last_ei))

    def _convert_benchmark_to_series(self, benchmark: IBenchmark) -> pd.Series:
        """
        extracts the company projected intensities or targets for a given scope
        :param feature: PROJECTED_EI or PROJECTED_TARGETS
        :param scope: a scope
        :return: pd.Series
        """
        return pd.Series({r.year: r.value for r in benchmark.projections}, name=(benchmark.region, benchmark.sector))

    def _get_projected_intensities(self, scope: EScope = EScope.S1S2) -> pd.Series:
        """
        Converts IBenchmarkScopes into dataframe for a scope
        :param scope: a scope
        :return: pd.Series
        """
        result = []
        for bm in self._EI_benchmarks.dict()[str(scope)]['benchmarks']:
            result.append(self._convert_benchmark_to_series(IBenchmark.parse_obj(bm)))
        df_bm = pd.DataFrame(result)
        df_bm.index.names = [self.column_config.REGION, self.column_config.SECTOR]

        return df_bm

    def _get_intensity_benchmarks(self, company_sector_region_info: pd.DataFrame,
                                  scope: EScope = EScope.S1S2) -> pd.DataFrame:
        """
        Overrides subclass method
        returns a Dataframe with production benchmarks per company_id given a region and sector.
        :param company_sector_region_info: DataFrame with at least the following columns :
        ColumnsConfig.COMPANY_ID, ColumnsConfig.SECTOR and ColumnsConfig.REGION
        :param scope: a scope
        :return: A DataFrame with company and intensity benchmarks per calendar year per row
        """
        benchmark_projection = self._get_projected_intensities(scope)  # TODO optimize performance
        sectors = company_sector_region_info[self.column_config.SECTOR]
        regions = company_sector_region_info[self.column_config.REGION]
        benchmark_regions = regions.copy()
        mask = benchmark_regions.isin(benchmark_projection.reset_index()[self.column_config.REGION])
        benchmark_regions.loc[~mask] = "Global"

        benchmark_projection = benchmark_projection.loc[list(zip(benchmark_regions, sectors)),
                                                        range(self.temp_config.CONTROLS_CONFIG.base_year,
                                                              self.temp_config.CONTROLS_CONFIG.target_end_year + 1)]
        benchmark_projection.index = sectors.index

        return benchmark_projection

class DataVaultWarehouse(DataWarehouse):

    def __init__(self,
                 engine: sqlalchemy.engine.base.Engine,
                 company_data: VaultCompanyDataProvider,
                 benchmark_projected_production: ProductionBenchmarkDataProvider,
                 benchmarks_projected_emissions_intensity: IntensityBenchmarkDataProvider,
                 ingest_schema: str = None,
                 column_config: Type[ColumnsConfig] = ColumnsConfig,
                 tempscore_config: Type[TemperatureScoreConfig] = TemperatureScoreConfig):
        super().__init__(company_data=company_data,
                         benchmark_projected_production=benchmark_projected_production,
                         benchmarks_projected_emissions_intensity=benchmarks_projected_emissions_intensity,
                         column_config=column_config,
                         tempscore_config=tempscore_config)
        self._engine=engine
        self._schema = ingest_schema or engine.dialect.default_schema_name or 'demo'
        # intensity_projections = pd.read_sql(f"select * from {self._schema}.{intensity_table}", self._engine)
        # intensity_projections['scope'] = 'S1+S2'
        # intensity_projections['source'] = self._schema
        
        # The DataVaultWarehouse provides three calculations per company:
        #    * Cumulative trajectory of emissions
        #    * Cumulative target of emissions
        #    * Cumulative budget of emissions (separately for each benchmark)
        for t in ['cumulative_emissions', 'cumulative_budget_1', 'overshoot_ratios', 'temperature_scores']:
            osc.drop_unmanaged_table(ingest_catalog, self._schema, t, engine, trino_bucket)
            osc.drop_unmanaged_data(self._schema, t, trino_bucket)

        qres = self._engine.execute(f"""
create table cumulative_emissions with (
    format = 'parquet',
    external_location = 's3a://{trino_bucket.name}/trino/{self._schema}/cumulative_emissions/'
) as
select C.company_name, C.company_id, '{company_data._schema}' as source, 'S1+S2' as scope,
       sum(ET.co2_intensity_trajectory_by_year * P.production_by_year) as cumulative_trajectory,
       sum(EI.co2_intensity_target_by_year * P.production_by_year) as cumulative_target
from {company_data._schema}.{company_data._company_table} C
     join {company_data._schema}.{company_data._production_table} P on P.company_name=C.company_name
     join {company_data._schema}.{company_data._intensity_table} EI on EI.company_name=C.company_name and EI.year=P.year
     join {company_data._schema}.{company_data._trajectory_table} ET on ET.company_name=C.company_name and ET.year=P.year
group by C.company_name, C.company_id, '{company_data._schema}', 'S1+S2'
""")
        # Need to fetch so table created above is established before using in query below
        qres.fetchall()
        qres = self._engine.execute(f"""
create table cumulative_budget_1 with (
    format = 'parquet',
    external_location = 's3a://{trino_bucket.name}/trino/{self._schema}/cumulative_budget_1/'
) as
select C.company_name, C.company_id, '{company_data._schema}' as source, 'S1+S2' as scope, 'benchmark_1' as benchmark,
       B.global_budget, B.benchmark_temp,
       sum(B.intensity * P.production_by_year) as cumulative_budget
from {company_data._schema}.{company_data._company_table} C
     join {company_data._schema}.{company_data._production_table} P on P.company_name=C.company_name
     join {self._schema}.benchmark_ei B on P.year=B.year and C.region=B.region and C.sector=B.sector
group by C.company_name, C.company_id, '{company_data._schema}', 'S1+S2', 'benchmark_1', B.global_budget, B.benchmark_temp
""")
        # Need to fetch so table created above is established before using in query below
        qres.fetchall()
        qres = self._engine.execute(f"""
create table overshoot_ratios with (
    format = 'parquet',
    external_location = 's3a://{trino_bucket.name}/trino/{self._schema}/overshoot_ratios/'
) as
select E.company_name, E.company_id, '{company_data._schema}' as source, 'S1+S2' as scope, 'benchmark_1' as benchmark,
       B.global_budget, B.benchmark_temp,
       E.cumulative_trajectory/B.cumulative_budget as trajectory_overshoot_ratio,
       E.cumulative_target/B.cumulative_budget as target_overshoot_ratio
from {self._schema}.cumulative_emissions E
     join {self._schema}.cumulative_budget_1 B on E.company_id=B.company_id
""")
        # Need to fetch so table created above is established before using in query below
        qres.fetchall()
        qres = self._engine.execute(f"""
create table temperature_scores with (
    format = 'parquet',
    external_location = 's3a://{trino_bucket.name}/trino/{self._schema}/temperature_scores/'
) as
select R.company_name, R.company_id, '{company_data._schema}' as source, 'S1+S2' as scope, 'benchmark_1' as benchmark,
       R.benchmark_temp + R.global_budget * (R.trajectory_overshoot_ratio-1) * 2.2/3664.0 as trajectory_temperature_score,
       R.benchmark_temp + R.global_budget * (R.target_overshoot_ratio-1) * 2.2/3664.0 as target_temperature_score
from {self._schema}.overshoot_ratios R
""")
        # Need to fetch so table created above is established before any might want to use later
        qres.fetchall()


    def get_preprocessed_company_data(self, company_ids: List[str]) -> List[ICompanyAggregates]:
        raise NotImplementedError

    def get_pa_temp_scores(self, probability: float, company_ids: List[str]) -> pd.Series:
        if probability < 0 or probability > 1:
            raise ValueError(f"probability value {probability} outside range [0.0, 1.0]")
        temp_scores = pd.read_sql(f"select company_id, target_temperature_score, trajectory_temperature_score from {self._schema}.temperature_scores",
                                  self._engine, index_col='company_id')
        # We may have company_ids in our portfolio not in our database, and vice-versa.
        # Return proper pa_temp_scores for what we can find, and np.nan for those we cannot
        retval = pd.Series(data=None, index=company_ids, dtype='float64')
        retval.loc[retval.index.intersection(temp_scores.index)] = temp_scores.target_temperature_score*probability + temp_scores.trajectory_temperature_score*(1-probability)
        return retval