"""
This file defines the constants used throughout the different classes. In order to redefine these settings whilst using
the module, extend the respective config class and pass it to the class as the "constants" parameter.
"""
import logging

from .interfaces import TemperatureScoreControls

import pint
import pint_pandas
from ITR.data.osc_units import ureg, Q_
from typing import List


class ColumnsConfig:
    # Define a constant for each column used in the
    COMPANY_ID = "company_id"
    COMPANY_LEI = "company_lei"
    COMPANY_ISIN = "company_isin"
    COMPANY_ISIC = "isic"
    COMPANY_MARKET_CAP = "company_market_cap"
    INVESTMENT_VALUE = "investment_value"
    COMPANY_ENTERPRISE_VALUE = "company_enterprise_value"
    COMPANY_EV_PLUS_CASH = "company_ev_plus_cash"
    COMPANY_TOTAL_ASSETS = "company_total_assets"
    SCOPE = "scope"
    START_YEAR = "start_year"
    VARIABLE = "variable"
    SLOPE = "slope"
    TIME_FRAME = "time_frame"
    TEMPERATURE_SCORE = "temperature_score"
    COMPANY_NAME = "company_name"
    OWNED_EMISSIONS = "owned_emissions"
    COUNTRY = 'country'
    SECTOR = 'sector'
    TEMPLATE_EXPOSURE = 'exposure'
    TEMPLATE_CURRENCY = 'currency'
    TEMPLATE_REPORT_DATE = 'report_date'
    EMISSIONS_METRIC = 'emissions_metric'
    PRODUCTION_METRIC = 'production_metric'    # The unit of production (i.e., power generated, tons of steel produced, vehicles manufactured, etc.)
    BASE_YEAR_PRODUCTION = 'base_year_production'
    GHG_SCOPE12 = 'ghg_s1s2'
    GHG_SCOPE3 = 'ghg_s3'
    TEMPLATE_SCOPE1 = 'em_s1'
    TEMPLATE_SCOPE2 = 'em_s2'
    TEMPLATE_SCOPE12 = 'em_s1s2'
    TEMPLATE_SCOPE3 = 'em_s3'
    TEMPLATE_SCOPE123 = 'em_s1s2s3'
    HISTORIC_DATA = "historic_data"
    TARGET_DATA = "target_data"
    TEMPLATE_PRODUCTION = 'production'
    COMPANY_REVENUE = 'company_revenue'
    CASH_EQUIVALENTS = 'company_cash_equivalents'
    BASE_YEAR = 'base_year'
    END_YEAR = 'end_year'
    ISIC = 'isic'
    INDUSTRY_LVL1 = "industry_level_1"
    INDUSTRY_LVL2 = "industry_level_2"
    INDUSTRY_LVL3 = "industry_level_3"
    INDUSTRY_LVL4 = "industry_level_4"
    REGION = 'region'
    CUMULATIVE_BUDGET = 'cumulative_budget'
    CUMULATIVE_TRAJECTORY = 'cumulative_trajectory'
    CUMULATIVE_TARGET = 'cumulative_target'
    TARGET_PROBABILITY = 'target_probability'
    BENCHMARK_TEMP = 'benchmark_temperature'
    BENCHMARK_GLOBAL_BUDGET = 'benchmark_global_budget'
    BASE_EI = 'ei_at_base_year'
    PROJECTED_EI = 'projected_intensities'
    PROJECTED_TARGETS = 'projected_targets'
    HISTORIC_PRODUCTIONS = 'historic_productions'
    HISTORIC_EMISSIONS = 'historic_emissions'
    HISTORIC_EI = 'historic_ei'

    TRAJECTORY_SCORE = 'trajectory_score'
    TRAJECTORY_OVERSHOOT = 'trajectory_overshoot_ratio'
    TARGET_SCORE = 'target_score'
    TARGET_OVERSHOOT = 'target_overshoot_ratio'

    # Output columns
    WEIGHTED_TEMPERATURE_SCORE = "weighted_temperature_score"
    CONTRIBUTION_RELATIVE = "contribution_relative"
    CONTRIBUTION = "contribution"


class SectorsConfig:
    STEEL = "Steel"
    ELECTRICITY = "Electricity Utilities"
    INFORMATION_TECHNOLOGY = "Information Technology"
    INDUSTRIALS = "Industrials"
    FINANCIALS = "Financials"
    HEALTH_CARE = "Health Care"
    AUTOMOBILE = "Autos"

    @classmethod
    def get_configured_sectors(cls) -> List[str]:
        """
        Get a list of sectors configured in the tool.
        :return: A list of sectors string values
        """
        return [SectorsConfig.STEEL, SectorsConfig.ELECTRICITY, SectorsConfig.AUTOMOBILE]


class VariablesConfig:
    EMISSIONS = "Emissions"
    PRODUCTIONS = "Productions"
    EMISSIONS_INTENSITIES = "Emissions Intensities"


class TargetConfig:
    COMPANY_ID = "company_id"
    COMPANY_LEI = "company_lei"
    COMPANY_ISIN = "company_isin"
    COMPANY_ISIC = "isic"
    NETZERO_DATE = 'netzero_date'
    TARGET_TYPE = 'target_type'
    TARGET_SCOPE = 'target_scope'
    TARGET_START_YEAR = 'target_start_year'
    TARGET_BASE_YEAR = 'target_base_year'
    TARGET_BASE_MAGNITUDE = 'target_base_year_qty'
    TARGET_BASE_UNITS = 'target_base_year_unit'
    TARGET_YEAR = 'target_year'
    TARGET_REDUCTION_VS_BASE = 'target_reduction_ambition'


class TabsConfig:
    FUNDAMENTAL = "fundamental_data"
    PROJECTED_EI = "projected_ei_in_Wh"
    PROJECTED_PRODUCTION = "projected_production"
    PROJECTED_TARGET = "projected_target"
    HISTORIC_DATA = "historic_data"
    TEMPLATE_INPUT_DATA = 'ITR input data'
    TEMPLATE_TARGET_DATA = 'ITR target input data'


class PortfolioAggregationConfig:
    COLS = ColumnsConfig


class TemperatureScoreConfig(PortfolioAggregationConfig):
    SCORE_RESULT_TYPE = 'score_result_type'
    # Unfortunately we need to cross over to interfaces.py
    CONTROLS_CONFIG = TemperatureScoreControls(
        base_year=2019,
        target_end_year=2050,
        projection_start_year=2010,
        projection_end_year=2019,
        tcre=Q_(2.2, ureg.delta_degC),
        carbon_conversion=Q_(3664.0, ureg('Gt CO2')),
        scenario_target_temperature=Q_(1.5, ureg.delta_degC)
    )


class LoggingConfig:
    FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    @classmethod
    def add_config_to_logger(cls, logger: logging.Logger):
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(cls.FORMAT)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
