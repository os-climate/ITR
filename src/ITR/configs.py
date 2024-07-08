"""This file defines the constants used throughout the different classes. In order to redefine these settings whilst using
the module, extend the respective config class and pass it to the class as the "constants" parameter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict

from .data.osc_units import EmissionsQuantity, Quantity, delta_degC_Quantity


def ITR_median(*args, **kwargs):
    method = pd.DataFrame.median
    return method(*args, **kwargs)


def ITR_mean(*args, **kwargs):
    method = pd.DataFrame.mean
    return method(*args, **kwargs)


class ColumnsConfig:
    # Define a constant for each column used in the dataframe
    COMPANY_ID = "company_id"
    COMPANY_LEI = "company_lei"
    COMPANY_ISIN = "company_isin"
    COMPANY_ISIC = "isic"
    COMPANY_CURRENCY = "currency"
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
    COUNTRY = "country"
    SECTOR = "sector"
    TEMPLATE_EXPOSURE = "exposure"
    TEMPLATE_FX_QUOTE = "fx_quote"
    TEMPLATE_FX_RATE = "fx_rate"
    TEMPLATE_REPORT_DATE = "report_date"
    EMISSIONS_METRIC = "emissions_metric"
    PRODUCTION_METRIC = "production_metric"  # The unit of production (i.e., power generated, tons of steel produced, vehicles manufactured, etc.)
    BASE_YEAR_PRODUCTION = "base_year_production"
    GHG_SCOPE12 = "ghg_s1s2"
    GHG_SCOPE3 = "ghg_s3"
    HISTORIC_DATA = "historic_data"
    TARGET_DATA = "target_data"
    TEMPLATE_PRODUCTION = "production"
    COMPANY_REVENUE = "company_revenue"
    COMPANY_CASH_EQUIVALENTS = "company_cash_equivalents"
    BASE_YEAR = "base_year"
    END_YEAR = "end_year"
    ISIC = "isic"
    INDUSTRY_LVL1 = "industry_level_1"
    INDUSTRY_LVL2 = "industry_level_2"
    INDUSTRY_LVL3 = "industry_level_3"
    INDUSTRY_LVL4 = "industry_level_4"
    REGION = "region"
    CUMULATIVE_BUDGET = "cumulative_budget"
    CUMULATIVE_SCALED_BUDGET = "cumulative_scaled_budget"
    CUMULATIVE_TRAJECTORY = "cumulative_trajectory"
    CUMULATIVE_TARGET = "cumulative_target"
    TRAJECTORY_EXCEEDANCE_YEAR = "trajectory_exceedance_year"
    TARGET_EXCEEDANCE_YEAR = "target_exceedance_year"
    TARGET_PROBABILITY = "target_probability"
    BENCHMARK_TEMP = "benchmark_temperature"
    BENCHMARK_GLOBAL_BUDGET = "benchmark_global_budget"
    BASE_EI = "ei_at_base_year"
    PROJECTED_EI = "projected_intensities"
    PROJECTED_TARGETS = "projected_targets"
    HISTORIC_PRODUCTIONS = "historic_productions"
    HISTORIC_EMISSIONS = "historic_emissions"
    HISTORIC_EI = "historic_ei"

    TRAJECTORY_SCORE = "trajectory_score"
    TRAJECTORY_OVERSHOOT = "trajectory_overshoot_ratio"
    TARGET_SCORE = "target_score"
    TARGET_OVERSHOOT = "target_overshoot_ratio"

    # Output columns
    WEIGHTED_TEMPERATURE_SCORE = "weighted_temperature_score"
    CONTRIBUTION_RELATIVE = "contribution_relative"
    CONTRIBUTION = "contribution"


class SectorsConfig:
    POWER_UTILITY = "Electricity Utilities"
    GAS_UTILITY = "Gas Utilities"
    UTILITY = "Utilities"
    STEEL = "Steel"
    ALUMINUM = "Aluminum"
    ENERGY = "Energy"
    OIL_AND_GAS = "Oil & Gas"
    COAL = "Coal"
    OIL = "Oil"
    GAS = "Gas"
    AUTOMOBILE = "Autos"
    TRUCKING = "Trucking"
    CEMENT = "Cement"
    BUILDINGS_CONSTRUCTION = "Construction Buildings"
    BUILDINGS_RESIDENTIAL = "Residential Buildings"
    BUILDINGS_COMMERCIAL = "Commercial Buildings"
    TEXTILES = "Textiles"
    CHEMICALS = "Chemicals"
    PLASTICS = "Petrochem & Plastics"
    AG_CHEM = "Ag Chem"
    CONSUMER_PRODUCTS = "Consumer Products"
    PHARMACEUTICALS = "Pharmaceuticals"
    FIBERS_AND_RUBBER = "Fiber & Rubber"
    INFORMATION_TECHNOLOGY = "Information Technology"
    INDUSTRIALS = "Industrials"
    FINANCIALS = "Financials"
    HEALTH_CARE = "Health Care"

    @classmethod
    def get_configured_sectors(cls) -> List[str]:
        """Get a list of sectors configured in the tool.
        :return: A list of sectors string values
        """
        return [
            SectorsConfig.POWER_UTILITY,
            SectorsConfig.GAS_UTILITY,
            SectorsConfig.UTILITY,
            SectorsConfig.STEEL,
            SectorsConfig.ALUMINUM,
            SectorsConfig.ENERGY,
            SectorsConfig.COAL,
            SectorsConfig.OIL,
            SectorsConfig.GAS,
            SectorsConfig.OIL_AND_GAS,
            SectorsConfig.AUTOMOBILE,
            SectorsConfig.TRUCKING,
            SectorsConfig.CEMENT,
            SectorsConfig.BUILDINGS_CONSTRUCTION,
            SectorsConfig.BUILDINGS_RESIDENTIAL,
            SectorsConfig.BUILDINGS_COMMERCIAL,
            SectorsConfig.TEXTILES,
            SectorsConfig.CHEMICALS,
            SectorsConfig.PLASTICS,
            SectorsConfig.AG_CHEM,
            SectorsConfig.CONSUMER_PRODUCTS,
            SectorsConfig.PHARMACEUTICALS,
            SectorsConfig.FIBERS_AND_RUBBER,
        ]


class VariablesConfig:
    EMISSIONS = "Emissions"
    PRODUCTIONS = "Productions"
    EMISSIONS_INTENSITIES = "Emissions Intensities"


class TargetConfig:
    COMPANY_ID = "company_id"
    COMPANY_LEI = "company_lei"
    COMPANY_ISIN = "company_isin"
    COMPANY_ISIC = "isic"
    NETZERO_DATE = "netzero_date"
    TARGET_TYPE = "target_type"
    TARGET_SCOPE = "target_scope"
    TARGET_START_YEAR = "target_start_year"
    TARGET_BASE_YEAR = "target_base_year"
    TARGET_BASE_MAGNITUDE = "target_base_year_qty"
    TARGET_BASE_UNITS = "target_base_year_unit"
    TARGET_YEAR = "target_year"
    TARGET_REDUCTION_VS_BASE = "target_reduction_ambition"


class TabsConfig:
    FUNDAMENTAL = "fundamental_data"
    PROJECTED_EI = "projected_ei"  # really "projected
    PROJECTED_PRODUCTION = "projected_production"
    PROJECTED_TARGET = "projected_target"
    HISTORIC_DATA = "historic_data"
    TEMPLATE_INPUT_DATA = "ITR input data"
    TEMPLATE_INPUT_DATA_V2 = "ITR V2 input data"
    TEMPLATE_ESG_DATA_V2 = "ITR V2 esg data"
    TEMPLATE_TARGET_DATA = "ITR target input data"


class PortfolioAggregationConfig:
    COLS = ColumnsConfig


@dataclass
class ProjectionControls:
    LOWER_PERCENTILE: float = 0.1
    UPPER_PERCENTILE: float = 0.9

    LOWER_DELTA: float = -0.10  # -0.15
    # We clamp projections to a very small positive number, lest we create nonsense
    UPPER_DELTA: float = +0.03  # +1e-3

    BASE_YEAR: int = 2019
    TARGET_YEAR: int = 2050
    TREND_CALC_METHOD: Callable[
        [pd.DataFrame, Optional[str], Optional[bool]], pd.DataFrame
    ] = ITR_median


class TemperatureScoreControls(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_year: int
    target_end_year: int
    tcre: delta_degC_Quantity
    carbon_conversion: EmissionsQuantity
    scenario_target_temperature: delta_degC_Quantity
    target_probability: float

    def __getitem__(self, item):
        return getattr(self, item)

    @property
    def tcre_multiplier(self) -> Quantity:
        return self.tcre / self.carbon_conversion


class TemperatureScoreConfig(PortfolioAggregationConfig):
    SCORE_RESULT_TYPE = "score_result_type"
    # FIXME: Sooner or later, mutable default arguments cause problems.
    CONTROLS_CONFIG = TemperatureScoreControls(
        base_year=ProjectionControls.BASE_YEAR,
        target_end_year=ProjectionControls.TARGET_YEAR,
        tcre="2.2 delta_degC",
        carbon_conversion="3664.0 Gt CO2",
        scenario_target_temperature="1.5 delta_degC",
        target_probability=0.5,
    )


class LoggingConfig:
    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @classmethod
    def add_config_to_logger(cls, logger: logging.Logger):
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(cls.FORMAT)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)


# Add these lines to any file that uses logging
# from ITR.configs import LoggingConfig
# import logging
# logger = logging.getLogger(__name__)
# LoggingConfig.add_config_to_logger(logger)
