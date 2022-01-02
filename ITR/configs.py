"""
This file defines the constants used throughout the different classes. In order to redefine these settings whilst using
the module, extend the respective config class and pass it to the class as the "constants" parameter.
"""
from .interfaces import TemperatureScoreControls

import pint
import pint_pandas
from ITR.data.osc_units import ureg, Q_

class ColumnsConfig:
    # Define a constant for each column used in the
    COMPANY_ID = "company_id"
    COMPANY_ISIN = "company_isin"
    COMPANY_ISIC = "isic"
    MARKET_CAP = "company_market_cap"
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
    PRODUCTION_METRIC = 'production_metric'    # The unit of production (i.e., power generated, tons of steel produced, vehicles manufactured, etc.)
    GHG_SCOPE12 = 'ghg_s1s2'    # This seems to be the base year PRODUCTION number, nothing at all to do with any quantity of actual S1S2 emissions
    GHG_SCOPE3 = 'ghg_s3'
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
    BASE_EI = 'emission_intensity_at_base_year'
    BASE_PRODUCTION = 'production_at_base_year'    # This would be a better name than GHG_SCOPE12
    PROJECTED_TRAJECTORIES = 'projected_ei_trajectories'
    PROJECTED_TARGETS = 'projected_ei_targets'
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


class PortfolioAggregationConfig:
    COLS = ColumnsConfig


class TemperatureScoreConfig(PortfolioAggregationConfig):
    TEMPERATURE_RESULTS = 'temperature_results'
    CONTROLS_CONFIG = TemperatureScoreControls(
        base_year=2019,
        target_end_year=2050,
        projection_start_year=2010,
        projection_end_year=2019,
        tcre=Q_(2.2, ureg.delta_degC),
        carbon_conversion=Q_(3664.0, ureg('Gt CO2')),
        scenario_target_temperature=Q_(1.5, ureg.delta_degC)
    )
