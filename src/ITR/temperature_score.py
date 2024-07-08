import itertools
import logging
import warnings  # needed until apply behaves better with Pint quantities in arrays
from typing import List, Optional, Tuple, Type

import pandas as pd

import ITR

from .configs import ColumnsConfig, LoggingConfig, TemperatureScoreConfig
from .data.data_warehouse import DataWarehouse
from .data.osc_units import Q_, delta_degC_Quantity, ureg
from .interfaces import (
    Aggregation,
    AggregationContribution,
    EScope,
    EScoreResultType,
    ETimeFrames,
    PortfolioCompany,
    ScoreAggregation,
    ScoreAggregations,
    ScoreAggregationScopes,
)
from .portfolio_aggregation import PortfolioAggregation, PortfolioAggregationMethod

logger = logging.getLogger(__name__)
LoggingConfig.add_config_to_logger(logger)

nan_delta_degC = Q_(pd.NA, "delta_degC")
nan_dimensionless = Q_(pd.NA, "dimensionless")


class TemperatureScore(PortfolioAggregation):
    """This class is provides a temperature score based on the climate goals.

    :param fallback_score: The temp score if a company is not found
    :param config: A class defining the constants that are used throughout this class. This parameter is only required
                    if you'd like to overwrite a constant. This can be done by extending the TemperatureScoreConfig
                    class and overwriting one of the parameters.
    """

    def __init__(
        self,
        time_frames: List[ETimeFrames],
        scopes: Optional[List[EScope]] = None,
        fallback_score: float = Q_(3.2, ureg.delta_degC),
        aggregation_method: PortfolioAggregationMethod = PortfolioAggregationMethod.WATS,
        budget_column: str = ColumnsConfig.CUMULATIVE_BUDGET,
        grouping: Optional[List] = None,
        config: Type[TemperatureScoreConfig] = TemperatureScoreConfig,
    ):
        super().__init__(config)
        self.c: Type[TemperatureScoreConfig] = config
        self.fallback_score = fallback_score

        self.time_frames = time_frames
        self.scopes = scopes

        self.aggregation_method = aggregation_method
        self.budget_column = budget_column
        self.grouping: list = []
        if grouping is not None:
            self.grouping = grouping

    def get_score(
        self, scorable_row: pd.Series
    ) -> Tuple[
        delta_degC_Quantity,
        delta_degC_Quantity,
        float,
        delta_degC_Quantity,
        float,
        EScoreResultType,
    ]:
        """Get the temperature score for a certain target based on the annual reduction rate and the regression parameters.

        :param scorable_row: The target as a row of a data frame
        :return: The temperature score, which is a tuple of (TEMPERATURE_SCORE, TRAJECTORY_SCORE, TRAJECTORY_OVERSHOOT,
                        TARGET_SCORE, TARGET_OVERSHOOT, TEMPERATURE_RESULTS])
        """
        # If both trajectory and target data missing assign default value
        if (
            ITR.isna(scorable_row[self.c.COLS.CUMULATIVE_TARGET])
            and ITR.isna(scorable_row[self.c.COLS.CUMULATIVE_TRAJECTORY])
        ) or scorable_row[self.budget_column].m <= 0:
            return (
                self.get_default_score(scorable_row),
                nan_delta_degC,
                nan_dimensionless,
                nan_delta_degC,
                nan_dimensionless,
                EScoreResultType.DEFAULT,
            )

        # If only target data missing assign only trajectory_score to final score
        elif (
            ITR.isna(scorable_row[self.c.COLS.CUMULATIVE_TARGET])
            or scorable_row[self.c.COLS.CUMULATIVE_TARGET] == 0
        ):
            target_overshoot_ratio = nan_dimensionless
            target_temperature_score = nan_delta_degC
            trajectory_overshoot_ratio = (
                scorable_row[self.c.COLS.CUMULATIVE_TRAJECTORY]
                / scorable_row[self.budget_column]
            )
            trajectory_temperature_score = scorable_row[self.c.COLS.BENCHMARK_TEMP] + (
                scorable_row[self.c.COLS.BENCHMARK_GLOBAL_BUDGET]
                * (trajectory_overshoot_ratio - 1.0)
                * self.c.CONTROLS_CONFIG.tcre_multiplier
            )
            score = trajectory_temperature_score
            return (
                score,
                trajectory_temperature_score,
                trajectory_overshoot_ratio,
                target_temperature_score,
                target_overshoot_ratio,
                EScoreResultType.TRAJECTORY_ONLY,
            )
        else:
            target_overshoot_ratio = (
                scorable_row[self.c.COLS.CUMULATIVE_TARGET]
                / scorable_row[self.budget_column]
            )
            trajectory_overshoot_ratio = (
                scorable_row[self.c.COLS.CUMULATIVE_TRAJECTORY]
                / scorable_row[self.budget_column]
            )

            target_temperature_score = scorable_row[self.c.COLS.BENCHMARK_TEMP] + (
                scorable_row[self.c.COLS.BENCHMARK_GLOBAL_BUDGET]
                * (target_overshoot_ratio - 1.0)
                * self.c.CONTROLS_CONFIG.tcre_multiplier
            )
            trajectory_temperature_score = scorable_row[self.c.COLS.BENCHMARK_TEMP] + (
                scorable_row[self.c.COLS.BENCHMARK_GLOBAL_BUDGET]
                * (trajectory_overshoot_ratio - 1.0)
                * self.c.CONTROLS_CONFIG.tcre_multiplier
            )

            # If trajectory data has run away (because trajectory projections are positive, not negative, use only target results
            if trajectory_overshoot_ratio > 10.0 or ITR.isna(
                trajectory_temperature_score
            ):
                score = target_temperature_score
                score_result_type = EScoreResultType.TARGET_ONLY
            else:
                score = target_temperature_score * scorable_row[
                    self.c.COLS.TARGET_PROBABILITY
                ] + trajectory_temperature_score * (
                    1 - scorable_row[self.c.COLS.TARGET_PROBABILITY]
                )
                score_result_type = EScoreResultType.COMPLETE

            return (
                score,
                trajectory_temperature_score,
                trajectory_overshoot_ratio,
                target_temperature_score,
                target_overshoot_ratio,
                score_result_type,
            )

    def get_ghc_temperature_score(
        self, row: pd.Series, company_data: pd.DataFrame
    ) -> delta_degC_Quantity:
        """Get the aggregated temperature score. S1+S2+S3 is an emissions weighted sum of S1+S2 and S3.

        :param company_data: The original data, grouped by company, scope category, and time frame
        :param row: The row to calculate the temperature score for (if the scope of the row isn't s1s2s3, it will return the original score)
        :return: The aggregated temperature score for a company
        """
        # TODO: Notify user when S1+S2+S3 is built up from S1+S2 and S3 score of different ScoreResultTypes
        # FIXME: the weighted average is achored on base_year weighting, not cumulative weighting

        # row.name is the MultiIndex tuple (company_id, scope)
        row_company_id = row.name
        if row[self.c.COLS.SCOPE] != EScope.S1S2S3:
            return row[self.c.COLS.TEMPERATURE_SCORE]
        df = company_data.loc[[row_company_id]]
        if df[
            df[self.c.COLS.SCOPE].eq(EScope.S1S2S3)
            & df[self.c.COLS.TIME_FRAME].eq(row[self.c.COLS.TIME_FRAME])
        ].size:
            return row[self.c.COLS.TEMPERATURE_SCORE]
        s1s2 = df[
            df[self.c.COLS.SCOPE].eq(EScope.S1S2)
            & df[self.c.COLS.TIME_FRAME].eq(row[self.c.COLS.TIME_FRAME])
        ]
        s3 = df[
            df[self.c.COLS.SCOPE].eq(EScope.S3)
            & df[self.c.COLS.TIME_FRAME].eq(row[self.c.COLS.TIME_FRAME])
        ]
        if s3.empty:
            # FIXME: should we return a DEFAULT temperature score if there's no S3 data?
            return s1s2[self.c.COLS.TEMPERATURE_SCORE]

        try:
            # If the s3 emissions are less than 40 percent, we'll ignore them altogether, if not, we'll weigh them
            # FIXME: These should use cumulative emissions, not the anachronistic ghg_s1s2 and ghg_s33!
            company_emissions = (
                s1s2[self.c.COLS.GHG_SCOPE12] + s3[self.c.COLS.GHG_SCOPE3]
            )
            if (s3[self.c.COLS.GHG_SCOPE3] / company_emissions < 0.4).all():
                return s1s2[self.c.COLS.TEMPERATURE_SCORE]
            else:
                return (
                    s1s2[self.c.COLS.TEMPERATURE_SCORE] * s1s2[self.c.COLS.GHG_SCOPE12]
                    + s3[self.c.COLS.TEMPERATURE_SCORE] * s3[self.c.COLS.GHG_SCOPE3]
                ) / company_emissions

        except ZeroDivisionError:
            raise ValueError("The mean of the S1+S2 plus the S3 emissions is zero")

    def get_default_score(self, target: pd.Series) -> delta_degC_Quantity:
        """:param target: The target as a row of a dataframe
        :return: The temperature score
        """
        return self.fallback_score

    def _prepare_data(self, data: pd.DataFrame, target_probability: float):
        """Prepare the data such that it can be used to calculate the temperature score.

        :param data: The original data set as a pandas data frame, indexed by (COMPANY_ID, SCOPE)
        :return: The extended data frame, indexed by COMPANY_ID
        """
        company_id_and_scope = [self.c.COLS.COMPANY_ID, self.c.COLS.SCOPE]
        companies = data.index.get_level_values(self.c.COLS.COMPANY_ID).unique()

        # If target score not provided, use non-specific probability
        data = data.fillna({self.c.COLS.TARGET_PROBABILITY: target_probability})

        # If scope S1S2S3 is in the list of scopes to calculate, we need to calculate the other two as well
        if self.scopes:
            scopes = self.scopes.copy()
            if EScope.S1S2S3 in self.scopes and EScope.S1S2 not in self.scopes:
                scopes.append(EScope.S1S2)
            if EScope.S1S2S3 in scopes and EScope.S3 not in scopes:
                scopes.append(EScope.S3)
        else:
            scopes = data.index.get_level_values(self.c.COLS.SCOPE).unique()
            # No need to append any scopes not found in any company data...

        df_combinations = pd.DataFrame(
            list(itertools.product(*[companies, self.time_frames, scopes])),
            columns=[self.c.COLS.COMPANY_ID, self.c.COLS.TIME_FRAME, self.c.COLS.SCOPE],
        )
        # index of scoring_data_left has MORE rows than data if score_combinations has more than one combo
        scoring_data_left = pd.merge(
            left=data, right=df_combinations, how="left", on=company_id_and_scope
        ).set_index("company_id")
        # index of scoring_data_inner may have FEWER rows than data if not all rows align
        scoring_data_inner = pd.merge(
            left=data, right=df_combinations, how="inner", on=company_id_and_scope
        ).set_index("company_id")
        # goal is to identify COMPANY_IDs that don't make it through INNER
        na_data = scoring_data_left[self.c.COLS.TIME_FRAME].isna()
        idx_difference = scoring_data_left[na_data].index.difference(
            scoring_data_left[~na_data].index
        )
        if idx_difference.size:
            logger.warning(
                f"Dropping companies with no relevant scope data: {idx_difference.to_list()}"
            )
        scoring_data = scoring_data_inner

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # See https://github.com/hgrecco/pint-pandas/issues/114
            (
                scoring_data[self.c.COLS.TEMPERATURE_SCORE],
                scoring_data[self.c.COLS.TRAJECTORY_SCORE],
                scoring_data[self.c.COLS.TRAJECTORY_OVERSHOOT],
                scoring_data[self.c.COLS.TARGET_SCORE],
                scoring_data[self.c.COLS.TARGET_OVERSHOOT],
                scoring_data[self.c.SCORE_RESULT_TYPE],
            ) = zip(*scoring_data.apply(lambda row: self.get_score(row), axis=1))

        # Fix up dtypes for the new columns we just added
        for col in [
            self.c.COLS.TEMPERATURE_SCORE,
            self.c.COLS.TRAJECTORY_SCORE,
            self.c.COLS.TARGET_SCORE,
        ]:
            scoring_data[col] = scoring_data[col].astype("pint[delta_degC]")
        for col in [self.c.COLS.TARGET_OVERSHOOT, self.c.COLS.TRAJECTORY_OVERSHOOT]:
            scoring_data[col] = scoring_data[col].astype("pint[dimensionless]")

        scoring_data = self.cap_scores(scoring_data)
        return scoring_data

    def _calculate_company_score(self, data):
        """Calculate the combined, weighted s1s2s3 scores for all companies.

        :param data: The original data set as a pandas data frame
        :return: The data frame, with an updated s1s2s3 temperature score
        """
        data[self.c.SCORE_RESULT_TYPE] = pd.Categorical(
            data[self.c.SCORE_RESULT_TYPE],
            ordered=True,
            categories=EScoreResultType.get_result_types(),
        )

        if False:
            # FIXME: Either we need to iterate across all the data, weighting S3 data when appropriate,
            # Or we can restrict to only "best" scores, returning a restricted set of data (and dropping
            # data for scopes whose results are not as good as those from other scopes).  What we cannot
            # do is to try to fill in the scores of the whole dataset from the restricted dataset.
            idx = (
                data[
                    [
                        self.c.COLS.TIME_FRAME,
                        self.c.COLS.GHG_SCOPE12,
                        self.c.COLS.GHG_SCOPE3,
                        self.c.COLS.TEMPERATURE_SCORE,
                        self.c.SCORE_RESULT_TYPE,
                    ]
                ]
                .groupby([self.c.COLS.TIME_FRAME])[self.c.SCORE_RESULT_TYPE]
                .transform("max")
                == data[self.c.SCORE_RESULT_TYPE]
            )

            company_timeframe_data = data[idx]  # noqa: F841

        # FIXME: from here to the end of the function, why not replace `data` with `company_timeframe_data`?
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data[self.c.COLS.TEMPERATURE_SCORE] = data.apply(
                lambda row: self.get_ghc_temperature_score(row, data),
                axis=1,  # used to iterate over company_timeframe_data
            ).astype("pint[delta_degC]")
        return data

    def calculate(
        self,
        data: Optional[pd.DataFrame] = None,
        data_warehouse: Optional[DataWarehouse] = None,
        portfolio: Optional[List[PortfolioCompany]] = None,
        target_probability: Optional[float] = None,
    ):
        """Calculate the temperature for a dataframe of company data. The columns in the data frame should be a combination
        of IDataProviderTarget and IDataProviderCompany.

        :param data: The data set (or None if the data should be retrieved)
        :param data_warehouse: A list of DataProvider instances. Optional, only required if data is empty.
        :param portfolio: A list of PortfolioCompany models. Optional, only required if data is empty.
        :return: A data frame containing all relevant information for the targets and companies
        """
        if portfolio is not None:
            logger.info(f"calculating temperature score for {len(portfolio)} companies")
        if target_probability is None:
            target_probability = (
                TemperatureScoreConfig.CONTROLS_CONFIG.target_probability
            )
        if data is None:
            if data_warehouse is not None and portfolio is not None:
                from . import utils

                data = utils.get_data(data_warehouse, portfolio)
            else:
                raise ValueError(
                    "You need to pass and either a data set or a datawarehouse and companies"
                )

        logger.info("temperature score preparing data")
        data = self._prepare_data(data, target_probability)
        logger.info("temperature score data prepared")

        if self.scopes:
            if EScope.S1S2S3 in self.scopes:
                # _check_column is for portfolio aggregation, and reports missing data but does not raise exceptions
                # self._check_column(data, self.c.COLS.GHG_SCOPE12)
                # self._check_column(data, self.c.COLS.GHG_SCOPE3)
                data = self._calculate_company_score(data)

            # We need to filter the scopes again, because we might have had to add a scope in the preparation step
            data = data[data[self.c.COLS.SCOPE].isin(self.scopes)]
        else:
            # We are happy to have computed all the scores that might be useful, according to the benchmark
            pass

        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        #     # See https://github.com/hgrecco/pint-pandas/issues/114
        #     data[self.c.COLS.TEMPERATURE_SCORE] = data[self.c.COLS.TEMPERATURE_SCORE].map(
        #         lambda x: Q_(round(x.m, 2), x.u)).astype('pint[delta_degC]')
        return data

    def _get_aggregations(
        self, data: pd.DataFrame, total_companies: int
    ) -> Tuple[Aggregation, pd.Series, pd.Series]:
        """Get the aggregated score over a certain data set. Also calculate the (relative) contribution of each company

        :param data: A data set, containing one row per company
        :return: An aggregated score and the relative and absolute contribution of each company
        """
        data = data.copy()
        weighted_scores = self._calculate_aggregate_score(
            data, self.c.COLS.TEMPERATURE_SCORE, self.aggregation_method
        )  # .astype('pint[delta_degC]')
        # https://github.com/pandas-dev/pandas/issues/50564 explains why we need fillna(1.0) to make sum work
        data[self.c.COLS.CONTRIBUTION_RELATIVE] = (
            weighted_scores / weighted_scores.fillna(1.0).sum()
        ).astype("pint[percent]")
        data[self.c.COLS.CONTRIBUTION] = weighted_scores
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data_contributions = (
                data.reset_index("company_id")[
                    [
                        "company_name",
                        "company_id",
                        "temperature_score",
                        "contribution_relative",
                        "contribution",
                    ]
                ]
                .sort_values(self.c.COLS.CONTRIBUTION_RELATIVE, ascending=False)
                .fillna(0)
                .to_dict(orient="records")
            )
        contribution_dicts = [
            {k: v if isinstance(v, str) else str(v) for k, v in contribution.items()}
            for contribution in data_contributions
        ]
        aggregations = (
            Aggregation(
                # https://github.com/pandas-dev/pandas/issues/50564 explains why we need fillna(0) to make sum work
                score=weighted_scores.fillna(0).sum(),
                # proportion is not declared by anything to be a percent, so we make it a number from 0..1
                proportion=len(weighted_scores) / total_companies,
                contributions=[
                    AggregationContribution.model_validate(contribution)
                    for contribution in contribution_dicts
                ],
            ),
            data[self.c.COLS.CONTRIBUTION_RELATIVE],
            data[self.c.COLS.CONTRIBUTION],
        )

        return aggregations

    def _get_score_aggregation(
        self, data: pd.DataFrame, time_frame: ETimeFrames, scope: EScope
    ) -> Optional[ScoreAggregation]:
        """Get a score aggregation for a certain time frame and scope, for the data set as a whole and for the different
        groupings.

        :param data: The whole data set
        :param time_frame: A time frame
        :param scope: A scope
        :return: A score aggregation, containing the aggregations for the whole data set and each individual group
        """
        filtered_data = data[data[self.c.COLS.SCOPE].eq(scope)]
        if scope == EScope.S3:
            na_s3 = filtered_data[self.c.COLS.GHG_SCOPE3].isna()
            filtered_data = filtered_data[~na_s3]
        filtered_data = filtered_data[
            filtered_data[self.c.COLS.TIME_FRAME].eq(time_frame)
        ].copy()
        filtered_data[self.grouping] = filtered_data[self.grouping].fillna("unknown")
        total_companies = len(filtered_data)
        if not filtered_data.empty:
            (
                score_aggregation_all,
                filtered_data[self.c.COLS.CONTRIBUTION_RELATIVE],
                filtered_data[self.c.COLS.CONTRIBUTION],
            ) = self._get_aggregations(filtered_data, total_companies)
            mask = filtered_data["score_result_type"].eq(EScoreResultType.DEFAULT)
            filtered_data.loc[mask, self.c.COLS.TEMPERATURE_SCORE] = self.fallback_score
            # https://github.com/pandas-dev/pandas/issues/50564 explains why we need fillna(0) to make sum work
            influence_percentage = (
                self._calculate_aggregate_score(
                    filtered_data,
                    self.c.COLS.CONTRIBUTION_RELATIVE,
                    self.aggregation_method,
                )
                .fillna(0)
                .sum()
            )
            score_aggregation = ScoreAggregation(
                grouped={},
                all=score_aggregation_all,
                influence_percentage=influence_percentage,
            )

            # If there are grouping column(s) we'll group in pandas and pass the results to the aggregation
            if len(self.grouping) == 0:
                return score_aggregation
            elif len(self.grouping) == 1:
                # Silence deprecation warning issuing from this change: https://github.com/pandas-dev/pandas/issues/42795
                self.grouping = self.grouping[0]

            grouped_data = filtered_data.groupby(self.grouping)
            for group_names, group in grouped_data:
                group_name_joined = (
                    group_names
                    if isinstance(group_names, str)
                    else "-".join([str(group_name) for group_name in group_names])
                )
                (
                    score_aggregation.grouped[group_name_joined],
                    _,
                    _,
                ) = self._get_aggregations(group.copy(), total_companies)
            return score_aggregation
        else:
            return None

    def aggregate_scores(self, data: pd.DataFrame) -> ScoreAggregations:
        """Aggregate scores to create a portfolio score per time_frame (short, mid, long).

        :param data: The results of the calculate method
        :return: A weighted temperature score for the portfolio
        """
        score_aggregations = ScoreAggregations()
        if self.scopes:
            for time_frame in self.time_frames:
                score_aggregation_scopes = ScoreAggregationScopes()
                for scope in self.scopes:
                    if data[
                        data[self.c.COLS.TIME_FRAME].eq(time_frame)
                        & data[self.c.COLS.SCOPE].eq(scope)
                    ].size:
                        score_aggregation_scopes.__setattr__(
                            scope.name,
                            self._get_score_aggregation(data, time_frame, scope),
                        )
                score_aggregations.__setattr__(
                    time_frame.value, score_aggregation_scopes
                )
        else:
            grouped_timeframes = data[
                data[self.c.COLS.TIME_FRAME].isin(self.time_frames)
            ].groupby(self.c.COLS.TIME_FRAME)
            for time_frame, timeframe_group in grouped_timeframes:
                score_aggregation_scopes = ScoreAggregationScopes()
                grouped_scopes = timeframe_group.groupby(self.c.COLS.SCOPE)
                for scope, scope_group in grouped_scopes:
                    score_aggregation_scopes.__setattr__(
                        scope.name,
                        self._get_score_aggregation(scope_group, time_frame, scope),
                    )
                score_aggregations.__setattr__(
                    time_frame.value, score_aggregation_scopes
                )

        return score_aggregations

    def cap_scores(self, scores: pd.DataFrame) -> pd.DataFrame:
        """Cap the temperature scores in the input data frame to a certain value, based on the scenario that's being used.
        This can either be for the whole data set, or only for the top X contributors.

        :param scores: The data set with the temperature scores
        :return: The input data frame, with capped scores
        """
        return scores

    def anonymize_data_dump(self, scores: pd.DataFrame) -> pd.DataFrame:
        """Anonymize the scores by deleting the company IDs, ISIN and renaming the companies.

        :param scores: The data set with the temperature scores
        :return: The input data frame, anonymized
        """
        scores.drop(
            columns=[self.c.COLS.COMPANY_ID, self.c.COLS.COMPANY_ISIN], inplace=True
        )
        for index, company_name in enumerate(scores[self.c.COLS.COMPANY_NAME].unique()):
            scores.loc[
                scores[self.c.COLS.COMPANY_NAME] == company_name,
                self.c.COLS.COMPANY_NAME,
            ] = "Company" + str(index + 1)
        return scores
