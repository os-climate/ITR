import warnings  # needed until apply behaves better with Pint quantities in arrays

from typing import Optional, Tuple, Type, List

import pandas as pd
import numpy as np
import itertools

from pint import Quantity
from .data.osc_units import ureg, Q_, PA_
from ITR.interfaces import EScope, ETimeFrames, EScoreResultType, Aggregation, AggregationContribution, \
    ScoreAggregation, \
    ScoreAggregationScopes, ScoreAggregations, PortfolioCompany
from ITR.portfolio_aggregation import PortfolioAggregation, PortfolioAggregationMethod
from ITR.configs import TemperatureScoreConfig
from ITR import utils
from ITR.data.data_warehouse import DataWarehouse


class TemperatureScore(PortfolioAggregation):
    """
    This class is provides a temperature score based on the climate goals.

    :param fallback_score: The temp score if a company is not found
    :param config: A class defining the constants that are used throughout this class. This parameter is only required
                    if you'd like to overwrite a constant. This can be done by extending the TemperatureScoreConfig
                    class and overwriting one of the parameters.
    """

    def __init__(self, time_frames: List[ETimeFrames], scopes: List[EScope],
                 fallback_score: float = Q_(3.2, ureg.delta_degC),
                 aggregation_method: PortfolioAggregationMethod = PortfolioAggregationMethod.WATS,
                 grouping: Optional[List] = None, config: Type[TemperatureScoreConfig] = TemperatureScoreConfig):
        super().__init__(config)
        self.c: Type[TemperatureScoreConfig] = config
        self.fallback_score = fallback_score

        self.time_frames = time_frames
        self.scopes = scopes

        self.aggregation_method: PortfolioAggregationMethod = aggregation_method
        self.grouping: list = []
        if grouping is not None:
            self.grouping = grouping

    def get_score(self, scorable_row: pd.Series) -> Tuple[
        Quantity['delta_degC'], Quantity['delta_degC'], float, Quantity['delta_degC'], float, EScoreResultType]:
        """
        Get the temperature score for a certain target based on the annual reduction rate and the regression parameters.

        :param scorable_row: The target as a row of a data frame
        :return: The temperature score, which is a tuple of (TEMPERATURE_SCORE, TRAJECTORY_SCORE, TRAJECTORY_OVERSHOOT,
                        TARGET_SCORE, TARGET_OVERSHOOT, TEMPERATURE_RESULTS])
        """

        # If both trajectory and target data missing assign default value
        if (np.isnan(scorable_row[self.c.COLS.CUMULATIVE_TARGET]) and
            np.isnan(scorable_row[self.c.COLS.CUMULATIVE_TRAJECTORY])) or \
                scorable_row[self.c.COLS.CUMULATIVE_BUDGET].m <= 0:
            return self.get_default_score(scorable_row), np.nan, np.nan, np.nan, np.nan, EScoreResultType.DEFAULT

        # If only target data missing assign only trajectory_score to final score
        elif np.isnan(scorable_row[self.c.COLS.CUMULATIVE_TARGET]) or scorable_row[self.c.COLS.CUMULATIVE_TARGET] == 0:
            target_overshoot_ratio = np.nan
            target_temperature_score = np.nan
            trajectory_overshoot_ratio = scorable_row[self.c.COLS.CUMULATIVE_TRAJECTORY] / scorable_row[
                self.c.COLS.CUMULATIVE_BUDGET]
            trajectory_temperature_score = scorable_row[self.c.COLS.BENCHMARK_TEMP] + \
                (scorable_row[self.c.COLS.BENCHMARK_GLOBAL_BUDGET] * (trajectory_overshoot_ratio - 1.0) *
                    self.c.CONTROLS_CONFIG.tcre_multiplier)
            score = trajectory_temperature_score
            return score, trajectory_temperature_score, trajectory_overshoot_ratio, \
                target_temperature_score, target_overshoot_ratio, EScoreResultType.TRAJECTORY_ONLY

        else:
            target_overshoot_ratio = scorable_row[self.c.COLS.CUMULATIVE_TARGET] / scorable_row[
                self.c.COLS.CUMULATIVE_BUDGET]
            trajectory_overshoot_ratio = scorable_row[self.c.COLS.CUMULATIVE_TRAJECTORY] / scorable_row[
                self.c.COLS.CUMULATIVE_BUDGET]

            target_temperature_score = scorable_row[self.c.COLS.BENCHMARK_TEMP] + \
                (scorable_row[self.c.COLS.BENCHMARK_GLOBAL_BUDGET] * (target_overshoot_ratio - 1.0) *
                    self.c.CONTROLS_CONFIG.tcre_multiplier)
            trajectory_temperature_score = scorable_row[self.c.COLS.BENCHMARK_TEMP] + \
                (scorable_row[self.c.COLS.BENCHMARK_GLOBAL_BUDGET] * (trajectory_overshoot_ratio - 1.0) *
                    self.c.CONTROLS_CONFIG.tcre_multiplier)
            score = target_temperature_score * scorable_row[self.c.COLS.TARGET_PROBABILITY] + \
                trajectory_temperature_score * (1 - scorable_row[self.c.COLS.TARGET_PROBABILITY])

            return score, trajectory_temperature_score, trajectory_overshoot_ratio, target_temperature_score, \
                target_overshoot_ratio, EScoreResultType.COMPLETE

    def get_ghc_temperature_score(self, row: pd.Series, company_data: pd.DataFrame) -> Quantity['delta_degC']:
        """
        Get the aggregated temperature score. S1+S2+S3 is an emissions weighted sum of S1+S2 and S3.

        :param company_data: The original data, grouped by company, time frame and scope category
        :param row: The row to calculate the temperature score for (if the scope of the row isn't s1s2s3, it will return the original score)
        :return: The aggregated temperature score for a company
        """
        # TODO: Notify user when S1+S2+S3 is built up from S1+S2 and S3 score of different ScoreResultTypes

        if row[self.c.COLS.SCOPE] != EScope.S1S2S3:
            return row[self.c.COLS.TEMPERATURE_SCORE]
        s1s2 = company_data.loc[(row[self.c.COLS.COMPANY_ID], row[self.c.COLS.TIME_FRAME], EScope.S1S2)]
        s3 = company_data.loc[(row[self.c.COLS.COMPANY_ID], row[self.c.COLS.TIME_FRAME], EScope.S3)]

        try:
            # If the s3 emissions are less than 40 percent, we'll ignore them altogether, if not, we'll weigh them
            if s3[self.c.COLS.GHG_SCOPE3] / (s1s2[self.c.COLS.GHG_SCOPE12] + s3[self.c.COLS.GHG_SCOPE3]) < 0.4:
                return s1s2[self.c.COLS.TEMPERATURE_SCORE]
            else:
                company_emissions = s1s2[self.c.COLS.GHG_SCOPE12] + s3[self.c.COLS.GHG_SCOPE3]
                return ((s1s2[self.c.COLS.TEMPERATURE_SCORE] * s1s2[self.c.COLS.GHG_SCOPE12] +
                         s3[self.c.COLS.TEMPERATURE_SCORE] * s3[self.c.COLS.GHG_SCOPE3]) / company_emissions)

        except ZeroDivisionError:
            raise ValueError("The mean of the S1+S2 plus the S3 emissions is zero")

    def get_default_score(self, target: pd.Series) -> Quantity['delta_degC']:
        """
        :param target: The target as a row of a dataframe
        :return: The temperature score
        """
        return self.fallback_score

    def _prepare_data(self, data: pd.DataFrame):
        """
        Prepare the data such that it can be used to calculate the temperature score.

        :param data: The original data set as a pandas data frame
        :return: The extended data frame
        """
        companies = data[self.c.COLS.COMPANY_ID].unique()

        # If scope S1S2S3 is in the list of scopes to calculate, we need to calculate the other two as well
        scopes = self.scopes.copy()
        if EScope.S1S2S3 in self.scopes and EScope.S1S2 not in self.scopes:
            scopes.append(EScope.S1S2)
        if EScope.S1S2S3 in scopes and EScope.S3 not in scopes:
            scopes.append(EScope.S3)

        score_combinations = pd.DataFrame(list(itertools.product(*[companies, scopes, self.time_frames])),
                                          columns=[self.c.COLS.COMPANY_ID, self.c.COLS.SCOPE, self.c.COLS.TIME_FRAME])
        scoring_data = pd.merge(left=data, right=score_combinations, how='outer', on=[self.c.COLS.COMPANY_ID])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # See https://github.com/hgrecco/pint-pandas/issues/114
            scoring_data[self.c.COLS.TEMPERATURE_SCORE], scoring_data[self.c.COLS.TRAJECTORY_SCORE], scoring_data[
                self.c.COLS.TRAJECTORY_OVERSHOOT], scoring_data[self.c.COLS.TARGET_SCORE], scoring_data[
                self.c.COLS.TARGET_OVERSHOOT], scoring_data[self.c.SCORE_RESULT_TYPE] = zip(*scoring_data.apply(
                    lambda row: self.get_score(row), axis=1))

        # Fix up dtypes for the new columns we just added
        for c in [self.c.COLS.TEMPERATURE_SCORE, self.c.COLS.TRAJECTORY_SCORE, self.c.COLS.TRAJECTORY_SCORE,
                  self.c.COLS.TARGET_SCORE]:
            scoring_data[c] = scoring_data[c].astype('pint[delta_degC]')

        scoring_data = self.cap_scores(scoring_data)
        return scoring_data

    def _calculate_company_score(self, data):
        """
        Calculate the combined s1s2s3 scores for all companies.

        :param data: The original data set as a pandas data frame
        :return: The data frame, with an updated s1s2s3 temperature score
        """
        # Calculate the GHC
        company_data = data[
            [self.c.COLS.COMPANY_ID, self.c.COLS.TIME_FRAME, self.c.COLS.SCOPE, self.c.COLS.GHG_SCOPE12,
             self.c.COLS.GHG_SCOPE3, self.c.COLS.TEMPERATURE_SCORE, self.c.SCORE_RESULT_TYPE]
        ].groupby([self.c.COLS.COMPANY_ID, self.c.COLS.TIME_FRAME, self.c.COLS.SCOPE]).mean()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data[self.c.COLS.TEMPERATURE_SCORE] = data.apply(
                lambda row: self.get_ghc_temperature_score(row, company_data), axis=1
            )
        return data

    def calculate(self, data: Optional[pd.DataFrame] = None,
                  data_warehouse: Optional[DataWarehouse] = None,
                  portfolio: Optional[List[PortfolioCompany]] = None):
        """
        Calculate the temperature for a dataframe of company data. The columns in the data frame should be a combination
        of IDataProviderTarget and IDataProviderCompany.

        :param data: The data set (or None if the data should be retrieved)
        :param data_warehouse: A list of DataProvider instances. Optional, only required if data is empty.
        :param portfolio: A list of PortfolioCompany models. Optional, only required if data is empty.
        :return: A data frame containing all relevant information for the targets and companies
        """
        if data is None:
            if data_warehouse is not None and portfolio is not None:
                data = utils.get_data(data_warehouse, portfolio)
            else:
                raise ValueError("You need to pass and either a data set or a datawarehouse and companies")

        data = self._prepare_data(data)

        if EScope.S1S2S3 in self.scopes:
            self._check_column(data, self.c.COLS.GHG_SCOPE12)
            self._check_column(data, self.c.COLS.GHG_SCOPE3)
            data = self._calculate_company_score(data)

        # We need to filter the scopes again, because we might have had to add a scope in the preparation step
        data = data[data[self.c.COLS.SCOPE].isin(self.scopes)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # See https://github.com/hgrecco/pint-pandas/issues/114
            data[self.c.COLS.TEMPERATURE_SCORE] = data[self.c.COLS.TEMPERATURE_SCORE].map(
                lambda x: Q_(round(x.m, 2), x.u)).astype('pint[delta_degC]')
        return data

    def _get_aggregations(self, data: pd.DataFrame, total_companies: int) -> Tuple[Aggregation, pd.Series, pd.Series]:
        """
        Get the aggregated score over a certain data set. Also calculate the (relative) contribution of each company

        :param data: A data set, containing one row per company
        :return: An aggregated score and the relative and absolute contribution of each company
        """
        data = data.copy()
        weighted_scores = self._calculate_aggregate_score(data, self.c.COLS.TEMPERATURE_SCORE,
                                                          self.aggregation_method).astype('pint[delta_degC]')
        data[self.c.COLS.CONTRIBUTION_RELATIVE] = pd.Series(weighted_scores / weighted_scores.sum(),
                                                            dtype='pint[percent]')
        data[self.c.COLS.CONTRIBUTION] = weighted_scores
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            contributions = data \
                .sort_values(self.c.COLS.CONTRIBUTION_RELATIVE, ascending=False) \
                .where(pd.notnull(data), 0) \
                .to_dict(orient="records")
        aggregations = Aggregation(
            score=weighted_scores.sum(),
            proportion=len(weighted_scores) / (total_companies / 100.0),
            contributions=[AggregationContribution.parse_obj(contribution) for contribution in contributions]
        ), \
                       data[self.c.COLS.CONTRIBUTION_RELATIVE], \
                       data[self.c.COLS.CONTRIBUTION]

        return aggregations

    def _get_score_aggregation(self, data: pd.DataFrame, time_frame: ETimeFrames, scope: EScope) -> \
            Optional[ScoreAggregation]:
        """
        Get a score aggregation for a certain time frame and scope, for the data set as a whole and for the different
        groupings.

        :param data: The whole data set
        :param time_frame: A time frame
        :param scope: A scope
        :return: A score aggregation, containing the aggregations for the whole data set and each individual group
        """
        filtered_data = data[(data[self.c.COLS.TIME_FRAME] == time_frame) &
                             (data[self.c.COLS.SCOPE] == scope)].copy()
        filtered_data[self.grouping] = filtered_data[self.grouping].fillna("unknown")
        total_companies = len(filtered_data)
        if not filtered_data.empty:
            score_aggregation_all, \
            filtered_data[self.c.COLS.CONTRIBUTION_RELATIVE], \
            filtered_data[self.c.COLS.CONTRIBUTION] = self._get_aggregations(filtered_data, total_companies)
            filtered_data['DEFAULT'] = 1.0 * (filtered_data[self.c.SCORE_RESULT_TYPE] == EScoreResultType.DEFAULT)
            score_aggregation = ScoreAggregation(
                grouped={},
                all=score_aggregation_all,
                influence_percentage=self._calculate_aggregate_score(
                    filtered_data, 'DEFAULT', self.aggregation_method).sum() * 100)

            # If there are grouping column(s) we'll group in pandas and pass the results to the aggregation
            if len(self.grouping) > 0:
                grouped_data = filtered_data.groupby(self.grouping)
                for group_names, group in grouped_data:
                    group_name_joined = group_names if type(group_names) == str else "-".join(
                        [str(group_name) for group_name in group_names])
                    score_aggregation.grouped[group_name_joined], _, _ = self._get_aggregations(group.copy(),
                                                                                                total_companies)
            return score_aggregation
        else:
            return None

    def aggregate_scores(self, data: pd.DataFrame) -> ScoreAggregations:
        """
        Aggregate scores to create a portfolio score per time_frame (short, mid, long).

        :param data: The results of the calculate method
        :return: A weighted temperature score for the portfolio
        """

        score_aggregations = ScoreAggregations()
        for time_frame in self.time_frames:
            score_aggregation_scopes = ScoreAggregationScopes()
            for scope in self.scopes:
                score_aggregation_scopes.__setattr__(scope.name, self._get_score_aggregation(data, time_frame, scope))
            score_aggregations.__setattr__(time_frame.value, score_aggregation_scopes)

        return score_aggregations

    def cap_scores(self, scores: pd.DataFrame) -> pd.DataFrame:
        """
        Cap the temperature scores in the input data frame to a certain value, based on the scenario that's being used. 
        This can either be for the whole data set, or only for the top X contributors.

        :param scores: The data set with the temperature scores
        :return: The input data frame, with capped scores
        """

        return scores

    def anonymize_data_dump(self, scores: pd.DataFrame) -> pd.DataFrame:
        """
        Anonymize the scores by deleting the company IDs, ISIN and renaming the companies.

        :param scores: The data set with the temperature scores
        :return: The input data frame, anonymized
        """
        scores.drop(columns=[self.c.COLS.COMPANY_ID, self.c.COLS.COMPANY_ISIN], inplace=True)
        for index, company_name in enumerate(scores[self.c.COLS.COMPANY_NAME].unique()):
            scores.loc[scores[self.c.COLS.COMPANY_NAME] == company_name, self.c.COLS.COMPANY_NAME] = 'Company' + str(
                index + 1)
        return scores
