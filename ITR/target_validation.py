import datetime
import itertools

import pandas as pd
from typing import Type, List, Tuple, Optional
from ITR.configs import PortfolioAggregationConfig
import logging

from ITR.interfaces import IDataProviderTarget, IDataProviderCompany, EScope, ETimeFrames


class TargetProtocol:
    """
    This class validates the targets, to make sure that only active, useful targets are considered. It then combines the targets with company-related data into a dataframe where there's one row for each of the nine possible target types (short, mid, long * S1+S2, S3, S1+S2+S3). This class follows the procedures outlined by the target protocol that is a part of the "Temperature Rating Methodology" (2020), which has been created by CDP Worldwide and WWF International.

    :param config: A Portfolio aggregation config
    """

    def __init__(self, config: Type[PortfolioAggregationConfig] = PortfolioAggregationConfig):
        self.c = config
        self.logger = logging.getLogger(__name__)
        self.s2_targets: List[IDataProviderTarget] = []
        self.target_data: pd.DataFrame = pd.DataFrame()
        self.company_data: pd.DataFrame = pd.DataFrame()
        self.data: pd.DataFrame = pd.DataFrame()

    def process(self, targets: List[IDataProviderTarget], companies: List[IDataProviderCompany]) -> pd.DataFrame:
        """
        Process the targets and companies, validate all targets and return a data frame that combines all targets and company data into a 9-box grid.

        :param targets: A list of targets
        :param companies: A list of companies
        :return: A data frame that combines the processed data
        """
        # Create multiindex on company, timeframe and scope for performance later on
        targets = self.prepare_targets(targets)
        self.target_data = pd.DataFrame.from_records([c.dict() for c in targets])

        # Create an indexed DF for performance purposes
        self.target_data.index = self.target_data.reset_index().set_index(
            [self.c.COLS.COMPANY_ID, self.c.COLS.TIME_FRAME, self.c.COLS.SCOPE]).index
        self.target_data = self.target_data.sort_index()

        self.company_data = pd.DataFrame.from_records([c.dict() for c in companies])
        self.group_targets()
        return pd.merge(left=self.data, right=self.company_data, how='outer', on=['company_id'])

    def validate(self, target: IDataProviderTarget) -> bool:
        """
        Validate a target, meaning it should:

        * Have a valid type
        * Not be finished
        * A valid end year

        :param target: The target to validate
        :return: True if it's a valid target, false if it isn't
        """
        return True



    def _prepare_target(self, target: IDataProviderTarget):
        """
        Prepare a target for usage later on in the process.

        :param target:
        :return:
        """

        return target

    def prepare_targets(self, targets: List[IDataProviderTarget]):


        return targets

    def _find_target(self, row: pd.Series, target_columns: List[str]) -> pd.Series:
        """
        Find the target that corresponds to a given row. If there are multiple targets available, filter them.

        :param row: The row from the data set that should be looked for
        :param target_columns: The columns that need to be returned
        :return: returns records from the input data, which contains company and target information, that meet specific criteria. For example, record of greatest emissions_in_scope
        """
        return row

    def group_targets(self):
        """
        Group the targets and create the 9-box grid (short, mid, long * s1s2, s3, s1s2s3).
        Group valid targets by category & filter multiple targets#
        Input: a list of valid targets for each company:
        For each company:

        Group all valid targets based on scope (S1+S2 / S3 / S1+S2+S3) and time frame (short / mid / long-term)
        into 6 categories.

        For each category: if more than 1 target is available, filter based on the following criteria
        -- Highest boundary coverage
        -- Latest base year
        -- Target type: Absolute over intensity
        -- If all else is equal: average the ambition of targets
        """

        grid_columns = [self.c.COLS.COMPANY_ID, self.c.COLS.TIME_FRAME, self.c.COLS.SCOPE]
        companies = self.company_data[self.c.COLS.COMPANY_ID].unique()
        scopes = [EScope.S1S2]
        empty_columns = [column for column in self.target_data.columns if column not in grid_columns]
        extended_data = pd.DataFrame(
            list(itertools.product(*[companies, ETimeFrames, scopes] + [[None]] * len(empty_columns))),
            columns=grid_columns + empty_columns)
        target_columns = extended_data.columns
        self.data = extended_data.apply(lambda row: self._find_target(row, target_columns), axis=1)
