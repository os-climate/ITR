# Go to folder "examples" and run this app with `python ITR_UI.py` and
# visit http://127.0.0.1:8050/ in your web browser


import pandas as pd
import numpy as np
import json
import pickle
import os
import base64
import io
import warnings
import ast

import dash
from dash import html
from dash import dcc

import dash_bootstrap_components as dbc # should be installed separately

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go

import ITR

from ITR.configs import ITR_median, ITR_mean
from ITR.data.data_warehouse import DataWarehouse
from ITR.portfolio_aggregation import PortfolioAggregationMethod
from ITR.temperature_score import TemperatureScore

from ITR.data.base_providers import BaseProviderProductionBenchmark, BaseProviderIntensityBenchmark
from ITR.data.template import TemplateProviderCompany
from ITR.interfaces import EScope, ETimeFrames, EScoreResultType, IEIBenchmarkScopes, IProductionBenchmarkScopes, ProjectionControls
# from ITR.configs import LoggingConfig

from ITR.data.osc_units import Q_, asPintSeries
from pint import Quantity
from pint_pandas import PintType

import logging

import sys
import argparse

# from pint import pint_eval
# pint_eval.tokenizer = pint_eval.uncertainty_tokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') # LoggingConfig.FORMAT
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

logger.info("Start!")

examples_dir ='' #'examples'
data_dir="data"
data_json_units_dir="json-units"
root = os.path.abspath('')

# Set input filename (from commandline or default)
parser = argparse.ArgumentParser()
parser.add_argument('file')
if len(sys.argv)>1:
    args = parser.parse_args()
    company_data_path = args.file
else:
    company_data_path = os.path.join(root, examples_dir, data_dir, "20220927 ITR V2 Sample Data.xlsx")

# Production benchmark (there's only one)
benchmark_prod_json_file = "benchmark_production_OECM.json"
benchmark_prod_json = os.path.join(root, examples_dir, data_dir, data_json_units_dir, benchmark_prod_json_file)
with open(benchmark_prod_json) as json_file:
    parsed_json = json.load(json_file)
prod_bms = IProductionBenchmarkScopes.parse_obj(parsed_json)
base_production_bm = BaseProviderProductionBenchmark(production_benchmarks=prod_bms)
logger.info('Load production benchmark from {}'.format(benchmark_prod_json_file))

# Emission intensities
benchmark_EI_OECM_PC_file = "benchmark_EI_OECM_PC.json"
benchmark_EI_OECM_S3_file = "benchmark_EI_OECM_S3.json"
benchmark_EI_OECM_file = "benchmark_EI_OECM.json" # Deprecated!
benchmark_EI_TPI_15_file = "benchmark_EI_TPI_1_5_degrees.json"
benchmark_EI_TPI_file = "benchmark_EI_TPI_2_degrees.json"
benchmark_EI_TPI_below_2_file = "benchmark_EI_TPI_below_2_degrees.json"

# loading dummy portfolio
df_portfolio = pd.read_excel(company_data_path, sheet_name="Portfolio")
companies = ITR.utils.dataframe_to_portfolio(df_portfolio)
logger.info('Load dummy portfolio from {}. You could upload your own portfolio using the template.'.format(company_data_path))

# matplotlib is integrated with Pint's units system: https://pint.readthedocs.io/en/0.18/plotting.html
# But not so plotly.  This function attempts to dequantify all units and return the magnitudes in their natural base units.

def dequantify_plotly(px_func, df, **kwargs):
    # `df` arrives with columns like "plot_in_x" and "plot_in_y"
    # `kwargs` arrives as a dict with things like 'x': 'plot_in_x' and 'y':'plot_in_y'
    new_df = df.copy()
    new_kwargs = dict(kwargs)
    for col in ['x', 'y']:
        s = df[kwargs[col]]
        if isinstance(s.dtype, PintType):
            pass
        elif s.map(lambda x: isinstance(x, Quantity)).any():
            item0 = s.values[0]
            s = s.astype(f"pint[{item0.u}]")
        else:
            assert kwargs[col] in df.columns
            new_kwargs[col] = kwargs[col]
            continue
        new_kwargs[col] = f"{kwargs[col]} (units = {str(s.pint.u)})"
        new_df[new_kwargs[col]] = ITR.nominal_values(s.pint.m)
        if ITR.HAS_UNCERTAINTIES:
            # uncertainties in production affect cumulative_budget
            # uncertainties in emissinos affect cumulative_usage
            new_kwargs[f"error_{col}"] = f"error_{kwargs[col]}"
            new_df[new_kwargs[f"error_{col}"]] = ITR.std_devs(s.pint.m)
    if 'hover_data' in kwargs:
        # No error terms in hover data
        for col in kwargs['hover_data']:
            s = df[col]
            if isinstance(s.dtype, PintType):
                new_df[col] = ITR.nominal_values(s.pint.m)
            elif s.map(lambda x: isinstance(x, Quantity)).any():
                item0 = s.values[0]
                s = s.astype(f"pint[{item0.u}]")
                new_df[col] = ITR.nominal_values(s.pint.m)
    # **kwargs typically {'x': 'cumulative_target', 'y': 'cumulative_budget', 'size': 'investment_value', 'color': 'sector', 'labels': {'color': 'Sector'}, 'hover_data': ['company_name', 'investment_value', 'temperature_score'], 'title': 'Overview of portfolio'}
    
    return px_func (new_df, **new_kwargs)


# nice cheatsheet for managing layout via className attribute: https://hackerthemes.com/bootstrap-cheatsheet/

# Define app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], # theme should be written in CAPITAL letters; list of themes https://www.bootstrapcdn.com/bootswatch/
                meta_tags=[{'name': 'viewport', # this thing makes layout responsible to mobile view
                            'content': 'width=device-width, initial-scale=1.0'}]
                )
app.title = "ITR Tool" # this puts text to the browser tab
server = app.server

filter_box = dbc.Row( # We are a row of the left-side column box
    children=[
        dbc.Row(
            [
                dbc.Col(
                    dbc.Label("\N{thermometer} Individual temperature score"),
                    width=9,
                ),
                dbc.Col(
                    [
                        dbc.Button("\N{books}", id="hover-target2", color="link", n_clicks=0),
                        dbc.Popover(dbc.PopoverBody(
                            "Focus on companies from portfolio with specific temperature score"), id="hover2",
                                    target="hover-target2", trigger="hover"),
                    ], width=2, align="center",
                ),
            ],
            align="center",
        ),
        dcc.RangeSlider(
            id="temp-score-range",
            min=0, max=8.5, value=[0, 8.5],
            step=0.5,
            marks={i / 10: str(i / 10) for i in range(0, 86, 5)},
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Label("\N{factory} Focus on a specific sector "),
                    width=9,
                ),
                dbc.Col(
                    [
                        dbc.Button("\N{books}", id="hover-target3", color="link", n_clicks=0),
                        dbc.Popover(dbc.PopoverBody(
                            "Scope of sectors could be different for different emission benchmark.\nScope of sectors covered by the tool is constantly growing."),
                                    id="hover3", target="hover-target3", trigger="hover"),
                    ], width=2,
                ),
            ],
            align="center",
        ),
        dcc.Dropdown(id="sector-dropdown",
                     options=[{'label': 'All Sectors', 'value': 'all_values'}],
                     value='all_values',
                     clearable=False,
                     placeholder="Select a sector"),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Label("\N{globe with meridians} Focus on a specific region "),
                    width=9,
                ),
                dbc.Col(
                    [
                        dbc.Button("\N{books}", id="hover-target4", color="link", n_clicks=0),
                        dbc.Popover(dbc.PopoverBody(
                            "Scope of countries could be different for different emission benchmark"),
                                    id="hover4", target="hover-target4", trigger="hover"),
                    ], width=2,
                ),
            ],
            align="center",
        ),
        dcc.Dropdown(id="region-dropdown",
                     options=[{'label': 'All Regions', 'value': 'all_values'}],
                     value='all_values',
                     clearable=False,
                     placeholder="Select a region"),
    ],
)

benchmark_box = dbc.Row(
    children=[
        dbc.Row(  # Select Benchmark
            [
                dbc.Col(
                    dbc.Label("\N{bar chart} Select Emissions Intensity benchmark "),
                    width=9,
                ),
                dbc.Col(
                    [
                        dbc.Button("\N{books}", id="hover-target5", color="link", n_clicks=0),
                        dbc.Popover(dbc.PopoverBody(
                            "This benchmark describes emission intensities projection for different regions and sectors"),
                                    id="hover5", target="hover-target5", trigger="hover"),
                    ], width=2,
                ),
            ],
            align="center",
        ),
        dcc.Dropdown(id="eibm-dropdown",
                     options=[  # 16.05.2022: make this dynamic
                         {'label': 'OECM (Prod-Centric) 1.5 degC', 'value': 'OECM_PC'},
                         {'label': 'OECM (Scope 3) 1.5 degC', 'value': 'OECM_S3'},
                         {'label': 'OECM (Deprecated) 1.5 degrees', 'value': 'OECM'},
                         {'label': 'TPI 1.5 degrees', 'value': 'TPI_15_degrees'},
                         {'label': 'TPI 2 degrees', 'value': 'TPI_2_degrees'},
                         {'label': 'TPI below 2 degrees', 'value': 'TPI_below_2_degrees'}
                     ],
                     value='OECM_S3',
                     clearable=False,
                     placeholder="Select Emissions Intensity benchmark"),
        html.Div(id='hidden-div', style={'display': 'none'}),
        html.Hr(),  # small space from the top
        dbc.Row(  # Mean / Median projection
            [
                dbc.Col(
                    dbc.Label("\N{level slider} Select method for projection"),
                    width=9,
                ),
                dbc.Col(
                    [
                        dbc.Button("\N{books}", id="hover-target6", color="link", n_clicks=0),
                        dbc.Popover(dbc.PopoverBody(
                            "Select method of averaging trend of emission intensities projections"),
                                    id="hover6", target="hover-target6", trigger="hover"),
                    ], width=2,
                ),
            ],
            align="center",
        ),
        dcc.RadioItems(
            id="projection-method",
            options=[
                {'label': 'Median', 'value': 'median'},
                {'label': 'Mean', 'value': 'mean'},
            ],
            value='median',
            inputStyle={"margin-right": "10px", "margin-left": "30px"},
            inline=True,
        ),
        html.Hr(),  # small space from the top
        dbc.Row(  # Scope selection
            [
                dbc.Col(
                    dbc.Label("\N{abacus} Select Scope(s) to Evaluate"),
                    width=9,
                ),
                dbc.Col(
                    [
                        dbc.Button("\N{books}", id="hover-target10", color="link", n_clicks=0),
                        dbc.Popover(dbc.PopoverBody(
                            "Select Scope(s) to process (S1, S1+S2, S3, S1+S1+S3, or All Scopes)"),
                                    id="hover10", target="hover-target10", trigger="hover"),
                    ], width=2,
                ),
            ],
            align="center",
        ),
        dcc.RadioItems(
            id="scope-options",
            options=[
                {'label': 'S1', 'value': 'S1'},
                {'label': 'S1+S2', 'value': 'S1S2'},
                {'label': 'S3', 'value': 'S3'},
                {'label': 'S1S2S3', 'value': 'S1S2S3'},
                {'label': 'All Scopes', 'value': ''},
            ],
            value='S1S2S3',
            inputStyle={"margin-right": "10px", "margin-left": "30px"},
            inline=True,
        ),
        html.Hr(),  # small space from the top
        dbc.Row(  # Winsorization of scenarios
            [
                dbc.Col(
                    dbc.Label("\N{wrench} Select winsorization value cap range"),
                    width=9,
                ),
                dbc.Col(
                    [
                        dbc.Button("\N{books}", id="hover-target8", color="link", n_clicks=0),
                        dbc.Popover(dbc.PopoverBody(
                            "Select which extreme datapoints of historical emission intensities you would like to exclude from calculations of projections"),
                                    id="hover8", target="hover-target8", trigger="hover"),
                    ], width=2,
                ),
            ],
            align="center",
        ),
        dcc.RangeSlider(
            id="scenarios-cutting",
            min=0, max=100,
            value=[ProjectionControls.LOWER_PERCENTILE * 100, ProjectionControls.UPPER_PERCENTILE * 100],
            step=10,
            marks={i: str(i) for i in range(0, 101, 10)},
            allowCross=False
        ),
        html.Hr(),  # small space from the top
        dbc.Row(  # Set ProjectionControls TARGET_YEAR
            [
                dbc.Col(
                    dbc.Label("\N{wrench} Select projection end year (2025-2050)"),
                    width=9,
                ),
                dbc.Col(
                    [
                        dbc.Button("\N{calendar}", id="hover-target9", color="link", n_clicks=0),
                        dbc.Popover(dbc.PopoverBody(
                            "Select the ending year of the benchmark analysis"),
                                    id="hover9", target="hover-target9", trigger="hover"),
                    ], width=4,
                ),
            ],
            align="center",
        ),
        dcc.Slider(
            id="target-year",
            min=2025, max=2050,
            value=2050,
            step=5,
            marks={i: str(i) for i in range(2025, 2051, 5)},
        ),
        html.Div([html.Span("Total benchmark budget through "), html.Span(id="target-bm-year"), html.Span(": "), html.Span(id="bm-budget")]),
        html.Div([html.Span("ITR of benchmark: "), html.Span(id="tempscore-ty")]),
    ],
)

itr_titlebar = dbc.Row(  # upload portfolio
    [
        dbc.Col(  # upload OS-C logo
            dbc.CardImg(
                src="https://os-climate.org/wp-content/uploads/sites/138/2021/10/OSC-Logo.png",
                className='align-self-center',
                # 'h-60 w-60 float-middle align-middle', # reducing size and alligning
                        bottom=False),
            width=2,
            align="center",
        ),
        dbc.Col(
            [
                html.H1(id="banner-title", children=[
                    html.A("OS-Climate Portfolio Alignment Tool", href="https://github.com/plotly/dash-svm",
                           style={"text-decoration": "none", "color": "inherit"})]),
                html.Div(
                    children='Prototype tool for calculating the Implied Temperature Rise of investor portfolio in the steel and electric utilities sectors \N{deciduous tree}'),
            ],
            width=8,
        ),
        dbc.Col([
            dbc.Spinner([html.H1(id="dummy-output-info", style={'color': 'white'}),
                         html.Data(id="spinner-warehouse"),
                         html.Data(id="spinner-eibm"),
                         html.Data(id="spinner-ty"),
                         html.Data(id="spinner-ts"),
                         html.Data(id="spinner-graphs"),
                         html.Data(id="spinner-xlsx"),
                         html.Data(id="spinner-reset"),
                         ], color="primary",
                        spinner_style={"width": "3rem", "height": "3rem"}),  # Spinner implementations
        ],
                width=1,
                ),
    ],
    justify='between',  # for this to work you need some space left (in total there 12 columns)
    align='center',
)

itr_filters_and_benchmarks = dbc.Col(
    [  # filters pane
        dbc.Card(
            [
                dbc.Row(
                    [  # Row with key figures
                        dbc.Col(html.H5("Filters", className="pf-filter")),  # PF score
                        dbc.Col(
                            html.Div(
                                dbc.Button("Reset filters",
                                           id="reset-filters-button",
                                           outline=True, color="dark", size="sm",
                                           className="me-md-2"
                                           ),
                                className="d-grid gap-2 d-md-flex justify-content-md-end"
                            )
                        ),
                    ]
                ),
                html.P("Select part of your portfolio", className="text-black-50"),
                filter_box,
            ], body=True
        ),
        html.Br(),
        dbc.Card(
            [
                html.H5("Benchmarks", className="macro-filters"),
                html.P("Here you could adjust benchmarks of calculations",
                       className="text-black-50"),
                benchmark_box,
            ], body=True
        ),
    ],
    width=3,
)

itr_main_figures = dbc.Col(
    [  # main pane
        dbc.Card([
            dbc.Row(  # Row with key figures
                [
                    dbc.Col(  # PF score
                        dbc.Card(
                            [
                                html.H1(id="output-info"),
                                html.Div('Portfolio-level temperature rating of selected companies',
                                         style={'color': 'black', 'fontSize': 16}),
                                html.Div('in delta degree Celsius',
                                         style={'color': 'grey', 'fontSize': 10}),
                            ], body=True
                        ),
                    ),
                    dbc.Col(  # Portfolio EVIC
                        dbc.Card(
                            [
                                html.H1(id="evic-info"),
                                html.Div('Enterprise Value incl. Cash of selected portfolio',
                                         style={'color': 'black', 'fontSize': 16}),
                                html.Div('in billions of template curr',
                                         style={'color': 'grey', 'fontSize': 10}),
                            ], body=True
                        ),
                    ),
                    dbc.Col(  # Portfolio notional
                        dbc.Card(
                            [
                                html.H1(id="pf-info"),
                                html.Div('Total Notional of a selected portfolio',
                                         style={'color': 'black', 'fontSize': 16}),
                                html.Div('in millions of template curr',
                                         style={'color': 'grey', 'fontSize': 10}),
                            ], body=True
                        ),
                    ),
                    dbc.Col(  # Number of companies
                        dbc.Card(
                            [
                                html.H1(id="comp-info"),
                                html.Div('Number of companies in the selected portfolio',
                                         style={'color': 'black', 'fontSize': 16}),
                                html.Div('# of companies', style={'color': 'grey', 'fontSize': 10}),
                            ], body=True
                        ),
                    ),
                ],
            ),
            dbc.Row(  # row with 2 graphs
                [
                    dbc.Col(dcc.Graph(id="co2-usage-vs-budget"), width=8),  # big bubble graph
                    dbc.Col(dcc.Graph(id="itr-coverage"), ),  # covered stacked bar graph
                ],
            ),
            dbc.Row(  # row with 2 graphs
                [
                    dbc.Col(dcc.Graph(id="industry-region-heatmap")),
                    dbc.Col(dcc.Graph(id="highest-ts-barchart")),
                ]
            ),
            dbc.Row(  # row with 1 bar graph
                [
                    dbc.Col(dcc.Graph(id="ts-aggr-barchart")),
                ]
            ),
        ], body=True),
        html.Br(),
    ],
    width=9,
)

itr_portfolio = dbc.Card(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.H5(
                        "Table below contains details about the members of the selected portfolio"),
                    width=10,
                ),
                dbc.Col(
                    html.Div(
                        [
                            dbc.Button("Export table to spreadsheet",
                                       id="export-to-excel",
                                       size="sm", className="me-md-2"
                                       ),
                            dcc.Download(id="download-dataframe-xlsx"),
                        ],
                        className="d-grid gap-2 d-md-flex justify-content-md-end"
                    ),
                    width=2,
                ),
            ],
            align="center",
        ),
        html.Br(),
        html.Div(id='display-excel-data'),
    ], body=True
)

# Define Layout
app.layout = dbc.Container(  # always start with container
    children=[
        html.Hr(),  # small space from the top
        itr_titlebar,           # This has static text and the spinner
        html.Hr(),
        dbc.Row(
            [
                itr_filters_and_benchmarks,
                itr_main_figures,
            ]
        ),
        itr_portfolio,
        html.Data(id="sector"),
        html.Data(id="region"),
        html.Data(id="scope-column"),
        html.Data(id="scope-options-ts"),
        html.Data(id="scope-options-reset"),
        dcc.Store(id="warehouse"),
        dcc.Store(id="warehouse-eibm"),
        dcc.Store(id="warehouse-ty"),
        dcc.Store(id="portfolio-df"),
        html.Data(id="sector-region-scope-ty"),
        html.Data(id="sector-dropdown-ts"),
        html.Data(id="sector-dropdown-reset"),
        html.Data(id="region-dropdown-ts"),
        html.Data(id="region-dropdown-reset"),
        # dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns])
    ],
    style={"max-width": "1500px"},
)

@app.callback(
    Output("warehouse", "data"),
    Output("spinner-warehouse", "value"),

    Input("banner-title", "children"),) # Just something to get us going...
def warehouse_new(banner_title):
    # load company data
    template_company_data = TemplateProviderCompany(company_data_path)
    Warehouse = DataWarehouse(template_company_data, benchmark_projected_production=None, benchmarks_projected_ei=None,
                              estimate_missing_data=DataWarehouse.estimate_missing_s3_data)
    return (json.dumps(pickle.dumps(Warehouse), default=str),
            "Spin-warehouse")

@app.callback(
    Output("warehouse-eibm", "data"),
    Output("sector-dropdown", "options"),
    Output("region-dropdown", "options"),
    Output("scope-options", "value"),
    Output("spinner-eibm", "value"),  # fake for spinner

    Input("warehouse", "data"),
    Input("eibm-dropdown", "value"),
    Input("projection-method", "value"),
    Input("scenarios-cutting", "value"),  # winzorization slide

    prevent_initial_call=True,)
# load default intensity benchmarks
def recalculate_individual_itr(warehouse_pickle_json, eibm, proj_meth, winz):
    '''
    Reload Emissions Intensity benchmark from a selected file
    :param eibm: Emissions Intensity benchmark identifier
    '''
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]  # to catch which widgets were pressed
    Warehouse = pickle.loads(ast.literal_eval(json.loads(warehouse_pickle_json)))

    if 'scenarios-cutting' in changed_id or 'projection-method' in changed_id:  # if winzorization params were changed
        if proj_meth == 'median':
            Warehouse.company_data.projection_controls.TREND_CALC_METHOD = ITR_median
        else:
            Warehouse.company_data.projection_controls.TREND_CALC_METHOD = ITR_mean
        Warehouse.company_data.projection_controls.LOWER_PERCENTILE = winz[0] / 100
        Warehouse.company_data.projection_controls.UPPER_PERCENTILE = winz[1] / 100
        # Trajectories are company-specific and don't depend on benchmarks
        Warehouse.update_trajectories()

    if 'eibm-dropdown' in changed_id or Warehouse.benchmarks_projected_ei is None:
        if eibm == 'OECM_PC':
            benchmark_file = benchmark_EI_OECM_PC_file
            scope = 'S1S2'
        elif eibm == 'OECM_S3':
            benchmark_file = benchmark_EI_OECM_S3_file
            scope = 'S1S2S3'
        elif eibm == 'TPI_2_degrees':
            benchmark_file = benchmark_EI_TPI_file
            scope = None
        elif eibm == 'TPI_15_degrees':
            benchmark_file = benchmark_EI_TPI_15_file
            scope = None
        elif eibm == 'OECM':
            benchmark_file = benchmark_EI_OECM_file
            scope = None
            logger.info('OECM benchmark is for backward compatibility only.  Use OECM_PC instead.')
        else:
            benchmark_file = benchmark_EI_TPI_below_2_file
            scope = None
        # load intensity benchmarks
        benchmark_EI = os.path.join(root, examples_dir, data_dir, data_json_units_dir, benchmark_file)
        with open(benchmark_EI) as json_file:
            parsed_json = json.load(json_file)
        EI_bm = BaseProviderIntensityBenchmark(EI_benchmarks=IEIBenchmarkScopes.parse_obj(parsed_json))
        Warehouse.update_benchmarks(base_production_bm, EI_bm)

    # Company fundamentals are in template_company_data.df_fundamentals
    df_fundamentals = Warehouse.company_data.df_fundamentals
    sectors = sorted(set(df_fundamentals.sector) & set(EI_bm._EI_df.index.get_level_values('sector')))
    regions = sorted(set(df_fundamentals.region) & set(EI_bm._EI_df.index.get_level_values('region')))
    # portfolio data is on the "Portfolio" page, and in df_portfolio

    return (json.dumps(pickle.dumps(Warehouse), default=str),
            [{"label": i, "value": i} for i in sectors] + [{'label': 'All Sectors', 'value': 'all_values'}],
            [{"label": i, "value": i} for i in regions] + [{'label': 'All Regions', 'value': 'all_values'}],
            scope,
            "Spin-eibm")

@app.callback(
    # In the future, also adjust sector-dropdown options
    Output("warehouse-ty", "data"),
    Output("sector-region-scope-ty", "value"),
    Output("bm-budget", "children"),
    Output("target-bm-year", "children"),
    Output("tempscore-ty", "children"),
    Output("spinner-ty", "value"),  # fake for spinner

    Input("warehouse-eibm", "data"),
    Input("sector-dropdown", "value"),
    Input("region-dropdown", "value"),
    Input("scope-options", "value"),
    Input("target-year", "value"),

    prevent_initial_call=True,)
def recalculate_target_year_ts(warehouse_pickle_json, sector, region, scope, target_year):
    '''
    When changing endpoint of benchmark budget calculations, update total budget and benchmark ITR resulting therefrom.
    We assume that 'Global' is the all-world, not rest-of-world, for any benchmark.
    Where possible, honor users' selection of SCOPE, REGION, and SCOPE_LIST, and where necessary, adjust
    REGION and SCOPE_LIST to accommodate SECTOR selection.
    '''

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]  # to catch which widgets were pressed
    Warehouse = pickle.loads(ast.literal_eval(json.loads(warehouse_pickle_json)))

    # All benchmarks use OECM production
    prod_bm = Warehouse.benchmark_projected_production
    EI_bm = Warehouse.benchmarks_projected_ei

    df_fundamentals = Warehouse.company_data.df_fundamentals
    sectors = sorted(set(df_fundamentals.sector) & set(EI_bm._EI_df.index.get_level_values('sector')))
    regions = sorted(set(df_fundamentals.region) & set(EI_bm._EI_df.index.get_level_values('region')))

    df = EI_bm._EI_df
    if sector=='all_values' or sector not in sectors:
        sector = None
        EI_sectors = df[df.index.get_level_values('sector').isin(sectors)]
        sector_scopes = df[df.index.get_level_values('sector').isin(sectors)][df.columns[0]].groupby(['sector', 'scope']).count()
        if not sector_scopes.index.get_level_values('sector').duplicated().any():
            # TPI is a scope-per-sector benchmarks, so looks best when we see all scopes
            scope = None
            EI_scopes = sector_scopes.index.get_level_values('scope').unique()
        else:
            EI_scopes = EI_sectors.index.get_level_values('scope').unique()
    else:
        EI_sectors = df.loc[sector]
        EI_scopes = EI_sectors.index.get_level_values('scope').unique()

    # So...our Warehouse doesn't retain side-effects, so we cannot "set-and-forget"
    # Instead, we have to re-make the change for the benefit of downstream users...
    EI_bm.projection_controls.TARGET_YEAR = target_year
    Warehouse.company_data.projection_controls.TARGET_YEAR = target_year

    if region=='all_values':
        # In this case, we want to deal exclusively with Global data, not sum things that will add to more than 1.0
        region = 'Global'
    if not scope or EScope[scope] not in EI_scopes:
        scope_list = None
    else:
        scope_list = [ EScope[scope] ]
    if sector:
        # Get some benchmark temperatures for < 2050 using OECM
        bm_base_prod_in_region = [ bm.base_year_production
                                   for bm in prod_bm._productions_benchmarks.AnyScope.benchmarks
                                   if bm.sector==sector and bm.region==region ][0]
        # There is no meaningful scope in production...
        base_prod_df = prod_bm._prod_df.loc[sector].droplevel(['scope']).apply(lambda col: col.mul(bm_base_prod_in_region))
        sector_scope = None if scope_list is None else scope_list[0]
        if sector_scope is None:
            if EScope.S1S2S3 in EI_scopes:
                sector_scope = EScope.S1S2S3
            else:
                if len(EI_scopes) != 1:
                    raise ValueError(f"Non-unique scope for sector {sector}")
                sector_scope = EI_scopes[0]
        elif sector_scope not in EI_scopes:
            raise ValueError(f"Scope {sector_scope} not in benchmark for sector {sector}")

        # FIXME: Methinks we need to reinterpret the OECM production benchmarks for TPI
        intensity_df = EI_bm._EI_df.loc[(sector, region, sector_scope)]
        target_year_cum_co2 = (base_prod_df.loc[base_prod_df.index.intersection(intensity_df.index)]
                               .mul(intensity_df)
                               .cumsum()
                               .astype('pint[Gt CO2e]'))
    elif 'Energy' in EI_bm._EI_df.index.get_level_values('sector'):
        # Get some benchmark temperatures for < 2050 using OECM
        bm_base_prod_in_region = [ bm.base_year_production
                                   for bm in prod_bm._productions_benchmarks.AnyScope.benchmarks
                                   if bm.sector=='Energy' and bm.region==region ][0]
        # In the OECM benchmark, Energy Scope 1, Scope 2, and Scope 3 actually account for all global emissions
        base_prod_df = prod_bm._prod_df.loc[('Energy', region, EScope.AnyScope)].mul(bm_base_prod_in_region)
        target_year_cum_co2 = (base_prod_df
                               .mul(EI_bm._EI_df.loc[('Energy', region, scope_list[0] if scope_list else EScope.S1S2S3)])
                               .cumsum()
                               .astype('pint[Gt CO2e]'))
    else:
        # We have to manually sum all the benchmark data...and it only works for 'Global' right now (and need to reflect such is selection options)
        region = 'Global'
        scope_list = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bm_base_prod_by_sector = pd.Series({bm.sector: bm.base_year_production
                                                for bm in prod_bm._productions_benchmarks.AnyScope.benchmarks
                                                if bm.region==region})
            bm_base_prod_by_sector.index.name = 'sector'
        # base_prod_df = prod_bm._prod_df.loc[(slice(None), region, EScope.AnyScope)].apply(lambda col: col.mul(bm_base_prod_by_sector))
        base_prod_df = prod_bm._prod_df.loc[(slice(None), region, EScope.AnyScope)].apply(lambda col: col.combine(bm_base_prod_by_sector,
                                                                                                                  lambda x, y: ITR.data.osc_units.ureg(f"{x} * {y}")))
        # TPI has Shipping vs. OECM's Trucking
        # TPI has Diversified Mining and Airlines (not yet imported from OECM)
        # TPI has Oil & Gas vs. OECM's separate Oil and Gas sectors
        common_sectors = base_prod_df.index.intersection(EI_bm._EI_df.loc[(slice(None), region, slice(None))].index.get_level_values('sector'))
        target_year_cum_co2 = (base_prod_df.loc[common_sectors]
                               .mul(EI_bm._EI_df.loc[(common_sectors, region, slice(None))])
                               .groupby(['region']).sum()
                               .squeeze()
                               .cumsum()
                               .astype('pint[Gt CO2e]'))
                               # The logic below needs everything condensed down to the 'Global' region (whatever scope that is)

    total_target_co2 = target_year_cum_co2.loc[EI_bm.projection_controls.TARGET_YEAR]
    total_final_co2 = target_year_cum_co2.loc[EI_bm._EI_df.columns[-1]]
    ts_cc = ITR.configs.TemperatureScoreConfig.CONTROLS_CONFIG
    # FIXME: Note that we cannot use ts_cc.scenario_target_temperature because that doesn't track the benchmark value
    # And we cannot make it track the benchmark value because then it becomes another global variable that would break Dash.
    target_year_ts = EI_bm._benchmark_temperature + (total_target_co2 - total_final_co2) * ts_cc.tcre_multiplier
    return (
        json.dumps(pickle.dumps(Warehouse), default=str),
        json.dumps([sector, region, scope]),
        f"{round(total_target_co2.m, 3)} Gt CO2e",
        f"{round(target_year_ts.m, 3)}˚C",
        EI_bm.projection_controls.TARGET_YEAR,
        "Spin-ty",
    )

@app.callback(
    Output("portfolio-df", "data"),
    Output("scope-column", "value"),
    Output("spinner-ts", "value"),  # fake for spinner

    Input("warehouse-ty", "data"),
    Input("sector-region-scope-ty", "value"),
    Input("target-bm-year", "children"),
    Input("tempscore-ty", "children"),

    prevent_initial_call=True,)
def calc_temperature_score(warehouse_pickle_json, sector_region_scope, *_):
    global companies

    Warehouse = pickle.loads(ast.literal_eval(json.loads(warehouse_pickle_json)))
    EI_bm = Warehouse.benchmarks_projected_ei
    sector, region, scope = json.loads(sector_region_scope)
    if not scope:
        scope_list = None
    else:
        scope_list = [ EScope[scope] ]
    temperature_score = TemperatureScore(
        time_frames = [ETimeFrames.LONG],
        scopes=scope_list, # None means "use the appropriate scopes for the benchmark
        # Options for the aggregation method are WATS, TETS, AOTS, MOTS, EOTS, ECOTS, and ROTS
        aggregation_method=PortfolioAggregationMethod.WATS
    )
    df = temperature_score.calculate(data_warehouse=Warehouse, portfolio=companies)
    # Filter out whatever is not in the benchmark
    bm_filter = EI_bm._EI_df.reset_index()[['sector', 'scope']].drop_duplicates() # Don't need rows from each region
    df = df.merge(bm_filter, on=['sector', 'scope'], how='inner')

    return (df.drop(columns=['historic_data', 'target_data']).to_json(orient='split', default_handler=str),
            'ghg_' + (EI_bm.scope_to_calc if hasattr(EI_bm, 'scope_to_calc') else EScope.S1S2).name.lower(),
            "Spin-ts",)

@app.callback(
    Output("co2-usage-vs-budget", "figure"),     # fig1
    Output("itr-coverage", "figure"),            # fig5
    Output("industry-region-heatmap", "figure"), # heatmap_fig
    Output("highest-ts-barchart", "figure"),     # high_score_fig
    Output("ts-aggr-barchart", "figure"), # port_score_diff_methods_fig
    Output("spinner-graphs", "value"),    # fake for spinner
    Output('output-info', 'children'),    # portfolio score
    Output('output-info', 'style'),       # conditional color
    Output('evic-info', 'children'),      # portfolio evic
    Output('pf-info', 'children'),        # portfolio notional
    Output('comp-info', 'children'),      # num of companies
    Output('display-excel-data', 'children'), # Table

    Input("portfolio-df", "data"),
    Input("temp-score-range", "value"),
    State("sector-dropdown", "value"),
    State("region-dropdown", "value"),
    State("scope-options", "value"),
    State("scope-column", "value"),

    prevent_initial_call=True,)
def update_graph(
        portfolio_json,
        te_sc,
        sec, reg, scope,
        scope_column,          # ghg_s1s2 or ghg_s3
):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]  # to catch which widgets were pressed
    amended_portfolio = pd.read_json(portfolio_json, orient='split')
    # Why does this get lost in translation?
    amended_portfolio.index.name='company_id'
    amended_portfolio = amended_portfolio.assign(scope=lambda x: x.scope.map(lambda y: EScope[y]),
                                                 time_frame=lambda x: x.time_frame.map(lambda y: ETimeFrames[y]),
                                                 score_result_type=lambda x: x.score_result_type.map(lambda y: EScoreResultType[y]),
                                                 temperature_score=lambda x: x.temperature_score.map(Q_).astype('pint[delta_degC]'),
                                                 ghg_s1s2=lambda x: x.ghg_s1s2.map(Q_).astype('pint[t CO2e]'),
                                                 # FIXME: we're going to have to deal with NULL ghg_s3, and possible ghg_s1s2
                                                 ghg_s3=lambda x: x.ghg_s3.map(Q_).astype('pint[t CO2e]'),
                                                 cumulative_budget=lambda x: x.cumulative_budget.map(Q_).astype('pint[t CO2e]'),
                                                 cumulative_trajectory=lambda x: x.cumulative_trajectory.map(Q_).astype('pint[t CO2e]'),
                                                 cumulative_target=lambda x: x.cumulative_target.map(Q_).astype('pint[t CO2e]'),
                                                 trajectory_score=lambda x: x.trajectory_score.map(Q_).astype('pint[delta_degC]'),
                                                 target_score=lambda x: x.target_score.map(Q_).astype('pint[delta_degC]'))

    temp_score_mask = (amended_portfolio.temperature_score >= Q_(te_sc[0], 'delta_degC')) & (
        amended_portfolio.temperature_score <= Q_(te_sc[1], 'delta_degC'))
    # Dropdown filters
    if sec == 'all_values':
        sec_mask = (amended_portfolio.sector != 'dummy')  # select all
    else:
        sec_mask = amended_portfolio.sector == sec
    if reg == 'all_values':
        reg_mask = (amended_portfolio.region != 'dummy')  # select all
    else:
        reg_mask = (amended_portfolio.region == reg)
    filt_df = amended_portfolio.loc[temp_score_mask & sec_mask & reg_mask]  # filtering
    if len(filt_df) == 0:  # if after filtering the dataframe is empty
        raise PreventUpdate

    if not scope:
        scope_list = None
    else:
        scope_list = [ EScope[scope] ]
    temperature_score = TemperatureScore(
        time_frames = [ETimeFrames.LONG],
        scopes=scope_list, # None means "use the appropriate scopes for the benchmark
        # Options for the aggregation method are WATS, TETS, AOTS, MOTS, EOTS, ECOTS, and ROTS
        aggregation_method=PortfolioAggregationMethod.WATS
    )
    aggregated_scores = temperature_score.aggregate_scores(filt_df)  # calc temp score for companies left in portfolio

    logger.info(f"ready to plot!\n{filt_df}")

    # Scatter plot
    filt_df.loc[:, 'cumulative_usage'] = (filt_df.cumulative_target.fillna(filt_df.cumulative_trajectory)
                                          +filt_df.cumulative_trajectory.fillna(filt_df.cumulative_target))/2.0
    fig1_kwargs = dict(x="cumulative_budget", y="cumulative_usage",
                       size="investment_value",
                       color="sector", labels={"color": "Sector"},
                       hover_data=["company_name", "investment_value", "temperature_score"],
                       title="Overview of portfolio",
                       log_x=True, log_y=True)
    if sec and sec != 'all_values':
        fig1_kwargs['text'] = 'company_name'
    fig1 = dequantify_plotly(px.scatter, filt_df, **fig1_kwargs)
    min_xy = min(ITR.nominal_values(filt_df.cumulative_budget.pint.m).min(), ITR.nominal_values(filt_df.cumulative_usage.pint.m).min())/2.0
    max_xy = max(ITR.nominal_values(filt_df.cumulative_budget.pint.m).max(), ITR.nominal_values(filt_df.cumulative_usage.pint.m).max())*2.0
    fig1.add_shape(dict(type='line', line_dash="dash", line_color="red", opacity=0.5, layer='below',
                        yref='y', xref='x',
                        y0=min_xy, y1=max_xy, x0=min_xy, x1=max_xy))
    fig1.update_layout({'legend_title_text': '', 'transition_duration': 500})
    fig1.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="center", x=0.5))

    # Covered companies analysis; if we pre-filter portfolio-df, then we'll never have "uncovered" companies here
    coverage = filt_df.reset_index('company_id')[['company_id', scope_column, 'cumulative_target']].copy()
    coverage['coverage_category'] = np.where(coverage[scope_column].isnull(),
                                             np.where(ITR.isnan(coverage['cumulative_target'].pint.m), "Not Covered",
                                                      "Covered only<Br>by target"),
                                             np.where(~ITR.isnan(coverage[scope_column].pint.m) & ITR.isnan(
                                                         coverage['cumulative_target'].pint.m),
                                                      "Covered only<Br>by emissions",
                                                      "Covered by<Br>emissions and targets"))
    dfg = coverage.groupby('coverage_category').count().reset_index()
    dfg['portfolio']='Portfolio'
    fig5 = dequantify_plotly (px.bar, dfg, x='portfolio',y="company_id", color="coverage_category",title="Coverage of companies in portfolio")
    fig5.update_xaxes(visible=False) # hide axis
    fig5.update_yaxes(visible=False) # hide axis
    fig5.update_layout({'legend_title_text': '','transition_duration':500, 'plot_bgcolor':'white'})
    fig5.update_layout(legend=dict(yanchor="middle",y=0.5,xanchor="left",x=1)) # location of legend

    # Heatmap
    trace = go.Heatmap(
        x=filt_df.sector,
        y=filt_df.region,
        z=ITR.nominal_values(filt_df.temperature_score.pint.m),
        type='heatmap',
        colorscale='Temps',
        zmin = 1.2, zmax = 2.5,
    )
    data = [trace]
    heatmap_fig = go.Figure(data=data)
    heatmap_fig.update_layout(title="Industry vs Region ratings")

    # visualizing worst performers
    df_high_score = filt_df.sort_values('temperature_score', ascending=False).groupby('sector').head(
        4)  # keeping only companies with highest score for the chart
    df_high_score['company_name'] = df_high_score['company_name'].str.split(' ').str[
        0]  # taking just 1st word for the bar chart
    high_score_fig = dequantify_plotly(px.bar, df_high_score,
                                       x="company_name",
                                       y="temperature_score",
                                       color="sector", title="Highest temperature scores by company")
    high_score_fig.update_traces(textposition='inside', textangle=0)
    high_score_fig.update_yaxes(title_text='Temperature score', range=[1, 8.5])
    high_score_fig.update_layout({'legend_title_text': '', 'transition_duration': 500})
    high_score_fig.update_layout(xaxis_title=None,
                                 legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="center", x=0.5))

    # Calculate different weighting methods
    # FIXME: this is utter confusion with respect to scopes!
    def agg_score(agg_method):
        if not scope:
            scope_list = None
        else:
            scope_list = [ EScope[scope] ]
        temperature_score = TemperatureScore(time_frames=[ETimeFrames.LONG],
                                             scopes=scope_list,
                                             aggregation_method=agg_method)  # Options for the aggregation method are WATS, TETS, AOTS, MOTS, EOTS, ECOTS, and ROTS
        aggregated_scores = temperature_score.aggregate_scores(filt_df)
        agg_zero = Q_(0.0, 'delta_degC')
        agg_score = agg_zero
        if aggregated_scores.long.S1S2:
            agg_score = aggregated_scores.long.S1S2.all.score
        elif aggregated_scores.long.S1:
            agg_score = aggregated_scores.long.S1.all.score
        if aggregated_scores.long.S1S2S3:
            agg_score = agg_score + aggregated_scores.long.S1S2S3.all.score
        elif aggregated_scores.long.S3:
            agg_score = agg_score + aggregated_scores.long.S3.all.score
        elif agg_score == agg_zero:
            return [agg_method.value, Q_(np.nan, 'delta_degC')]
        return [agg_method.value, agg_score]

    agg_temp_scores = [agg_score(i) for i in PortfolioAggregationMethod]
    methods, scores = list(map(list, zip(*agg_temp_scores)))
    if ITR.HAS_UNCERTAINTIES:
        scores_n, scores_s = [*map(list, zip(*map(lambda x: (x.m.n, x.m.s) if isinstance(x.m, ITR.UFloat) else (x.m, 0.0), scores)))]
        if sum(scores_s) == 0:
            df_temp_score = pd.DataFrame(
                data={0: pd.Series(methods, dtype='string'),
                      1: pd.Series([round (n, 2) for n in scores_n]),
                      2: pd.Series(['magnitude'] * len(scores_n))}
            )
        else:
            df_temp_score = pd.concat([pd.DataFrame(
                data={0: pd.Series(methods, dtype='string'),
                      1: pd.Series([round(n-s, 2) for n,s in zip(scores_n, scores_s)]),
                      2: pd.Series(['nominal'] * len(scores_n))}),
                                       pd.DataFrame(
                data={0: pd.Series(methods, dtype='string'),
                      1: pd.Series([round (2*s, 2) for s in scores_s]),
                      2: pd.Series(['std_dev'] * len(scores_s))})])
    else:
        scores_n = list(map(lambda x: x.m, scores))
        df_temp_score = pd.DataFrame(
            data={0: pd.Series(methods, dtype='string'),
                  1: pd.Series([round (n, 2) for n in scores_n]),
                  2: pd.Series(['magnitude'] * len(scores_n))}
        )
        
    # Separate column for names on Bar chart
    # Highlight WATS and TETS
    Weight_Dict = {'WATS': 'Investment<Br>weighted',  # <Br> is needed to wrap x-axis label
                   'TETS': 'Total emissions<Br>weighted',
                   'EOTS': "Enterprise Value<Br>weighted",
                   'ECOTS': "Enterprise Value<Br>+ Cash weighted",
                   'AOTS': "Total Assets<Br>weighted",
                   'ROTS': "Revenues<Br>weigted",
                   'MOTS': 'Market Cap<Br>weighted'}
    df_temp_score['Weight_method'] = df_temp_score[0].map(Weight_Dict)  # Mapping code to text
    # Creating barchart, plotting values of column `1`
    port_score_diff_methods_fig = px.bar( df_temp_score, x='Weight_method', y=1, color=2, text=1,
                                          title="Score by weighting scheme <br><sup>Assess the influence of weighting schemes on scores</sup>")
    port_score_diff_methods_fig.update_traces(textposition='inside', textangle=0)
    port_score_diff_methods_fig.update_yaxes(title_text='Temperature score', range=[0.5, 3])
    port_score_diff_methods_fig.update_xaxes(title_text=None, tickangle=0)
    port_score_diff_methods_fig.add_annotation(x=0.5, y=2.6, text="Main methodologies", showarrow=False)
    port_score_diff_methods_fig.add_shape(
        dict(type="rect", x0=-0.45, x1=1.5, y0=0, y1=2.7, line_dash="dot", line_color="LightSeaGreen"),
        row="all",
        col="all",
    )
    port_score_diff_methods_fig.add_hline(y=2, line_dash="dot", line_color="red",
                                          annotation_text="Critical value")  # horizontal line
    port_score_diff_methods_fig.update_layout(transition_duration=500)

    # input for the dash table
    df_for_output_table = filt_df.reset_index('company_id')[
        ['company_name', 'company_id', 'region', 'sector', 'cumulative_budget', 'investment_value',
         'trajectory_score', 'trajectory_exceedance_year',
         'target_score', 'target_exceedance_year',
         'temperature_score', 'scope']].copy()
    for col in ['temperature_score', 'trajectory_score', 'target_score', 'cumulative_budget']:
        df_for_output_table[col] = ITR.nominal_values(df_for_output_table[col].pint.m).round(2)  # f"{q:.2f~#P}"
        # pd.to_numeric(...).round(2)
    df_for_output_table['investment_value'] = df_for_output_table['investment_value'].apply(
        lambda x: "${:,.1f} Mn".format((x / 1000000)))  # formating column
    df_for_output_table['scope'] = df_for_output_table['scope'].map(str)
    df_for_output_table.rename(
        columns={'company_name': 'Name', 'company_id': 'ISIN', 'region': 'Region', 'sector': 'Industry',
                 'cumulative_budget': 'Emissions budget', 'investment_value': 'Notional',
                 'trajectory_score': 'Historical emissions score',
                 'trajectory_exceedance_year': 'Year historic emissions > 2050 budget',
                 'target_score': 'Target score',
                 'target_exceedance_year': 'Year target emissions > 2050 budget',
                 'temperature_score': 'Weighted temperature score',
                 'scope': 'Scope'}, inplace=True)

    # FIXME: this is utter confusion with respect to scopes!
    if aggregated_scores.long.S1S2:
        scores = aggregated_scores.long.S1S2.all.score.m
    elif aggregated_scores.long.S1:
        scores = aggregated_scores.long.S1.all.score.m
    elif aggregated_scores.long.S1S2S3:
        scores = aggregated_scores.long.S1S2S3.all.score.m
    elif aggregated_scores.long.S3:
        scores = aggregated_scores.long.S3.all.score.m
    else:
        raise ValueError("No aggregated scores")
    if ITR.HAS_UNCERTAINTIES and isinstance(scores, ITR.UFloat):
        scores = scores.n

    return (
        fig1, fig5,
        heatmap_fig, high_score_fig,
        port_score_diff_methods_fig,
        "Spin-graph",            # fake for spinner
        "{:.2f}".format(scores), # portfolio score
        {'color': 'ForestGreen'} if scores < 2 else {'color': 'Red'}, # conditional color
        str(round((filt_df.company_ev_plus_cash.sum())/10**9,0)), # sum of total EVIC for companies in portfolio
        str(round((filt_df.investment_value.sum())/10**6,1)), # portfolio notional
        str(len(filt_df)), # num of companies
        dbc.Table.from_dataframe(df_for_output_table,
                                 striped=True,
                                 bordered=True,
                                 hover=True,
                                 responsive=True,
                                 ),
    )

@app.callback(  # export table to spreadsheet
    Output("download-dataframe-xlsx", "data"),
    Output("spinner-xlsx", "value"),

    Input("export-to-excel", "n_clicks"),
    Input("portfolio-df", "data"),
    Input("temp-score-range", "value"),
    State("sector-dropdown", "value"),
    State("region-dropdown", "value"),
    State("scope-options", "value"),

    prevent_initial_call=True,)
def download_xlsx(n_clicks, portfolio_json, te_sc, sec, reg, scope):
    if n_clicks is None:
        raise PreventUpdate
    amended_portfolio = pd.read_json(portfolio_json, orient='split')
    # Why does this get lost in translation?
    amended_portfolio.index.name = 'company_id'
    amended_portfolio = amended_portfolio.assign(scope=lambda x: x.scope.map(lambda y: EScope[y]),
                                                 time_frame=lambda x: x.time_frame.map(lambda y: ETimeFrames[y]),
                                                 score_result_type=lambda x: x.score_result_type.map(lambda y: EScoreResultType[y]),
                                                 temperature_score=lambda x: x.temperature_score.map(Q_).astype('pint[delta_degC]'),
                                                 ghg_s1s2=lambda x: x.ghg_s1s2.map(Q_).astype('pint[t CO2e]'),
                                                 # FIXME: we're going to have to deal with NULL ghg_s3, and possible ghg_s1s2
                                                 ghg_s3=lambda x: x.ghg_s3.map(Q_).astype('pint[t CO2e]'),
                                                 cumulative_budget=lambda x: x.cumulative_budget.map(Q_).astype('pint[t CO2e]'),
                                                 cumulative_trajectory=lambda x: x.cumulative_trajectory.map(Q_).astype('pint[t CO2e]'),
                                                 cumulative_target=lambda x: x.cumulative_target.map(Q_).astype('pint[t CO2e]'),
                                                 trajectory_score=lambda x: x.trajectory_score.map(Q_).astype('pint[delta_degC]'),
                                                 target_score=lambda x: x.target_score.map(Q_).astype('pint[delta_degC]'),
                                                 benchmark_temperature=lambda x: x.benchmark_temperature.map(Q_).astype('pint[delta_degC]'),
                                                 benchmark_global_budget=lambda x: x.benchmark_global_budget.map(Q_).astype('pint[Gt CO2e]'))
    temp_score_mask = (amended_portfolio.temperature_score >= Q_(te_sc[0], 'delta_degC')) & (
        amended_portfolio.temperature_score <= Q_(te_sc[1], 'delta_degC'))
    # Dropdown filters
    if sec == 'all_values':
        sec_mask = (amended_portfolio.sector != 'dummy')  # select all
    else:
        sec_mask = amended_portfolio.sector == sec
    if reg == 'all_values':
        reg_mask = (amended_portfolio.region != 'dummy')  # select all
    else:
        reg_mask = (amended_portfolio.region == reg)
    filt_df = amended_portfolio.loc[temp_score_mask & sec_mask & reg_mask].pint.dequantify()  # filtering
    if len(filt_df) == 0:  # if after filtering the dataframe is empty
        raise PreventUpdate
    return (dcc.send_data_frame(filt_df.to_excel, "ITR_calculations.xlsx", sheet_name="ITR_calculations"),
            "Spin-xlsx")

@app.callback( # reseting dropdowns
    Output("temp-score-range", "value"),
    Output("projection-method", "value"),
    Output("scenarios-cutting", "value"),
    Output("target-year", "value"),
    Output("spinner-reset", "value"),

    Input("reset-filters-button", "n_clicks"),
    State("portfolio-df", "data"),

    prevent_initial_call=True,)
def reset_filters(n_clicks_reset, portfolio_json):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0] # to catch which widgets were pressed
    if n_clicks_reset is None:
        raise PreventUpdate

    amended_portfolio = pd.read_json(portfolio_json, orient='split')[['sector', 'region']]
    # Don't really need to do this...but why is it lost in translation?
    amended_portfolio.index.name = 'company_id'

    if (ProjectionControls.TREND_CALC_METHOD != ITR_median
        or ProjectionControls.LOWER_PERCENTILE != 0.1
        or ProjectionControls.UPPER_PERCENTILE != 0.9
        or ProjectionControls.TARGET_YEAR != 2050):
        ProjectionControls.TREND_CALC_METHOD=ITR_median
        ProjectionControls.LOWER_PERCENTILE = 0.1
        ProjectionControls.UPPER_PERCENTILE = 0.9

    # All the other things that are reset do not actually change the portfolio itself
    # (thought they may change which parts of the portfolio are plotted next)

    return ( # if button is clicked, reset filters
        [0,4],                  # Elsewhere we have this as 8.5
        'median',
        [ProjectionControls.LOWER_PERCENTILE*100,ProjectionControls.UPPER_PERCENTILE*100],
        2050,
        "Spin-reset",
    )

@app.callback(
    Output('dummy-output-info', 'children'),  # fake for spinner

    Input('spinner-eibm', 'value'),
    Input('spinner-ts', 'value'),
    Input('spinner-ty', 'value'),
    Input('spinner-graphs', 'value'),
    Input('spinner-xlsx', 'value'),
    Input('spinner-reset', 'value'),)
def spinner_concentrator(*_):
    return 'Spin!'

if __name__ == "__main__":
    app.run_server(use_reloader=False, debug=False)
