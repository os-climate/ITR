# Go to folder "examples" and run this app with `python ITR_UI.py` and
# visit http://127.0.0.1:8050/ in your web browser


import pandas as pd
import numpy as np
import json
import pickle
import os
from uuid import uuid4
import base64
import io
import warnings
import ast

import dash
from dash import html, dcc
from dash import DiskcacheManager

import dash_bootstrap_components as dbc # should be installed separately

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import diskcache
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

from ITR.data.osc_units import ureg, Q_, asPintSeries
from pint import Quantity
from pint_pandas import PintType

import logging

import sys
import argparse

launch_uid = uuid4()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') # LoggingConfig.FORMAT
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

logger.info("Start!")

cache = diskcache.Cache("./.webassets-cache")
background_callback_manager = DiskcacheManager(cache, cache_by=[lambda: launch_uid], expire=600)

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

# Production benchmark (there's only one, and we have to stretch it from OECM to cover TPI)
benchmark_prod_json_file = "benchmark_production_OECM.json"
benchmark_prod_json = os.path.join(root, examples_dir, data_dir, data_json_units_dir, benchmark_prod_json_file)
with open(benchmark_prod_json) as json_file:
    parsed_json = json.load(json_file)

oil_prod_bm = [bm for bm in parsed_json['AnyScope']['benchmarks'] if bm['sector']=='Oil'][0]
oil_base_prod = ureg(oil_prod_bm['base_year_production']).to('MJ')
oil_prod_series = pd.DataFrame({**{'year': [oil_prod['year'] for oil_prod in oil_prod_bm['projections_nounits']]},
                                **{'value': [oil_prod['value'] for oil_prod in oil_prod_bm['projections_nounits']]}}).set_index('year').squeeze()
oil_prod_series = oil_prod_series.add(1).cumprod().mul(oil_base_prod.m)

gas_prod_bm = [bm for bm in parsed_json['AnyScope']['benchmarks'] if bm['sector']=='Gas'][0]
gas_base_prod = ureg(gas_prod_bm['base_year_production']).to('MJ')
gas_prod_series = pd.DataFrame({**{'year': [gas_prod['year'] for gas_prod in gas_prod_bm['projections_nounits']]},
                                **{'value': [gas_prod['value'] for gas_prod in gas_prod_bm['projections_nounits']]}}).set_index('year').squeeze()
gas_prod_series = gas_prod_series.add(1).cumprod().mul(gas_base_prod.m)

oil_and_gas_prod_series = oil_prod_series.add(gas_prod_series).div(oil_base_prod.m + gas_base_prod.m)
oil_and_gas_prod_series = oil_and_gas_prod_series.div(oil_and_gas_prod_series.shift(1))
oil_and_gas_prod_series.iloc[0] = 1.0
oil_and_gas_prod_series = oil_and_gas_prod_series.sub(1)

oil_and_gas_bm = dict(oil_prod_bm)
oil_and_gas_bm['sector'] = 'Oil & Gas'
oil_and_gas_bm['base_year_production'] = f"{oil_base_prod + gas_base_prod:~P}"
oil_and_gas_bm['projections_nounits'] = [ {'year': year, 'value': value} for year,value in oil_and_gas_prod_series.to_dict().items() ]
parsed_json['AnyScope']['benchmarks'].append(oil_and_gas_bm)

# coal_prod_bm = dict([bm for bm in parsed_json['AnyScope']['benchmarks'] if bm['sector']=='Coal'][0])
# coal_base_prod = ureg(coal_prod_bm['base_year_production']).to('MJ')

# coal_prod_bm['sector'] = 'Diversified Mining'
# coal_prod_bm['base_year_production'] = f"{coal_base_prod:~P}"
# parsed_json['AnyScope']['benchmarks'].append(coal_prod_bm)

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
benchmark_EI_TPI_2deg_high_efficiency_file = "benchmark_EI_TPI_2_degrees_high_efficiency.json"
benchmark_EI_TPI_2deg_shift_improve_file = "benchmark_EI_TPI_2_degrees_shift_improve.json"

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
                            'content': 'width=device-width, initial-scale=1.0'}],
                background_callback_manager=background_callback_manager,
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
                     options=[{'label': 'All Sectors', 'value': ''}],
                     value='',
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
                     options=[{'label': 'All Regions', 'value': ''}],
                     value='',
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
                         {'label': 'OECM (Deprecated) 1.5 degrees', 'value': 'OECM'},
                         {'label': 'OECM (Prod-Centric) 1.5 degC', 'value': 'OECM_PC'},
                         {'label': 'OECM (Scope 3) 1.5 degC', 'value': 'OECM_S3'},
                         {'label': 'TPI 1.5 degrees (No Autos)', 'value': 'TPI_15_degrees'},
                         {'label': 'TPI 2 degrees (HE)', 'value': 'TPI_2_degrees_high_efficiency'},
                         {'label': 'TPI 2 degrees (SI)', 'value': 'TPI_2_degrees_shift_improve'},
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
                {'label': 'S2', 'value': 'S2'},
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
        html.Div(id="bm-budgets-target-year")
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
        dbc.Col([html.Span(id="loading-template-data", hidden=False, children="loading template data...")]),
        dbc.Col([
            dbc.Spinner(
                [
                    html.H1(id="dummy-output-info", style={'color': 'white'}),
                    html.Data(id="spinner-warehouse"),
                    html.Data(id="spinner-eibm"),
                    html.Data(id="spinner-ty"),
                    html.Data(id="spinner-ty-ts"),
                    html.Data(id="spinner-ts"),
                    html.Data(id="spinner-graphs"),
                    html.Data(id="spinner-xlsx"),
                    html.Data(id="spinner-reset"),
                ],
                color="primary",
                spinner_style={"width": "3rem", "height": "3rem"},
                # value="Finished",
            ),  # Spinner/Progress implementations
        ], width=1, ),
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
#        html.Data(id="scope-column"),
        html.Data(id="scope-options-ty"),
        html.Data(id="scope-options-ts"),
#         html.Data(id="scope-options-reset"),
        dcc.Store(id="warehouse"),
        dcc.Store(id="warehouse-eibm"),
        dcc.Store(id="warehouse-ty"),
        dcc.Store(id="portfolio-df"),
        html.Data(id="sector-region-scope-ty-ts"),
#        html.Data(id="sector-dropdown-ty"),
#        html.Data(id="sector-dropdown-reset"),
#        html.Data(id="region-dropdown-ty"),
#        html.Data(id="region-dropdown-reset"),
        html.Data(id="show-oecm-bm"),
        html.Data(id="bm-budget"),
        html.Data(id="bm-1e-budget"),
        html.Data(id="bm-2e-budget"),
        html.Data(id="tempscore-ty-ts"),
        html.Data(id="reset-sector-region-scope"),
        # dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns])
    ],
    style={"max-width": "1500px"},
)

@app.callback(
    Output("warehouse", "data"),
    Output("loading-template-data", "hidden"),
    Output("spinner-warehouse", "value"),

    Input("banner-title", "children"),) # Just something to get us going...
def warehouse_new(banner_title):
    # load company data
    template_company_data = TemplateProviderCompany(company_data_path, projection_controls = ProjectionControls())
    Warehouse = DataWarehouse(template_company_data, benchmark_projected_production=None, benchmarks_projected_ei=None,
                              estimate_missing_data=DataWarehouse.estimate_missing_s3_data)
    return (json.dumps(pickle.dumps(Warehouse), default=str),
            True,
            "Spin-warehouse")

@dash.callback(
    output=(Output("warehouse-eibm", "data"),        # Warehouse initialized with benchmark
            # We cannot set scope_options based on benchmark, because that may depend on sector
            Output("show-oecm-bm", "value"),
            Output("spinner-eibm", "value"),),  # fake for spinner

    inputs=(Input("warehouse", "data"),
            Input("eibm-dropdown", "value"),
            Input("projection-method", "value"),
            Input("scenarios-cutting", "value"), # winzorization slider

            State("target-year", "value"),),

    background=True,

    prevent_initial_call=True,)
# load default intensity benchmarks
def recalculate_individual_itr(warehouse_pickle_json, eibm, proj_meth, winz, target_year):
    '''
    Reload Emissions Intensity benchmark from a selected file
    :param warehouse_pickle_json: Pickled JSON version of Warehouse containing only company data
    :param eibm: Emissions Intensity benchmark identifier
    :param proj_meth: Trajectory projection method (median or mean)
    :param winz: Winsorization parameters (limit of outlier data)
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
        show_oecm_bm = 'no'
        if eibm == 'OECM_PC':
            benchmark_file = benchmark_EI_OECM_PC_file
            show_oecm_bm = 'yes'
        elif eibm == 'OECM_S3':
            benchmark_file = benchmark_EI_OECM_S3_file
            show_oecm_bm = 'yes'
        elif eibm.startswith('TPI_2_degrees'):
            benchmark_file = benchmark_EI_TPI_file
        elif eibm == 'TPI_15_degrees':
            benchmark_file = benchmark_EI_TPI_15_file
        elif eibm == 'OECM':
            benchmark_file = benchmark_EI_OECM_file
            logger.info('OECM benchmark is for backward compatibility only.  Use OECM_PC instead.')
        else:
            benchmark_file = benchmark_EI_TPI_below_2_file
        # load intensity benchmarks
        benchmark_EI = os.path.join(root, examples_dir, data_dir, data_json_units_dir, benchmark_file)
        with open(benchmark_EI) as json_file:
            parsed_json = json.load(json_file)
        if eibm.startswith('TPI_2_degrees'):
            if '_high_efficiency' in eibm:
                extra_EI = os.path.join(root, examples_dir, data_dir, data_json_units_dir, benchmark_EI_TPI_2deg_high_efficiency_file)
            else:
                extra_EI = os.path.join(root, examples_dir, data_dir, data_json_units_dir, benchmark_EI_TPI_2deg_shift_improve_file)                
            with open(extra_EI) as json_file:
                extra_json = json.load(json_file)
                for scope_name in EScope.get_scopes():
                    if scope_name in extra_json:
                        if scope_name not in parsed_json:
                            parsed_json[scope_name] = extra_json[scope_name]
                        else:
                            parsed_json[scope_name]['benchmarks'] += extra_json[scope_name]['benchmarks']
        # Initialize with the default ProjectionControls...
        EI_bm = BaseProviderIntensityBenchmark(EI_benchmarks=IEIBenchmarkScopes.parse_obj(parsed_json))
        Warehouse.update_benchmarks(base_production_bm, EI_bm)
        # ...then trim TARGET_YEAR to fit whatever has been set.
        EI_bm.projection_controls.TARGET_YEAR = target_year
        Warehouse.company_data.projection_controls.TARGET_YEAR = target_year

    return (json.dumps(pickle.dumps(Warehouse), default=str),
            show_oecm_bm,
            "Spin-eibm")

@dash.callback(
    # In the future, also adjust sector-dropdown options
    output = (Output("warehouse-ty", "data"),
              Output("sector-dropdown", "options"),
              Output("sector-dropdown", "value"),
              Output("region-dropdown", "options"),
              Output("region-dropdown", "value"),
              Output("scope-options-ty", "value"),
              Output("scope-options", "value"),
              Output("spinner-ty", "value"),),  # fake for spinner

    inputs = (Input("warehouse-eibm", "data"),
              Input("target-year", "value"),
              Input("reset-sector-region-scope", "children"),

              State("sector-dropdown", "value"),
              State("region-dropdown", "value"),
              State("scope-options", "value"),),

    background=True,

    prevent_initial_call=True,)
def recalculate_warehouse_target_year(warehouse_pickle_json, target_year, reset_sector_region_scope, sector, region, scope, *_):
    '''
    When changing endpoint of benchmark budget calculations, update total budget and benchmark ITR resulting therefrom.
    We assume that 'Global' is the all-world, not rest-of-world, for any benchmark.
    Where possible, honor users' selection of SCOPE, REGION, and SCOPE_LIST, and where necessary, adjust
    REGION and SCOPE_LIST to accommodate SECTOR selection.  But don't "listen" for new user SECTOR/REGION/SCOPE
    choices.
    '''

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]  # to catch which widgets were pressed
    Warehouse = pickle.loads(ast.literal_eval(json.loads(warehouse_pickle_json)))

    # All benchmarks use OECM production
    prod_bm = Warehouse.benchmark_projected_production
    EI_bm = Warehouse.benchmarks_projected_ei

    df_fundamentals = Warehouse.company_data.df_fundamentals
    sectors = sorted(set(df_fundamentals.sector) & set(EI_bm._EI_df.index.get_level_values('sector')))
    regions = sorted(set(df_fundamentals.region) & set(EI_bm._EI_df.index.get_level_values('region')) | {'Global'})

    df = EI_bm._EI_df
    # Our Warehouse doesn't retain side-effects, so we cannot "set-and-forget"
    # Instead, we have to re-make the change for the benefit of downstream users...
    EI_bm.projection_controls.TARGET_YEAR = target_year
    Warehouse.company_data.projection_controls.TARGET_YEAR = target_year

    if sector not in sectors:
        sector = ''
        EI_sectors = df[df.index.get_level_values('sector').isin(sectors)]
        sector_scopes = EI_sectors[df.columns[0]].groupby(['sector', 'scope']).count()
        if not sector_scopes.index.get_level_values('sector').duplicated().any():
            # TPI is a scope-per-sector benchmarks, so looks best when we see all scopes
            scope = ''
            EI_scopes = sector_scopes.index.get_level_values('scope').unique()
        else:
            EI_scopes = EI_sectors.index.get_level_values('scope').unique()
    else:
        EI_sectors = df.loc[sector]
        EI_scopes = EI_sectors.index.get_level_values('scope').unique()

    if region not in regions:
        region = ''

    if not scope or EScope[scope] not in EI_scopes:
        scope = ''

    return (
        json.dumps(pickle.dumps(Warehouse), default=str),
        [{"label": i, "value": i} for i in sectors] + [{'label': 'All Sectors', 'value': ''}], sector,
        [{"label": i, "value": i} for i in regions] + [{'label': 'All Regions', 'value': ''}], region,
        json.dumps([{"label": scope.name, "value": scope.name} for scope in sorted(EI_scopes)] + [{'label': 'All Scopes', 'value': ''}]),
        scope,
        "Spin-ty",
    )

# Some words on the current concept of Scope Math
# 
# For a single sector, any and all scope combinations give "valid" footprint descriptions...
# ...and we can compare against the scope footprints of the sector.
# 
# Beyond the sector is the Acvitity (Primary Energy, Secondary Energy, End Use)
# We can aggregate within an Activity as we do within a Sector, noting that End Use has special S3 rules.
# ...and if we have the same scope data for each sector and the overall activity,
# we can compare the activity-aggregated sectors with the bnechmark budget (on a per-scope or all-scope basis).
#
# If we look across multiple Acivities, scopes give the rules:
# The Scope 1 footprint can be computed as the sum of S1 scope emissions across all sectors (company aggregates and benchmark budget)
# The Scope 2 footprint can be computed as the sum of S2 scope emissions across all sectors (company aggregates and benchmark budget)
# The S1+S2 footprint can be computed as sum(S1) + abs(sum(S2) - Electric Utilties S1) (company aggregates and benchmark budget)
# If there are no Electricity Utilities in the sector, then it just aggregates all S1+S2 across all sectors
# There's no meaningful S3 aggregation in the multi-acvtivity case
# For S1+S2+S3, we separate Energy from Power from End Use, then...
# The S1+S2+S3 footprint for Energy is meaningful (if Energy is in the mix)
# The S1+S2+S3 footprint for Utilities is meaningful (if Utilities are in the mix)
# The S1+S2+S3 are meaningful for End Use as follows: sum(S1+S2) + S3(cement, buildings, aviation, shipping, roads)
# The overall S1+S2+S3 budget is also meaningful

def get_co2_per_sector_region_scope(prod_bm, ei_df, sector, region, scope_list) -> pd.Series:
    '''
    Given a production benchmark PROD_BM and an emissions intensity benchmark EI_DF, compute series of
    benchmark emissions for SECTOR, REGION, SCOPE.  SCOPE may be given by SCOPE_LIST
    or inferred from benchmark scope when SCOPE_LIST is None.
    '''
    EI_sectors = ei_df.loc[sector]
    EI_scopes = EI_sectors.index.get_level_values('scope').unique()
    # Get some benchmark temperatures for < 2050 using OECM; should be only one so can use list comprehension to search
    bm_sector_prod_in_region = [ bm.base_year_production
                                 for bm in prod_bm._productions_benchmarks.AnyScope.benchmarks
                                 # All benchmarks have a 'Global' region defined, but not all are defined for all regions (or sectors)
                                 if bm.sector==sector and bm.region==region][0]
    # There is no meaningful scope in production...
    base_prod_ser = prod_bm._prod_df.droplevel(['scope']).loc[(sector, region)].mul(bm_sector_prod_in_region)
    sector_scope = None if scope_list is None else scope_list[0]
    if sector_scope is None:
        if EScope.S1S2S3 in EI_scopes:
            sector_scope = EScope.S1S2S3
        elif EScope.S1S2 in EI_scopes:
            sector_scope = EScope.S1S2
        else:
            if len(EI_scopes) != 1:
                raise ValueError(f"Non-unique scope for sector {sector}")
            sector_scope = EI_scopes[0]
    elif sector_scope not in EI_scopes:
        raise ValueError(f"Scope {sector_scope.name} not in benchmark for sector {sector}")

    intensity_df = ei_df.loc[(sector, region, sector_scope)]
    target_year_cum_co2 = base_prod_ser.mul(intensity_df).cumsum().astype('pint[Gt CO2e]')
    return target_year_cum_co2

energy_activities = {'Energy', 'Coal', 'Gas', 'Oil', 'Oil & Gas'}
utility_activities = {'Utilities', 'Electricity Utilities', 'Gas Utilities'}
end_use_s3_activities = {'Cement', 'Residential Buildings', 'Commercial Buildings', 'Airlines', 'Shipping', 'Autos'}


def try_get_co2(prod_bm, ei_df, sectors, region, scope_list):
    result = []
    if 'Energy' in sectors:
        sectors = (sectors - energy_activities) | {'Energy'}
    elif 'Oil & Gas' in sectors:
        sectors = (sectors - energy_activities) | {'Oil & Gas'}
    elif 'Utilities' in sectors:
        sectors = (sectors - utility_activities) | {'Utilities'}
    for sector in sectors:
        try:
            result.append(get_co2_per_sector_region_scope(prod_bm, ei_df, sector, region, scope_list ))
        except ValueError:
            pass
    if len(result) > 1 and scope_list and scope_list[0] == EScope.S3:
        if 'Energy' in ei_df.index and sectors - end_use_s3_activities:
            # We are in OECM...Scope 3 doesn't overlap across end_use_s3_activities
            # But if there are any sectors outside that (such as 'Gas' + end_use_s3_activities), raise error
            raise ValueError(f"Scope 3 only meaningful for single-activity footprints (sectors were {sectors})")
        # Else we are in TPI, which has no S3 overlaps
    return result

# For OECM, calculate multi-sector footprints in a given activity
def get_co2_in_sectors_region_scope(prod_bm, ei_df, sectors, region, scope_list) -> pd.DataFrame:
    '''
    Aggregate benchmark emissions across SECTORS.  If SCOPE_LIST is none, we feel our way through them (TPI).
    Otherwise, caller is responsible for setting SCOPE_LIST to the specific scope requested (and in the TPI case
    this may limit what sectors can be aggregated).
    '''
    energy_sectors = sectors & energy_activities
    utility_sectors = sectors & utility_activities
    end_use_sectors = sectors - energy_sectors - utility_sectors

    if (not energy_sectors) + (not utility_sectors) + (not end_use_sectors) == 2:
        # All sectors are in a single activity, so aggregate by scope.  Caller sets scope!
        if len(sectors)==1:
            return get_co2_per_sector_region_scope(prod_bm, ei_df, next(iter(sectors)), region, scope_list)

        if scope_list and (scope_list[0] in [ EScope.S1S2, EScope.S1S2S3 ]):
            co2_s1s2_elements = try_get_co2(prod_bm, ei_df, sectors, region, [ EScope.S1S2 ] )
            if co2_s1s2_elements:
                total_co2_s1s2 = pd.concat(co2_s1s2_elements, axis=1).sum(axis=1).astype('pint[Gt CO2e]')
            else:
                raise ValueError(f"no benchmark emissions for {sectors} in scope")
            if sectors & {'Utilities'}:
                power_co2_s1 = get_co2_per_sector_region_scope(prod_bm, ei_df, 'Utilities', region, [ EScope.S1 ])
            elif sectors & {'Electricity Utilities'}:
                power_co2_s1 = get_co2_per_sector_region_scope(prod_bm, ei_df, 'Electricity Utilities', region, [ EScope.S1 ])
            else:
                power_co2_s1 = 0.0 * total_co2_s1s2
            total_co2 = total_co2_s1s2 - power_co2_s1
            if scope_list[0] == EScope.S1S2S3:
                total_co2_s3_elements = []
                if end_use_sectors:
                    total_co2_s3_elements = try_get_co2(prod_bm, ei_df, end_use_sectors & end_use_s3_activities, region, [ EScope.S3 ] )
                elif energy_sectors:
                    total_co2_s3_elements = try_get_co2(prod_bm, ei_df, energy_sectors, region, [ EScope.S3 ] )
                else:
                    total_co2_s3_elements = try_get_co2(prod_bm, ei_df, utility_sectors, region, [ EScope.S3 ] )
                if total_co2_s3_elements:
                    total_co2_s3 = pd.concat(total_co2_s3_elements, axis=1).sum(axis=1).astype('pint[Gt CO2e]')
                    total_co2 = total_co2 + total_co2_s3
            return total_co2
        # At this point, either scope_list is None, meaning fish out TPI metrics, or S1, S2, or S3, all of which aggregate within activity under OECM
        return pd.concat( try_get_co2(prod_bm, ei_df, sectors, region, scope_list ), axis=1).sum(axis=1).astype('pint[Gt CO2e]')
    # Multi-activity case, so SCOPE sets the rules for OECM
    if scope_list:
        if scope_list[0] in [ EScope.S1S2, EScope.S1S2S3 ]:
            co2_s1s2_elements = try_get_co2(prod_bm, ei_df, sectors, region, [ EScope.S1S2 ] )
            if co2_s1s2_elements:
                total_co2_s1s2 = pd.concat(co2_s1s2_elements, axis=1).sum(axis=1).astype('pint[Gt CO2e]')
            else:
                raise ValueError(f"No S1S2 data for sectors {sectors}")
            # Now subtract out the S1 from Power Generation so we don't double-count that
            if sectors & {'Utilities'}:
                power_co2_s1 = get_co2_per_sector_region_scope(prod_bm, ei_df, 'Utilities', region, [ EScope.S1])
            elif sectors & {'Electricity Utilities'}:
                power_co2_s1 = get_co2_per_sector_region_scope(prod_bm, ei_df, 'Electricity Utilities', region, [ EScope.S1])
            else:
                power_co2_s1 = 0.0 * total_co2_s1s2
            total_co2 = total_co2_s1s2 - power_co2_s1
            if EScope.S1S2S3 in scope_list:
                total_co2_s3_elements = try_get_co2(prod_bm, ei_df, sectors & end_use_s3_activities, region, [ EScope.S3] )
                if total_co2_s3_elements:
                    total_co2_s3 = pd.concat(total_co2_s3_elements, axis=1).sum(axis=1).astype('pint[Gt CO2e]')
                    total_co2 = total_co2 + total_co2_s3
            return total_co2
    # Must be S1 or S2 by itself (or scope_list is None, meaning whatever scope can be found).  try_get_co2 will deal with S3 case.
    total_co2 = pd.concat(try_get_co2(prod_bm, ei_df, sectors, region, scope_list ), axis=1).sum(axis=1).astype('pint[Gt CO2e]')
    return total_co2

@dash.callback(
    output = (Output("scope-options-ts", "value"),
              Output("sector-region-scope-ty-ts", "children"),
              Output("bm-budget", "children"),
              Output("bm-1e-budget", "children"),
              Output("bm-2e-budget", "children"),
              Output("tempscore-ty-ts", "children"),
              Output("spinner-ty-ts", "value"),),  # fake for spinner

    inputs = (Input("warehouse-ty", "data"),
              Input("sector-dropdown", "options"),
              Input("sector-dropdown", "value"),
              Input("region-dropdown", "value"),
              Input("scope-options", "value"),),

    background=True,

    prevent_initial_call=True,)
def recalculate_target_year_ts(warehouse_pickle_json, sectors_dict_list, sector, region, scope):
    '''
    Recalculate budget amounts and aligned temperature score for Warehouse after setting target year.
    These calculations are for display purposes.  Downstream processes make their own decisions with respect to upstream Warehouse.
    '''

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]  # to catch which widgets were pressed
    Warehouse = pickle.loads(ast.literal_eval(json.loads(warehouse_pickle_json)))

    # All benchmarks use OECM production
    prod_bm = Warehouse.benchmark_projected_production
    EI_bm = Warehouse.benchmarks_projected_ei
    bm_energy_calcs = False
    df_fundamentals = Warehouse.company_data.df_fundamentals
    zero_co2 = Q_(0.0, 'Gt CO2e')

    df_ei = EI_bm._EI_df
    if not scope:
        if 'Energy' in df_ei.index:
            if EI_bm._EI_benchmarks['S1S2'].production_centric:
                scope_list = [ EScope.S1S2 ]
            else:
                scope_list = [ EScope.S1S2S3 ]
        else:
            scope_list = None
    else:
        scope_list = [ EScope[scope] ]
    if not region:
        region = 'Global'

    target_year_1e_cum_co2 = None
    target_year_2e_cum_co2 = None
    if sector:
        sectors = { sector }
        target_year_cum_co2 = get_co2_per_sector_region_scope(prod_bm, df_ei, sector, region, scope_list)
    else:
        sectors = { x['value'] for x in sectors_dict_list if x['value'] }
        if 'Energy' in df_ei.index:
            # We are in OECM
            if scope_list[0] in [ EScope.S1, EScope.S2 , EScope.S3 ]:
                # For Scope 1 and Scope 2, we can just aggregate companies across all sectors (and error if S3)
                target_year_cum_co2 = get_co2_in_sectors_region_scope(prod_bm, df_ei, sectors, region, scope_list)
            else:
                # Single or Multiple activities...
                
                # Separate out Energy (1e), Utilities (2e), and End Use
                energy_sectors = sectors & energy_activities
                utility_sectors = sectors & utility_activities
                end_use_sectors = sectors - energy_sectors - utility_sectors

                if energy_sectors:
                    target_year_1e_cum_co2 = get_co2_in_sectors_region_scope(prod_bm, df_ei, energy_sectors, region, scope_list)
                if utility_sectors:
                    target_year_2e_cum_co2 = get_co2_in_sectors_region_scope(prod_bm, df_ei, utility_sectors, region, scope_list)
                if end_use_sectors:
                    target_year_cum_co2 = get_co2_in_sectors_region_scope(prod_bm, df_ei, end_use_sectors, region, scope_list)
                else:
                    if target_year_1e_cum_co2 is not None:
                        if target_year_2e_cum_co2:
                            target_year_cum_co2 = target_year_2e_cum_co2
                            target_year_2e_cum_co2 = None
                        else:
                            target_year_cum_co2 = target_year_1e_cum_co2
                            target_year_1e_cum_co2 = None
                    elif target_year_2e_cum_co2 is not None:
                        target_year_cum_co2 = target_year_2e_cum_co2
                        target_year_2e_cum_co2 = None
        else:
            # We are in TPI
            target_year_cum_co2 = get_co2_in_sectors_region_scope(prod_bm, df_ei, sectors, region, scope_list)

        assert EI_bm.projection_controls.TARGET_YEAR in target_year_cum_co2.index

    total_target_co2 = target_year_cum_co2.loc[EI_bm.projection_controls.TARGET_YEAR]
    total_final_co2 = target_year_cum_co2.loc[df_ei.columns[-1]]
    if target_year_1e_cum_co2 is not None:
        target_year_1e = target_year_1e_cum_co2.loc[EI_bm.projection_controls.TARGET_YEAR]
    if target_year_2e_cum_co2 is not None:
        target_year_2e = target_year_2e_cum_co2.loc[EI_bm.projection_controls.TARGET_YEAR]
    ts_cc = ITR.configs.TemperatureScoreConfig.CONTROLS_CONFIG
    # FIXME: Note that we cannot use ts_cc.scenario_target_temperature because that doesn't track the benchmark value
    # And we cannot make it track the benchmark value because then it becomes another global variable that would break Dash.
    target_year_ts = EI_bm._benchmark_temperature + (total_target_co2 - total_final_co2) * ts_cc.tcre_multiplier
    EI_scopes = set()
    for sector in sectors:
        EI_scopes = EI_scopes | set(df_ei.loc[sector].index.get_level_values('scope'))
    return (
        json.dumps([{"label": scope.name, "value": scope.name} for scope in sorted(EI_scopes)] + [{'label': 'All Scopes', 'value': ''}]),
        json.dumps([sector, region, scope]),
        f"{round(total_target_co2.m, 3)} Gt CO2e",
        f"{round(target_year_1e.m, 3)} Gt CO2e" if target_year_1e_cum_co2 is not None else "", # bm-1e-budget
        f"{round(target_year_2e.m, 3)} Gt CO2e" if target_year_2e_cum_co2 is not None else "", # bm-2e-budget
        f"{round(target_year_ts.m, 3)}˚C",
        # EI_bm.projection_controls.TARGET_YEAR,
        "Spin-ty",
    )

@app.callback(
    Output("scope-options", "options"),

    Input("scope-options-ty", "value"),
    Input("scope-options-ts", "value"),

    prevent_initial_call=True,)
def update_scope_options(scope_options_ty, scope_options_ts):
    new_scope_options = None
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]  # to catch which widgets were pressed
    if "scope-options-ts" in changed_id:
        new_scope_options = scope_options_ts
    else:
        new_scope_options = scope_options_ty
    return json.loads(new_scope_options)

@app.callback(
    Output("bm-budgets-target-year", "children"),

    Input("show-oecm-bm", "value"),
    Input("target-year", "value"),
    Input("bm-budget", "children"),
    Input("bm-1e-budget", "children"),
    Input("bm-2e-budget", "children"),
    Input("tempscore-ty-ts", "children"),

    prevent_initial_call=True,)
def bm_budget_year_target(show_oecm, target_year, bm_end_use_budget, bm_1e_budget, bm_2e_budget, tempscore_ty):
    if show_oecm=="yes":
        children = [ html.Div([html.Span(f"Benchmark totals through {target_year}")]), ]
        if not bm_1e_budget.startswith('0.0 '):
            children.append(html.Div([html.Span(f"Primary energy budget: {bm_1e_budget}")]))
        if not bm_2e_budget.startswith('0.0 '):
            children.append(html.Div([html.Span(f"Secondary energy budget: {bm_2e_budget}")]))
        if not bm_end_use_budget.startswith('0.0 '):
            children.append(html.Div([html.Span(f"End-use budget: {bm_end_use_budget}")]))
    else:
        children = [html.Div([html.Span(f"Benchmark budget through {target_year}: {bm_end_use_budget}")])]
    return children + [html.Div([html.Span(f"ITR of benchmark: {tempscore_ty}")])]

@dash.callback(
    output = (Output("portfolio-df", "data"),
              Output("spinner-ts", "value"),),  # fake for spinner

    inputs = (Input("warehouse-ty", "data"),),

    background=True,

    prevent_initial_call=True,)
def calc_temperature_score(warehouse_pickle_json, *_):
    global companies

    Warehouse = pickle.loads(ast.literal_eval(json.loads(warehouse_pickle_json)))
    EI_bm = Warehouse.benchmarks_projected_ei
    temperature_score = TemperatureScore(
        time_frames = [ETimeFrames.LONG],
        scopes=None, # None means "use the appropriate scopes for the benchmark
        # Options for the aggregation method are WATS, TETS, AOTS, MOTS, EOTS, ECOTS, and ROTS
        aggregation_method=PortfolioAggregationMethod.WATS
    )
    df = temperature_score.calculate(data_warehouse=Warehouse, portfolio=companies)
    return (df.drop(columns=['historic_data', 'target_data']).to_json(orient='split', default_handler=str),
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
    Input("sector-dropdown", "value"),
    Input("region-dropdown", "value"),
    Input("scope-options", "value"),

    prevent_initial_call=True,)
def update_graph(
        portfolio_json,
        te_sc,
        sec, reg, scope,
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
    if not sec:
        sec_mask = (amended_portfolio.sector != 'dummy')  # select all
    else:
        sec_mask = amended_portfolio.sector == sec
    if not reg:
        reg_mask = (amended_portfolio.region != 'dummy')  # select all
    else:
        reg_mask = (amended_portfolio.region == reg)
    if not scope:
        scope_mask = (amended_portfolio.scope != 'dummy')
    else:
        scope_mask = (amended_portfolio.scope == EScope[scope])
    filt_df = amended_portfolio.loc[temp_score_mask & sec_mask & reg_mask & scope_mask]  # filtering
    if len(filt_df) == 0:  # if after filtering the dataframe is empty
        # breakpoint()
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
    if sec:
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
    coverage = filt_df.reset_index('company_id')[['company_id', 'scope', 'ghg_s1s2', 'ghg_s3', 'cumulative_target']]
    coverage['ghg'] = coverage.apply(lambda x: x.ghg_s1s2+x.ghg_s3 if x.scope==EScope.S1S2S3 else x.ghg_s3 if x.scope==EScope.S3 else x.ghg_s1s2, axis=1).astype('pint[t CO2e]')
    coverage['coverage_category'] = np.where(coverage['ghg'].isnull(),
                                             np.where(ITR.isnan(coverage['cumulative_target'].pint.m), "Not Covered",
                                                      "Covered only<Br>by target"),
                                             np.where(~ITR.isnan(coverage['ghg'].pint.m) & ITR.isnan(
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
    elif aggregated_scores.long.S2:
        scores = aggregated_scores.long.S2.all.score.m
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

    State("portfolio-df", "data"),
    State("temp-score-range", "value"),
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
    if sec is None:
        sec_mask = (amended_portfolio.sector != 'dummy')  # select all
    else:
        sec_mask = amended_portfolio.sector == sec
    if reg is None:
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
    Output("reset-sector-region-scope", "children"),
    Output("spinner-reset", "value"),

    Input("reset-filters-button", "n_clicks"),

    prevent_initial_call=True,)
def reset_filters(n_clicks_reset):

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
        "Reset Sector, Region, Scope",
        "Spin-reset",
    )

@app.callback(
    Output("dummy-output-info", "children"),  # fake for spinner

    Input("spinner-eibm", "value"),
    Input("spinner-ts", "value"),
    Input("spinner-ty", "value"),
    Input("spinner-ty-ts", "value"),
    Input("spinner-graphs", "value"),
    Input("spinner-xlsx", "value"),
    Input("spinner-reset", "value"),)
def spinner_concentrator(*_):
    return 'Spin!'

if __name__ == "__main__":
    app.run_server(use_reloader=False, debug=True)
