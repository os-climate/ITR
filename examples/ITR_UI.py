# Go to folder "examples" and run this app with `python ITR_UI.py` and
# visit http://127.0.0.1:8050/ in your web browser


import pandas as pd
import numpy as np
import json
import os
import base64
import io

import dash
from dash import html
from dash import dcc

import dash_bootstrap_components as dbc # should be installed separately

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go

import ITR

from ITR.data.data_warehouse import DataWarehouse
from ITR.portfolio_aggregation import PortfolioAggregationMethod
from ITR.temperature_score import TemperatureScore

from ITR.data.base_providers import BaseProviderProductionBenchmark, BaseProviderIntensityBenchmark
from ITR.data.template import TemplateProviderCompany
from ITR.interfaces import EScope, ETimeFrames, IEIBenchmarkScopes, IProductionBenchmarkScopes, ProjectionControls
# from ITR.configs import LoggingConfig

from ITR.data.osc_units import Q_
from pint import Quantity
from pint_pandas import PintType

import logging

import sys
import argparse


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
    company_data_path = os.path.join(root, examples_dir, data_dir, "20220720 ITR Tool Sample Data.xlsx")


# load company data
template_company_data = TemplateProviderCompany(company_data_path)

# load production benchmarks
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

temperature_score: TemperatureScore = None # created during `recalculate_individual_itr`
EI_bm: BaseProviderIntensityBenchmark = None # Emissions Intensity benchmarks, created during `recalculate_individual_itr`

# load default intensity benchmarks
def recalculate_individual_itr(eibm):
    '''
    Reload Emissions Intensity benchmark from a selected file
    :param eibm: Emissions Intensity benchmark identifier
    '''
    global temperature_score, EI_bm

    if eibm == 'OECM_PC':
        benchmark_file = benchmark_EI_OECM_PC_file
    elif eibm == 'OECM_S3':
        benchmark_file = benchmark_EI_OECM_S3_file
    elif eibm == 'TPI_2_degrees':
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
    EI_bm = BaseProviderIntensityBenchmark(EI_benchmarks=IEIBenchmarkScopes.parse_obj(parsed_json))
    Warehouse = DataWarehouse(template_company_data, base_production_bm, EI_bm)
    temperature_score = TemperatureScore(
                            time_frames = [ETimeFrames.LONG],
                            scopes=[EI_bm.scope_to_calc],
                            # Options for the aggregation method are WATS, TETS, AOTS, MOTS, EOTS, ECOTS, and ROTS
                            aggregation_method=PortfolioAggregationMethod.WATS
                            )
    df = temperature_score.calculate(data_warehouse=Warehouse, portfolio=companies)
    return df


initial_portfolio = recalculate_individual_itr('OECM_PC')
amended_portfolio_global = initial_portfolio.copy()
filt_df = initial_portfolio.copy()

# matplotlib is integrated with Pint's units system: https://pint.readthedocs.io/en/0.18/plotting.html
# But not so plotly.  This function attempts to dequantify all units and return the magnitudes in their natural base units.

def dequantify_plotly(px_func, df, **kwargs):
    new_df = df.copy()
    for col in ['x', 'y']:
        s = df[kwargs[col]]
        if isinstance(s.dtype, PintType):
            new_df[kwargs[col]] = s.values.quantity.to_base_units().m
        elif s.map(lambda x: isinstance(x, Quantity)).any():
            item0 = s.values[0]
            s = s.astype(f"pint[{item0.u}]")
            new_df[kwargs[col]] = s.values.quantity.m
    if 'hover_data' in kwargs:
        for col in kwargs['hover_data']:
            s = df[col]
            if isinstance(s.dtype, PintType):
                new_df[col] = s.values.quantity.to_base_units().m
            elif s.map(lambda x: isinstance(x, Quantity)).any():
                item0 = s.values[0]
                s = s.astype(f"pint[{item0.u}]")
                new_df[col] = s.values.quantity.m

    return px_func (new_df, **kwargs)


# nice cheatsheet for managing layout via className attribute: https://hackerthemes.com/bootstrap-cheatsheet/

# Define app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], # theme should be written in CAPITAL letters; list of themes https://www.bootstrapcdn.com/bootswatch/
                meta_tags=[{'name': 'viewport', # this thing makes layout responsible to mobile view
                            'content': 'width=device-width, initial-scale=1.0'}]
                )
app.title = "ITR Tool" # this puts text to the browser tab
server = app.server

controls = dbc.Row( # always do in rows ...
    [
        dbc.Col(  # ... and then split to columns
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
                    id="temp-score",
                    min=0, max=4, value=[0, 4],
                    step=0.5,
                    marks={i / 10: str(i / 10) for i in range(0, 40, 5)},
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
                             options=[{"label": i, "value": i} for i in initial_portfolio["sector"].unique()] + [
                                 {'label': 'All Sectors', 'value': 'all_values'}],
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
                             options=[{"label": i if i != "Global" else "Other", "value": i} for i in
                                      initial_portfolio["region"].unique()] + [
                                         {'label': 'All Regions', 'value': 'all_values'}],
                             value='all_values',
                             clearable=False,
                             placeholder="Select a region"),

            ],
        ),
    ],
)

macro = dbc.Row(
    [
        dbc.Col(
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
                             value='OECM_PC',
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
                dbc.Row(  # Winzorization of scenarios
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
                    marks={i: str(i) for i in range(0, 100, 10)},
                    allowCross=False
                ),
            ],
        ),
    ],
)


# Define Layout
app.layout = dbc.Container(  # always start with container
    children=[
        html.Hr(),  # small space from the top
        dbc.Row(  # upload portfolio
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
                    dbc.Spinner([html.H1(id="dummy-output-info", style={'color': 'white'})], color="primary",
                                spinner_style={"width": "3rem", "height": "3rem"}),  # Spinner implementations
                ],
                    width=1,
                ),
            ],
            justify='between',  # for this to work you need some space left (in total there 12 columns)
            align='center',
        ),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    [  # filters pane
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    dbc.Row(
                                        [  # Row with key figures
                                            dbc.Col(html.H5("Filters", className="pf-filter")),  # PF score
                                            dbc.Col(
                                                html.Div(
                                                    dbc.Button("Reset filters",
                                                               id="reset-filters-but",
                                                               outline=True, color="dark", size="sm",
                                                               className="me-md-2"
                                                               ),
                                                    className="d-grid gap-2 d-md-flex justify-content-md-end"
                                                )
                                            ),
                                        ]
                                    ),
                                    html.P("Select part of your portfolio", className="text-black-50"),
                                    controls,
                                ]
                            )
                        ),
                        html.Br(),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Benchmarks", className="macro-filters"),
                                    html.P("Here you could adjust benchmarks of calculations",
                                           className="text-black-50"),
                                    macro,
                                ]
                            )
                        ),
                    ],
                    width=3,
                ),
                dbc.Col(
                    [  # main pane
                        dbc.Card(
                            dbc.CardBody([

                                dbc.Row(  # Row with key figures
                                    [
                                        dbc.Col(  # PF score
                                            dbc.Card(dbc.CardBody(
                                                [
                                                    html.H1(id="output-info"),
                                                    html.Div('Portfolio-level temperature rating of selected companies',
                                                             style={'color': 'black', 'fontSize': 16}),
                                                    html.Div('in delta degree Celcius',
                                                             style={'color': 'grey', 'fontSize': 10}),
                                                ]
                                            )
                                            ),
                                        ),
                                        dbc.Col(  # Portfolio EVIC
                                            dbc.Card(dbc.CardBody(
                                                [
                                                    html.H1(id="evic-info"),
                                                    html.Div('Enterprise Value incl. Cash of selected portfolio',
                                                             style={'color': 'black', 'fontSize': 16}),
                                                    html.Div('in billions of template curr',
                                                             style={'color': 'grey', 'fontSize': 10}),
                                                ]
                                            )
                                            ),
                                        ),
                                        dbc.Col(  # Portfolio notional
                                            dbc.Card(dbc.CardBody(
                                                [
                                                    html.H1(id="pf-info"),
                                                    html.Div('Total Notional of a selected portfolio',
                                                             style={'color': 'black', 'fontSize': 16}),
                                                    html.Div('in millions of template curr',
                                                             style={'color': 'grey', 'fontSize': 10}),
                                                ]
                                            )
                                            ),
                                        ),
                                        dbc.Col(  # Number of companies
                                            dbc.Card(dbc.CardBody(
                                                [
                                                    html.H1(id="comp-info"),
                                                    html.Div('Number of companies in the selected portfolio',
                                                             style={'color': 'black', 'fontSize': 16}),
                                                    html.Div('# of companies', style={'color': 'grey', 'fontSize': 10}),
                                                ]
                                            )
                                            ),
                                        ),
                                    ],
                                ),
                                dbc.Row(  # row with 2 graphs
                                    [
                                        dbc.Col(dcc.Graph(id="graph-2"), width=8),  # big bubble graph
                                        dbc.Col(dcc.Graph(id="graph-6"), ),  # covered graph
                                    ],
                                ),
                                dbc.Row(  # row with 2 graphs
                                    [
                                        dbc.Col(dcc.Graph(id="graph-3")),
                                        dbc.Col(dcc.Graph(id="graph-4")),
                                    ]
                                ),
                                dbc.Row(  # row with 1 bar graph
                                    [
                                        dbc.Col(dcc.Graph(id="graph-5")),
                                    ]
                                ),
                            ])
                        ),
                        html.Br(),
                    ],
                    width=9,
                ),
            ]
        ),
        dbc.Row(  # Table
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
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
                            html.Div(id='container-button-basic'),
                        ]
                    ),
                ),
            )
        )
    ],
    style={"max-width": "1500px"},
)


@app.callback(
    [
        Output("graph-2", "figure"),
        Output("graph-6", "figure"),
        Output("graph-3", "figure"),
        Output("graph-4", "figure"),
        Output("graph-5", "figure"),
        Output('dummy-output-info', 'children'),  # fake for spinner
        Output('output-info', 'children'),  # portfolio score
        Output('output-info', 'style'),  # conditional color
        Output('evic-info', 'children'),  # portfolio evic
        Output('pf-info', 'children'),  # portfolio notional
        Output('comp-info', 'children'),  # num of companies
        Output('container-button-basic', 'children'),  # Table
    ],
    [
        Input("temp-score", "value"),
        Input("sector-dropdown", "value"),
        Input("region-dropdown", "value"),
        Input("eibm-dropdown", "value"),
        Input('projection-method', 'value'),
        Input("scenarios-cutting", "value"),  # winzorization slide
    ],
)
def update_graph(
        te_sc,
        sec, reg,
        eibm,
        proj_meth,
        winz,
):
    global amended_portfolio_global, initial_portfolio, filt_df, temperature_score
    global companies, company_data_path, template_company_data
    global base_production_bm, EI_bm

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]  # to catch which widgets were pressed

    if 'scenarios-cutting' or 'projection-method' in changed_id:  # if winzorization params were changed
        if proj_meth == 'median':
            template_company_data.projection_controls.TREND_CALC_METHOD = staticmethod(pd.DataFrame.median)
        else:
            template_company_data.projection_controls.TREND_CALC_METHOD = staticmethod(pd.DataFrame.mean)
        template_company_data.projection_controls.LOWER_PERCENTILE = winz[0] / 100
        template_company_data.projection_controls.UPPER_PERCENTILE = winz[1] / 100
        template_company_data = TemplateProviderCompany(excel_path=company_data_path)

    # we need to recalculate temperature score as we changed benchmark
    amended_portfolio_global = recalculate_individual_itr(eibm)

    temp_score_mask = (amended_portfolio_global.temperature_score >= Q_(te_sc[0], 'delta_degC')) & (
                amended_portfolio_global.temperature_score <= Q_(te_sc[1], 'delta_degC'))
    # Dropdown filters
    if sec == 'all_values':
        sec_mask = (amended_portfolio_global.sector != 'dummy')  # select all
    else:
        sec_mask = amended_portfolio_global.sector == sec
    if reg == 'all_values':
        reg_mask = (amended_portfolio_global.region != 'dummy')  # select all
    else:
        reg_mask = (amended_portfolio_global.region == reg)
    filt_df = amended_portfolio_global.loc[temp_score_mask & sec_mask & reg_mask]  # filtering
    if len(filt_df) == 0:  # if after filtering the dataframe is empty
        raise PreventUpdate
    aggregated_scores = temperature_score.aggregate_scores(filt_df)  # calc temp score for companies left in pf

    # Scatter plot
    fig1 = dequantify_plotly(px.scatter, filt_df, x="cumulative_target", y="cumulative_budget",
                             size="investment_value",
                             color="sector", labels={"color": "Sector"},
                             hover_data=["company_name", "investment_value", "temperature_score"],
                             title="Overview of portfolio")
    fig1.update_layout({'legend_title_text': '', 'transition_duration': 500})
    fig1.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="center", x=0.5))

    # Covered companies analysis
    scope_column = 'ghg_' + EI_bm.scope_to_calc.name.lower()
    coverage = filt_df[['company_id', scope_column, 'cumulative_target']].copy()
    zeroE = Q_(0, 't CO2')
    coverage['coverage_category'] = np.where(coverage[scope_column].isnull(),
                                             np.where(coverage['cumulative_target'] == zeroE, "Not Covered",
                                                      "Covered only<Br>by target"),
                                             np.where((coverage[scope_column] > zeroE) & (
                                                         coverage['cumulative_target'] == zeroE),
                                                      "Covered only<Br>by emissions",
                                                      "Covered by<Br>emissions and targets"))
    dfg = coverage.groupby('coverage_category').count().reset_index()
    dfg['portfolio']='Portfolio'
    fig5 = dequantify_plotly (px.bar, dfg, x='portfolio',y="company_id", color="coverage_category",text='company_id',title="Coverage of companies in portfolio")
    fig5.update_xaxes(visible=False) # hide axis
    fig5.update_yaxes(visible=False) # hide axis
    fig5.update_layout({'legend_title_text': '','transition_duration':500, 'plot_bgcolor':'white'})
    fig5.update_layout(legend=dict(yanchor="middle",y=0.5,xanchor="left",x=1)) # location of legend

    # Heatmap
    trace = go.Heatmap(
        x=filt_df.sector,
        y=filt_df.region,
        z=filt_df.temperature_score.map(lambda x: x.m),
        type='heatmap',
        colorscale='Temps',
        zmin = 1.49, zmax = 2.9,
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
                                       y="temperature_score", text="temperature_score",
                                       color="sector", title="Highest temperature scores by company")
    high_score_fig.update_traces(textposition='inside', textangle=0)
    high_score_fig.update_yaxes(title_text='Temperature score', range=[1, 4])
    high_score_fig.update_layout({'legend_title_text': '', 'transition_duration': 500})
    high_score_fig.update_layout(xaxis_title=None,
                                 legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="center", x=0.5))

    # Calculate different weighting methods
    def agg_score(agg_method):
        global EI_bm

        temperature_score = TemperatureScore(time_frames=[ETimeFrames.LONG],
                                             scopes=[EI_bm.scope_to_calc],
                                             aggregation_method=agg_method)  # Options for the aggregation method are WATS, TETS, AOTS, MOTS, EOTS, ECOTS, and ROTS
        aggregated_scores = temperature_score.aggregate_scores(filt_df)
        if aggregated_scores.long.S1S2:
            agg_s1s2 = [agg_method.value,aggregated_scores.long.S1S2.all.score]
        else:
            agg_s1s2 = []
        if aggregated_scores.long.S3:
            agg_s3 = [agg_method.value,aggregated_scores.long.S3.all.score]
        else:
            agg_s3 = []
    
        return agg_s1s2 + agg_s3

    agg_temp_scores = [agg_score(i) for i in PortfolioAggregationMethod]
    methods, scores = list(map(list, zip(*agg_temp_scores)))
    df_temp_score = pd.DataFrame(
        data={0: pd.Series(methods, dtype='string'), 1: pd.Series(scores, dtype='pint[delta_degC]')})
    df_temp_score[1] = pd.to_numeric(df_temp_score[1].astype('pint[delta_degC]').values.quantity.m).round(
        2)  # rounding score
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
    port_score_diff_methods_fig = dequantify_plotly(px.bar, df_temp_score, x='Weight_method', y=1, text=1,
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
    df_for_output_table = filt_df[
        ['company_name', 'company_id', 'region', 'sector', 'cumulative_budget', 'investment_value', 'trajectory_score',
         'target_score', 'temperature_score']].copy()
    df_for_output_table['temperature_score'] = df_for_output_table['temperature_score'].astype(
        'pint[delta_degC]').values.quantity.m  # f"{q:.2f~#P}"
    df_for_output_table['trajectory_score'] = pd.to_numeric(
        df_for_output_table['trajectory_score'].astype('pint[delta_degC]').values.quantity.m).round(2)
    df_for_output_table['target_score'] = pd.to_numeric(
        df_for_output_table['target_score'].astype('pint[delta_degC]').values.quantity.m).round(2)
    df_for_output_table['cumulative_budget'] = pd.to_numeric(
        df_for_output_table['cumulative_budget'].astype('pint[Mt CO2]').values.quantity.m).round(2)
    df_for_output_table['investment_value'] = df_for_output_table['investment_value'].apply(
        lambda x: "${:,.1f} Mn".format((x / 1000000)))  # formating column
    df_for_output_table.rename(
        columns={'company_name': 'Name', 'company_id': 'ISIN', 'region': 'Region', 'sector': 'Industry',
                 'cumulative_budget': 'Emissions budget', 'investment_value': 'Notional',
                 'trajectory_score': 'Historical emissions score', 'target_score': 'Target score',
                 'temperature_score': 'Weighted temperature score'}, inplace=True)

    if aggregated_scores.long.S1S2:
        scores = aggregated_scores.long.S1S2.all.score.m
    elif aggregated_scores.long.S3:
        scores = aggregated_scores.long.S3.all.score.m
    else:
        raise ValueError("No aggregated scores")

    return (
        fig1, fig5,
        heatmap_fig, high_score_fig,
        port_score_diff_methods_fig,
        "{:.2f}".format(scores), # fake for spinner
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
    Input('export-to-excel', 'n_clicks'),
    prevent_initial_call=True,
)

def download_xlsx(n_clicks):
    return dcc.send_data_frame(filt_df.to_excel, "ITR_calculations.xlsx", sheet_name="ITR_calculations")

@app.callback( # reseting dropdowns
    [
        Output("temp-score", "value"),
        Output("sector-dropdown", "value"),
        Output("sector-dropdown", "options"), # update sector dropdown options as it could be different depending on the selected scenario
        Output("region-dropdown", "value"),
        Output("region-dropdown", "options"), # update region dropdown options as it could be different depending on the selected scenario
        Output("projection-method", "value"),
        Output("scenarios-cutting", "value"),
    ],
    [
        Input('reset-filters-but', 'n_clicks'),
        Input("eibm-dropdown", "value")
    ]
)

def reset_filters(n_clicks_reset, eibm):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0] # to catch which widgets were pressed
    if n_clicks_reset is None and 'eibm-dropdown' not in changed_id:
        raise PreventUpdate

    ProjectionControls.TREND_CALC_METHOD=staticmethod(pd.DataFrame.median)
    ProjectionControls.LOWER_PERCENTILE = 0.1
    ProjectionControls.UPPER_PERCENTILE = 0.9
    template_company_data = TemplateProviderCompany(excel_path=company_data_path)
    amended_portfolio_global = recalculate_individual_itr(eibm)
    initial_portfolio = amended_portfolio_global

    return ( # if button is clicked, reset filters
        [0,4],
        'all_values',
        [{"label": i, "value": i} for i in amended_portfolio_global["sector"].unique()] + [{'label': 'All Sectors', 'value': 'all_values'}],
        'all_values',
        [{"label": i if i != "Global" else "Other", "value": i} for i in amended_portfolio_global["region"].unique()] + [{'label': 'All Regions', 'value': 'all_values'}],
        'median',
        [ProjectionControls.LOWER_PERCENTILE*100,ProjectionControls.UPPER_PERCENTILE*100],
    )

if __name__ == "__main__":
    app.run_server(debug=True)
