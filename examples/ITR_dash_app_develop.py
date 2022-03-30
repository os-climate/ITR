# Run this app with `python ITR_dash_app.py` and
# visit http://127.0.0.1:8050/ in your web browser
# and pray.


import pandas as pd
import numpy as np
import json
import os
import base64
import datetime
import io

import dash
from dash import html
from dash import dcc
from dash import dash_table

import dash_bootstrap_components as dbc # should be installed separately

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go

import ITR

from ITR.data.data_warehouse import DataWarehouse
from ITR.portfolio_aggregation import PortfolioAggregationMethod
from ITR.temperature_score import TemperatureScore
from ITR.configs import ColumnsConfig, TemperatureScoreConfig

from ITR.data.base_providers import BaseProviderProductionBenchmark, BaseProviderIntensityBenchmark
from ITR.data.template import TemplateProviderCompany
from ITR.interfaces import ICompanyData, EScope, ETimeFrames, PortfolioCompany, IEIBenchmarkScopes, IProductionBenchmarkScopes

from ITR.data.osc_units import ureg, Q_, PA_
from pint import Quantity
from pint_pandas import PintType

from ITR.utils import get_project_root
pkg_root = get_project_root()


# Initial calculations
print('Start!!!!!!!!!')

directory1 ='' #'examples'
directory2="data"
directory3="json-units"

# company_json_file = "fundamental_data.json"
benchmark_prod_json_file = "benchmark_production_OECM.json"
benchmark_EI_OECM_file = "benchmark_EI_OECM.json"
benchmark_EI_TPI_file = "benchmark_EI_TPI_2_degrees.json"
benchmark_EI_TPI_below_2_file = "benchmark_EI_TPI_below_2_degrees.json"

root = os.path.dirname(os.path.abspath("__file__"))
print(f"root = {root}; pkg_root = {pkg_root}")

# root = os.path.dirname(os.path.abspath(__file__))
# company_json = os.path.join(root, directory1, directory2, directory3, company_json_file)
benchmark_prod_json = os.path.join(root, directory1, directory2, directory3, benchmark_prod_json_file)
benchmark_EI_OECM = os.path.join(root, directory1, directory2, directory3, benchmark_EI_OECM_file)
benchmark_EI_TPI = os.path.join(root, directory1, directory2, directory3, benchmark_EI_TPI_file)
benchmark_EI_TPI_below_2 = os.path.join(root, directory1, directory2, directory3, benchmark_EI_TPI_below_2_file)

# load production benchmarks
with open(benchmark_prod_json) as json_file:
    parsed_json = json.load(json_file)
prod_bms = IProductionBenchmarkScopes.parse_obj(parsed_json)
base_production_bm = BaseProviderProductionBenchmark(production_benchmarks=prod_bms)

# load intensity benchmarks

# OECM
with open(benchmark_EI_OECM) as json_file:
    parsed_json = json.load(json_file)
ei_bms = IEIBenchmarkScopes.parse_obj(parsed_json)
OECM_EI_bm = BaseProviderIntensityBenchmark(EI_benchmarks=ei_bms)

# TPI
with open(benchmark_EI_TPI) as json_file:
    parsed_json = json.load(json_file)
ei_bms = IEIBenchmarkScopes.parse_obj(parsed_json)
TPI_EI_bm = BaseProviderIntensityBenchmark(EI_benchmarks=ei_bms)

# TPI below 2
with open(benchmark_EI_TPI_below_2) as json_file:
    parsed_json = json.load(json_file)
ei_bms = IEIBenchmarkScopes.parse_obj(parsed_json)
TPI_below_2_EI_bm = BaseProviderIntensityBenchmark(EI_benchmarks=ei_bms)

# load company data
# presently projections are assigned to companies based on a single benchmark.
# To support multiple benchmarks we have to copy the company data (we cannot .copy because of ABC)
# Next step is probably to access projections via a dictionary indexed by benchmark name
template_company_data_OECM = TemplateProviderCompany(excel_path="data/20220215 ITR Tool Sample Data.xlsx")
template_company_data_TPI = TemplateProviderCompany(excel_path="data/20220215 ITR Tool Sample Data.xlsx")
template_company_data_TPI2 = TemplateProviderCompany(excel_path="data/20220215 ITR Tool Sample Data.xlsx")

OECM_warehouse = DataWarehouse(template_company_data_OECM, base_production_bm, OECM_EI_bm)
TPI_warehouse = DataWarehouse(template_company_data_TPI, base_production_bm, TPI_EI_bm)
TPI_below_2_warehouse = DataWarehouse(template_company_data_TPI2, base_production_bm, TPI_below_2_EI_bm)


# dummy_portfolio = "example_portfolio.csv"
dummy_portfolio = "template_portfolio.csv"
df_portfolio = pd.read_csv(os.path.join(directory1,directory2,dummy_portfolio), encoding="iso-8859-1", sep=';')
print('got till here 1')
companies = ITR.utils.dataframe_to_portfolio(df_portfolio)
temperature_score = TemperatureScore(time_frames = [ETimeFrames.LONG],scopes=[EScope.S1S2],aggregation_method=PortfolioAggregationMethod.WATS) # Options for the aggregation method are WATS, TETS, AOTS, MOTS, EOTS, ECOTS, and ROTS

portfolio_data = ITR.utils.get_data(OECM_warehouse, companies)
amended_portfolio_global = temperature_score.calculate(portfolio_data)
initial_portfolio = amended_portfolio_global
print('got till here 2')

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
        dbc.Col( # ... and then split to columns
         children=[   
            #         dbc.Row(
            #             [
            #                 dbc.Col( # Carbon budget slider
            #                     dbc.Label("\N{scroll} Benchmark carbon budget"), 
            #                     width=9, # max is 12 per column
            #                     ),
            #                 dbc.Col(
            #                     [
            #                     dbc.Button("\N{books}",id="hover-target1", color="link", n_clicks=0, className="text-right"),
            #                     dbc.Popover(dbc.PopoverBody("And here's some amazing content. Cool!"),id="hover1",target="hover-target1",trigger="hover"), 
            #                     ], width=2,
            #                     ),                           
            #             ],
            #             align="center",
            #         ),
            #         dcc.RangeSlider(
            #             id="carb-budg",
            #             min=initial_portfolio.cumulative_budget.min(),max=initial_portfolio.cumulative_budget.max(),
            #             value=[initial_portfolio.cumulative_budget.min(), initial_portfolio.cumulative_budget.max()],
            #             tooltip={'placement': 'bottom'},
            #             marks={i*(10**8): str(i) for i in range(0, int(initial_portfolio.cumulative_budget.max()/(10**8)), 10)},
            #         ),
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Label("\N{thermometer} Individual temperature score"), 
                                width=9,
                                ),
                            dbc.Col(
                                [
                                dbc.Button("\N{books}",id="hover-target2", color="link", n_clicks=0),
                                dbc.Popover(dbc.PopoverBody("And here's some amazing content. Cool!"),id="hover2",target="hover-target2",trigger="hover"), 
                                ], width=2, align="center",
                                ),                           
                        ],
                        align="center",
                    ),
                    dcc.RangeSlider(
                        id="temp-score",
                        min = 0, max = 4, value=[0,4],
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
                                dbc.Button("\N{books}",id="hover-target3", color="link", n_clicks=0),
                                dbc.Popover(dbc.PopoverBody("And here's some amazing content. Cool!"),id="hover3",target="hover-target3",trigger="hover"), 
                                ], width=2,
                                ),                           
                        ],
                        align="center",
                    ),
                    dcc.Dropdown(id="sector-dropdown",
                                options=[{"label": i, "value": i} for i in initial_portfolio["sector"].unique()] + [{'label': 'All Sectors', 'value': 'all_values'}],
                                 value = 'all_values',
                                 clearable =False,
                                 placeholder="Select a sector"),            
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Label("\N{globe with meridians} Focus on a specific region "), 
                                width=9,
                                ),
                            dbc.Col(
                                [
                                dbc.Button("\N{books}",id="hover-target4", color="link", n_clicks=0),
                                dbc.Popover(dbc.PopoverBody("And here's some amazing content. Cool!"),id="hover4",target="hover-target4",trigger="hover"), 
                                ], width=2,
                                ),                           
                        ],
                        align="center",
                    ),
                    dcc.Dropdown(id="region-dropdown",
                                 options=[{"label": i, "value": i} for i in initial_portfolio["region"].unique()] + [{'label': 'All Regions', 'value': 'all_values'}],
                                 value = 'all_values',
                                 clearable =False,
                                 placeholder="Select a region"),            
                  
        ],
        ),
    ],
)

macro = dbc.Row(
    [
        dbc.Col(
         children=[   
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Label("\N{bar chart} Select Benchmark "), 
                                width=9,
                                ),
                            dbc.Col(
                                [
                                dbc.Button("\N{books}",id="hover-target5", color="link", n_clicks=0),
                                dbc.Popover(dbc.PopoverBody("And here's some amazing content. Cool!"),id="hover5",target="hover-target5",trigger="hover"), 
                                ], width=2,
                                ),                           
                        ],
                        align="center",
                    ),                    
                    dcc.Dropdown(id="scenario-dropdown",
                                options=[
                                    {'label': 'OECM 1.5 degrees', 'value': 'OECM'},
                                    {'label': 'TPI 2 degrees', 'value': 'TPI_2_degrees'},
                                    {'label': 'TPI below 2 degrees', 'value': 'TPI_below_2_degrees'}                                    
                                ],
                                value='OECM',
                                clearable =False,
                                placeholder="Select emission scenario"),  
                    html.Div(id='hidden-div', style={'display':'none'})
         ],
        ),
    ],
)


# Define Layout
app.layout = dbc.Container( # always start with container
                children=[
                    # dcc.Store(id='memory-output'), # not used, but the idea is to use as clipboard to store dataframe
                    html.Hr(), # small space from the top
                    dbc.Row( # upload portfolio
                        [
                            dbc.Col(
                                dbc.CardImg(
                                    src="https://os-climate.org/wp-content/uploads/sites/138/2021/10/OSC-Logo.png",
                                    className='h-60 w-60 float-right align-middle', # reducing size and alligning
                                    bottom=False),
                                width = 2,
                            ),
                            dbc.Col(
                                [
                                    html.H1(id="banner-title",children=[html.A("OS-Climate Portfolio Alignment Tool",href="https://github.com/plotly/dash-svm",style={"text-decoration": "none","color": "inherit"})]),
                                    html.Div(children='Prototype tool for calculating the Implied Temperature Rise of investor portfolio in the steel and electric utilities sectors \N{deciduous tree}'),                                    
                                ],
                                width = 6,
                            ),
                            dbc.Col([
                                dcc.Upload(
                                    id='upload-data',
                                    children=html.Div(
                                        dbc.Button('Upload portfolio', size="lg", color="primary",className='align-bottom',),
                                        ),
                                    multiple=False # Allow multiple files to be uploaded
                                ),
                                ],
                                width=2,
                            ), 
                            dbc.Col(html.Div(dbc.Button('Get template (needs implementation)', size="lg", color="secondary",
                                                        href="https://docs.faculty.ai/user-guide/apps/examples/dash_file_upload_download.html",
                                                        download="dash_file_upload_download.html",
                                                        external_link=True,
                                            ),
                                    ),
                                    width=2,
                                    className='align-middle',
                            )
                        ],
                        # no_gutters=False, # deprecated, creates spaces btw components
                        justify='center', # for this to work you need some space left (in total there 12 columns)
                        align = 'center',
                    ),
                    # dbc.Row( # the row below is commented out, but left just in case to reverse upload functionality
                    #     [
                    #         dbc.Col(
                    #             [dbc.InputGroup(
                    #                         [dbc.InputGroupAddon("Put the URL of a csv portfolio here:", addon_type="prepend"),
                    #                         dbc.Input(id="input-url",value = 'data/example_portfolio_main.csv',),
                    #                         ]
                    #                 ),
                    #             ],
                    #             width = 9,
                    #         ),
                    #         dbc.Col(dbc.Button("Upload new portfolio", id="run-url", color="primary", ),   
                    #                 width=3,
                    #         ),                         
                    #     ]
                    # ),                    
                    html.Hr(),
                    dbc.Row(
                    [
                        dbc.Col([ # filters pane
                            dbc.Card(dbc.CardBody(
                                                [
                                                    dbc.Row([ # Row with key figures
                                                        dbc.Col(html.H5("Filters", className="pf-filter")), # PF score
                                                        dbc.Col(
                                                            html.Div(
                                                                dbc.Button("Reset filters", 
                                                                            id="reset-filters-but", 
                                                                            outline=True, color="dark",size="sm",className="me-md-2"
                                                                        ),
                                                                    className="d-grid gap-2 d-md-flex justify-content-md-end"
                                                                    )
                                                                ),
                                                    ]),
                                                    html.P("Select part of your portfolio", className="text-black-50"),
                                                    controls,
                                                ]
                                            )
                                    ),     
                            html.Br(),
                            dbc.Card(dbc.CardBody(
                                                [
                                                    html.H5("Scenario assumptions", className="macro-filters"),
                                                    html.P("Here you could adjust basic assumptions of calculations", className="text-black-50"),
                                                    macro,
                                                ]
                                            )
                                    ),       
                                ],
                            width=3,
                        ),
                        dbc.Col([ # main pane
                                dbc.Row([ # Row with key figures
                                        dbc.Col( # PF score
                                            dbc.Card(dbc.CardBody(
                                                                [
                                                                    html.H1(id="output-info"),
                                                                    html.P('Portfolio-level temperature rating of selected companies'),
                                                                ]
                                                            )
                                                    ),       
                                            ),
                                        dbc.Col( # Portfolio EVIC
                                            dbc.Card(dbc.CardBody(
                                                                [
                                                                    html.H1(id="evic-info"),
                                                                    html.P('Enterprise Value incl. Cash of selected portfolio in Bn'),
                                                                ]
                                                            )
                                                    ),       
                                            ),
                                        dbc.Col( # Portfolio notional
                                            dbc.Card(dbc.CardBody(
                                                                [
                                                                    html.H1(id="pf-info"),
                                                                    html.P('Total Notional of a selected portfolio in Mn'),
                                                                ]
                                                            )
                                                    ),       
                                            ),                                                                                        
                                        dbc.Col( # Number of companies
                                            dbc.Card(dbc.CardBody(
                                                                [
                                                                    html.H1(id="comp-info"),
                                                                    html.P('Number of companies in the selected portfolio'),
                                                                ]
                                                            )
                                                    ),       
                                            ),                                                                                        
                                    ],
                                ),
                                dbc.Row([dbc.Col(dcc.Graph(id="graph-2"),width=8), # big bubble graph
                                         dbc.Col(dcc.Graph(id="graph-6"),), # covered graph
                                    ],
                                ),
                                dbc.Row([ # 2 graphs
                                    dbc.Col(dcc.Graph(id="graph-3", 
                                                      # style={"height": "70vh", "max-height": "90vw",'title': 'Dash Data Visualization'},
                                              ),
                                    ),
                                    dbc.Col(dcc.Graph(id="graph-4", 
                                                    # style={"height": "70vh", "max-height": "90vw",'title': 'Dash Data Visualization'},
                                            ),
                                    ),
                                ]),
                                dbc.Row([ # 2 graphs
                                    dbc.Col(dcc.Graph(id="graph-5", 
                                                      # style={"height": "70vh", "max-height": "90vw",'title': 'Dash Data Visualization'},
                                              ),
                                    ),
                                ]),                                
                                html.Br(),
                                dbc.Card(dbc.CardBody( # Table
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        html.H5("Table below contains details about the members of the selected portfolio"),
                                                        width=10,
                                                        ),
                                                    dbc.Col(
                                                        html.Div(
                                                                [
                                                                dbc.Button("\N{books}",id="hover-target7", color="link", n_clicks=0, className="text-right"),
                                                                dbc.Popover(dbc.PopoverBody([
                                                                                    html.P("Emissions budget: ..."),
                                                                                    html.P("Trajectory score: ..."),
                                                                                    html.P("Target score: ..."),
                                                                                    html.P("Temperature score: ..."),
                                                                                    ]
                                                                                    ),
                                                                            id="hover7",target="hover-target7",trigger="hover"), 
                                                                ],
                                                                className="d-grid gap-2 d-md-flex justify-content-md-end",
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

                            ]
                        ),
                    ]
                    )
                ],
            style={"max-width": "1500px", 
                    # "margin": "auto"
                    },
            )
print('got till here 4')



def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename: # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('iso-8859-1')),sep=';')
        elif 'xls' in filename: # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        # print(df)
        return df
    except Exception as e:
        print(e)


@app.callback(
    [
    Output("graph-2", "figure"), Output("graph-6", "figure"),Output("graph-3", "figure"), Output("graph-4", "figure"), Output("graph-5", "figure"), 
    Output('output-info','children'), # portfolio score
    Output('output-info','style'), # conditional color
    Output('evic-info','children'), # portfolio evic
    Output('pf-info','children'), # portfolio notional
    Output('comp-info','children'), # num of companies
    # Output('carb-budg', 'min'), Output('carb-budg', 'max'), # this was an adjusting of min-max of a slider
    Output('container-button-basic', 'children'), # Table
    ],
    [
#        Input('memory-output', 'data'), # here is our imported csv in memory
        Input("scenario-dropdown", "value"),
        # Input("carb-budg", "value"), # carbon budget
        Input("temp-score", "value"),
        # Input("run-url", "n_clicks"), 
        # Input("input-url", "n_submit"),
        Input("sector-dropdown", "value"), 
        Input("region-dropdown", "value"),
        Input('upload-data', 'contents'),
    ],
    [
        # State("input-url", "value"), # url functionality
        State('upload-data', 'filename'), # upload functionality
        ],
)

def update_graph(
                # df_store,
                scenario,
                # ca_bu, 
                te_sc, 
                sec, reg,
                list_of_contents, list_of_names, # related to upload
                # url,
                ):

    global amended_portfolio_global, initial_portfolio, temperature_score, companies

    print('got till here 5')

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0] # to catch which widgets were pressed
    if 'upload-data' in changed_id: # if "upload new pf" button was clicked    
        df_portfolio = parse_contents(list_of_contents, list_of_names)
        # df_portfolio = pd.read_csv(url, encoding="iso-8859-1", sep=';')
        companies = ITR.utils.dataframe_to_portfolio(df_portfolio)
        portfolio_data = ITR.utils.get_data(OECM_warehouse, companies)
        initial_portfolio = temperature_score.calculate(portfolio_data)
        initial_portfolio = initial_portfolio.sort_values(by='temperature_score', ascending=False)
        filt_df = initial_portfolio
        amended_portfolio_global = filt_df
        aggregated_scores = temperature_score.aggregate_scores(filt_df)

    else: # no new portfolio
        if scenario == 'OECM':
            portfolio_data = ITR.utils.get_data(OECM_warehouse, companies)
        elif scenario == 'TPI_2_degrees':
            portfolio_data = ITR.utils.get_data(TPI_warehouse, companies)
        else:
            portfolio_data = ITR.utils.get_data(TPI_below_2_warehouse, companies)

        amended_portfolio_global = temperature_score.calculate(portfolio_data)
        initial_portfolio = amended_portfolio_global

        # carbon_mask = (initial_portfolio.cumulative_budget >= ca_bu[0]) & (initial_portfolio.cumulative_budget <= ca_bu[1])
        temp_score_mask = (initial_portfolio.temperature_score >= Q_(te_sc[0],'delta_degC')) & (initial_portfolio.temperature_score <= Q_(te_sc[1],'delta_degC'))

        # Dropdown filters
        if sec == 'all_values':
            sec_mask = (initial_portfolio.sector != 'dummy') # select all
        else:
            sec_mask = initial_portfolio.sector == sec
        if reg == 'all_values':
            reg_mask = (initial_portfolio.region != 'dummy') # select all
        else:
            reg_mask = (initial_portfolio.region == reg)
        filt_df = initial_portfolio.loc[temp_score_mask & sec_mask & reg_mask] # filtering
        filt_df = filt_df.sort_values(by='temperature_score', ascending=False)
        if len(filt_df) == 0: # if after filtering the dataframe is empty
            raise PreventUpdate
        amended_portfolio_global = filt_df
        aggregated_scores = temperature_score.aggregate_scores(filt_df) # calc temp score for companies left in pf


    # Calculate different weighting methods
    def agg_score(agg_method):
        temperature_score = TemperatureScore(time_frames = [ETimeFrames.LONG],
                                             scopes=[EScope.S1S2],
                                             aggregation_method=agg_method) # Options for the aggregation method are WATS, TETS, AOTS, MOTS, EOTS, ECOTS, and ROTS
        aggregated_scores = temperature_score.aggregate_scores(filt_df)
        return [agg_method.value,aggregated_scores.long.S1S2.all.score]

    agg_temp_scores = [agg_score(i) for i in PortfolioAggregationMethod]
    methods, scores = list(map(list, zip(*agg_temp_scores)))
    df_temp_score = pd.DataFrame(data={0:pd.Series(methods,dtype='string'), 1:pd.Series(scores, dtype='pint[delta_degC]')})
    # Separate column for names on Bar chart
    # Highlight WATS and TETS
    Weight_Dict = {'WATS': 'Investment<Br>weighted', # <Br> is needed to wrap x-axis label
                   'TETS': 'Total emissions<Br>weighted', 
                   'EOTS': "Enterprise Value<Br>weighted", 
                   'ECOTS': "Enterprise Value<Br>+ Cash weighted", 
                   'AOTS': "Total Assets<Br>weighted", 
                   'ROTS': "Revenues<Br>weigted",
                   'MOTS': 'Market Cap<Br>weighted'}
    df_temp_score['Weight_method'] = df_temp_score[0].map(Weight_Dict) # Mapping code to text
    # 1 is the label of the row we will be graphing
    # .map(lambda x: Q_(round(x.m, 2), x.u))
    df_temp_score[1]=df_temp_score[1].astype('pint[delta_degC]')
    # Creating barchart, plotting values of column `1`
    fig4 = dequantify_plotly (px.bar, df_temp_score, x='Weight_method', y=1, text=1,title = "Score by weighting scheme <br><sup>Assess the influence of weighting schemes on scores</sup>")
    fig4.update_traces(textposition='inside', textangle=0)
    fig4.update_yaxes(title_text='Temperature score', range = [1,3])
    fig4.update_xaxes(title_text=None, tickangle=0)
    fig4.add_annotation(x=0.5, y=2.6,text="Main methodologies",showarrow=False)
    fig4.add_shape(
        dict(type="rect", x0=-0.45, x1=1.5, y0=0, y1=2.7, line_dash="dot",line_color="LightSeaGreen"),
        row="all",
        col="all",
    )
    fig4.add_hline(y=2, line_dash="dot",line_color="red",annotation_text="Critical value") # horizontal line
    fig4.update_layout(transition_duration=500)




    # Scatter plot
    fig1 = dequantify_plotly (px.scatter, filt_df, x="cumulative_target", y="cumulative_budget", 
                              size="investment_value", 
                              color = "sector", labels={"color": "Sector"}, 
                              hover_data=["company_name", "investment_value", "temperature_score"],
                              title="Overview of portfolio")
    fig1.update_layout({'legend_title_text': '','transition_duration':500})
    fig1.update_layout(legend=dict(orientation = "h",yanchor="bottom",y=1,xanchor="center",x=0.5))
    

    # Covered companies analysis
    coverage=filt_df[['company_id','ghg_s1s2','cumulative_target']].copy()
    zeroE = Q_(0, 't CO2')
    coverage['coverage_category'] = np.where(coverage['ghg_s1s2'].isnull(),
                                             np.where(coverage['cumulative_target']==zeroE, "Not Covered", "Covered only<Br>by target"),
                                             np.where((coverage['ghg_s1s2'] >zeroE) & (coverage['cumulative_target']==zeroE),
                                                      "Covered only<Br>by emissions",
                                                      "Covered by<Br>emissions and targets"))
    dfg=coverage.groupby('coverage_category').count().reset_index()
    dfg['portfolio']='Portfolio' # 1 column to have just 1 bar. I didn't figure out how to do it more ellegant
    fig5 = dequantify_plotly (px.bar, dfg, x='portfolio',y="company_id", color="coverage_category",text='company_id',title="Coverage of companies in portfolio")
    fig5.update_xaxes(visible=False) # hide axis
    fig5.update_yaxes(visible=False) # hide axis
    fig5.update_layout({'legend_title_text': '','transition_duration':500, 'plot_bgcolor':'white'})
    fig5.update_layout(legend=dict(yanchor="middle",y=0.5,xanchor="left",x=1)) # location of legend

    # Heatmap
    trace = go.Heatmap(
                    x = filt_df.sector,
                    y = filt_df.region,
                    z = filt_df.temperature_score.map(lambda x: x.m),
                    type = 'heatmap',
                    colorscale = 'Temps',
                    )
    data = [trace]
    fig2 = go.Figure(data = data)
    fig2.update_layout(title = "Industry vs Region ratings")

    fig3 = dequantify_plotly (px.bar, filt_df.query("temperature_score > @Q_(2, 'delta_degC')"), 
                              x="company_name", y="temperature_score", 
                              text ="temperature_score",
                              color="sector",title="Highest temperature scores by company")
    fig3.update_traces(textposition='inside', textangle=0)
    fig3.update_yaxes(title_text='Temperature score', range = [1,4])
    fig3.update_layout({'legend_title_text': '','transition_duration':500})
    fig3.update_layout(xaxis_title = None,legend=dict(orientation = "h",yanchor="bottom",y=1,xanchor="center",x=0.5))
    
    # Carbon budget slider update
    # drop_d_min = initial_portfolio.cumulative_budget.min()
    # drop_d_max = initial_portfolio.cumulative_budget.max()

    df=amended_portfolio_global[['company_name', 'company_id','region','sector','cumulative_budget','investment_value','trajectory_score', 'target_score','temperature_score']].copy()  
    df['temperature_score']=df['temperature_score'].astype('pint[delta_degC]').values.quantity.m
    df['trajectory_score']=df['trajectory_score'].astype('pint[delta_degC]').values.quantity.m
    df['target_score']=df['target_score'].astype('pint[delta_degC]').values.quantity.m
    df['cumulative_budget'] = df['cumulative_budget'].astype('pint[Mt CO2]').values.quantity.m
    df['investment_value'] = df['investment_value'].apply(lambda x: "${:,.1f} Mn".format((x/1000000))) # formating column
    df.rename(columns={'company_name':'Name', 'company_id':'ISIN','region':'Region','sector':'Industry','cumulative_budget':'Emissions budget','investment_value':'Notional','trajectory_score':'Historical emissions score', 'target_score':'Target score','temperature_score':'Weighted temperature score'}, inplace=True)

    return (
        fig1, fig5, fig2, fig3, fig4,
        "{:.2f}".format(aggregated_scores.long.S1S2.all.score), # portfolio score
        {'color': 'ForestGreen'} if aggregated_scores.long.S1S2.all.score.m < 2 else {'color': 'Red'}, # conditional color
        str(round((filt_df.company_ev_plus_cash.sum())/10**9,0)), 
        str(filt_df.investment_value.sum()/10**6),
        str(len(filt_df)), # num of companies
        # str(len(filt_df.sector.unique())),  # num of sectors in pf
        # drop_d_min, drop_d_max, # Carbon budget slider update
        dbc.Table.from_dataframe(df,
                                striped=True,
                                bordered=True,
                                hover=True,
                                responsive=True,
                            ),
    )


@app.callback( # reseting dropdowns
    [
    # Output("carb-budg", "value"), # Carbon budget slider update
    Output("temp-score", "value"),
    Output("sector-dropdown", "value"),
    Output("region-dropdown", "value"),
    ],
    [Input('reset-filters-but', 'n_clicks')]
)

def reset_filters(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    return ( # if button is clicked, reset filters
        # [initial_portfolio.cumulative_budget.min(), initial_portfolio.cumulative_budget.max()], # Carbon budget slider update
        [0,4],
        'all_values',
        'all_values',
    )

if __name__ == "__main__":
    app.run_server(debug=True)
