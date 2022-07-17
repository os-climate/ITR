# Run this app with `python ITR_dash_app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
from dash import html
from dash import dcc
import dash_core_components as dcc
import dash_html_components as html
import dash_table


from dash.dependencies import Input, Output, State

import plotly.express as px
import pandas as pd

import ITR
from ITR.data.excel import ExcelProvider
from ITR.portfolio_aggregation import PortfolioAggregationMethod
from ITR.temperature_score import TemperatureScore
from ITR.interfaces import ETimeFrames, EScope

import urllib.request
import os


import base64
import datetime
import io

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


provider = ExcelProvider(path="data/data_provider_example.xlsx")
df_portfolio = pd.read_csv("data/example_portfolio.csv", encoding="iso-8859-1", sep=';')
companies = ITR.utils.dataframe_to_portfolio(df_portfolio)



def agg_score(agg_method):
    temperature_score = TemperatureScore(time_frames = [ETimeFrames.LONG],scopes=[EScope.S1S2],aggregation_method=agg_method) # Options for the aggregation method are WATS, TETS, AOTS, MOTS, EOTS, ECOTS, and ROTS
    amended_portfolio = temperature_score.calculate(data_providers=[provider], portfolio=companies)
    aggregated_scores = temperature_score.aggregate_scores(amended_portfolio)
    return [agg_method.value,aggregated_scores.long.S1S2.all.score]

agg_temp_scores = [agg_score(i) for i in PortfolioAggregationMethod]
fig_compare_weight = px.bar(pd.DataFrame(agg_temp_scores), x=0, y=1, text=1, title = "Comparing of different weighting methods applied to portfolio")


temperature_score = TemperatureScore(time_frames = [ETimeFrames.LONG],scopes=[EScope.S1S2],aggregation_method=PortfolioAggregationMethod.WATS) # Options for the aggregation method are WATS, TETS, AOTS, MOTS, EOTS, ECOTS, and ROTS
amended_portfolio = temperature_score.calculate(data_providers=[provider], portfolio=companies)
amended_portfolio_short=amended_portfolio[['company_name', 'time_frame', 'scope', 'temperature_score']]

fig1 = px.scatter(amended_portfolio, x="cumulative_target", y="cumulative_budget", size="temperature_score", color = "sector",title="Overview of portfolio")
fig2 = px.scatter(amended_portfolio, x="sector", y="region", size="temperature_score")
fig3 = px.bar(amended_portfolio.query("temperature_score > 2"), x="company_name", y="temperature_score", text ="temperature_score", color="sector",title="Worst contributors")


grouping = ['sector', 'region']
temperature_score.grouping = grouping
grouped_portfolio = temperature_score.calculate(data_providers=[provider], portfolio=companies)
grouped_aggregations = temperature_score.aggregate_scores(grouped_portfolio)

def generate_table(dataframe, max_rows=10):
    return html.Table([
                        html.Thead(
                            html.Tr([html.Th(col) for col in dataframe.columns])
                        ),
                        html.Tbody([
                            html.Tr([
                                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
                            ]) for i in range(min(len(dataframe), max_rows))
                        ])
                    ])

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8'))) # Assume that the user uploaded a CSV file
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded)) # Assume that the user uploaded an excel file
    except Exception as e:
        print(e)
        return html.Div(['There was an error processing this file.'])

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        dash_table.DataTable(data=df.to_dict('records'),
                            columns=[{'name': i, 'id': i} for i in df.columns]
        ),

        html.Hr(),  # horizontal line
        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={'whiteSpace': 'pre-wrap','wordBreak': 'break-all'})
    ])



app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
                        children=[
                            html.H1(children='ITR Tool',
                                    style={'textAlign': 'center'}),
                            html.Div(children='Calculation of temperature score for the provided portfolio of financial instruments', 
                                     style={'textAlign': 'center'}),

                            dcc.Upload(
                                id='upload-data',
                                children=html.Div(['Drag and Drop or ',html.A('Select Files')]),
                                style={'width': '100%','height': '60px','lineHeight': '60px',
                                    'borderWidth': '1px','borderStyle': 'dashed','borderRadius': '5px',
                                    'textAlign': 'center','margin': '10px'},
                                multiple=False # Allow multiple files to be uploaded
                            ),
                            html.Div(id='output-data-upload'),


                            dcc.Graph(id='Overview',figure=fig1),                            
                            dcc.Graph(id='Sector - Region',figure=fig2),                            
                            dcc.Graph(id='Worst scores',figure=fig3),
                            dcc.Graph(id='Compare_weights',figure=fig_compare_weight),                                                        
                            # html.H1(children='US Agriculture Exports (2011)'),
                            # generate_table(amended_portfolio_short),


                         ], # style={'columnCount': 2}
                        )



@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))

def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

if __name__ == '__main__':
    app.run_server(debug=True) # automatic reloading

