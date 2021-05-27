#### Resources ####
# Client side callbacks with js: https://github.com/covid19-dash/covid-dashboard/blob/master/app.py
# https://devcenter.heroku.com/articles/git
# https://devcenter.heroku.com/articles/git
# https://github.com/emcarthur/neanderthal-heritability-app
# https://neanderthal-heritability.herokuapp.com/
# https://stackoverflow.com/questions/47949173/deploy-a-python-dash-app-to-heroku-using-conda-environments-instead-of-virtua

#### IMPORT DEPENDENCIES ####

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import random
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import skewnorm
import plotly.figure_factory as ff
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

#### FUNCTIONS FOR  AF TRAJECTORY VIZ ####

# Scale an array between two values
def scaleToRange(data,desiredRange=(0,1), realRange=False):
    if realRange == False:
        realRange=(min(data),max(data))
    data = np.array(data)
    data = data * (desiredRange[1] - desiredRange[0]) / (realRange[1] - realRange[0])

    if realRange == False:
        realRange_min = min(data)
    else:
        realRange_min = realRange[0] * (desiredRange[1] - desiredRange[0]) / (realRange[1] - realRange[0])
    data = data - realRange_min + desiredRange[0]
    return(np.round(data,3))

# Convert 0-1 value to string rgb using colormap
def to_cmap_rgb(value):
    c = cmap(value)
    color = f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})'
    return color

# Dash vs solid line based on final AF in array
def linestyle(array):
    if array[-1] < 0.05:
        return 'dash'
    else:
        return 'solid'

# Select the distribution of fitness weights (represented as colors) based on input values
def selectFitnessWeights(intialDistSkew, directionality, selectionAmount, count):
    # initialDistSkew between -2 to +2
    # directionality is a string: rd, ri, both, or none
    # selectionAmount 0 to +0.1
    # count 5 to 100

    fitness_norm = fitness_norm = [skewnorm.ppf(x, intialDistSkew) for x in np.linspace(0.01,0.99,count)]
    fitness_01 = scaleToRange(fitness_norm, realRange=(-2.3,2.3))
    if directionality == 'rd':
        fitness_weights = selectionAmount/((2.5)**3)*-np.array(fitness_norm)**3
    if directionality == 'ri':
        fitness_weights = selectionAmount/((2.5)**3)*np.array(fitness_norm)**3
    if directionality == 'both':
        fitness_weights = selectionAmount/((2.5)**3)*abs(np.array(fitness_norm)**3)
    if directionality == 'none':
        fitness_weights = selectionAmount/((2.5)**3)*-abs(np.array(fitness_norm)**3)

    return (fitness_weights, fitness_norm, fitness_01)

#### DATA READ AND INITIALIZE ####

af_df = pd.read_csv("simulations.csv") # read in
af_df['fitness_weight'] = np.round(af_df['fitness_weight'],3) #!remove
cmap = LinearSegmentedColormap.from_list('BgR',['#0024af','#e2dee6','#b70000']) # generate color map
fitness_weights, fitness_norm, fitness_01 = selectFitnessWeights(0,'rd',0.01,20) # intial values



#### CREATE APP COMPONENTS ####
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True) # Dash app with bootstrap css
server = app.server


# Question 1 about skew of neanderthal alleles with respect to trait-association
question1 = dbc.Row([
    html.H3("(1) At the time of introgression, how were the majority of Neanderthal alleles associated with the trait?",style={'font-size':"14px", "margin-top": '10px'}),
    html.P('More Meanderthal alleles were:' ,style={'font-size':'11px'}),
    html.Div(dcc.Slider(
        id='intialDistSkew', min=-3, max=3,
        value=0, step=0.01, marks={-3:'Risk-decreasing', 0:'Equally risk-decreasing and increasing',3:'Risk-increasing'}),style = {'width': '80%', 'padding-left': '15%', 'padding-right':'5%'}),
],style={'padding-bottom':'70px','padding-left':'10px','padding-right':'10px'})

# Question 2 about fitness relationship to trait
question2 = dbc.Row([
    html.H3("(2) Was there a directionality bias in the trait-associated introgressed alleles that were beneficial to human fitness in the Eurasian environment?", style={'font-size':"14px"}),
    dcc.RadioItems(
        options=[
            {'label': 'Yes, risk-DECREASING (blue) alleles were beneficial', 'value': 'rd'},
            {'label': 'Yes, risk-INCREASING (red) alleles were beneficial (e.g. sunburn)', 'value': 'ri'},
            {'label': 'No, all trait-associated variation was BENEFICIAL regardless of direction of effect', 'value': 'both'},
            {'label': 'No, all trait-associated variation was HARMFUL regardless of direction of effect', 'value': 'none'}
        ],
        value='rd', id='directionality', style={'font-size':'11px'}
    )
],style={'padding-bottom':'10px','padding-left':'10px','padding-right':'10px'})

# Question 3 about strength of selection
question3 = dbc.Row([
    html.H3("(3) What was the strength of selection on trait-associated variation?", style={'font-size':"14px"}),
    html.Div(dcc.Slider(
        id='selectionAmount', min=0, max=0.05,
        value=0.01, step=0.0001, marks={0:'No selection',0.05:'Strong selection'}),style = {'width': '80%', 'padding-left': '15%', 'padding-right':'5%'}),
],style={'padding-bottom':'50px','padding-left':'10px','padding-right':'10px'})

# Question 4 about number of variants
question4 = dbc.Row([
    html.H3("(4) How many variants do you want to visualize the allele frequency trajectory for?", style={'font-size':"14px", "margin-top": '15px'}),
    html.Div([html.P('Between 5-100:',style={'padding-right':'10px'}),dcc.Input(id='count', type='number', min=5, max=100, step=1, value=20,style={'height':'16px'}),html.P('(Tip: The left figure is easier to interpret with a small sample, while the right is better with larger numbers.)',style={'font-size':'8px'})],style={'font-size':'11px','display':'flex'}),
],style={'padding-bottom':'10px','padding-left':'10px','padding-right':'10px'})

# Optional buttons for examples
optional = dbc.Row([
    html.H3("Or try some example settings:", style={'font-size':"14px", "margin-top": '15px'}),
    dbc.Button("Loss of trait-associated variants leading to heritability depletion (Most traits)", outline=True, color="secondary", className="mr-1",size='sm',style={'text-size':'8px'}, id='example1'),
    dbc.Button("Maintenence of trait-associated variants leading to heritability enrichment with bi-directionality (Autoimmunity, White blood cell count)", outline=True, color="secondary", className="mr-1",size='sm',style={'text-size':'8px'},id='example2'),
    dbc.Button("Loss of risk-conferring variants leading to heritability depletion with remaining alleles conferring uni-directional effects (Schizophrenia, Anorexia)", outline=True, color="secondary", className="mr-1",size='sm',style={'text-size':'8px'}, id='example3'),
    dbc.Button("Maintenence of risk-conferring variants leading to heritability enrichment with remaining alleles conferring uni-directional effects (Sunburn, Balding)", outline=True, color="secondary", className="mr-1",size='sm',style={'text-size':'8px'},id='example4'),
], style={'padding-left':'10px','padding-right':'10px'})

# Combine Q1-4 into one set of controls
controls = dbc.Card([
    html.H3('Controls:',style={'font-size':'18px'}),
    question1,
    question2,
    question3,
    question4,
    dbc.Button("Submit", color="secondary", id='submit'),
    html.Hr(),
    optional,
], body=True)

# Jumbotron header with instructions
jumbotron = dbc.Jumbotron([
    html.H3("Visualizing the theoretical evolutionary trajectory of trait-associated Neanderthal-introgressed alleles", style={'font-size':'22px'}),
    html.P("2-4% of modern Eurasian genomes are inherited from our Neanderthal ancestors. Since introgression, these variants have had different evolutionary paths. Some variants were likely harmful and lost through drift or selection. Other variants may have provided adaptive benefits to humans as they migrated out of Africa. ", className="lead", style={'font-size':'13px'}),
    html.P("In our paper: McArthur, Rinker & Capra 'Quantifying the contribution of Neanderthal introgression to the heritability of complex traits'[Link], we propose a model that variation associated with different traits experienced different evolutionary histories leading to patterns in GWAS we see today. ", className="lead", style={'font-size':'13px'}),
    html.P("We built this tool to explore and visualize some different theoretical trajectories of variants associated with traits. Play around with some of the parameters or examples on the left to see what might have happened to introgressed variants since hybridization depending on their strength and direction of trait-association.", className="lead", style={'font-size':'13px'}),
],style={'padding':'1rem'})


#### APP LAYOUT ####
app.layout = dbc.Container([
    dbc.Row(
                [
                    dbc.Col([
                        dbc.Row(dbc.Col(jumbotron, width=12)),
                        dbc.Row([
                            dbc.Col(dcc.Loading(id = "loading-icon", children=[html.Div(dcc.Graph(id='af_graph'))], type="circle"),width=7,style={'padding':'0px'}),
                            dbc.Col(dcc.Loading(id = "loading-icon2", children=[html.Div(dcc.Graph(id='dist_graph'))], type="circle"),width=5,style={'padding':'0px'}),
                        ])
                    ], width=9),
                    dbc.Col(controls, width=3),
                ],
            ),
],fluid=True)

#### CALLBACKS AND INTERACTIVE CONTENT ####

# Loading figure 1
@app.callback(Output("loading-icon", "children"), Input("dummy", "children"))
# Loading figure 2
@app.callback(Output("loading-icon2", "children"), Input("dummy",'children'))

# Main callback which updates when you click "submit" or any of the example buttons and outputs the two graphs (and updates the parameter control panel)
@app.callback(
    [Output("af_graph", "figure"),
    Output("dist_graph", "figure"),
    Output("intialDistSkew", "value"),
    Output("directionality", "value"),
    Output("selectionAmount", "value"),
    Output("count", "value")],
    [Input("submit","n_clicks"),
    Input("example1","n_clicks"),
    Input("example2","n_clicks"),
    Input("example3","n_clicks"),
    Input("example4","n_clicks")],
    [State("intialDistSkew", "value"),
    State("directionality", "value"),
    State("selectionAmount", "value"),
    State("count", "value"),
    ])
def update_graph(btn1, btn2, btn3, btn4, btn5, intialDistSkew,directionality, selectionAmount,count):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0] # what button was last pressed

    if 'submit' in changed_id: # if submit was last pressed, will use the input state for the 4 parameters
        if count is None: # in case the user sets an invalid "count" value
            count = 20
    elif 'example1' in changed_id: # if example 1 was last pressed, set the 4 parameters
        intialDistSkew=0
        directionality='none'
        selectionAmount=0.02
        count=30
    elif 'example2' in changed_id: # if example 2 was last pressed, set the 4 parameters
        intialDistSkew=0
        directionality='both'
        selectionAmount=0.02
        count=30
    elif 'example3' in changed_id: # if example 2 was last pressed, set the 4 parameters
        intialDistSkew=1.2
        directionality='rd'
        selectionAmount=0.05
        count=30
    elif 'example4' in changed_id: # if example 2 was last pressed, set the 4 parameters
        intialDistSkew=1.5
        directionality='ri'
        selectionAmount=0.025
        count=30

    # Pick the variants from the fitness distribution generated from the 4 input parameters
    fitness_weights, fitness_norm, fitness_01 = selectFitnessWeights(intialDistSkew,directionality,selectionAmount,count)
    af_df['used'] = False
    fitness_remaining = []

    # FIGURE 1
    af_fig = go.Figure()

    for idx, i in enumerate(fitness_weights): # loop through all the variants to plot them
        minVal = min(abs(i - af_df[af_df['used'] == False]['fitness_weight'])) # find the unused simulation with the fitness_weight closest to the desired weight (i)
        selectedVar = random.choice(af_df[(abs(i - af_df['fitness_weight']) == minVal) &  (af_df['used'] == False)].index)
        af_df.loc[selectedVar,'used'] = True # mark that you have used that simulation
        af = af_df.loc[selectedVar,[str(x) for x in list(range(500))]].values # select that simulation
        af_fig.add_trace( # plot that simulation
            go.Scatter(x=list(range(500)), y=af, opacity=0.5, mode='lines',line=dict(color=to_cmap_rgb(fitness_01[idx]),dash=linestyle(af)), name=f'Variant #{idx}: AF: {round(af[-1],2)}', showlegend=False ), #color=cmap(fitness_01[idx]) hoverinfo='skip'
            )
        if af[-1] > 0.05: # if variant was not "lost/rare" add it to the variants that remain
            fitness_remaining.append(fitness_norm[idx])

    af_fig.update_layout( # update figure 1 layout
        title="Allele frequency trajectory of introgressed variants",
        template='simple_white',
        xaxis_title="Generations",
        yaxis_title="Allele frequency",
        font=dict(
            size=9),
    )
    af_fig.update_xaxes( # update figure 1 axis
        ticktext=["At time of introgression", "At time of modern humans"],
        tickvals=[50, 450],
        fixedrange=True,
        ticklen=0
    )
    af_fig.update_yaxes( # update figure 1 axis
        range=[0,1],
        tickvals=[0,0.05,0.2, 0.5,1],
        ticktext=["Lost alleles","Rare","Common","High Frequency", "Fixed"],
        fixedrange=True,
        ticklen=0
    )

    # FIGURE 2
    if len(fitness_remaining) > 1:
        hist_data = [fitness_remaining, fitness_norm]
        group_labels = ['Remaining introgressed alleles in modern Eurasians','Introgressed alleles in hybrids']
        colors = ['green', 'black']
    else:
        hist_data = [fitness_norm]
        group_labels = ['Introgressed alleles in hybrids']
        colors = ['black']

    dist_fig = ff.create_distplot(hist_data, group_labels, show_hist=False, colors=colors)

    dist_fig.update_layout( # update figure 2 layout
        title="Trait-associated distribution of introgressed alleles",
        template='simple_white',
        xaxis_title="Trait-association direction",
        yaxis_title="Density",
        font=dict(
            size=9),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
            ),
    )
    dist_fig.update_xaxes( # update figure 2 axis
        range=[-3,3],
        ticktext=["Protective", "No trait-association","Risk"],
        tickvals=[-2.5, 0, 2.5],
        fixedrange=True,
        ticklen=0
    )
    dist_fig.update_yaxes( # update figure 2 axis
        tickvals=[],
        fixedrange=True
    )

    return af_fig, dist_fig, intialDistSkew, directionality, selectionAmount, count # update the figures and the parameter panels

if __name__ == "__main__":
    app.run_server()
