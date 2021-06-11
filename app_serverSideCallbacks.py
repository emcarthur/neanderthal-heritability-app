#### Resources ####
# Client side callbacks with js: https://github.com/covid19-dash/covid-dashboard/blob/master/app.py
# https://devcenter.heroku.com/articles/git
# https://devcenter.heroku.com/articles/git
# https://github.com/emcarthur/neanderthal-heritability-app
# https://neanderthal-heritability.herokuapp.com/
# https://stackoverflow.com/questions/60938972/dash-implementing-a-trace-highlight-callback
# https://stackoverflow.com/questions/47949173/deploy-a-python-dash-app-to-heroku-using-conda-environments-instead-of-virtua
# to push app to github & heroku git push origin main and git push heroku main

# client side callbacks: https://replit.com/@StephenTierney/EnlightenedImpartialModule
#https://community.plotly.com/t/is-it-possible-to-update-just-layout-not-whole-figure-of-graph-in-callback/8300/22
# Just a minor addition to your great example: You have to make a deep copy of the figure object. Otherwise you might run into issues when adding elements to an existing list, e.g. such as traces.
#I was able to completely decouple any figure modification (e.g. adding descriptive shapes, traces, annotations etc.) while keeping the original (large) figure data in a dcc.Store object and not resend it on every update (of the auxiliary data).
#https://dash.plotly.com/clientside-callbacks
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
from scipy.stats.kde import gaussian_kde
from plotly.subplots import make_subplots




app = dash.Dash(__name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}],
    suppress_callback_exceptions=True
)
server = app.server

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
def selectFitnessWeights(intialDistSkew, riskDecreasingPressure, riskIncreasingPressure, count):
    # initialDistSkew between -2 to +2
    # riskDecreasingPressure & riskIncreasingPressure is between -0.05 and 0.05
    # count 5 to 100

    fitness_norm = fitness_norm = [skewnorm.ppf(x, intialDistSkew) for x in np.linspace(0.01,0.99,count)]
    fitness_01 = scaleToRange(fitness_norm, realRange=(-2.3,2.3))
    fitness_weights = [ riskIncreasingPressure/((2.5)**3)*x**3 if x > 0 else -riskDecreasingPressure/((2.5)**3)*x**3 for x in fitness_norm]

    return (fitness_weights, fitness_norm, fitness_01)

#### DATA READ AND INITIALIZE ####

af_df = pd.read_csv("simulations.csv") # read in
af_df['fitness_weight'] = np.round(af_df['fitness_weight'],3) #!remove
cmap = LinearSegmentedColormap.from_list('BgR',['#0024af','#e2dee6','#b70000']) # generate color map
#fitness_weights, fitness_norm, fitness_01 = selectFitnessWeights(0,-0.03,-0.03,30) # intial values


#### STYLES ####
# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 62.5,
    "left": 0,
    "bottom": 0,
    "width": "27rem",
    "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    "padding": "0.5rem 1rem",
    "background-color": "#f8f9fa",
}

SIDEBAR_HIDEN = {
    "position": "fixed",
    "top": 62.5,
    "left": "-27rem",
    "bottom": 0,
    "width": "27rem",
    "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    "padding": "0rem 0rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "transition": "margin-left .5s",
    "margin-left": "29rem",
    "margin-right": "2rem",
    "padding": "1rem 1rem",
}

CONTENT_STYLE1 = {
    "transition": "margin-left .5s",
    "margin-left": "2rem",
    "margin-right": "2rem",
    "padding": "1rem 1rem",
}

#### LAYOUT ITEMS ####

## Sidebar questions ##

# Question 1 about skew of neanderthal alleles with respect to trait-association
question1 = dbc.FormGroup([
    dbc.Label("At the time of introgression ➀ , how were the majority of Neanderthal alleles associated with the trait?",style={'font-size':"13px",'font-weight':'bold'},html_for="initialDistSkew"),
    html.P('More Neanderthal alleles were:' ,style={'font-size':'11px'}),
    dcc.Slider(
        id='intialDistSkew', min=-3, max=3,
        value=0, step=0.01,
        marks={-3:{'label':'Risk-decreasing', 'style':{'font-size':'12px','width':'20px','text-align':'center'}},
        0:{'label':'Equally risk-increasing and decreasing', 'style':{'font-size':'12px'}},
        3:{'label':'Risk-increasing', 'style':{'font-size':'12px'}}}
        ,),
],style={'margin-bottom':'30px'})

# Question 2 about fitness relationship to trait
question2 = dbc.FormGroup([
    dbc.Label("What were the selective pressures on risk-DECREASING alleles?",style={'font-size':"13px",'font-weight':'bold'},html_for="riskDecreasingPressure"),
    dcc.Slider(
        id='riskDecreasingPressure', min=-0.05, max=0.05,
        value=-0.01, step=0.0001,
        marks={-0.05:{'label':'Negative', 'style':{'font-size':'12px','width':'20px','text-align':'center'}},
        0:{'label':'Neutral', 'style':{'font-size':'12px'}},
        0.05:{'label':'Positive', 'style':{'font-size':'12px'}}}
        ,),
])

# Question 3 about fitness relationship to trait
question3 = dbc.FormGroup([
    dbc.Label("What were the selective pressures on risk-INCREASING alleles?",style={'font-size':"13px",'font-weight':'bold'},html_for="riskIncreasingPressure"),
    dcc.Slider(
        id='riskIncreasingPressure', min=-0.05, max=0.05,
        value=-0.01, step=0.0001,
        marks={-0.05:{'label':'Negative', 'style':{'font-size':'12px','width':'20px','text-align':'center'}},
        0:{'label':'Neutral', 'style':{'font-size':'12px'}},
        0.05:{'label':'Positive', 'style':{'font-size':'12px'}}}
        ,),
])

# Question 4 about number of variants
question4 = dbc.FormGroup(
    [
        dbc.Label("How many variants to visualize? (5-100)", html_for="count", width=9, style={'font-size':"13px",'font-weight':'bold'}),
        dbc.Col(dbc.Input(type="number", min=5,max=100,step=1, value=8, id='count',style={'font-size':"12px",'margin-top':'5px'}), width=3),
    ],
    row=True,
)

# Optional buttons for examples
optional = html.Div([
    html.H3("Or try some example settings:", style={'font-size':"13px", 'font-weight':'bold',"margin-top": '0px'}),
    dbc.Row([
        dbc.Col(dbc.Button("Loss of risk-conferring variants leading to heritability depletion with remaining alleles conferring uni-directional effects (Schizophrenia, Anorexia)", outline=True, color="secondary", className="mr-1",size='sm',style={'font-size':'10px'}, id='scz_example',block=True),width=6,style={'padding':'2px'}),
        dbc.Col(dbc.Button("Maintenence of risk-conferring variants leading to heritability enrichment with remaining alleles conferring uni-directional effects (Sunburn, Balding)", outline=True, color="secondary", className="mr-1",size='sm',style={'font-size':'10px'},id='sunburn_example',block=True),width=6,style={'padding':'2px'}),
    ]),
    dbc.Row([
        dbc.Col(dbc.Button("Loss of trait-associated variants leading to heritability depletion (Most traits)", outline=True, color="secondary", className="mr-1",size='sm',style={'font-size':'10px'}, id='most_example',block=True),width=6,style={'padding':'2px'}),
        dbc.Col(dbc.Button("Maintenence of trait-associated variants leading to heritability enrichment with bi-directionality (Autoimmunity, White blood cell count)", outline=True, color="secondary", className="mr-1",size='sm',style={'font-size':'10px'},id='wbc_example',block=True),width=6,style={'padding':'2px'}),
    ]),
], style={'display':'inline'})

# Combine Q1-4 into one set of controls
controls = html.Div([
    question1,
    question2,
    question3,
    question4,
    dbc.Button("Submit", color="success", id='submit',size='sm',block=True),
    html.Hr(),
    optional,
])


## Menu/NavBar/Sidebar ##

dropdown = dbc.DropdownMenu(
            [
                dbc.DropdownMenuItem("Home", href="/"),
                dbc.DropdownMenuItem("Methods details", href="/methods"),
                dbc.DropdownMenuItem("GitHub Code", href="https://github.com/emcarthur/neanderthal-heritability-app"),
                dbc.DropdownMenuItem("Paper", href="https://www.biorxiv.org/content/10.1101/2020.06.08.140087v1"),
            ],
            in_navbar=True,
            label="Menu items",
            className="ml-auto", #className="ml-auto flex-nowrap mt-3 mt-md-0",
            right=True,
        )

navbar = dbc.Navbar(
    [
            dbc.Row(
                [
                    dbc.Col(dbc.Button("▸ Toggle sidebar controls", color="secondary", className="mr-1", id="btn_sidebar")),
                ],
                align="center",
                no_gutters=True,
            ),
        dropdown
    ],
    color="dark",
    dark=True,
)

sidebar = html.Div([
        controls,
    ],
    id="sidebar",
    style=SIDEBAR_STYLE,
)

content = html.Div(
    id="page-content",
    style=CONTENT_STYLE
)

app.layout = html.Div(
    [
        dcc.Store(id='side_click'),
        dcc.Location(id="url"),
        navbar,
        sidebar,
        content,
    ],
)
#jumbotron = dbc.Row([dbc.Col(dcc.Markdown('''
#### Visualizing the theoretical evolutionary trajectory of trait-associated Neanderthal-introgressed alleles

# <div class="col-md-12">
#    <div class="pull-left"><img src="YourImage.png"/></div>
#    <div class="pull-left">Your text goes here......</div>
# </div>
jumbotron =  dbc.Row([
        html.H3("Visualizing the theoretical evolutionary trajectory of trait-associated Neanderthal-introgressed alleles", style={'font-size':'20px'}),
        dbc.Row([
            html.Div([
                html.Img(src=app.get_asset_url("tree.png"),alt="tree",style={'width':'600px','height':'92px','float':'right'}),
                html.Div("2-4% of modern Eurasian genomes are inherited from our Neanderthal ancestors. Some introgressed variants were likely harmful and lost through drift or selection. Other variants may have provided adaptive benefits to humans as they migrated out of Africa.", className="lead", style={'font-size':'13px','margin-bottom':'8px'}),
                html.Div("We propose a model that, since hybridization ➀, introgressed variation associated with different traits experienced different evolutionary histories leading to patterns in GWAS we see today ➁.", className="lead", style={'font-size':'13px','margin-bottom':'8px'}),
                html.Div("We built this tool to explore and visualize some different theoretical trajectories of variants associated with traits. Toggle the sidebar controls (upper left) to explore!", className="lead", style={'font-size':'13px'}),
            ])
        ],style={'margin-right':'0px','margin-left':'0px'}),
        #html.P("", className="lead", style={'font-size':'13px'}),
    ])

main_page = dbc.Container(
            [
                jumbotron,
                html.Hr(),
                dbc.Row([
                    dbc.Col(html.H2("➀ Trait-associated distribution of introgressed variants at hybridization", className="display-4", style={'font-size':'14px','font-weight':'bold','text-align':'center'}),width=3),
                    dbc.Col(html.H2("Allele frequency trajectory of introgressed variants", className="display-4", style={'font-size':'14px','font-weight':'bold','text-align':'center'}),width=6),
                    dbc.Col(html.H2("➁ Trait-associated distribution of REMAINING introgressed variants", className="display-4", style={'font-size':'14px','font-weight':'bold','text-align':'center'}),width=3),
                ]),
                dbc.Row([
                    dbc.Col([dcc.Loading(id = "loading-icon", children=[html.Div(dcc.Graph(id='dist1_graph'))], type="circle"),
                                html.Div(html.Img(src=app.get_asset_url("arrow1.png"),alt="arrow",style={'height':'60px','width':'auto'}), style={'text-align':'right','display':'block',}),
                                html.Div(html.Img(src=app.get_asset_url("legend.png"),alt="legend",style={'height':'100px','width':'auto'}), style={'text-align':'left','display':'block','padding':'5px'})],
                                width=3,style={'padding':'0px'}),
                    dbc.Col(dcc.Loading(id = "loading-icon", children=[html.Div(dcc.Graph(id='af_graph'))], type="circle"),width=6),
                    dbc.Col([dcc.Loading(id = "loading-icon", children=[html.Div(dcc.Graph(id='dist2_graph'))], type="circle"),
                        html.Div([
                            html.Img(src=app.get_asset_url("arrow2.png"),alt="arrow",style={'height':'60px','width':'auto','display':'inline-block'}),
                            html.Div("Resulting heritability and directionality patterns in GWAS", style={'font-size':'13px','font-weight':'bold','text-align':'left','margin-left':'10px'})
                        ],style={'display':'flex','align-items':'center'}),
                        dcc.Loading(id = "loading-icon", children=[html.Div(dcc.Graph(id='arrow_graph'),style={'display':'inline-block','text-align':'center'})], type="circle")
                    ], width=3,style={'padding':'0px'}),
                ])
            ], style={'width':'100%','max-width':'none'}
        )


@app.callback(
    [Output("sidebar", "style"),
    Output("page-content", "style"),
    Output("side_click", "data"),],
    [Input("btn_sidebar", "n_clicks")],
    [State("side_click", "data")],
)
def toggle_sidebar(n, nclick):
    if n:
        if nclick == "SHOW":
            sidebar_style = SIDEBAR_HIDEN
            content_style = CONTENT_STYLE1
            cur_nclick = "HIDDEN"
        else:
            sidebar_style = SIDEBAR_STYLE
            content_style = CONTENT_STYLE
            cur_nclick = "SHOW"
    else:
        sidebar_style = SIDEBAR_HIDEN
        content_style = CONTENT_STYLE1
        cur_nclick = 'HIDDEN'

    return sidebar_style, content_style, cur_nclick


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname in ["/", ""]:
        return main_page
    elif pathname == "/methods":
        return html.P("Methods text to come here!")
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )

#### CALLBACKS AND INTERACTIVE CONTENT ####

# Loading figure 1
@app.callback(Output("loading-icon", "children"), Input("dummy", "children"))

# Main callback which updates when you click "submit" or any of the example buttons and outputs the two graphs (and updates the parameter control panel)
@app.callback(
    [Output("af_graph", "figure"),
    Output("dist1_graph", "figure"),
    Output("dist2_graph", "figure"),
    Output("arrow_graph", "figure"),
    Output("intialDistSkew", "value"),
    Output("riskDecreasingPressure", "value"),
    Output("riskIncreasingPressure", "value"),
    Output("count", "value")],
    [Input("submit","n_clicks"),
    Input("most_example","n_clicks"),
    Input("wbc_example","n_clicks"),
    Input("scz_example","n_clicks"),
    Input("sunburn_example","n_clicks")],
    [State("intialDistSkew", "value"),
    State("riskDecreasingPressure", "value"),
    State("riskIncreasingPressure", "value"),
    State("count", "value"),
    ])
def update_graph(btn1, btn2, btn3, btn4, btn5, intialDistSkew,riskDecreasingPressure, riskIncreasingPressure,count):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0] # what button was last pressed

    if 'submit' in changed_id: # if submit was last pressed, will use the input state for the 4 parameters
        if count is None: # in case the user sets an invalid "count" value
            count = 20
    elif 'most_example' in changed_id: # if example 1 was last pressed, set the 4 parameters
        intialDistSkew=0
        riskDecreasingPressure=-0.02
        riskIncreasingPressure=-0.02
        count=30
    elif 'wbc_example' in changed_id: # if example 2 was last pressed, set the 4 parameters
        intialDistSkew=0
        riskDecreasingPressure=0.02
        riskIncreasingPressure=0.02
        count=30
    elif 'scz_example' in changed_id: # if example 2 was last pressed, set the 4 parameters
        intialDistSkew=1.2
        riskDecreasingPressure=0.05
        riskIncreasingPressure=-0.05
        count=30
    elif 'sunburn_example' in changed_id: # if example 2 was last pressed, set the 4 parameters
        intialDistSkew=1.5
        riskDecreasingPressure=-0.025
        riskIncreasingPressure=0.025
        count=30

    # Pick the variants from the fitness distribution generated from the 4 input parameters
    fitness_weights, fitness_norm, fitness_01 = selectFitnessWeights(intialDistSkew,riskDecreasingPressure,riskIncreasingPressure,count)
    af_df['used'] = False
    fitness_remaining = []
    fitness_remaining01 = []

    # FIGURE 1
    af_fig = go.Figure()

    for idx, i in enumerate(fitness_weights): # loop through all the variants to plot them
        minVal = min(abs(i - af_df[af_df['used'] == False]['fitness_weight'])) # find the unused simulation with the fitness_weight closest to the desired weight (i)
        selectedVar = random.choice(af_df[(abs(i - af_df['fitness_weight']) == minVal) &  (af_df['used'] == False)].index)
        af_df.loc[selectedVar,'used'] = True # mark that you have used that simulation
        af = af_df.loc[selectedVar,[str(x) for x in list(range(10))]].values # select that simulation
        af_fig.add_trace( # plot that simulation
            go.Scatter(x=list(range(10)), y=af, opacity=0.5, mode='lines',line=dict(color=to_cmap_rgb(fitness_01[idx]),dash=linestyle(af)), name=f'Variant #{idx}: AF: {round(af[-1],2)}', showlegend=False ), #color=cmap(fitness_01[idx]) hoverinfo='skip'
            )
        if af[-1] > 0.01: # if variant was not "lost/rare" add it to the variants that remain
            fitness_remaining.append(fitness_norm[idx])
            fitness_remaining01.append(fitness_01[idx])

    if len(fitness_remaining) == 0:
        fitness_remaining = [0.011, -0.01]
        fitness_remaining01 = [0.49,0.51]
    elif len(fitness_remaining) == 1:
        fitness_remaining.append(0)
        fitness_remaining01.append(0.5)

    af_fig.update_layout( # update figure 1 layout
        template='simple_white',
        xaxis_title="Generations",
        yaxis_title="Allele frequency",
        height=360,
        margin=dict(l=40,r=20,t=0,b=10),
        font=dict(
            size=9),
    )
    af_fig.update_xaxes( # update figure 1 axis
        ticktext=["➀ At time of introgression", "➁ At time of modern humans"],
        tickvals=[300, 1700],
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
    xx = np.linspace(-4,4,500)
    kernel = gaussian_kde(fitness_norm)
    kde_y = kernel(xx)

    dist1_fig = make_subplots(rows=2,cols=1,row_heights=[0.8,0.2],shared_xaxes=True,vertical_spacing=0.04)
    dist1_fig.add_trace(go.Scatter(x=xx, y=kde_y, mode='lines',line=dict(color=to_cmap_rgb(0.5),dash='solid'), showlegend=False,hoverinfo='skip',fill='tozeroy'),row=1,col=1)#, #color=cmap(fitness_01[idx]) )
    dist1_fig.add_trace(go.Scatter(x=xx[xx <-1.5], y=kde_y[xx < -1.5], mode='lines',line=dict(color=to_cmap_rgb(0.2),dash='solid'), showlegend=False,hoverinfo='skip',fill='tozeroy'),row=1,col=1)#, #color=cmap(fitness_01[idx]) )
    dist1_fig.add_trace(go.Scatter(x=xx[xx >1.5], y=kde_y[xx > 1.5], mode='lines',line=dict(color=to_cmap_rgb(0.8),dash='solid'), showlegend=False,hoverinfo='skip',fill='tozeroy'),row=1,col=1)#, #color=cmap(fitness_01[idx]) )
    dist1_fig.add_vline(x=0,line_width=1.2,line_dash="dot", line_color='black',row=1,col=1)
    for i,j in zip(fitness_norm,fitness_01):
        dist1_fig.add_trace(go.Box(
            x=np.array(i),
            y=np.array(0),
            marker_symbol='line-ns-open',
            marker_color=to_cmap_rgb(j),
            boxpoints='all',
            jitter=0,
            fillcolor='rgba(255,255,255,0)',
            line_color='rgba(255,255,255,0)',
            hoverinfo='skip',
            showlegend=False,
        ), row=2, col=1)


    dist1_fig.update_layout( # update figure 2 layout
        template='simple_white',
        yaxis_title="Density",
        xaxis2_title="Trait-association direction",
        yaxis1={'tickvals':[],'fixedrange':True},
        yaxis2 = {'tickvals':[0],'ticktext':['%s Variants' % count],'fixedrange':True},
        font=dict(
            size=9),
        margin=dict(l=10,r=10,t=10,b=10),
        height=140,
    )
    dist1_fig.update_xaxes( # update figure 2 axis
        range=[-4,4],
        ticktext=["Protective", "No trait-association","Risk"],
        tickvals=[-3.5, 0, 3.5],
        fixedrange=True,
        ticklen=0
    )

# FIGURE 2
    # FIGURE 2
    xx = np.linspace(-4,4,500)
    kernel = gaussian_kde(fitness_remaining)
    kde_y = kernel(xx)

    dist2_fig = make_subplots(rows=2,cols=1,row_heights=[0.8,0.2],shared_xaxes=True,vertical_spacing=0.04)
    dist2_fig.add_trace(go.Scatter(x=xx, y=kde_y, mode='lines',line=dict(color=to_cmap_rgb(0.5),dash='solid'), showlegend=False,hoverinfo='skip',fill='tozeroy'),row=1,col=1)#, #color=cmap(fitness_01[idx]) )
    dist2_fig.add_trace(go.Scatter(x=xx[xx <-1.5], y=kde_y[xx < -1.5], mode='lines',line=dict(color=to_cmap_rgb(0.2),dash='solid'), showlegend=False,hoverinfo='skip',fill='tozeroy'),row=1,col=1)#, #color=cmap(fitness_01[idx]) )
    dist2_fig.add_trace(go.Scatter(x=xx[xx >1.5], y=kde_y[xx > 1.5], mode='lines',line=dict(color=to_cmap_rgb(0.8),dash='solid'), showlegend=False,hoverinfo='skip',fill='tozeroy'),row=1,col=1)#, #color=cmap(fitness_01[idx]) )
    dist2_fig.add_vline(x=0,line_width=1.2,line_dash="dot", line_color='black',row=1,col=1)
    for i,j in zip(fitness_remaining,fitness_remaining01):
        dist2_fig.add_trace(go.Box(
            x=np.array(i),
            y=np.array(0),
            marker_symbol='line-ns-open',
            marker_color=to_cmap_rgb(j),
            boxpoints='all',
            jitter=0,
            fillcolor='rgba(255,255,255,0)',
            line_color='rgba(255,255,255,0)',
            hoverinfo='skip',
            showlegend=False,
        ), row=2, col=1)


    dist2_fig.update_layout( # update figure 2 layout
        template='simple_white',
        yaxis_title="Density",
        xaxis2_title="Trait-association direction",
        yaxis1={'tickvals':[],'fixedrange':True},
        yaxis2 = {'tickvals':[0],'ticktext':['%s Remaining<br>Variants' % len(fitness_remaining)],'fixedrange':True},
        font=dict(
            size=9),
        margin=dict(l=10,r=10,t=10,b=10),
        height=140,
    )
    dist2_fig.update_xaxes( # update figure 2 axis
        range=[-4,4],
        ticktext=["Protective", "No trait-association","Risk"],
        tickvals=[-3.5, 0, 3.5],
        fixedrange=True,
        ticklen=0
    )


    x_arrow_val = (sum(kde_y[xx <-1.5]) + sum(kde_y[xx >1.5])) - 11
    if x_arrow_val > 0:
        x_arrow_val = min(max(2,x_arrow_val),10)
        x_arrow_start = -0.2
    else:
        x_arrow_val = max(min(-2,x_arrow_val),-10)
        x_arrow_start = 0.2

    if (sum(kde_y[xx <-1.5]) == 0) or (sum(kde_y[xx <-1.5]) == 0):
        y_arrow_val = -10
    else:
        y_arrow_val = max(sum(kde_y[xx <-1.5])/sum(kde_y[xx >1.5]),sum(kde_y[xx >1.5])/sum(kde_y[xx <-1.5]))
        y_arrow_val = 15*np.log10(y_arrow_val/5)
    if y_arrow_val > 0:
        y_arrow_val = min(max(2,y_arrow_val),10)
        y_arrow_start = -0.2
    else:
        y_arrow_val = max(min(-2,y_arrow_val),-10)
        y_arrow_start = 0.2

    arrow_fig = go.Figure()
    arrow_fig.add_annotation(
      x=x_arrow_val,  # arrows' head
      y=y_arrow_val,  # arrows' head
      ax=x_arrow_start,  # arrows' tail
      ay=y_arrow_start,  # arrows' tail
      xref='x',
      yref='y',
      axref='x',
      ayref='y',
      text='',  # if you want only the arrow
      showarrow=True,
      arrowhead=2,
      arrowsize=1,
      arrowwidth=3,
      arrowcolor='gray'
    )
    arrow_fig.update_layout(
        template='simple_white',
        yaxis_title="Directionality<br>(distribution skew)",
        xaxis_title="Heritability contribution<br>(weight in distribution tails)",
        font=dict(
            size=8),
        margin=dict(l=10,r=10,t=10,b=10),
        height=150,
        width=230,
    )
    arrow_fig.update_xaxes( # update figure 2 axis
        range=[-10,10],
        ticktext=["Depletion", "Enrichment"],
        tickvals=[-8, 8],
        fixedrange=True,
        ticklen=0,
        showline=False,
        mirror=True,
    )
    arrow_fig.update_yaxes( # update figure 2 axis
        range=[-10,10],
        ticktext=["Bi-directional", "Uni-directional"],
        tickvals=[-8, 8],
        fixedrange=True,
        ticklen=0,
        showline=False,
        mirror=True,
    )

    arrow_fig.add_vline(x=0,line_width=1.2, line_color='black',opacity=1)
    arrow_fig.add_hline(y=0,line_width=1.2,line_color='black',opacity=1)

    af_fig.write_html("test4.html", include_plotlyjs='directory')


    return af_fig, dist1_fig, dist2_fig, arrow_fig, intialDistSkew, riskDecreasingPressure, riskIncreasingPressure, count # update the figures and the parameter panels


if __name__ == "__main__":
    app.run_server(debug=True)
