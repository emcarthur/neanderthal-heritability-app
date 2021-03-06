#### IMPORT DEPENDENCIES ####
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, ClientsideFunction
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
import plotly.io
import time

### Initialize App ###
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1'}],
                suppress_callback_exceptions=True
                )
server = app.server
app.title = 'Neanderthal Heritability App'

#### Figure Storage ####
# Note: now that these are generated and exported to JSON, you only have to import them
afFigStorage = plotly.io.read_json("afFigStorage.json")
distFigStorage = plotly.io.read_json("distFigStorage.json")
arrowFigStorage = plotly.io.read_json("arrowFigStorage.json")

#from make_figures import make_afFig, make_distFig, make_arrowFig
#afFigStorage = make_afFig()
#afFigStorage.write_json("afFigStorage.json")
#distFigStorage = make_distFig()
#distFigStorage.write_json("distFigStorage.json")
#arrowFigStorage = make_arrowFig()
#arrowFigStorage.write_json("arrowFigStorage.json")

#### FUNCTIONS FOR  AF TRAJECTORY VIZ ####

# Scale an array between two values
def scaleToRange(data, desiredRange=(0, 1), realRange=False):
    if realRange == False:
        realRange = (min(data), max(data))
    data = np.array(data)
    data = data * (desiredRange[1] - desiredRange[0]
                   ) / (realRange[1] - realRange[0])

    if realRange == False:
        realRange_min = min(data)
    else:
        realRange_min = realRange[0] * (desiredRange[1] -
                                        desiredRange[0]) / (realRange[1] - realRange[0])
    data = data - realRange_min + desiredRange[0]
    return(np.round(data, 3))


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
    fitness_norm = fitness_norm = [skewnorm.ppf(
        x, intialDistSkew) for x in np.linspace(0.01, 0.99, count)]
    fitness_01 = scaleToRange(fitness_norm, realRange=(-2.3, 2.3))
    fitness_weights = [riskIncreasingPressure / ((2.5)**3) * x**3 if x >
                       0 else -riskDecreasingPressure / ((2.5)**3) * x**3 for x in fitness_norm]

    return (fitness_weights, fitness_norm, fitness_01)


# Select the simulation index from the simulations.csv based on the fitness weights
def selectIndexes(fitness_weights, fitness_norm, fitness_01, af_df):
    af_df['used'] = False
    sim_idxs = []
    fitness_remaining = []
    fitness_remaining01 = []
    # loop through all the variants to plot them:
    for idx, i in enumerate(fitness_weights):
        # find the unused simulation with the fitness_weight closest to the desired weight (i):
        minVal = min(abs(i - af_df[af_df['used'] == False]['fitness_weight']))
        selectedVar = random.choice(
            af_df[(abs(i - af_df['fitness_weight']) == minVal) & (af_df['used'] == False)].index)
        # mark that you have used that simulation:
        af_df.loc[selectedVar, 'used'] = True
        sim_idxs.append(selectedVar)
        # if variant was not "lost/rare" add it to the variants that remain
        if af_df.loc[selectedVar][-2] > 0.01:
            fitness_remaining.append(fitness_norm[idx])
            fitness_remaining01.append(fitness_01[idx])
    return sim_idxs, fitness_remaining, fitness_remaining01


#### DATA READ AND INITIALIZE ####
af_df = pd.read_csv("simulations.csv") # read in
af_df['fitness_weight'] = np.round(af_df['fitness_weight'],3) # for some randomness
cmap = LinearSegmentedColormap.from_list('BgR',['#0024af','#e2dee6','#b70000']) # generate color map

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
    dbc.Label("At the time of introgression ??? , how were the majority of Neanderthal alleles associated with the trait?", style={
              'font-size': "13px", 'font-weight': 'bold'}, html_for="initialDistSkew"),
    html.P('More Neanderthal alleles were:', style={'font-size': '11px'}),
    dcc.Slider(
        id='intialDistSkew', min=-3, max=3,
        value=0, step=0.01,
        marks={-3: {'label': 'Risk-decreasing', 'style': {'font-size': '12px', 'width': '20px', 'text-align': 'center'}},
               0: {'label': 'Equally risk-increasing and decreasing', 'style': {'font-size': '12px'}},
               3: {'label': 'Risk-increasing', 'style': {'font-size': '12px'}}},),
], style={'margin-bottom': '30px'})

# Question 2 about fitness relationship to trait
question2 = dbc.FormGroup([
    dbc.Label([html.Span("What were the selective pressures on risk-"), html.Span('DECREASING', style={'color': '#0024af'}), html.Span(
        ' alleles?')], style={'font-size': "13px", 'font-weight': 'bold'}, html_for="riskDecreasingPressure"),
    dcc.Slider(
        id='riskDecreasingPressure', min=-0.03, max=0.03,
        value=-0.003, step=0.0001,
        marks={-0.03: {'label': 'Negative', 'style': {'font-size': '12px', 'width': '20px', 'text-align': 'center'}},
               0: {'label': 'Neutral', 'style': {'font-size': '12px'}},
               0.03: {'label': 'Positive', 'style': {'font-size': '12px'}}},),
])

# Question 3 about fitness relationship to trait
question3 = dbc.FormGroup([
    dbc.Label([html.Span("What were the selective pressures on risk-"), html.Span('INCREASING', style={'color': '#b70000'}), html.Span(
        ' alleles?')], style={'font-size': "13px", 'font-weight': 'bold'}, html_for="riskIncreasingPressure"),
    dcc.Slider(
        id='riskIncreasingPressure', min=-0.03, max=0.03,
        value=-0.003, step=0.0001,
        marks={-0.03: {'label': 'Negative', 'style': {'font-size': '12px', 'width': '20px', 'text-align': 'center'}},
               0: {'label': 'Neutral', 'style': {'font-size': '12px'}},
               0.03: {'label': 'Positive', 'style': {'font-size': '12px'}}},),
])

# Question 4 about number of variants
question4 = dbc.FormGroup(
    [
        dbc.Label("How many variants to visualize? (5-50)", html_for="count",
                  width=9, style={'font-size': "13px", 'font-weight': 'bold'}),
        dbc.Col(dbc.Input(type="number", min=5, max=50, step=1, value=10, id='count', style={
                'font-size': "12px", 'margin-top': '5px'}), width=3),
    ],
    row=True,
)

# Optional buttons for examples
optional = html.Div([
    html.H3("Or try some example settings:", style={
            'font-size': "13px", 'font-weight': 'bold', "margin-top": '0px'}),
    dbc.Row([
        dbc.Col(dbc.Button("Loss of risk-conferring variants leading to heritability depletion with remaining alleles conferring uni-directional effects (Schizophrenia, Anorexia)",
                           outline=True, color="secondary", className="mr-1", size='sm', style={'font-size': '10px'}, id='scz_example', block=True), width=6, style={'padding': '2px'}),
        dbc.Col(dbc.Button("Maintenance of risk-conferring variants leading to heritability enrichment with remaining alleles conferring uni-directional effects (Sunburn, Balding)",
                           outline=True, color="secondary", className="mr-1", size='sm', style={'font-size': '10px'}, id='sunburn_example', block=True), width=6, style={'padding': '2px'}),
    ]),
    dbc.Row([
        dbc.Col(dbc.Button("Loss of trait-associated variants leading to heritability depletion (Most traits)", outline=True, color="secondary",
                           className="mr-1", size='sm', style={'font-size': '10px'}, id='most_example', block=True), width=6, style={'padding': '2px'}),
        dbc.Col(dbc.Button("Maintenance of trait-associated variants leading to heritability enrichment with bi-directionality (Autoimmunity, White blood cell count)",
                           outline=True, color="secondary", className="mr-1", size='sm', style={'font-size': '10px'}, id='wbc_example', block=True), width=6, style={'padding': '2px'}),
    ]),
], style={'display': 'inline'})

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
        dbc.DropdownMenuItem(
            "GitHub Code", href="https://github.com/emcarthur/neanderthal-heritability-app"),
        dbc.DropdownMenuItem(
            "NCOMMS Paper", href="https://www.nature.com/articles/s41467-021-24582-y"),
        dbc.DropdownMenuItem(
            "Nature EcoEvo Community article", href="https://natureecoevocommunity.nature.com/posts/uncovering-the-legacy-of-our-neanderthal-ancestors"),
        dbc.DropdownMenuItem(
            "bioRxiv Paper", href="https://www.biorxiv.org/content/10.1101/2020.06.08.140087v1"),
    ],
    in_navbar=True,
    label="Menu items",
    className="ml-auto",  # className="ml-auto flex-nowrap mt-3 mt-md-0",
    right=True,
)

# Header jumbotron with info about the app
jumbotron = dbc.Row([
    html.H3("Visualizing the theoretical evolutionary trajectory of trait-associated Neanderthal-introgressed alleles",
            style={'font-size': '20px'}),
    dbc.Row([
            html.Div([
                html.Img(src=app.get_asset_url("tree.png"), alt="tree", style={
                         'width': '600px', 'height': '92px', 'float': 'right'}),
                html.Div("2-4% of modern Eurasian genomes are inherited from our Neanderthal ancestors. Some introgressed variants were likely harmful and lost through drift or selection. Other variants may have provided adaptive benefits to humans as they migrated out of Africa.",
                         className="lead", style={'font-size': '13px', 'margin-bottom': '8px'}),
                html.Div("We propose a model that, since hybridization ???, introgressed variation associated with different traits experienced different evolutionary histories leading to patterns in GWAS we see today ???.",
                         className="lead", style={'font-size': '13px', 'margin-bottom': '8px'}),
                html.Div(["We built this tool to visualize some different theoretical ",
                          html.Span("(and highly oversimplified!!)", id="tooltip-target",
                                    style={"textDecoration": "underline", "cursor": "pointer"}),
                          dbc.Tooltip("DISCLAIMER!: This is a simple qualitative model. It is NOT a forward-time demographically-aware simulation of realistic evolution since introgression. We did not optimize any parameters, nor did we consider pleiotropy or LD. It is meant for illustrative purposes only and a tool for developing intuition for our model.",
                                      target="tooltip-target", style={'font-size': '10px'}),
                          " trajectories of variants associated with traits. Toggle the sidebar controls (upper left) to explore!"], className="lead", style={'font-size': '13px'}),
            ])
            ], style={'margin-right': '0px', 'margin-left': '0px'}),
])

# Content of main page: jumbotron, then three columns with figures
main_page = dbc.Container(
    [
        jumbotron,
        html.Hr(),
        dbc.Row([
            dbc.Col(html.H2("??? Trait-associated distribution of introgressed variants at hybridization",
                            className="display-4", style={'font-size': '14px', 'font-weight': 'bold', 'text-align': 'center'}), width=3),
            dbc.Col(
                [html.H2("Allele frequency trajectory of introgressed variants", className="display-4", style={'font-size': '14px', 'font-weight': 'bold', 'text-align': 'center'}),
                 html.Div(dcc.Loading(id="loading", children=[
                     ""], type="circle"), style={'visibility': 'visible'}),
                 ], width=6),
            dbc.Col(html.H2("??? Trait-associated distribution of REMAINING introgressed variants", className="display-4",
                            style={'font-size': '14px', 'font-weight': 'bold', 'text-align': 'center'}), width=3),
        ]),
        dbc.Row([
            dbc.Col([dcc.Graph(id='dist1_graph2'),
                     dcc.Store('distFigStorage1',
                               data=distFigStorage),
                     dcc.Store('distFig1_dataStorage'),
                     html.Div(html.Img(src=app.get_asset_url("arrow1.png"), alt="arrow", style={
                              'height': '60px', 'width': 'auto'}), style={'text-align': 'right', 'display': 'block', }),
                     html.Div(html.Img(src=app.get_asset_url("legend.png"), alt="legend", style={
                              'height': '100px', 'width': 'auto'}), style={'text-align': 'left', 'display': 'block', 'padding': '5px'}),
                     ], width=3, style={'padding': '0px'}),
            dbc.Col([
                dcc.Graph(id='af_graph2'),
                dcc.Store(id='afFigStorage', data=afFigStorage),
                dcc.Store(id='afFig_dataStorage'),
            ], width=6),
            dbc.Col([dcc.Graph(id='dist2_graph2'),
                     dcc.Store('distFigStorage2', data=distFigStorage),
                     dcc.Store('distFig2_dataStorage'),
                     html.Div([
                         html.Img(src=app.get_asset_url("arrow2.png"), alt="arrow", style={
                                  'height': '60px', 'width': 'auto', 'display': 'inline-block'}),
                         html.Div("Resulting heritability and directionality patterns in GWAS", style={
                                  'font-size': '13px', 'font-weight': 'bold', 'text-align': 'left', 'margin-left': '10px'})
                     ], style={'display': 'flex', 'align-items': 'center'}),
                     html.Div(dcc.Graph(id='arrow_graph'), style={
                              'display': 'inline-block', 'text-align': 'center'}),
                     dcc.Store(id='arrowFigStorage', data=arrowFigStorage),
                     dcc.Store(id='arrowFig_dataStorage'),
                     ], width=3, style={'padding': '0px'}),
        ])
    ], style={'width': '100%', 'max-width': 'none'}
)

# Content of methods page
methods = dbc.Container([
    html.H3("Methods", style={'font-size': '22px'}),
    dcc.Markdown('''
    _DISCLAIMER!: This is a simple qualitative model. It is NOT a forward-time demographically-aware simulation of realistic evolution since introgression. We did not optimize any parameters, nor did we consider pleiotropy or LD. It is meant for illustrative purposes only and a tool for developing intuition for our model._
    ''', className="lead", style={'font-size': '13px'}),
    html.H3("Simulating allele frequency trajectories",
            style={'font-size': '18px'}),
    dcc.Markdown('''
    With the limitations in mind, this is how we created the allele frequency trajectories:

    - We "simulated" an effective population size of 1861 individuals for 2000 generations. We randomly shuffled 186 "introgressed alleles" into the population. Therefore, at generation 0, the "introgressed" allele frequency is 5% `(186/[2 x 1861])`.
    - At each subsequent generation, we randomly sampled a pair of individuals from the population (parents) and randomly chose one copy of the allele from each parent to pass on to the offspring (simulating segregation).
    - This was done 1861 times for each generation to keep the effective population size consistent (1861 new offspring each generation). At each generation, we counted the number of introgressed alleles over the total number of alleles to record the allele frequency.
    - Repeat for 2000 generations!
    - To "simulate" selection, fitness weights were applied to parents. For example, if the introgressed allele (a) was less fit than the ancestral allele (A) with a weight of 0.01, the fitness of a parent with AA genotype would be "100% fit" while parent Aa and aa would be 99% and 98% as fit, respectively. When randomly selecting parents to produce offspring, the random selection is weighted by these fitnesses: `[1, 0.99, 0.98]`.

    This procedure and visualization was inspired by [learnPopGen's Shiny app for visualizing Drift & Selection](https://phytools.shinyapps.io/drift-selection/).
    ''', className="lead", style={'font-size': '13px'}),
    dbc.Row([dbc.Col([
        html.H3("Linking trait-associations with fitness",
                style={'font-size': '18px'}),
        dcc.Markdown('''
                Now that we have the allele frequency trajectories linked to fitness weights, we need to determine how this is related to trait-association (denoted by red vs. blue). Different traits affected fitness, and thus selection, differently. So this relationship between trait-association and fitness is determined by the user in the controls. This is controlled by a simple cubic equation and is easiest to visualize in the right figure.
                ''', className="lead", style={'font-size': '13px'}),
        html.H3("Web application creation & code",
                style={'font-size': '18px'}),
        dcc.Markdown('''
        If you are looking for the methods to create the website, check out [the code here at my GitHub](https://github.com/emcarthur/neanderthal-heritability-app). I want to acknowledge the powerful, well-documented, and open-source tools used to create this app:

        - [Plotly Dash](https://plotly.com/dash/) (framework for building reactive web applications and interactive figures)
        - [Dash Bootstrap](https://dash-bootstrap-components.opensource.faculty.ai/) (library of Bootstrap components for Dash)
        - [Inkscape](https://inkscape.org/) (open source vector graphics editor)

        ''', className="lead", style={'font-size': '13px'})
    ], width=6),
        dbc.Col(html.Div(html.Img(src=app.get_asset_url("legend2.jpg"),
                                  alt="legend", style={'width': '80%'})), width=6),
    ]),
])

# Navigation bar at the top
navbar = dbc.Navbar(
    [
        dbc.Row(
            [
                dbc.Col(dbc.Button("??? Toggle sidebar controls",
                                   color="secondary", className="mr-1", id="btn_sidebar")),
            ],
            align="center",
            no_gutters=True,
        ),
        dropdown
    ],
    color="dark",
    dark=True,
)

# Side bar with controls
sidebar = html.Div([
        controls,
    ],
    id="sidebar",
    style=SIDEBAR_STYLE,
)


# Page content, will update depending on if sidebar controls are toggled
content = html.Div(
    id="page-content",
    style=CONTENT_STYLE
)

# Final layout of all components
app.layout = html.Div(
    [
        dcc.Store(id='side_click'),
        dcc.Location(id="url"),
        navbar,
        sidebar,
        content,
    ],
)
#### CALLBACKS AND INTERACTIVE CONTENT ####

# Return 404 if page not available
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname in ["/", ""]:
        return main_page
    elif pathname == "/methods":
        return methods
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


# Determines the content of the page depending on if the side bar is out
@app.callback(
    [Output("sidebar", "style"),
     Output("page-content", "style"),
     Output("side_click", "data"), ],
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

# For a loading bar when any button is clicked and when the site first loads (very hacky but a way of getting around the client side callbacks)
@app.callback(
    Output("loading", "children"),
    [Input("submit", "n_clicks"),
     Input("most_example", "n_clicks"),
     Input("wbc_example", "n_clicks"),
     Input("scz_example", "n_clicks"),
     Input("sunburn_example", "n_clicks")],)
def loading_bar(btn1, btn2, btn3, btn4, btn5):
    if not dash.callback_context.triggered:
        time.sleep(2)  # at first website load
    else:
        time.sleep(1)
    return ""

# Main callback which updates when you click "submit" or any of the example buttons and outputs the FOUR graphs (and updates the parameter control panel)
@app.callback(
    [Output("intialDistSkew", "value"),
     Output("riskDecreasingPressure", "value"),
     Output("riskIncreasingPressure", "value"),
     Output("count", "value"),
     Output('afFig_dataStorage', 'data'),
     Output('distFig1_dataStorage', 'data'),
     Output('distFig2_dataStorage', 'data'),
     Output('arrowFig_dataStorage', 'data'), ],
    [Input("submit", "n_clicks"),
     Input("most_example", "n_clicks"),
     Input("wbc_example", "n_clicks"),
     Input("scz_example", "n_clicks"),
     Input("sunburn_example", "n_clicks")],
    [State("intialDistSkew", "value"),
     State("riskDecreasingPressure", "value"),
     State("riskIncreasingPressure", "value"),
     State("count", "value"),
     ])
def update_graph(btn1, btn2, btn3, btn4, btn5, intialDistSkew, riskDecreasingPressure, riskIncreasingPressure, count):
    # what button was last pressed
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'submit' in changed_id:  # if submit was last pressed, will use the input state for the 4 parameters
        if count is None:  # in case the user sets an invalid "count" value
            count = 20
    elif 'most_example' in changed_id:  # if example 1 was last pressed, set the 4 parameters
        intialDistSkew = 0
        riskDecreasingPressure = -0.007
        riskIncreasingPressure = -0.007
        count = 30
    elif 'wbc_example' in changed_id:  # if example 2 was last pressed, set the 4 parameters
        intialDistSkew = 0
        riskDecreasingPressure = 0.02
        riskIncreasingPressure = 0.02
        count = 30
    elif 'scz_example' in changed_id:  # if example 2 was last pressed, set the 4 parameters
        intialDistSkew = 1.2
        riskDecreasingPressure = 0.03
        riskIncreasingPressure = -0.03
        count = 30
    elif 'sunburn_example' in changed_id:  # if example 2 was last pressed, set the 4 parameters
        intialDistSkew = 1.5
        riskDecreasingPressure = -0.025
        riskIncreasingPressure = 0.025
        count = 30

    count = max(5, min(50, count))
    # Pick the variants from the fitness distribution generated from the 4 input parameters
    fitness_weights, fitness_norm, fitness_01 = selectFitnessWeights(
        intialDistSkew, riskDecreasingPressure, riskIncreasingPressure, count - 2)
    sim_idxs, fitness_remaining, fitness_remaining01 = selectIndexes(
        fitness_weights, fitness_norm, fitness_01, af_df.copy(deep=True))

    # Randoly pick 2 allele frequency trajectories that do not get lost (to avoid a no data error for the distribution generated in the second distribution figure)
    sim_idxs += random.sample(range(150, 158), 2)
    fitness_norm = np.append(fitness_norm, [-0.08, 0.08])
    fitness_remaining = np.append(fitness_remaining, [-0.08, 0.08])
    fitness_01 = np.append(fitness_01, [0.51, 0.49])
    fitness_remaining01 = np.append(fitness_remaining01, [0.51, 0.49])

    # Calculate the distribution for the left and right distribution figures
    xx = np.linspace(-4, 4, 100)
    kernel = gaussian_kde(fitness_norm)
    distribution1_yVals = kernel(xx)
    kernel = gaussian_kde(fitness_remaining)
    distribution2_yVals = kernel(xx)

    # Determine the direction and trajectory of the arrow for the arrow figure (bottom right)
    x_arrow_val = 5 * \
        (sum(distribution2_yVals[xx < -1.5]) +
         sum(distribution2_yVals[xx > 1.5])) - 10
    if x_arrow_val > 0:
        x_arrow_val = min(max(2, x_arrow_val), 10)
        x_arrow_start = -0.2
    else:
        x_arrow_val = max(min(-2, x_arrow_val), -10)
        x_arrow_start = 0.2

    if (sum(distribution2_yVals[xx < -1.5]) == 0) or (sum(distribution2_yVals[xx > 1.5]) == 0):
        y_arrow_val = -10
    elif(sum(distribution2_yVals[xx < -1.5]) < 0.01) and (sum(distribution2_yVals[xx > 1.5]) < 0.01):
        y_arrow_val = -10
    else:
        y_arrow_val = max(sum(distribution2_yVals[xx < -1.5]) / sum(distribution2_yVals[xx > 1.5]), sum(
            distribution2_yVals[xx > 1.5]) / sum(distribution2_yVals[xx < -1.5]))
        y_arrow_val = 13 * np.log10(y_arrow_val / 6)

    if y_arrow_val > 0:
        y_arrow_val = min(max(2, y_arrow_val), 10)
        y_arrow_start = -0.2
    else:
        y_arrow_val = max(min(-2, y_arrow_val), -10)
        y_arrow_start = 0.2

    # Update the data that should be visualized for each of the figures. These will go into storage and then get used by the client side callbacks
    af_fig_callbackData = (sim_idxs, [to_cmap_rgb(x) for x in fitness_01])
    dist_fig1_callbackData = (distribution1_yVals, fitness_norm, [
                              to_cmap_rgb(x) for x in fitness_01])
    dist_fig2_callbackData = (distribution2_yVals, fitness_remaining, [
                              to_cmap_rgb(x) for x in fitness_remaining01])
    arrow_fig_callbackData = (
        x_arrow_val, x_arrow_start, y_arrow_val, y_arrow_start)

    # update the figures and the parameter panels
    return intialDistSkew, riskDecreasingPressure, riskIncreasingPressure, count, af_fig_callbackData, dist_fig1_callbackData, dist_fig2_callbackData, arrow_fig_callbackData

#### CLIENT-SIDE CALLBACKS ####
# Update the allele frequency (center) figure using the updated stored data
app.clientside_callback(
    ClientsideFunction(
        namespace='clientsideAF',
        function_name='update_affigure'
    ),
    output=Output('af_graph2', 'figure'),
    inputs=[Input('afFig_dataStorage', 'data')],
    state=[State('afFigStorage', 'data')],
)

# Update the first distribution (left) figure using the updated stored data
app.clientside_callback(
    ClientsideFunction(
        namespace='clientsideDist',
        function_name='update_distfigure'
    ),
    output=Output('dist1_graph2', 'figure'),
    inputs=[Input('distFig1_dataStorage', 'data')],
    state=[State('distFigStorage1', 'data')],
)

# Update the second distribution (right) figure using the updated stored data
app.clientside_callback(
    ClientsideFunction(
        namespace='clientsideDist',
        function_name='update_distfigure'
    ),
    output=Output('dist2_graph2', 'figure'),
    inputs=[Input('distFig2_dataStorage', 'data')],
    state=[State('distFigStorage2', 'data')],
)

# Update the arrow (right bottom) figure using the updated stored data
app.clientside_callback(
    ClientsideFunction(
        namespace='clientsideArrow',
        function_name='update_Arrowfigure'
    ),
    output=Output('arrow_graph', 'figure'),
    inputs=[Input('arrowFig_dataStorage', 'data')],
    state=[State('arrowFigStorage', 'data')],
)

if __name__ == "__main__":
    app.run_server(debug=False)
