import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.stats.kde import gaussian_kde
from plotly.subplots import make_subplots
from matplotlib.colors import LinearSegmentedColormap

af_df = pd.read_csv("simulations.csv")
af_df = pd.concat([af_df,pd.read_csv("simulations_noPressure.csv")],ignore_index=True)
af_fig = go.Figure()
fitness_weights = af_df['fitness_weight'].values
cmap = LinearSegmentedColormap.from_list('BgR',['#0024af','#e2dee6','#b70000']) # generate color map

def linestyle(array):
    if array[-1] < 0.05:
        return 'dash'
    else:
        return 'solid'

def to_cmap_rgb(value):
    c = cmap(value)
    color = f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})'
    return color

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

def make_afFig():

    for i in range(len(af_df)): # loop through all the variants to plot them
        af = af_df.loc[i,[str(x) for x in list(range(2000))]].values # select that simulation
        af_fig.add_trace( # plot that simulation
            go.Scatter(x=list(range(2000)), y=af, opacity=0.5, mode='lines',line=dict(color='black',dash=linestyle(af)), showlegend=False )
            )

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

    return af_fig

def make_distFig():
    xx = np.linspace(-4,4,100)
    kde_y = np.zeros(100)

    dist_fig = make_subplots(rows=2,cols=1,row_heights=[0.8,0.2],shared_xaxes=True,vertical_spacing=0.04)
    dist_fig.add_trace(go.Scatter(x=xx, y=kde_y, mode='lines',line=dict(color=to_cmap_rgb(0.5),dash='solid'), showlegend=False,hoverinfo='skip',fill='tozeroy'),row=1,col=1)
    dist_fig.add_trace(go.Scatter(x=xx[xx <-1.5], y=kde_y[xx < -1.5], mode='lines',line=dict(color=to_cmap_rgb(0.2),dash='solid'), showlegend=False,hoverinfo='skip',fill='tozeroy'),row=1,col=1)
    dist_fig.add_trace(go.Scatter(x=xx[xx >1.5], y=kde_y[xx > 1.5], mode='lines',line=dict(color=to_cmap_rgb(0.8),dash='solid'), showlegend=False,hoverinfo='skip',fill='tozeroy'),row=1,col=1)
    dist_fig.add_vline(x=0,line_width=1.2,line_dash="dot", line_color='black',row=1,col=1)


    dist_fig.update_layout( # update figure 2 layout
        template='simple_white',
        yaxis_title="Density",
        xaxis2_title="Trait-association direction",
        yaxis1={'tickvals':[],'fixedrange':True},
        yaxis2 = {'tickvals':[0],'ticktext':['Variants'],'fixedrange':True},
        font=dict(
            size=9),
        margin=dict(l=10,r=10,t=10,b=10),
        height=140,
    )
    dist_fig.update_xaxes( # update figure 2 axis
        range=[-4,4],
        ticktext=["Protective", "No trait-association","Risk"],
        tickvals=[-3.5, 0, 3.5],
        fixedrange=True,
        ticklen=0
    )
    return dist_fig

def make_arrowFig():
    arrow_fig = go.Figure()
    arrow_fig.add_annotation(
      x=0,  # arrows' head
      y=0,  # arrows' head
      ax=0,  # arrows' tail
      ay=0,  # arrows' tail
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

    return arrow_fig
