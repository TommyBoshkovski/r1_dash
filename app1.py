import os
import pathlib

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import dash_daq as daq
import scipy.io as sio
import numpy as np
import plotly.express as px
import plotly
import pandas as pd

import pandas as pd

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=0.9"}],
    external_stylesheets=['https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css','https://codepen.io/chriddyp/pen/bWLwgP.css']
)
server = app.server
app.config["suppress_callback_exceptions"] = True

APP_PATH = str(pathlib.Path(__file__).parent.resolve())

strength5 = pd.read_csv('data/strength_5.csv')
strength2 = pd.read_csv('data/strength_2.csv')

nodes_names = ['lateralorbitofrontal_1_R','lateralorbitofrontal_2_R', 'lateralorbitofrontal_3_R', 'lateralorbitofrontal_4_R', 'parsorbitalis_1_R', 'frontalpole_1_R', 'medialorbitofrontal_1_R', 'medialorbitofrontal_2_R', 'medialorbitofrontal_3_R', 'parstriangularis_1_R', 'parstriangularis_2_R', 'parsopercularis_1_R', 'parsopercularis_2_R', 'rostralmiddlefrontal_1_R', 'rostralmiddlefrontal_2_R', 'rostralmiddlefrontal_3_R', 'rostralmiddlefrontal_4 _R', 'rostralmiddlefrontal_5 _R', 'rostralmiddlefrontal_6_R', 'superiorfrontal_1_R', 'superiorfrontal_2_R', 'superiorfrontal_3_R', 'superiorfrontal_4_R', 'superiorfrontal_5_R', 'superiorfrontal_6_R', 'superiorfrontal_7_R', 'superiorfrontal_8_R', 'caudalmiddlefrontal_1_R', 'caudalmiddlefrontal_2_R', 'caudalmiddlefrontal_3_R', 'precentral_1_R', 'precentral_2_R', 'precentral_3_R', 'precentral_4_R', 'precentral_5_R', 'precentral_6_R', 'paracentral_1_R', 'paracentral_2_R', 'paracentral_3_R', 'rostralanteriorcingulate_1_R', 'caudalanteriorcingulate_1_R', 'posteriorcingulate_1_R', 'posteriorcingulate_2_R', 'isthmuscingulate_1_R', 'postcentral_1_R', 'postcentral_2_R', 'postcentral_3_R', 'postcentral_4_R', 'postcentral_5_R', 'supramarginal_1_R', 'supramarginal_2_R', 'supramarginal_3_R', 'supramarginal_4_R', 'superiorparietal_1_R', 'superiorparietal_2_R', 'superiorparietal_3_R', 'superiorparietal_4_R', 'superiorparietal_5_R', 'superiorparietal_6_R', 'superiorparietal_7_R', 'inferiorparietal_1_R', 'inferiorparietal_2_R', 'inferiorparietal_3_R', 'inferiorparietal_4_R', 'inferiorparietal_5_R', 'inferiorparietal_6_R', 'precuneus_1_R', 'precuneus_2_R', 'precuneus_3_R', 'precuneus_4_R', 'precuneus_5_R', 'cuneus_1_R', 'cuneus_2_R', 'pericalcarine_1_R', 'pericalcarine_2_R', 'lateraloccipital_1_R', 'lateraloccipital_2_R', 'lateraloccipital_3_R', 'lateraloccipital_4_R', 'lateraloccipital_5_R', 'lingual_1_R', 'lingual_2_R', 'lingual_3_R', 'fusiform_1_R', 'fusiform_2_R', 'fusiform_3_R', 'fusiform_4_R', 'parahippocampal_1_R', 'entorhinal_1_R', 'temporalpole_1_R', 'inferiortemporal_1_R', 'inferiortemporal_2_R', 'inferiortemporal_3_R', 'inferiortemporal_4_R', 'middletemporal_1_R', 'middletemporal_2_R', 'middletemporal_3_R', 'middletemporal_4_R', 'bankssts_1_R', 'superiortemporal_1_R', 'superiortemporal_2_R', 'superiortemporal_3_R', 'superiortemporal_4_R', 'superiortemporal_5_R', 'transversetemporal_1_R', 'insula_1_R', 'insula_2_R', 'insula_3_R', 'lateralorbitofrontal_1_L', 'lateralorbitofrontal_2_L', 'lateralorbitofrontal_3_L', 'lateralorbitofrontal_4_L', 'parsorbitalis_1_L', 'frontalpole_1_L', 'medialorbitofrontal_1_L', 'medialorbitofrontal_2_L', 'parstriangularis_1_L', 'parsopercularis_1_L', 'parsopercularis_2_L', 'rostralmiddlefrontal_1_L', 'rostralmiddlefrontal_2_L', 'rostralmiddlefrontal_3_L', 'rostralmiddlefrontal_4 _L', 'rostralmiddlefrontal_5 _L', 'rostralmiddlefrontal_6_L', 'superiorfrontal_1_L', 'superiorfrontal_2_L', 'superiorfrontal_3_L', 'superiorfrontal_4_L', 'superiorfrontal_5_L', 'superiorfrontal_6_L', 'superiorfrontal_7_L', 'superiorfrontal_8_L', 'superiorfrontal_9_L', 'caudalmiddlefrontal_1_L', 'caudalmiddlefrontal_2_L', 'caudalmiddlefrontal_3_L', 'precentral_1_L', 'precentral_2_L', 'precentral_3_L', 'precentral_4_L', 'precentral_5_L', 'precentral_6_L', 'precentral_7_L', 'precentral_8_L', 'paracentral_1_L', 'paracentral_2_L', 'rostralanteriorcingulate_1_L', 'caudalanteriorcingulate_1_L', 'posteriorcingulate_1_L', 'posteriorcingulate_2_L', 'isthmuscingulate_1_L', 'postcentral_1_L', 'postcentral_2_L', 'postcentral_3_L', 'postcentral_4_L', 'postcentral_5_L', 'postcentral_6_L', 'postcentral_7_L', 'supramarginal_1_L', 'supramarginal_2_L', 'supramarginal_3_L', 'supramarginal_4_L', 'supramarginal_5_L', 'superiorparietal_1_L', 'superiorparietal_2_L', 'superiorparietal_3_L', 'superiorparietal_4_L', 'superiorparietal_5_L', 'superiorparietal_6_L', 'superiorparietal_7_L', 'inferiorparietal_1_L', 'inferiorparietal_2_L', 'inferiorparietal_3_L', 'inferiorparietal_4_L', 'inferiorparietal_5_L', 'precuneus_1_L', 'precuneus_2_L', 'precuneus_3_L', 'precuneus_4_L', 'precuneus_5_L', 'cuneus_1_L', 'pericalcarine_1_L', 'lateraloccipital_1_L', 'lateraloccipital_2_L', 'lateraloccipital_3_L', 'lateraloccipital_4_L', 'lateraloccipital_5_L', 'lingual_1_L', 'lingual_2_L', 'lingual_3_L', 'lingual_4_L', 'fusiform_1_L', 'fusiform_2_L', 'fusiform_3_L', 'fusiform_4_L', 'parahippocampal_1_L', 'entorhinal_1_L', 'temporalpole_1_L', 'inferiortemporal_1_L', 'inferiortemporal_2_L', 'inferiortemporal_3_L', 'inferiortemporal_4_L', 'middletemporal_1_L', 'middletemporal_2_L', 'middletemporal_3_L', 'middletemporal_4_L', 'bankssts_1_L', 'bankssts_2_L', 'superiortemporal_1_L', 'superiortemporal_2_L', 'superiortemporal_3_L', 'superiortemporal_4_L', 'superiortemporal_5_L', 'transversetemporal_1_L', 'insula_1_L', 'insula_2_L', 'insula_3_L', 'insula_4_L']

yeo_net_names=['vis','sm','da','va','lim','fp','dmn']
ve_net_names=['pm', 'asc1', 'asc2', 'pss','ps', 'lim', 'ins']

gyeo5 = sio.loadmat('data/yeo_gradient5.mat')
geconomo5 = sio.loadmat('data/economo_gradient5.mat')
gyeo2 = sio.loadmat('data/yeo_gradient2.mat')
geconomo2 = sio.loadmat('data/economo_gradient2.mat')
yeo =np.array([5, 5, 5, 5, 7, 5, 5, 5, 5, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 2, 2, 2, 2, 2, 2, 2, 2, 2, 7, 4, 4, 4, 7, 2, 2, 2, 2, 2, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 7, 7, 7, 7, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 5, 5, 5, 5, 7, 5, 5, 5, 7, 4, 4, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 7, 4, 4, 4, 7, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7, 5, 5, 5, 5, 5, 5, 7, 7, 7, 7, 7, 7, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4])
ve = np.array([4, 4, 4, 4, 4, 3, 4, 4, 4, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 4, 4, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 6, 6, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 5, 7, 7, 7, 4, 4, 4, 4, 4, 3, 4, 4, 2, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 6, 6, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 5, 7, 7, 7, 7])
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
yeo_colors = dict()
yeo_colors['sm'] = '#1f77b4'
yeo_colors['va'] = '#ff7f0e'
yeo_colors['vis'] = '#2ca02c'
yeo_colors['fp'] = '#d62728'
yeo_colors['dmn'] = '#9467bd'
yeo_colors['lim'] = '#8c564b'
yeo_colors['da'] = '#e377c2'

ve_colors = dict()
ve_colors['pm'] = '#1f77b4'
ve_colors['asc1'] = '#ff7f0e'
ve_colors['asc2'] = '#2ca02c'
ve_colors['pss'] = '#d62728'
ve_colors['ps'] = '#9467bd'
ve_colors['lim'] = '#8c564b'
ve_colors['ins'] = '#e377c2'


def build_banner():
    return html.Div(
        id="banner",
        className="banner",
        children=[
            html.Div(
                id="banner-text",
                children=[
                    html.H5("The R1-weighted connectome"),
                    html.H6("Complementing brain networks with a myelin-sensitive measure", style={"paddingBottom": "10px"}),
                    html.H6(["This dashboard is associated with the following preprint available on biorXiv: ",html.A("Boshkovski, T., et al. 2020",href="https://www.biorxiv.org/content/10.1101/2020.08.06.237941v1")], style={"fontSize": "10pt"}),
                ],
            ),
        ],
    )
def build_tabs():
    return html.Div(
        id="tabs",
        className="tabs",
        children=[
            dcc.Tabs(
                id="app-tabs",
                value="tab2",
                className="custom-tabs",
                children=[
                    dcc.Tab(
                        id="Specs-tab",
                        label="Threshold at least 2 streamlines",
                        value="tab1",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                    dcc.Tab(
                        id="Control-chart-tab",
                        label="Threshold at least 5 streamlines",
                        value="tab2",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                    ),
                ],
            )
        ],
    )
def build_graphs(tab):
    if (tab == 'tab1'):
        return html.Div(
            id="strength",
            #className="row",
            children=[
                html.Div(),
                html.Div(
                    className="five columns",
                    children=[dcc.Graph(figure=gradient('yeo',2), id='g1', clear_on_unhover =True)]
                    ),
                html.Div(
                    className="five columns",
                    children=[dcc.Graph(figure=gradient('economo',2), id='g2', clear_on_unhover =True)]
                    ),
                html.Div(
                    className="eleven columns",
                    children=[dcc.Graph(figure = pl_strength(strength2,'all','yeo'), id='gstr')]
                    )
                ]
                )
    else:
        return html.Div(
            id="strength",
            #className="row",
            children=[
                html.Div(),
                html.Div(
                    className="five columns",
                    children=[dcc.Graph(figure=gradient('yeo',5), id='g1', clear_on_unhover =True)]
                    ),
                html.Div(
                    className="five columns",
                    children=[dcc.Graph(figure=gradient('economo',5), id='g2', clear_on_unhover =True)]
                    ),
                html.Div(
                    className="eleven columns",
                    children=[dcc.Graph(figure = pl_strength(strength5,'all','yeo'), id='gstr')]
                    )
                ]
                )

def pl_strength(df,yeo,grad):
    fig = make_subplots(rows=2, cols=1)
    dff = df.sort_values(by='NOS',ascending=False)
    df1 = df.sort_values(by='R1',ascending=False)
    if(grad == 'yeo'):
        if(yeo=='all'):
            fig.add_trace(go.Bar(x = dff.labels, y = dff.NOS, marker_color="#7470b3", name="NOS"), row=1, col=1)
            fig.add_trace(go.Bar(x = df1.labels, y = df1.R1, marker_color="#7470b3", name="R1"), row=2, col=1)
        elif(yeo=='sm'):
            fig.add_trace(go.Bar(x = dff.labels, y = dff.NOS, marker_color=dff.color1, name="NOS"), row=1, col=1)
            fig.add_trace(go.Bar(x = df1.labels, y = df1.R1, marker_color=df1.color1, name="R1"), row=2, col=1)
        elif(yeo=='va'):
            fig.add_trace(go.Bar(x = dff.labels, y = dff.NOS, marker_color=dff.color2, name="NOS"), row=1, col=1)
            fig.add_trace(go.Bar(x = df1.labels, y = df1.R1, marker_color=df1.color2, name="R1"), row=2, col=1)
        elif(yeo=='vis'):
            fig.add_trace(go.Bar(x = dff.labels, y = dff.NOS, marker_color=dff.color3, name="NOS"), row=1, col=1)
            fig.add_trace(go.Bar(x = df1.labels, y = df1.R1, marker_color=df1.color3, name="R1"), row=2, col=1)
        elif(yeo=='fp'):
            fig.add_trace(go.Bar(x = dff.labels, y = dff.NOS, marker_color=dff.color4, name="NOS"), row=1, col=1)
            fig.add_trace(go.Bar(x = df1.labels, y = df1.R1, marker_color=df1.color4, name="R1"), row=2, col=1)
        elif(yeo=='dmn'):
            fig.add_trace(go.Bar(x = dff.labels, y = dff.NOS, marker_color=dff.color5, name="NOS"), row=1, col=1)
            fig.add_trace(go.Bar(x = df1.labels, y = df1.R1, marker_color=df1.color5, name="R1"), row=2, col=1)
        elif(yeo=='lim'):
            fig.add_trace(go.Bar(x = dff.labels, y = dff.NOS, marker_color=dff.color6, name="NOS"), row=1, col=1)
            fig.add_trace(go.Bar(x = df1.labels, y = df1.R1, marker_color=df1.color6, name="R1"), row=2, col=1)
        else:
            fig.add_trace(go.Bar(x = dff.labels, y = dff.NOS, marker_color=dff.color7), row=1, col=1)
            fig.add_trace(go.Bar(x = df1.labels, y = df1.R1, marker_color=df1.color7, name="R1"), row=2, col=1)
    else:
        if(yeo=='all'):
            fig.add_trace(go.Bar(x = dff.labels, y = dff.NOS, marker_color="#7470b3", name="NOS"), row=1, col=1)
            fig.add_trace(go.Bar(x = df1.labels, y = df1.R1, marker_color="#7470b3", name="R1"), row=2, col=1)
        elif(yeo=='pm'):
            fig.add_trace(go.Bar(x = dff.labels, y = dff.NOS, marker_color=dff.color1_ve, name="NOS"), row=1, col=1)
            fig.add_trace(go.Bar(x = df1.labels, y = df1.R1, marker_color=df1.color1_ve, name="R1"), row=2, col=1)
        elif(yeo=='ins'):
            fig.add_trace(go.Bar(x = dff.labels, y = dff.NOS, marker_color=dff.color2_ve, name="NOS"), row=1, col=1)
            fig.add_trace(go.Bar(x = df1.labels, y = df1.R1, marker_color=df1.color2_ve, name="R1"), row=2, col=1)
        elif(yeo=='lim'):
            fig.add_trace(go.Bar(x = dff.labels, y = dff.NOS, marker_color=dff.color3_ve, name="NOS"), row=1, col=1)
            fig.add_trace(go.Bar(x = df1.labels, y = df1.R1, marker_color=df1.color3_ve, name="R1"), row=2, col=1)
        elif(yeo=='ps'):
            fig.add_trace(go.Bar(x = dff.labels, y = dff.NOS, marker_color=dff.color4_ve, name="NOS"), row=1, col=1)
            fig.add_trace(go.Bar(x = df1.labels, y = df1.R1, marker_color=df1.color4_ve, name="R1"), row=2, col=1)
        elif(yeo=='pss'):
            fig.add_trace(go.Bar(x = dff.labels, y = dff.NOS, marker_color=dff.color5_ve, name="NOS"), row=1, col=1)
            fig.add_trace(go.Bar(x = df1.labels, y = df1.R1, marker_color=df1.color5_ve, name="R1"), row=2, col=1)
        elif(yeo=='asc1'):
            fig.add_trace(go.Bar(x = dff.labels, y = dff.NOS, marker_color=dff.color6_ve, name="NOS"), row=1, col=1)
            fig.add_trace(go.Bar(x = df1.labels, y = df1.R1, marker_color=df1.color6_ve, name="R1"), row=2, col=1)
        else:
            fig.add_trace(go.Bar(x = dff.labels, y = dff.NOS, marker_color=dff.color7_ve, name="NOS"), row=1, col=1)
            fig.add_trace(go.Bar(x = df1.labels, y = df1.R1, marker_color=df1.color7_ve, name="R1"), row=2, col=1)
    fig.update_yaxes(title_text="NOS strength", tickfont=dict(size=12, color='white'), title_font=dict(size=14, color='white'), row=1, col=1)
    fig.update_yaxes(title_text="R1 [s<sup>-1</sup>]", tickfont=dict(size=12, color='white'), title_font=dict(size=14, color='white'), row=2, col=1)
    fig.update_xaxes(showticklabels=False, tickfont=dict(size=10, color='white'), title_font=dict(size=14))
    fig.update_xaxes(showticklabels=False, title_text="Nodes", title_font=dict(size=14, color='white'), tickfont=dict(size=10, color='white'), row=2, col=1)
    fig.update_layout(height=400, paper_bgcolor = '#1e2131', plot_bgcolor = '#1e2131', showlegend=False)
    return fig


 
def gradient(gtype,gtresh):
    
    fig = go.Figure()
    labels1_yeo = []
    labels3_yeo = []
    labels1_economo = []
    labels3_economo = []

    if(gtresh==2):
        gyeo=gyeo2
        geconomo = geconomo2
    else:
        gyeo=gyeo5
        geconomo = geconomo5
    for i in range(0,7):
        labels1_yeo.extend(gyeo['labels1_yeo'][0][i])
        labels3_yeo.extend(gyeo['labels3_yeo'][0][i])
        labels1_economo.extend(geconomo['labels1_economo'][0][i])
        labels3_economo.extend(geconomo['labels3_economo'][0][i])
    clr1yeo = [yeo_colors[str(x)] for x in labels1_yeo]
    clr1ve = [ve_colors[str(x)] for x in labels1_economo]
    R1_NOS_rank_yeo = np.squeeze(gyeo['R1_NOS_rank_yeo'])
    R1NOS_Indx_yeo = np.squeeze(gyeo['R1NOS_Indx_yeo'])

    R1_NOS_rank_ve = np.squeeze(geconomo['R1_NOS_rank_economo'])
    R1NOS_Indx_ve = np.squeeze(geconomo['R1NOS_Indx_economo'])

    if(gtype=='yeo'):
        fig.add_trace(go.Bar(x=labels1_yeo, y=R1_NOS_rank_yeo[R1NOS_Indx_yeo-1],marker_color=clr1yeo))
        fig.update_yaxes(tickfont=dict(size=15, color='white'), title_font=dict(size=18, color='white'))
        fig.update_xaxes(title='Yeo\'s 7 networks functional classes', tickfont=dict(size=15, color='white'), title_font=dict(size=18, color='white'))

    else:
        fig.add_trace(go.Bar(x=labels1_economo, y=R1_NOS_rank_ve[R1NOS_Indx_ve-1], marker_color=clr1ve))
        fig.update_yaxes(tickfont=dict(size=15, color='white'), title_font=dict(size=18, color='white'))
        fig.update_xaxes(title='von Economo cytoarchitectonic classes', tickfont=dict(size=15, color='white'), title_font=dict(size=18, color='white'))
    fig.update_layout(height=300,paper_bgcolor = '#1e2131', plot_bgcolor = '#1e2131', showlegend=False)
    return fig



app.layout = html.Div(
    id="big-app-container",
    style={
        "zoom": 0.97,
    },
    children=[
        build_banner(),
        html.Div(
            id="app-container",
            style={'backgroundColor': "#1e2132"},
            children=[
                build_tabs(),
                # Main app
                html.Div(id="app-content", children=[]),
            ],
        ),
    ],
)


@app.callback(
    [Output("app-content", "children")],
    [Input("app-tabs", "value")],
)
def render_tab_content(tab_switch):
    return [build_graphs(tab_switch)]


@app.callback(
    dash.dependencies.Output('gstr', 'figure'),
    [dash.dependencies.Input('g1', 'hoverData'),
    dash.dependencies.Input('g2', 'hoverData'),
    dash.dependencies.Input("app-tabs", "value")])
def update_strength_yeo(hoverData, hoverData2, tab):
    if(hoverData==None and hoverData2==None):
        if(tab == 'tab1'):
            return pl_strength(strength2,'all','yeo')
        else:
            return pl_strength(strength5,'all','yeo')

    elif(hoverData != None and hoverData2==None):
        yeo = hoverData['points'][0]['label']
        if(tab == 'tab1'):
            return pl_strength(strength2,yeo,'yeo')
        else:
            return pl_strength(strength5,yeo,'yeo')

    elif(hoverData == None and hoverData2 != None):
        ve = hoverData2['points'][0]['label']
        if(tab == 'tab1'):
            return pl_strength(strength2,ve,'ve')
        else:
            return pl_strength(strength5,ve,'ve')


# Running the server
if __name__ == "__main__":
    app.run_server(debug=False)
