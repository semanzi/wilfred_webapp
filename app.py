#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 11:13:28 2022

@author: sean
"""
import pandas as pd
import numpy as np
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_cytoscape as cyto
import dash_table
import networkx as nx
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
from sklearn.preprocessing import StandardScaler
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots

cyto.load_extra_layouts()

data = pd.read_csv('assets/data/study_inclusion_table_type_resolved.csv', low_memory=False)
sys_revs = pd.read_csv('assets/data/systematic_reviews_extended_data.csv', low_memory=False)

outcomes_list = list()
for i in range(len(sys_revs)):
    if sys_revs['Mortality'][i] == 'yes':
        a = 'mortality, '
    else:
        a = ''
    if sys_revs['Mechanical_Ventilation'][i] == 'yes':
        b = 'mechanical ventilation, '
    else:
        b = ''
    if sys_revs['ICU'][i] == 'yes':
        c = 'ICU, '
    else:
        c = ''
    if sys_revs['Hospital_los'][i] == 'yes':
        d = 'hospital length of stay, '
    else:
        d = ''
    if sys_revs['Quality_of_life'][i] == 'yes':
        e = 'quality of life, '
    else:
        e = ''
    if sys_revs['Adverse_Events'][i] == 'yes':
        f = 'adverse events, '
    else:
        f = ''
    if sys_revs['Blood_markers'][i] == 'yes':
        g = 'blood markers, '
    else:
        g = ''
    check = isinstance(sys_revs['Other'][i], str)
    if check == True:
        h = sys_revs['Other'][i]
    else:
        h = ''
    
    out_str = a + b + c + d + e + f + g + h
    outcomes_list.append(out_str)

sys_revs['Outcomes'] = outcomes_list    

################### Create network #######################

def create_nx_graph(nodeData,edgeData):
    ## Initiate the graph object
    G = nx.Graph()
    
    ## Tranform the data into the correct format for use with NetworkX
    # Node tuples (ID, dict of attributes)
    idList = nodeData['id'].tolist()
    labels =  pd.DataFrame(nodeData['Label'])
    labelDicts = labels.to_dict(orient='records')
    nodeTuples = [tuple(r) for r in zip(idList,labelDicts)]
    
    # Edge tuples (Source, Target, dict of attributes)
    sourceList = edgeData['source'].tolist()
    targetList = edgeData['target'].tolist()
    edgeTuples = [tuple(r) for r in zip(sourceList,targetList)]
    
    ## Add the nodes and edges to the graph
    G.add_nodes_from(nodeTuples)
    G.add_edges_from(edgeTuples)
    
    return G

def create_analysis(G):
    #Graph metrics
    g_met_dict = dict()
    g_met_dict['num_chars'] = G.number_of_nodes()
    g_met_dict['num_inter'] = G.number_of_edges()
    g_met_dict['density'] = nx.density(G)
    
    #Node metrics
    e_cent = nx.eigenvector_centrality(G)
    page_rank = nx.pagerank(G)
    degree = nx.degree(G)
    between = nx.betweenness_centrality(G)
    
    # Extract the analysis output and convert to a suitable scale and format
    e_cent_size = pd.DataFrame.from_dict(e_cent, orient='index',
                                         columns=['cent_value'])
    e_cent_size.reset_index(drop=True, inplace=True)
    #e_cent_size = e_cent_size*100
    page_rank_size = pd.DataFrame.from_dict(page_rank, orient='index',
                                            columns=['rank_value'])
    page_rank_size.reset_index(drop=True, inplace=True)
    #page_rank_size = page_rank_size*1000
    degree_list = list(degree)
    degree_dict = dict(degree_list)
    degree_size = pd.DataFrame.from_dict(degree_dict, orient='index',
                                         columns=['deg_value'])
    degree_size.reset_index(drop=True, inplace=True)
    g_met_dict['avg_deg'] = degree_size.iloc[:,0].mean()
    between_size = pd.DataFrame.from_dict(between, orient='index',
                                          columns=['betw_value'])
    between_size.reset_index(drop=True, inplace=True)
    
    dfs = [e_cent_size,page_rank_size,degree_size,between_size]
    analysis_df = pd.concat(dfs, axis=1)
    cols = list(analysis_df.columns)
    an_arr = analysis_df.to_numpy(copy=True)
    scaler = StandardScaler()
    an_scaled = scaler.fit_transform(an_arr)
    an_df = pd.DataFrame(an_scaled)
    an_st = an_df.copy(deep=True)
    an_st.columns = cols
    an_df.columns = cols
    an_mins = list(an_df.min())
    for i in range(len(an_mins)):
        an_df[cols[i]] -= an_mins[i] - 1
        an_df[cols[i]] *= 12
    
    return an_df, an_st

p_id_list = list(range(len(data)))
sr_id_list = list(range(len(data),(len(data)+len(sys_revs)),1))
p_id_df = pd.DataFrame(p_id_list)
p_id_df.columns = ['Id']
data = pd.concat([p_id_df,data], axis=1)
sr_id_df = pd.DataFrame(sr_id_list)
sr_id_df.columns = ['Id']
sys_revs = pd.concat([sr_id_df,sys_revs],axis=1)

rows = list()
seq = list(range(14,64,1))
rows.append(0)
for i in range(len(seq)):
    rows.append(seq[i])

adj_mat = data.iloc[:,rows].copy(deep=True)

edge_list = list()
for i in range(len(adj_mat)):
    for j in range(1,len(adj_mat.columns)):
        #print('row' + str(i) + ", " + 'col' + str(j))
        if adj_mat.iloc[i,j] == 1:
            edge_list.append({'source': adj_mat.iloc[i,0], 'target': sys_revs.iloc[(j-1),0]})
edge_df = pd.DataFrame(edge_list)
edge_dict = edge_df.to_dict(orient='index')

label = list()
for i in range(len(data)):
    label.append('')
label = pd.DataFrame(label, columns=['Label'])
data = pd.concat([data, label],axis=1)
p_nodes = data[['Id','Author', 'Label', 'Year', 'Type',]]
review_nodes = sys_revs[['Id','Author', 'Author','Date_Published']]
rev_type = ['sys rev'] * len(review_nodes)
rev_type = pd.DataFrame(rev_type)
review_nodes = pd.concat([review_nodes, rev_type], axis=1)
review_nodes.columns = ['Id','Author', 'Label', 'Year','Type']
nodes_df = pd.concat([p_nodes, review_nodes]).reset_index(drop=True)
nodes_df.columns = ['id', 'Authors', 'Label', 'Year/Date published', 'Type']
nodes_dict = nodes_df.to_dict(orient='index')

nx_graph = create_nx_graph(nodes_df,edge_df)
an_df, an_st = create_analysis(nx_graph)

type_list = list(nodes_df['Type'].unique())
col_list = ['#fc1c03','#fc7b03','#166e1f','#2ca4a8','#1713f0','#a313bf']
node_colour = list()
for i in range(0,len(nodes_df),1):
    if nodes_df['Type'][i] == type_list[0]:
        node_colour.append(col_list[0])
    elif nodes_df['Type'][i] == type_list[1]:
        node_colour.append(col_list[1])
    elif nodes_df['Type'][i] == type_list[2]:
        node_colour.append(col_list[2])
    elif nodes_df['Type'][i] == type_list[3]:
        node_colour.append(col_list[3])
    elif nodes_df['Type'][i] == type_list[4]:
        node_colour.append(col_list[4])
    elif nodes_df['Type'][i] == type_list[5]:
        node_colour.append(col_list[5])

pri_connection_dict_list = list()
for i in range(len(data)):
    temp_list = list()
    for j in range(14,65,1):
        if data.iloc[i,j] == 1:
            #temp_list.append(list(sys_revs.iloc[(j-14),[0,1,2,3,19]]))
            temp_list.append(list(sys_revs.iloc[(j-14),:]))
    connection_df = pd.DataFrame(temp_list)
    connection_dict = connection_df.to_dict(orient='index')
    pri_connection_dict_list.append(connection_dict)
        
sys_rev_connection_dict_list = list()
for i in range(14,64,1):
    temp_list = list()
    for j in range(len(data)):
        if data.iloc[j,i] == 1:
            temp_list.append(list(data.iloc[j,[0,1,2,3,4,8,9]]))
    connection_df = pd.DataFrame(temp_list)
    #connection_df.columns = ['Id','Author','Year','Title','Journal','Date published','Study type']
    connection_dict = connection_df.to_dict(orient='index')
    sys_rev_connection_dict_list.append(connection_dict)

node_colour_df = pd.DataFrame(node_colour)
node_colour_df.reset_index(inplace=True,drop=True)
node_colour_df.columns = ['Node_colour']
an_df.reset_index(inplace=True,drop=True)
nodes_df.reset_index(inplace=True,drop=True)
nodes_df = pd.concat([nodes_df, an_df, node_colour_df],axis=1).reset_index(drop=True)
nodes_dict = nodes_df.to_dict(orient='index')

connection_dict_list = pri_connection_dict_list + sys_rev_connection_dict_list

for i in range(len(nodes_dict)):
    nodes_dict[i]['connections'] = connection_dict_list[i]

p_study_info = list()
for i in range(len(data)):
    info_df = data.iloc[i,[0,4,8]]
    info_dict = info_df.to_dict()
    p_study_info.append(info_dict)
    
s_rev_info = list()
for i in range(len(sys_revs)):
    info_df = sys_revs.iloc[i,[0,3,4,5,6,7,8,9,19,20]]
    info_dict = info_df.to_dict()
    s_rev_info.append(info_dict)
    
study_info_dict_list = p_study_info + s_rev_info
for i in range(len(nodes_dict)):
    nodes_dict[i]['info'] = study_info_dict_list[i]

elements = list()
for i in range(len(edge_dict)):
    elements.append({'data': edge_dict[i]})

for i in range(len(nodes_dict)):
    elements.append({'data': nodes_dict[i]})

############### Create sankey ####################

def create_sankey(data):
    yr_type_df = data[['Year', 'Type']]
    count_df = yr_type_df.value_counts()
    uni_year = pd.DataFrame(data['Year'].drop_duplicates())
    uni_year.sort_values('Year',inplace=True)
    uni_year_list = uni_year['Year'].to_list()
    uni_type = pd.DataFrame(data['Type'].drop_duplicates())
    uni_type.sort_values('Type',inplace=True)
    uni_type_list = uni_type['Type'].to_list()
    uni_sys = list(data.columns)
    uni_sys_df = pd.DataFrame(uni_sys,columns=["sys_author"])
    uni_sys_df = uni_sys_df.iloc[13:63]
    uni_sys_df.sort_values("sys_author", inplace=True)
    uni_vals = uni_type_list + uni_year_list
    
    # color = ['#b26efa','#957aff','#7085ff','#398fff','#0097ff','#009eff','#00a4ff','#00a9ff',
    #              '#00adff','#00b1f7','#00b3ea','#00b6dd','#00b7ce','#00b9c0',
    #              '#00b9b2','#00baa4','#00ba97','#23ba8b','#23628F','#8F6A23','#6B238F','#4F8F23']
    
    color = ['#6d92ce','#175666','#abb590','#07000c','#8e4429','#2b2b20','#5476f2','#cef9b8','#0c0326',
             '#aed125','#d5e212','#aa5c8f','#146d9e','#e20641','#3a2b15','#595355','#78d39b','#d3d8a9','#3f73c1',
             '#293ce5','#b58682','#303435','#423637','#d9bdef','#303502','#f2eb2b','#89db0f','#6a6b68','#888989',
             '#330707','#8c98af','#5b00aa','#968b87','#6f968e','#9c6ed8','#e05cbf','#24cc97','#68b8ed','#062e30',
             '#382f37','#8b16a3','#535926','#005126','#543b75','#2d3e5b','#455b14','#a398e2','#875f69','#031e44',
             '#858f93','#23628F','#8F6A23','#6B238F','#4F8F23','#b26efa','#957aff','#7085ff','#398fff','#0097ff',
             '#009eff','#00a4ff','#00a9ff','#00adff','#00b1f7','#00b3ea','#00b6dd','#00b7ce','#00b9c0','#00b9b2',
             '#00baa4','#00ba97','#23ba8b'
             ]
    
    source_list = list()
    target_list = list()
    for i in range(len(yr_type_df)):
        for j in range(len(uni_vals)):
            if yr_type_df.iloc[i,0] == uni_vals[j]:
                target_list.append(j)
    for i in range(len(yr_type_df)):
        for j in range(len(uni_vals)):
            if yr_type_df.iloc[i,1] == uni_vals[j]:
                source_list.append(j)
    
    type_df = data.iloc[:,9]
    sys_df = data.iloc[:,14:64]
    type_sys_df = pd.concat([type_df,sys_df],axis=1)
    uni_sys_df = pd.DataFrame(uni_sys,columns=["sys_author"])
    uni_sys_df = uni_sys_df.iloc[14:64]
    uni_sys_list = uni_sys_df["sys_author"].to_list()
    uni_type_sys_vals = uni_sys_list + uni_type_list
    
    source_list_two = list()
    target_list_two = list()
    for i in range(len(type_sys_df)):
        for j in range(len(uni_type_sys_vals)):
            if type_sys_df.iloc[i,0] == uni_type_sys_vals[j]:
                for k in range(1,51):
                    if type_sys_df.iloc[i,k] >= 1:
                        target_list_two.append(j)
    for i in range(len(type_sys_df)):
        for j in range(1,51):
            if type_sys_df.iloc[i,j] >= 1:
                source_list_two.append(j-1)
    
    #source_list_two_adj = [x + 18 for x in source_list_two]           
    #target_list_two_adj = [x + 22 for x in target_list_two]
    #uni_vals = uni_vals + uni_sys_list
    
    source_list_adj = [x + 50 for x in source_list]
    target_list_adj = [x + 50 for x in target_list]
    uni_vals = uni_type_sys_vals + uni_year_list
    
    source_list = source_list_two + source_list_adj
    target_list = target_list_two + target_list_adj
    
    s_t_df = pd.DataFrame([source_list, target_list])
    s_t_df = s_t_df.transpose()
    s_t_uni = pd.DataFrame(s_t_df.value_counts(),columns=['value'])
    s_t_uni.reset_index(inplace=True)
    s_t_uni.columns = ['source', 'target', 'value']
    link_col = ['#23628F','#8F6A23','#6B238F','#4F8F23']
    link_col_list = list()
    for i in range(len(s_t_uni)):
        if s_t_uni.iloc[i,0] == 50 or s_t_uni.iloc[i,1] == 50:
            link_col_list.append(link_col[0])
        elif s_t_uni.iloc[i,0] == 51 or s_t_uni.iloc[i,1] == 51:
            link_col_list.append(link_col[1])
        elif s_t_uni.iloc[i,0] == 52 or s_t_uni.iloc[i,1] == 52:
            link_col_list.append(link_col[2])
        elif s_t_uni.iloc[i,0] == 53 or s_t_uni.iloc[i,1] == 53:
            link_col_list.append(link_col[3])
    
    source_ser = s_t_uni['source'].squeeze()
    target_ser = s_t_uni['target'].squeeze()
    value_ser = s_t_uni['value'].squeeze()
    
    node_dict = {'color': color,
                 'label': uni_vals,
                 'line': {'color': 'black', 'width': 0.5},
                 'pad': 50,
                 'thickness': 50,
                 #'hovertemplate': "%{label} <br>Value x: %{value} <br>Value y: %{categorycount} <extra></extra>"
                 }
    link_dict = {'color': link_col_list,
                 'line': {'color': 'black', 'width': 0.2},
                 'source': source_ser,
                 'target': target_ser,
                 'value': value_ser}
    layout_dict = {'font': {'size': 20, 'color': 'black'},
                   'hoverlabel': {'bgcolor': 'white', 'font': {'color': 'black'}}}
    
    return node_dict, link_dict, layout_dict    

sank_nodes, sank_links, sank_layout = create_sankey(data)
sank_fig = go.Figure(go.Sankey(link=sank_links,node=sank_nodes))
sank_fig.update_layout(sank_layout)
sank_fig.update_traces(hoverinfo="all", selector=dict(type='sankey'))

################ Create timeline ###################

def create_timeline(data, sys_revs):
    low_date_bound = pd.to_datetime('31/01/1999', format='%d/%m/%Y')
    mid_date_bound = pd.to_datetime('05/01/2019', format='%d/%m/%Y')
    
    case_rep = data[data['Type'] == 'case report study']
    case_rep_df = case_rep[['Author', 'Date published']]
    case_rep_df.reset_index(inplace=True, drop=True)
    rand_3 = pd.DataFrame(np.random.uniform(0.1, 0.9, len(case_rep_df)), columns=['y'])
    case_rep_df = pd.concat([case_rep_df, rand_3], axis=1)
    case_rep_df['Date published'] = pd.to_datetime(case_rep_df['Date published'], format='%d/%m/%Y')
    case_rep_df.sort_values('Date published', inplace=True)
    case_rep_early_df = case_rep_df[case_rep_df['Date published'] < low_date_bound]
    case_rep_early_df.reset_index(inplace=True, drop=True)
    case_rep_mid_df = case_rep_df[(case_rep_df['Date published'] >= low_date_bound) & (case_rep_df['Date published'] < mid_date_bound)]
    case_rep_mid_df.reset_index(inplace=True, drop=True)
    case_rep_later_df = case_rep_df[case_rep_df['Date published'] >= mid_date_bound]
    case_rep_later_df.reset_index(inplace=True, drop=True)
    
    case_ser = data[data['Type'] == 'case series reports']
    case_ser_df = case_ser[['Author', 'Date published']]
    case_ser_df.reset_index(inplace=True, drop=True)
    rand_4 = pd.DataFrame(np.random.uniform(1.1, 1.9, len(case_ser_df)), columns=['y'])
    case_ser_df = pd.concat([case_ser_df, rand_4], axis=1)
    case_ser_df['Date published'] = pd.to_datetime(case_ser_df['Date published'], format='%d/%m/%Y')
    case_ser_df.sort_values('Date published', inplace=True)
    case_ser_early_df = case_ser_df[case_ser_df['Date published'] < low_date_bound]
    case_ser_early_df.reset_index(inplace=True, drop=True)
    case_ser_mid_df = case_ser_df[(case_ser_df['Date published'] >= low_date_bound) & (case_ser_df['Date published'] < mid_date_bound)]
    case_ser_mid_df.reset_index(inplace=True, drop=True)
    case_ser_later_df = case_ser_df[case_ser_df['Date published'] >= mid_date_bound]
    case_ser_later_df.reset_index(inplace=True, drop=True)
    
    non_rct = data[data['Type'] == 'non rct']
    non_rct_df = non_rct[['Author', 'Date published']]
    non_rct_df.reset_index(inplace=True, drop=True)
    rand_1 = pd.DataFrame(np.random.uniform(2.1, 2.9, len(non_rct_df)), columns=['y'])
    non_rct_df = pd.concat([non_rct_df, rand_1], axis=1)
    non_rct_df['Date published'] = pd.to_datetime(non_rct_df['Date published'], format='%d/%m/%Y')
    non_rct_df.sort_values('Date published', inplace=True)
    non_rct_early_df = non_rct_df[non_rct_df['Date published'] < low_date_bound]
    non_rct_early_df.reset_index(inplace=True, drop=True)
    non_rct_mid_df = non_rct_df[(non_rct_df['Date published'] >= low_date_bound) & (non_rct_df['Date published'] < mid_date_bound)]
    non_rct_mid_df.reset_index(inplace=True, drop=True)
    non_rct_later_df = non_rct_df[non_rct_df['Date published'] >= mid_date_bound]
    non_rct_later_df.reset_index(inplace=True, drop=True)
    
    rct = data[data['Type'] == 'rct']
    rct_df = rct[['Author', 'Date published']]
    rct_df.reset_index(inplace=True, drop=True)
    rand_2 = pd.DataFrame(np.random.uniform(3.1, 3.9, len(rct_df)), columns=['y'])
    rct_df = pd.concat([rct_df, rand_2], axis=1)
    rct_df['Date published'] = pd.to_datetime(rct_df['Date published'], format='%d/%m/%Y')
    rct_df.sort_values('Date published', inplace=True)
    rct_early_df = rct_df[rct_df['Date published'] < low_date_bound]
    rct_early_df.reset_index(inplace=True, drop=True)
    rct_mid_df = rct_df[(rct_df['Date published'] >= low_date_bound) & (rct_df['Date published'] < mid_date_bound)]
    rct_mid_df.reset_index(inplace=True, drop=True)
    rct_later_df = rct_df[rct_df['Date published'] >= mid_date_bound]
    rct_later_df.reset_index(inplace=True, drop=True)
    
    rand_5 = pd.DataFrame(np.random.uniform(4.1, 4.9, len(sys_revs)), columns=['y'])
    sys_revs_df = pd.concat([sys_revs, rand_5], axis=1)
    sys_revs_df['Date_Published'] = pd.to_datetime(sys_revs_df['Date_Published'], format='%d/%m/%Y')
    sys_revs_df.sort_values('Date_Published', inplace=True)
    sys_revs_early_df = sys_revs_df[sys_revs_df['Date_Published'] < low_date_bound]
    sys_revs_early_df.reset_index(inplace=True, drop=True)
    sys_revs_mid_df = sys_revs_df[(sys_revs_df['Date_Published'] >= low_date_bound) & (sys_revs_df['Date_Published'] < mid_date_bound)]
    sys_revs_mid_df.reset_index(inplace=True, drop=True)
    sys_revs_later_df = sys_revs_df[sys_revs_df['Date_Published'] >= mid_date_bound]
    sys_revs_later_df.reset_index(inplace=True, drop=True)
    
    
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.0, column_widths=[0.1,0.1,0.8], x_title='Publication Date', y_title='Study type')
    fig.add_trace(go.Scatter(x=non_rct_early_df['Date published'], y=non_rct_early_df['y'],
                             mode='markers',
                             name='Non-RCT',
                             marker_color='#03b1fc',
                             marker_symbol='circle',
                             legendgroup='nonrct',
                             hovertemplate = 
                             '<b>Authors:</b> %{text}'+
                             '<br><b>Date of publication:</b> %{x}<extra></extra>',
                             text = list(non_rct_early_df['Author']),
                             showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=case_rep_early_df['Date published'], y=case_rep_early_df['y'],
                             mode='markers',
                             name='Case report/study',
                             marker_color='#fc8403',
                             marker_symbol='diamond',
                             legendgroup='caserep',
                             hovertemplate = 
                             '<b>Authors:</b> %{text}'+
                             '<br><b>Date of publication:</b> %{x}<extra></extra>',
                             text = list(case_rep_early_df['Author']),
                             showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=case_ser_early_df['Date published'], y=case_ser_early_df['y'],
                             mode='markers',
                             name='Case series/reports',
                             marker_color='#316906',
                             marker_symbol='pentagon',
                             legendgroup='caseser',
                             hovertemplate = 
                             '<b>Authors:</b> %{text}'+
                             '<br><b>Date of publication:</b> %{x}<extra></extra>',
                             text = list(case_ser_early_df['Author']),
                             showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=non_rct_mid_df['Date published'], y=non_rct_mid_df['y'],
                             mode='markers',
                             name='Non-RCT',
                             marker_color='#03b1fc',
                             marker_symbol='circle',
                             legendgroup='nonrct',
                             hovertemplate = 
                             '<b>Authors:</b> %{text}'+
                             '<br><b>Date of publication:</b> %{x}<extra></extra>',
                             text = list(non_rct_mid_df['Author']),
                             showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=rct_mid_df['Date published'], y=rct_mid_df['y'],
                             mode='markers',
                             name='RCT',
                             marker_color='#8803fc',
                             marker_symbol='square',
                             legendgroup='rct',
                             hovertemplate = 
                             '<b>Authors:</b> %{text}'+
                             '<br><b>Date of publication:</b> %{x}<extra></extra>',
                             text = list(rct_mid_df['Author']),
                             showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=case_rep_mid_df['Date published'], y=case_rep_mid_df['y'],
                             mode='markers',
                             name='Case report/study',
                             marker_color='#fc8403',
                             marker_symbol='diamond',
                             legendgroup='caserep',
                             hovertemplate = 
                             '<b>Authors:</b> %{text}'+
                             '<br><b>Date of publication:</b> %{x}<extra></extra>',
                             text = list(case_rep_mid_df['Author']),
                             showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=case_ser_mid_df['Date published'], y=case_ser_mid_df['y'],
                             mode='markers',
                             name='Case series/reports',
                             marker_color='#316906',
                             marker_symbol='pentagon',
                             legendgroup='caseser',
                             hovertemplate = 
                             '<b>Authors:</b> %{text}'+
                             '<br><b>Date of publication:</b> %{x}<extra></extra>',
                             text = list(case_ser_mid_df['Author']),
                             showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=non_rct_later_df['Date published'], y=non_rct_later_df['y'],
                             mode='markers',
                             name='Non-RCT',
                             marker_color='#03b1fc',
                             marker_symbol='circle',
                             hovertemplate = 
                             '<b>Authors:</b> %{text}'+
                             '<br><b>Date of publication:</b> %{x}<extra></extra>',
                             text = list(non_rct_later_df['Author']),
                             legendgroup='nonrct'), row=1, col=3)
    fig.add_trace(go.Scatter(x=rct_later_df['Date published'], y=rct_later_df['y'],
                             mode='markers',
                             name='RCT',
                             marker_color='#8803fc',
                             marker_symbol='square',
                             hovertemplate = 
                             '<b>Authors:</b> %{text}'+
                             '<br><b>Date of publication:</b> %{x}<extra></extra>',
                             text = list(rct_later_df['Author']),
                             legendgroup='rct'), row=1, col=3)
    fig.add_trace(go.Scatter(x=case_rep_later_df['Date published'], y=case_rep_later_df['y'],
                             mode='markers',
                             name='Case report/study',
                             marker_color='#fc8403',
                             marker_symbol='diamond',
                             hovertemplate = 
                             '<b>Authors:</b> %{text}'+
                             '<br><b>Date of publication:</b> %{x}<extra></extra>',
                             text = list(case_rep_later_df['Author']),
                             legendgroup='caserep'), row=1, col=3)
    fig.add_trace(go.Scatter(x=case_ser_later_df['Date published'], y=case_ser_later_df['y'],
                             mode='markers',
                             name='Case series/reports',
                             marker_color='#316906',
                             marker_symbol='pentagon',
                             hovertemplate = 
                             '<b>Authors:</b> %{text}'+
                             '<br><b>Date of publication:</b> %{x}<extra></extra>',
                             text = list(case_ser_later_df['Author']),
                             legendgroup='caseser'), row=1, col=3)
    fig.add_trace(go.Scatter(x=sys_revs_later_df['Date_Published'], y=sys_revs_later_df['y'],
                             mode='markers',
                             name='Systematic reviews',
                             marker_color='#ada924',
                             marker_symbol='hexagram',
                             hovertemplate = 
                             '<b>Authors:</b> %{text}'+
                             '<br><b>Date of publication:</b> %{x}<extra></extra>',
                             text = list(sys_revs_later_df['Author']),
                             legendgroup='sysrev'), row=1, col=3)
    
    fig.update_yaxes(range=[0,5])
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=3)
    #fig.update_yaxes(title_text='Study type', row=1, col=1)
    #fig.update_xaxes(title_text='Publication Date', row=1, col=2)
    fig.update_layout(showlegend=True)
    
    return fig

time_graph = create_timeline(data, sys_revs)
#fig.show(renderer="browser")

default_stylesheet = [
                {'selector': 'node',
                 'style': {
                         'width': 'data(deg_value)',
                         'height': 'data(deg_value)',
                         'background-color': 'data(Node_colour)',
                         'content': 'data(Label)',
                         'font-size': '35px',
                         'text-outline-color': 'white',
                         'text-outline-opacity': '1',
                         'text-outline-width': '8px',
                         # 'text-background-color': 'white',
                         # 'text-background-opacity': '1',
                         # 'text-background-shape': 'round-rectangle',
                         # 'text-background-padding': '20px'
                     }
                 },
                {'selector': 'edge',
                 'style': {
                         #'line-color': 'data(color_cent)'
                     }
                 }
            ]

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.Div(children=[
        html.H1(className='site-title, text-center',children=['Data Explorer']),
        html.H3(className='site-title, text-center',children=['Systematic reviews of convalescent plasma in COVID-19 continue to be poorly conducted and reported: a systematic review']),
        html.P(className='site-title, text-center',children=['Whear, R., Bethel, A., Abbott, R., Rogers, M., Orr, N., Manzi, S., Ukoumunne, O. C., Stein, K. & Thompson Coon, J. Systematic reviews of convalescent plasma in COVID-19 continue to be poorly conducted and reported: a systematic review. [published online ahead of print, 2022 Aug 4]. J Clin Epidemiol. 2022;S0895-4356(22)00178-0. doi:10.1016/j.jclinepi.2022.07.005'])
        ]),
    html.Div(className='spacer', children=[]),
    html.Div(className='center-items', children=[
        html.Div(className='text-center', children=[
            html.H3(children=['Network of systematic reviews and primary studies']),
            html.Div(children=[
                html.P(className='explan-text', children=['The network below describes the connections '
                             'between the systematic reviews and the primary '
                             'studies incliuded in those reviews. You can '
                             'click on any of the nodes in the network to '
                             'display information about that study on the '
                             'right-hand side. All of the associated studies '
                             'will then be listed in the tables below.']),
                ]),
            ]),
        ]),
    html.Div(className='row', children=[
        html.Div(className='col-8', children=[
                cyto.Cytoscape(
                    id='static-net',
                    className='net-obj',
                    elements=elements,
                    style={'width':'100%', 'height':'800px'},
                    layout={'name': 'cose',
                            'padding': 30,
                            #'quality': 'proof',
                            'nodeRepulsion': '7000',
                            #'gravity': '0.01',
                            'gravityRange': '6.0',
                            'nestingFactor': '0.8',
                            'edgeElasticity': '50',
                            'idealEdgeLength': '200',
                            'nodeDimensionsIncludeLabels': 'true',
                            'numIter': '6000',
                            },
                    stylesheet=default_stylesheet
                    )
                ]),
        html.Div(className='col-4', children=[
            html.Div(className='center-items', children=[
                html.Div(className='text-center', children=[
                    html.H3(children=['Network legend']),
                    ]),
                html.Div(className='text-center', children=[
                    html.Img(src=app.get_asset_url('images/network_legend.png'))
                    ]),
                html.Div(className='text-center', children=[
                    html.H3(children=['Study information']),
                    ]),
                html.Div(className='data-container', children=[
                    html.Ul(id='study-info', className='list-text')
                    ]),
                ]),
            ]),
        ]),
    html.Div([
        html.H3(children=['Download network image']),
        html.P(children=['To download an image file of the network as '
                         'displayed above, select an image file type '
                         'from the dropdown below then click the '
                         'download image button']),
        dcc.Dropdown(className='drops', id='image-select',
                     options=[
                         {'label': 'svg', 'value': 'svg'},
                         {'label': 'jpeg', 'value': 'jpeg'},
                         {'label': 'png', 'value': 'png'}],
                     value='svg',
                     multi=False,
                     clearable=False
                     ),
        html.Div(className='button-container', children=[
            html.Button('Download image', className='button', id='btn-dl')
            ]),
        ]),
    html.Div([
        html.Div(className='center-items', children=[
            html.Div(className='text-center', children=[
                html.H2(children=['Systematic review information']),
                html.P(className='explan-text', children=['The table below contains information about the '
                                 'systematic reviews associated with the study '
                                 'you have selected in the network diagram above. '
                                 'If you have selected a primary study, then all '
                                 'of the systematic reviews associated with that '
                                 'primary study will be displayed.']),
                ]),
            ]),
        dash_table.DataTable(id='rev-table',
                             style_data={
                                 'whiteSpace': 'normal',
                                 'height': 'auto',
                                 },
                             #columns=c_col,
                             #data=c_dict
                             )
        ]),
    html.Div([
        html.Div(className='center-items', children=[
            html.Div(className='text-center', children=[
                html.H2(children=['Primary study information']),
                html.P(className='explan-text', children=['The table below contains information about the '
                                  'primary studies associated with the study you '
                                  'have selected in the network diagram above. '
                                  'If you have selected a systematic review, then all '
                                  'of the primary studies associated with that '
                                  'systematic review will be displayed']),
                ]),
            ]),
        dash_table.DataTable(id='study-table',
                             style_data={
                                 'whiteSpace': 'normal',
                                 'height': 'auto',
                                 },
                             #columns=t_col,
                             #data=t_dict
                             )
        ]),
    html.Div([
        html.Div(className='center-items', children=[
            html.Div(className='text-center', children=[
                html.H2('Publication timeline'),
                html.P(className='explan-text', children=['The timeline below shows the publication date of the '
                       'systematic reviews and the primary studies from all of '
                       'those systematic reviews by study type'])
                ])
            ]),
        html.Div([
            dcc.Graph(figure=time_graph,
                      style={'height': '800px'})
            ]),
        ]),
    html.Div([
        html.Div(className='center-items', children=[
            html.Div(className='text-center', children=[
                html.H2('Primary studies year of publication and study type'),
                html.P(className='explan-text', children=['In the sankey diagram below the systematic review first author surname, '
                       'the study type and year of publication for all of the primary studies in the '
                       'systematic reviews are described']),
                html.P(className='explan-text', children=['When you hover on the sankey diagram below information about the nodes '
                                                          'and links will appear. In the systematic review author surname column, the '
                                                          'outgoing flow count represents the number of primary studies arising from '
                                                          'that systematic review. In the primary study design column, the incoming '
                                                          'flow count value represents the number of systematic reviews with that study '
                                                          'design. The figure on the left hand side represents the total number of studies '
                                                          'of the design. The incoming flow count in the primary study publication year '
                                                          'column gives the number of primary study design types published in that year. '
                                                          'The value on the left hand side is the number of studies published in that year.'])
                ]),
            ]),
        html.Div(className='row', children=[
            html.Div(className='col-4', children=[
                html.H3('Systematic review author surname')
                ]),
            html.Div(className='col-4', children=[
                html.Div(className='center-items', children=[
                    html.Div(className='text-center', children=[
                        html.H3('Primary study Design')
                        ]),
                    ]),
                ]),
            html.Div(className='col-4', children=[
                html.Div(className='center-items', children=[
                    html.Div(className='text-center', children=[
                        html.H3('Primary study publication year')
                        ]),
                    ]),
                ]),
            ]),
        html.Div([
            dcc.Graph(figure=sank_fig,
                      style={'height': '1400px'})
            ]),
        html.Div(className='row', children=[
            html.Div(className='col-4', children=[
                html.H3('Systematic review author surname')
                ]),
            html.Div(className='col-4', children=[
                html.Div(className='center-items', children=[
                    html.Div(className='text-center', children=[
                        html.H3('Primary study Design')
                        ]),
                    ]),
                ]),
            html.Div(className='col-4', children=[
                html.Div(className='center-items', children=[
                    html.Div(className='text-center', children=[
                        html.H3('Primary study publication year')
                        ]),
                    ]),
                ]),
            ]),
        ]),
    ])

@app.callback(Output("static-net", "generateImage"),
              [Input("btn-dl", "n_clicks"),
               Input("image-select", "value")])
def get_image(btn_clicks, img_val):
    if btn_clicks is None:
        raise PreventUpdate
    else:
        ftype = img_val
        action = 'download'
        fname = 'network_image_'+str(btn_clicks)
        return {'type': ftype,
                'action': action,
                'filename': fname,
                'bg': 'white'}

@app.callback([Output('rev-table', 'data'),
                Output('rev-table', 'columns'),
                Output('study-table', 'data'),
                Output('study-table', 'columns'),
                Output('study-info', 'children')],
              [Input('static-net', 'tapNodeData')])
def populate_tables(data):
    if data is None:
        raise PreventUpdate
    else:
        connected_dict = data['connections']
        sub_keys = ['id', 'Authors', 'Type', 'Year/Date published', 'cent_value', 'betw_value', 'deg_value', 'rank_value']
        tapped_dict = {k: data[k] for k in sub_keys}
        connected_df = pd.DataFrame.from_dict(connected_dict, orient='index')
        tapped_df = pd.DataFrame(tapped_dict, index=[0])
        if tapped_df.iloc[0,2] == 'sys rev':
            connected_df.columns = ['ID', 'Authors', 'Year', 'Title', 'Journal', 'Date published', 'Type']
            tapped_df.columns = ['ID', 'Authors', 'Type', 'Year/Date published', 'Centrality', 'Betweenness', 'Degree', 'Rank']
            c_col=[{"name": i, "id": i} for i in connected_df.columns]
            t_col=[{"name": i, "id": i} for i in tapped_df.columns]
            c_dict = connected_df.to_dict(orient='records')
            t_dict = tapped_df.to_dict(orient='records')
            
            tapped_info_dict = data['info']
            tapped_list = list()
            tapped_list.append(html.Li(html.B('Systematic Review')))
            tapped_list.append(html.Li(children=[html.B('Date published: '), str(tapped_dict['Year/Date published'])]))
            tapped_list.append(html.Li(children=[html.B('Journal title: '), str(tapped_info_dict['Journal_Title'])]))
            tapped_list.append(html.Li(children=[html.B('Impact factor: '), str(tapped_info_dict['Journal_Impact_Factor'])]))
            tapped_list.append(html.Li(children=[html.B('PRISMA requirement: '), tapped_info_dict['Prisma_checklist_required']]))
            tapped_list.append(html.Li(children=[html.B('PRISMA cited: '), tapped_info_dict['Prisma_cited']]))
            tapped_list.append(html.Li(children=[html.B('Justification: '), tapped_info_dict['Justification']]))
            tapped_list.append(html.Li(children=[html.B('IS involved: '), tapped_info_dict['IS_involved']]))
            tapped_list.append(html.Li(children=[html.B('Outcomes used: '), tapped_info_dict['Outcomes']]))
            tapped_list.append(html.Li(children=[html.B('AMSTAR rating: '), tapped_info_dict['Amstar_rating']]))
    
            return t_dict, t_col, c_dict, c_col, tapped_list
        else:
            connected_df.columns = ['ID', 'Authors', 'Date published', 'Journal', 'Impact factor', 'PRISMA required',
                                    'PRISMA cited', 'Protocol', 'Justification', 'Justification', 'IS involved',
                                    'om1', 'om2', 'om3', 'om4', 'om5', 'om6', 'om7', 'om8', 'AMSTAR rating', 'Outcomes']
            connected_cols_sub = ['ID', 'Authors', 'Date published', 'AMSTAR rating']
            connected_sub_df = connected_df[connected_cols_sub]
            tapped_df.columns = ['ID', 'Authors', 'Type', 'Year/Date published', 'Centrality', 'Betweenness', 'Degree', 'Rank']
            c_col=[{"name": i, "id": i} for i in connected_sub_df.columns]
            t_col=[{"name": i, "id": i} for i in tapped_df.columns]
            c_dict = connected_sub_df.to_dict(orient='records')
            t_dict = tapped_df.to_dict(orient='records')
            
            tapped_info_dict = data['info']
            tapped_list = list()
            tapped_list.append(html.Li(html.B('Primary study')))
            tapped_list.append(html.Li(children=[html.B('Study type: '), tapped_dict['Type']]))
            tapped_list.append(html.Li(children=[html.B('Journal title: '), tapped_info_dict['Journal title']]))
            tapped_list.append(html.Li(children=[html.B('Date published: '), tapped_info_dict['Date published']]))
            tapped_list.append(html.Li(children=['Included in ', str(len(connected_df)), ' systematic review(s)']))
            
            
            return c_dict, c_col, t_dict, t_col, tapped_list

    

if __name__ == "__main__":
    app.run_server(debug=True)
    
    
    