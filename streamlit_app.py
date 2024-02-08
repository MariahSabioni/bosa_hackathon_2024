#######################
# Import libraries
import numpy as np
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from streamlit_dynamic_filters import DynamicFilters
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from preprocess_df import *

#######################
# Page configuration
st.set_page_config(
    page_title="[Insert cool name]",
    page_icon="*",
    layout="wide",
    initial_sidebar_state="expanded"
    )
alt.themes.enable("dark")


#######################
# Load data
#df_reshaped = pd.read_csv('data/us-population-2010-2019-reshaped.csv')
df_original = pd.read_csv('data/400_m_hurdles_v7.csv')
df = df_original.copy()
df = preprocess_df(df)

#######################
# Sidebar
with st.sidebar:
    st.title('Performance Hub')
    st.caption('-----')
    
    default_event = df['event'].iloc[0]
    selected_event = st.selectbox('Select an event', df.event.unique(), index=0)

    filtered_athletes = df[df.event == default_event]['athlete'].unique()
    default_athlete = filtered_athletes[18]
    selected_athlete = st.selectbox('Select an athlete', filtered_athletes, index=18)
    
    athlete_df = df[(df.event == selected_event)&(df.athlete == selected_athlete)]

#######################
# Plots

# Scatterplot
def make_athlete_scatterplot(athlete_df):
    time_df = athlete_df.groupby(['athlete','competition','round','race', 'date'])['total_time'].min().reset_index()
    time_df = time_df.sort_values(by='date')
    fig = go.Figure()
    fig=px.scatter(
        time_df,
        x='date',
        y='total_time',
        color='competition',
        hover_data=['round'],
    )
    fig.update_layout(
        xaxis_title='Race date',
        yaxis_title='Race time (s)',
        showlegend=False,
        hovermode="x unified",
        title='All performances')
    return fig

# Lineplot
def make_compare_split_lineplot(athlete_race_df, df_compare):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            name=selected_athlete+' - '+selected_race,
            x=athlete_race_df['hurdle_id'],
            y=athlete_race_df['velocity'],
            line=dict(color='red'),
            mode='lines'
        ),)
    
    for athlete in df_compare.athlete.unique():
        df_compare_a = df_compare[df_compare['athlete']==athlete]
        for race in df_compare_a.race.unique():
            df_compare_r = df_compare_a[df_compare_a['race']==race]
            fig.add_trace(
                go.Scatter(
                    name=athlete+' - '+race,
                    x=df_compare_r['hurdle_id'],
                    y=df_compare_r['velocity'],
                    mode='lines'
                ),)

    fig.update_layout(
        yaxis_title='Velocity (m/s)',
        xaxis=dict( 
            tickvals=athlete_race_df['hurdle_id'],
            ticktext=athlete_race_df['hurdle_id_label'],
            title='Split'
        ),
    )
    return fig

def make_athlete_velocity_lineplot(athlete_race_df, athlete_race_df_2, athlete_df):
    df_pb = athlete_df[athlete_df['is_pb'] == True]
    pb_race = df_pb.race.iloc[0]
    df_grouped = (
        athlete_df[['split', 'hurdle_id','velocity']].groupby(['split', 'hurdle_id'])
        .agg(['mean', 'std', 'count'])
    )
    df_grouped = df_grouped.droplevel(axis=1, level=0).reset_index()
    df_grouped = df_grouped.sort_values(by="hurdle_id", ascending=True)
    fig = go.Figure([
    go.Scatter(
        name=selected_athlete+' - PB @'+pb_race,
        x=df_pb['hurdle_id'],
        y=df_pb['velocity'],
        line=dict(color='rgb(247,199,106)', width=0.7), #yellow
        mode='lines+markers'
    ),
    go.Scatter(
        name=selected_athlete+' - mean',
        x=df_grouped['hurdle_id'],
        y=round(df_grouped['mean'], 2),
        mode='lines',
        line=dict(color='rgba(255,255,255, 0.5)'),
    ),
    go.Scatter(
        name='mean+sd',
        x=df_grouped['hurdle_id'],
        y=round(df_grouped['std']+df_grouped['mean'], 2),
        mode='lines',
        marker=dict(color='#444'),
        line=dict(width=0),
        showlegend=False
    ),
    go.Scatter(
        name='mean-sd',
        x=df_grouped['hurdle_id'],
        y=round(-df_grouped['std']+df_grouped['mean'], 2),
        marker=dict(color='#444'),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(68, 68, 68, 0.5)',
        fill='tonexty',
        showlegend=False
    ),
    go.Scatter(
        name=selected_athlete+' - '+selected_race,
        x=athlete_race_df['hurdle_id'],
        y=athlete_race_df['velocity'],
        line=dict(color='#41b7b9', width=0.7), #light_blue
        mode='lines+markers',
    ),
    go.Scatter(
        name=selected_athlete+' - '+selected_race_2,
        x=athlete_race_df_2['hurdle_id'],
        y=athlete_race_df_2['velocity'],
        line=dict(color='rgb(48,118,137)', width=0.7), #dark_blue
        mode='lines+markers',
    ),
    ])
    fig.update_layout(
        title='Velocity statistics',
        xaxis=dict(
            tickvals=athlete_race_df['hurdle_id'],
            ticktext=athlete_race_df['hurdle_id_label'],
            title='Split'
        ),
        yaxis_title='Velocity (m/s)',
        legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    return fig

def make_athlete_time_lineplot(athlete_race_df, athlete_race_df_2, athlete_df):
    df_pb = athlete_df[athlete_df['is_pb'] == True]
    pb_race = df_pb.race.iloc[0]
    df_grouped = (
        athlete_df[['split', 'hurdle_id', 'interval']].groupby(['split', 'hurdle_id'])
        .agg(['mean', 'std', 'count'])
    )
    df_grouped = df_grouped.droplevel(axis=1, level=0).reset_index()
    df_grouped = df_grouped.sort_values(by="hurdle_id", ascending=True)
    fig = go.Figure([
    go.Scatter(
        name=selected_athlete+' - PB @'+pb_race,
        x=df_pb['hurdle_id'],
        y=df_pb['interval'],
        line=dict(color='rgb(247,199,106)', width=0.7), #yellow
        mode='lines+markers'
    ),
    go.Scatter(
        name=selected_athlete+' - mean',
        x=df_grouped['hurdle_id'],
        y=round(df_grouped['mean'], 2),
        mode='lines',
        line=dict(color='rgba(255,255,255, 0.5)'),
    ),
    go.Scatter(
        name='mean+sd',
        x=df_grouped['hurdle_id'],
        y=round(df_grouped['std']+df_grouped['mean'], 2),
        mode='lines',
        marker=dict(color='#444'),
        line=dict(width=0),
        showlegend=False
    ),
    go.Scatter(
        name='mean-sd',
        x=df_grouped['hurdle_id'],
        y=round(-df_grouped['std']+df_grouped['mean'], 2),
        marker=dict(color='#444'),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(68, 68, 68, 0.5)',
        fill='tonexty',
        showlegend=False
    ),
    go.Scatter(
        name=selected_athlete+' - '+selected_race,
        x=athlete_race_df['hurdle_id'],
        y=athlete_race_df['interval'],
        line=dict(color='#41b7b9', width=0.7), #light_blue
        mode='lines+markers',
    ),
    go.Scatter(
        name=selected_athlete+' - '+selected_race_2,
        x=athlete_race_df_2['hurdle_id'],
        y=athlete_race_df_2['interval'],
        line=dict(color='rgb(48,118,137)', width=0.7), #dark_blue
        mode='lines+markers',
    ),
    ])
    fig.update_layout(
        title='Time statistics',
        xaxis=dict(
            tickvals=athlete_race_df['hurdle_id'],
            ticktext=athlete_race_df['hurdle_id_label'],
            title='Split'
        ),
        yaxis_title='Time (s)',
        legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    return fig

def make_athlete_cumtime_lineplot(athlete_race_df, athlete_race_df_2, athlete_df,):
    df_pb = athlete_df[athlete_df['is_pb'] == True]
    pb_race = df_pb.race.iloc[0]
    df_grouped = (
        athlete_df[['split', 'hurdle_id', 'hurdle_timing']].groupby(['split', 'hurdle_id'])
        .agg(['mean', 'std', 'count'])
    )
    df_grouped = df_grouped.droplevel(axis=1, level=0).reset_index()
    df_grouped = df_grouped.sort_values(by="hurdle_id", ascending=True)
    fig = go.Figure([
    go.Scatter(
        name=selected_athlete+' - PB @'+pb_race,
        x=df_pb['hurdle_id'],
        y=df_pb['hurdle_timing'],
        line=dict(color='rgb(247,199,106)', width=0.7), #yellow
        mode='lines+markers',
    ),
    go.Scatter(
        name=selected_athlete+' - mean',
        x=df_grouped['hurdle_id'],
        y=round(df_grouped['mean'], 2),
        mode='lines',
        line=dict(color='rgba(255,255,255, 0.5)'),
    ),
    go.Scatter(
        name=selected_athlete+' - '+selected_race,
        x=athlete_race_df['hurdle_id'],
        y=athlete_race_df['hurdle_timing'],
        line=dict(color='#41b7b9', width=0.7), #light_blue
        mode='lines+markers',
    ),
    go.Scatter(
        name=selected_athlete+' - '+selected_race_2,
        x=athlete_race_df_2['hurdle_id'],
        y=athlete_race_df_2['hurdle_timing'],
        line=dict(color='rgb(48,118,137)', width=0.7), #dark_blue
        mode='lines+markers',
    ),
    ],)
    fig.update_layout(
        title='Cummulative time statistics',
        xaxis=dict(
            tickvals=athlete_race_df['hurdle_id'],
            ticktext=athlete_race_df['hurdle_id_label'],
            title='Split'
        ),
        yaxis=dict(title='Cummulative time (s)',),
        legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    return fig

def make_cluster_velocity_lineplot(athlete_race_df, cluster_best_df):
    # df_grouped = (
    #     athlete_df[['split', 'hurdle_id','velocity']].groupby(['split', 'hurdle_id'])
    #     .agg(['mean', 'std', 'count'])
    # )
    # df_grouped = df_grouped.droplevel(axis=1, level=0).reset_index()
    # df_grouped = df_grouped.sort_values(by="hurdle_id", ascending=True)
    cluster_best_race = cluster_best_df.race.iloc[0]
    fig = go.Figure([
    # go.Scatter(
    #     name=selected_athlete+' - Cluster Best @'+cluster_best_race,
    #     x=cluster_best_df['hurdle_id'],
    #     y=cluster_best_df['velocity'],
    #     line=dict(color='rgb(247,199,106)', width=0.7), #yellow
    #     mode='lines+markers'
    # ),
    go.Scatter(
        name='Cluster - mean',
        x=athlete_race_df['hurdle_id'],
        y=round(athlete_race_df['mean_velocity'], 2),
        mode='lines',
        line=dict(color='rgba(255,255,255, 0.5)'),
    ),
    go.Scatter(
        name='Cluster - mean+sd',
        x=athlete_race_df['hurdle_id'],
        y=round(athlete_race_df['std_velocity']+athlete_race_df['mean_velocity'], 2),
        mode='lines',
        marker=dict(color='#444'),
        line=dict(width=0),
        showlegend=False
    ),
    go.Scatter(
        name='Cluster - mean-sd',
        x=athlete_race_df['hurdle_id'],
        y=round(-athlete_race_df['std_velocity']+athlete_race_df['mean_velocity'], 2),
        marker=dict(color='#444'),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(68, 68, 68, 0.5)',
        fill='tonexty',
        showlegend=False
    ),
    go.Scatter(
        name=selected_athlete+' - '+selected_race,
        x=athlete_race_df['hurdle_id'],
        y=athlete_race_df['velocity'],
        line=dict(color='#41b7b9', width=0.7), #light_blue
        mode='lines+markers',
    ),
    ])
    fig.update_layout(
        title='Velocity statistics',
        xaxis=dict(
            tickvals=athlete_race_df['hurdle_id'],
            ticktext=athlete_race_df['hurdle_id_label'],
            title='Split'
        ),
        yaxis_title='Velocity (m/s)',
        legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    return fig

def make_cluster_time_lineplot(athlete_race_df, cluster_best_df):
    # df_grouped = (
    #     athlete_df[['split', 'hurdle_id','velocity']].groupby(['split', 'hurdle_id'])
    #     .agg(['mean', 'std', 'count'])
    # )
    # df_grouped = df_grouped.droplevel(axis=1, level=0).reset_index()
    # df_grouped = df_grouped.sort_values(by="hurdle_id", ascending=True)
    cluster_best_race = cluster_best_df.race.iloc[0]
    fig = go.Figure([
    go.Scatter(
        name=selected_athlete+' - Cluster Best @'+cluster_best_race,
        x=cluster_best_df['hurdle_id'],
        y=cluster_best_df['interval'],
        line=dict(color='rgb(247,199,106)', width=0.7), #yellow
        mode='lines+markers'
    ),
    go.Scatter(
        name=selected_athlete+' - mean',
        x=athlete_race_df['hurdle_id'],
        y=round(athlete_race_df['mean_intervals'], 2),
        mode='lines',
        line=dict(color='rgba(255,255,255, 0.5)'),
    ),
    go.Scatter(
        name='mean+sd',
        x=athlete_race_df['hurdle_id'],
        y=round(athlete_race_df['std_intervals']+athlete_race_df['mean_intervals'], 2),
        mode='lines',
        marker=dict(color='#444'),
        line=dict(width=0),
        showlegend=False
    ),
    go.Scatter(
        name='mean-sd',
        x=athlete_race_df['hurdle_id'],
        y=round(-athlete_race_df['std_intervals']+athlete_race_df['mean_intervals'], 2),
        marker=dict(color='#444'),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(68, 68, 68, 0.5)',
        fill='tonexty',
        showlegend=False
    ),
    go.Scatter(
        name=selected_athlete+' - '+selected_race,
        x=athlete_race_df['hurdle_id'],
        y=athlete_race_df['interval'],
        line=dict(color='#41b7b9', width=0.7), #light_blue
        mode='lines+markers',
    ),
    ])
    fig.update_layout(
        title='Time statistics',
        xaxis=dict(
            tickvals=athlete_race_df['hurdle_id'],
            ticktext=athlete_race_df['hurdle_id_label'],
            title='Split'
        ),
        yaxis_title='Time (s)',
        legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    return fig


# Bar chart
def make_performance_barchart(athlete_df):
    df_pb = athlete_df[athlete_df['is_pb'] == True]
    pb_race = df_pb.race.iloc[0]
    pb_performance=df_pb.total_time.iloc[0]
    pb_strides = df_pb.total_strides.iloc[0]
    df_latest = athlete_df[athlete_df['date'] == athlete_df['date'].max()]
    latest_race = df_latest.race.iloc[0]
    latest_strides = df_latest.total_strides.iloc[0]
    latest_performance=df_latest.total_time.iloc[0]

    performances = [latest_performance, pb_performance]
    races = [latest_race, pb_race]
    colors= ['#41b7b9', 'rgb(247,199,106)']
    labels = ['Latest performance: '+str(latest_performance)+' s; Total strides: '+str(latest_strides),
              'PB performance: '+str(pb_performance)+' s; Total strides: '+str(pb_strides),]

    fig = go.Figure(data=[go.Bar(x=performances, y=races,orientation='h',text=labels)])
    fig.update_traces(marker_color=colors,textposition='inside',textfont_size=14)
    fig.update_layout(title_text='Latest vs PB Performances', height=300,
                      xaxis=dict(range=[min(performances)-10, max(performances)+10]))

    return fig

def make_performance_compare_barchart(athlete_race_df, athlete_race_df_2, athlete_df):
    df_pb = athlete_df[athlete_df['is_pb'] == True]
    pb_race = df_pb.race.iloc[0]
    pb_performance=df_pb.total_time.iloc[0]
    pb_strides = df_pb.total_strides.iloc[0]
    pb_date = df_pb.date.iloc[0]
    selected_performance=athlete_race_df.total_time.iloc[0]
    selected_strides = athlete_race_df.total_strides.iloc[0]
    selected_date = athlete_race_df.date.iloc[0]
    selected_performance_2=athlete_race_df_2.total_time.iloc[0]
    selected_strides_2 = athlete_race_df_2.total_strides.iloc[0]
    selected_date_2 = athlete_race_df_2.date.iloc[0]

    performances = [pb_performance, selected_performance, selected_performance_2]
    races = [
        'PB: '+pb_race + ' - '+pb_date,
        'Baseline race: '+selected_race+' - '+selected_date, 
        'Compare race: '+selected_race_2+' - '+selected_date_2]
    colors= ['rgb(247,199,106)', '#41b7b9', 'rgb(48,118,137)']
    labels = ['Total time: '+str(pb_performance)+' s; Total strides: '+str(pb_strides),
              'Total time: '+str(selected_performance)+' s; Total strides: '+str(selected_strides),
              'Total time: '+str(selected_performance_2)+' s; Total strides: '+str(selected_strides_2),]

    fig = go.Figure(data=[go.Bar(x=performances, y=races,orientation='h',text=labels)])
    fig.update_traces(marker_color=colors,textposition='inside',textfont_size=14)
    fig.update_layout(title_text='Races Performances', height=300,
                      xaxis=dict(range=[min(performances)-10, max(performances)+10]),
                      yaxis=dict(categoryorder='array', categoryarray=races[::-1])  # Set the desired order
)

    return fig

def make_athlete_stride_barplot(athlete_race_df, athlete_race_df_2, athlete_df):
    df_pb = athlete_df[athlete_df['is_pb'] == True]
    pb_race = df_pb.race.iloc[0]
    df_grouped = (
        athlete_df[['split', 'hurdle_id', 'strides']].groupby(['split', 'hurdle_id'])
        .agg(pd.Series.mode)
    )
    df_grouped = df_grouped.reset_index()
    df_grouped = df_grouped.sort_values(by="hurdle_id", ascending=True)
    fig = go.Figure([
    go.Bar(
        name=selected_athlete+' - PB @'+pb_race,
        x=df_pb['hurdle_id'],
        y=df_pb['strides'],
        marker_color='rgb(247,199,106)', #yellow
        # textfont=dict(color='rgb(247,199,106)'),
        # text=df_pb['strides'],  # labels
        # textposition='outside',  # set text position
    ),
    go.Bar(
        name=selected_athlete+' - most common',
        x=df_grouped['hurdle_id'],
        y=df_grouped['strides'],
        marker_color= 'rgba(255,255,255, 0.5)',
        # textfont=dict(color='rgb(255,255,255)'),
        # text=df_grouped['strides'],  # labels
        # textposition='outside',  # set text position
    ),
    go.Bar(
        name=selected_athlete+' - '+selected_race,
        x=athlete_race_df['hurdle_id'],
        y=athlete_race_df['strides'],
        marker_color='#41b7b9', #light_blue
        # textfont=dict(color='#41b7b9'),
        # text=athlete_race_df['strides'],  # labels
        # textposition='outside',  # set text position
    ),
    go.Bar(
        name=selected_athlete+' - '+selected_race_2,
        x=athlete_race_df_2['hurdle_id'],
        y=athlete_race_df_2['strides'],
        marker_color= 'rgb(48,118,137)', #dark_blue
        # textfont=dict(color='rgb(48,118,137)'),
        # text=athlete_race_df_2['strides'],  # labels
        # textposition='outside',  # set text position
    ),
    ])
    #fig.update_traces(textposition='inside',textfont_size=14)
    fig.update_layout(
        title='Strides statistics',
        xaxis=dict(
            tickvals=athlete_race_df['hurdle_id'],
            ticktext=athlete_race_df['hurdle_id_label'],
            title='Split'
        ),
        yaxis=dict(showgrid=True,range=[athlete_df.strides.min()-1, athlete_df.strides.max()+1]),
        yaxis_title='Strides',
        legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    
    return fig

# Special chart
def make_special_chart(filtered_data):

    # Create subplots with one row and two columns
    fig = make_subplots(rows=1, cols=1,
                        specs=[[{"secondary_y": True}]])

    # Add bar plot for split times
    fig.add_trace(go.Bar(
        x=filtered_data['hurdle_id'],
        y=filtered_data['interval'],
        name='Split Time',
        marker_color='#41b7b9', 
        textfont=dict(color='#41b7b9'),
        text=filtered_data['interval'],  # labels
        textposition='outside',  # set text position
    ), row=1, col=1)

    # Add line plot for velocity
    fig.add_trace(go.Scatter(
        x=filtered_data['hurdle_id'],
        y=filtered_data['velocity'],
        name='Velocity',
        line=dict(color='white'),
        textfont=dict(color='white'),
        text=filtered_data['velocity'],  # labels
        textposition='top center',  # Set text position
        mode='lines+markers+text',
    ), row=1, col=1, secondary_y=True)

    # Add line plot for strides
    fig.add_trace(go.Scatter(
        x=filtered_data['hurdle_id'],
        y=filtered_data['strides'],
        name='number of strides',
        mode='markers',
        marker=dict(
        symbol='line-ew',  # Set marker symbol to horizontal line
        size=24 ,  # Adjust marker size
        line=dict(width=4, color='rgb(48,118,137)')  # Set marker line width and color
    )    ), row=1, col=1)

    # Update layout
    fig.update_layout(
        title=('Split times, velocity & strides'),
        xaxis=dict(
            tickvals=athlete_race_df['hurdle_id'],
            ticktext=athlete_race_df['hurdle_id_label'],
            title='Split'
        ),
        yaxis=dict(title='Split Time', domain=[0, 1]),
        yaxis2=dict(
            overlaying='y',
            side='right',
            title='Velocity (m/s)',
            showgrid=False,
            domain=[0, 1]
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
        ),
    )

    return fig

#######################
# Dashboard Main Panel
with st.container():

    # st.header("Athlete Stats")

    st.subheader('Performance History')

    # performances_barchart = make_performance_barchart(athlete_df,)
    # st.plotly_chart(performances_barchart, use_container_width=True)
    
    history_lineplot = make_athlete_scatterplot(athlete_df,)
    st.plotly_chart(history_lineplot, use_container_width=True)

with st.container():

    st.subheader('Race Stats')

    filtered_races = df[(df.event == selected_event)&(df.athlete == selected_athlete)].sort_values(by='date', ascending=False)['race'].unique()
    default_race = filtered_races[0]
    selected_race = st.selectbox('Select race:', filtered_races, index=0)
    
    athlete_race_df = df[(df.event == selected_event)&(df.athlete == selected_athlete)&(df.race == selected_race)]

    race_summary_plot = make_special_chart(athlete_race_df)
    st.plotly_chart(race_summary_plot, use_container_width=True)

    st.dataframe(athlete_race_df,
                column_order=("split", "hurdle_id_label", "velocity", "strides", "interval", "hurdle_timing", "temporary_place"),
                hide_index=True,
                width=None,
                use_container_width=True,
                column_config={
                    "split": st.column_config.TextColumn(
                        "Split",
                    ),
                    "hurdle_id_label": st.column_config.TextColumn(
                        "Split ID",
                    ),
                    "velocity": st.column_config.ProgressColumn(
                        "velocity",
                        format="%f"+" m/s",
                        min_value=min(athlete_race_df.velocity),
                        max_value=max(athlete_race_df.velocity),
                    ),
                    "strides": st.column_config.ProgressColumn(
                        "Strides",
                        format="%f",
                        min_value=min(athlete_race_df.strides),
                        max_value=max(athlete_race_df.strides),
                    ),
                    "interval": st.column_config.ProgressColumn(
                        "Split time",
                        format="%f"+" s",
                        min_value=min(athlete_race_df.interval),
                        max_value=max(athlete_race_df.interval),
                     ),
                    "hurdle_timing": st.column_config.ProgressColumn(
                        "Total time",
                        format="%f"+" s",
                        min_value=min(athlete_race_df.hurdle_timing),
                        max_value=max(athlete_race_df.hurdle_timing),
                     ),
                    "temporary_place": st.column_config.TextColumn(
                        "Position",
                    )
                    }
                )
   
with st.container():

    st.subheader('Race Comparison (Intra-Athlete)')

    st.write('Baseline race:     ' +selected_race)

    filtered_races_2 = df[(df.event == selected_event)&(df.athlete == selected_athlete)&(df.race != selected_race)]['race'].unique()
    default_race_2 = filtered_races[0]
    selected_race_2 = st.selectbox('Select race to compare to baseline:', filtered_races_2, index=0)
    
    athlete_race_df_2 = df[(df.event == selected_event)&(df.athlete == selected_athlete)&(df.race == selected_race_2)]

    performances_barchart_compare = make_performance_compare_barchart(athlete_race_df, athlete_race_df_2, athlete_df)
    st.plotly_chart(performances_barchart_compare, use_container_width=True)

    split_lineplot = make_athlete_velocity_lineplot(athlete_race_df,athlete_race_df_2, athlete_df,)
    st.plotly_chart(split_lineplot, use_container_width=True)

    time_lineplot = make_athlete_time_lineplot(athlete_race_df,athlete_race_df_2, athlete_df,)
    st.plotly_chart(time_lineplot, use_container_width=True)

    # cum_time_lineplot = make_athlete_cumtime_lineplot(athlete_race_df, athlete_race_df_2, athlete_df,)
    # st.plotly_chart(cum_time_lineplot, use_container_width=True)

    stride_barplot = make_athlete_stride_barplot(athlete_race_df,athlete_race_df_2, athlete_df,)
    st.plotly_chart(stride_barplot, use_container_width=True)
     
with st.container():

    st.subheader("Race Comparison (Intra-Cluster)")

    # cluster_default = athlete_race_df.cluster.iloc[0]
    # cluster_list = sorted(df.cluster.unique())
    # selected_cluster = st.selectbox('Select cluster to compare:', [str(x) for x in cluster_list], index=cluster_default)
    cluster_df = df[df['cluster'] == athlete_race_df.cluster.iloc[0]]
    cluster_best_df = cluster_df[cluster_df['total_time'] == cluster_df['total_time'].min()]

    velocity_cluster_plot = make_cluster_velocity_lineplot(athlete_race_df, cluster_best_df)
    st.plotly_chart(velocity_cluster_plot, use_container_width=True)

    # st.write('Feature importance')
    # st.dataframe(athlete_race_df.pivot(columns='hurdle_id_label', values='feature_importance',).reset_index(),
    #             hide_index=True,
    #             width=None,
    #             use_container_width=True,
    #             )
   

    # time_cluster_plot = make_cluster_time_lineplot(athlete_race_df, cluster_best_df)
    # st.plotly_chart(time_cluster_plot, use_container_width=True)

with st.container():

    dynamic_filters = DynamicFilters(df, filters=['athlete', 'race'])
    st.subheader("Race Comparison (Inter-Athlete)")
    dynamic_filters.display_filters(location='columns', num_columns=2, gap='medium')
    compare_df = dynamic_filters.filter_df()

    st.subheader('Splits')
    split_compare_lineplot = make_compare_split_lineplot(athlete_race_df, compare_df,)
    st.plotly_chart(split_compare_lineplot, use_container_width=True)

with st.expander('About', expanded=True):
    st.write('''
        ---
        ''')
