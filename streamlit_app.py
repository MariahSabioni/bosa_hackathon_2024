#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from streamlit_dynamic_filters import DynamicFilters
import plotly.graph_objs as go
from plotly.subplots import make_subplots

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
df_original = pd.read_csv('data/400_m_hurdles_v3.csv')
df = df_original.copy()
df.rename(columns={'name':'athlete'}, inplace=True)
pd.to_datetime(df['date'], format='%Y-%m-%d')
df['race'] = df['competition'].str.split('(').str[0].str.split().apply(lambda x: x[0] + '-' + ''.join(word[0] for word in x[1:]))+'-'+df['heat']
df['split'] = df['hurdle_id'].astype('str')

# Find PB
df['total_time'] = df.groupby(['athlete', 'race'])['hurdle_timing'].transform('max')
pb = df.groupby(['athlete'])['total_time'].transform(min) == df['total_time']
df['is_pb'] = pb

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
    time_df = athlete_df.groupby(['athlete','competition','heat','race', 'date'])['total_time'].min().reset_index()
    time_df = time_df.sort_values(by='date')
    fig = go.Figure()
    fig=px.scatter(
        time_df,
        x='date',
        y='total_time',
        color='competition',
        hover_data=['heat']
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
        xaxis_title='Split',
        yaxis_title='Velocity (m/s)',
    )
    return fig

def make_athlete_velocity_lineplot(athlete_race_df, athlete_race_df_2, athlete_df):
    df_pb = athlete_df[athlete_df['is_pb'] == True]
    pb_race = df_pb.race.iloc[0]
    df_grouped = (
        athlete_df[['split', 'hurdle_id', 'velocity']].groupby(['split', 'hurdle_id'])
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

def make_athlete_cumtime_areaplot(athlete_race_df, athlete_race_df_2, athlete_df,):
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
    ])
    fig.update_layout(
        title='Cummulative time statistics',
        xaxis=dict(
            tickvals=athlete_race_df['hurdle_id'],
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
    df_latest = athlete_df[athlete_df['date'] == athlete_df['date'].max()]
    latest_race = df_latest.race.iloc[0]
    latest_performance=df_latest.total_time.iloc[0]

    performances = [latest_performance, pb_performance]
    races = [latest_race, pb_race]
    colors= ['#41b7b9', 'rgb(247,199,106)']
    labels = ['Latest performance: '+str(latest_performance)+' s','PB performance: '+str(pb_performance)+' s', ]

    fig = go.Figure(data=[go.Bar(x=performances, y=races,orientation='h',text=labels)])
    fig.update_traces(marker_color=colors,textposition='inside',textfont_size=14)
    fig.update_layout(title_text='Latest vs PB Performances', height=300,
                      xaxis=dict(range=[min(performances)-10, max(performances)+10]))

    return fig

def make_split_bar_chart(df_filtered, df_compare):

    df_bar = pd.concat([df_filtered, df_compare])
    df_bar['athlete_race'] = df_bar['athlete'] + ' - ' + df_bar['race']
    athlete_race=df_bar.athlete_race.unique()


    for x in athlete_race:
        for hurdle_id in df_bar.hurdle_id.unique():
            fig = go.Figure(go.Bar(x=x, y=dur, name=hurdle_id,orientation='h',),)

    fig.update_layout(barmode='stack', xaxis={'categoryorder':'category ascending'},)

    # fig.update_layout(
    #     barmode="stack",)
    # fig.update_layout(
    #     yaxis = dict(
    #         tickmode = 'array',
    #         tickvals = [int(i) for i in range(int(data_seq["round_num"].max()+1))],
    #         ticktext = [('Round '+str(i)) for i in range(int(data_seq["round_num"].max()+1))]
    #     ),
    #     xaxis=dict(
    #         visible= False,
    #         showticklabels=False
    #     )
    # )
    fig.update_yaxes(autorange="reversed")
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
        textfont=dict(color='rgb(247,199,106)'),
        text=df_pb['strides'],  # labels
        textposition='outside',  # set text position
    ),
    go.Bar(
        name=selected_athlete+' - most common',
        x=df_grouped['hurdle_id'],
        y=round(df_grouped['strides'], 2),
        marker_color= 'rgba(255,255,255, 0.5)',
        textfont=dict(color='rgb(255,255,255)'),
        text=round(df_grouped['strides'], 2),  # labels
        textposition='outside',  # set text position
    ),
    go.Bar(
        name=selected_athlete+' - '+selected_race,
        x=athlete_race_df['hurdle_id'],
        y=athlete_race_df['strides'],
        marker_color='#41b7b9', #light_blue
        textfont=dict(color='#41b7b9'),
        text=athlete_race_df['strides'],  # labels
        textposition='outside',  # set text position
    ),
    go.Bar(
        name=selected_athlete+' - '+selected_race_2,
        x=athlete_race_df_2['hurdle_id'],
        y=athlete_race_df_2['strides'],
        marker_color= 'rgb(48,118,137)', #dark_blue
        textfont=dict(color='rgb(48,118,137)'),
        text=athlete_race_df_2['strides'],  # labels
        textposition='outside',  # set text position
    ),
    ])
    fig.update_traces(textposition='inside',textfont_size=14)
    fig.update_layout(
        title='Strides statistics',
        xaxis=dict(
            tickvals=athlete_race_df['hurdle_id'],
            title='Split'
        ),
        yaxis=dict(range=[athlete_df.strides.min(), athlete_df.strides.max()]),
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
        line=dict(color='rgb(247,199,106)'),
        textfont=dict(color='rgb(247,199,106)'),
        text=filtered_data['velocity'],  # labels
        textposition='top center',  # Set text position
        mode='lines+markers+text',
    ), row=1, col=1, secondary_y=True)

    # Add line plot for strides
    fig.add_trace(go.Scatter(
        x=filtered_data['hurdle_id'],
        y=filtered_data['strides'],
        name='number of strides',
        line=dict(color='rgb(48,118,137)')
    ), row=1, col=1)

    # Update layout
    fig.update_layout(
        title=('Split times, velocity & strides'),
        xaxis=dict(
            tickvals=filtered_data['hurdle_id'],
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

    st.header("Athlete Stats")

    st.subheader('Performance History')

    performances_barchart = make_performance_barchart(athlete_df,)
    st.plotly_chart(performances_barchart, use_container_width=True)
    
    history_lineplot = make_athlete_scatterplot(athlete_df,)
    st.plotly_chart(history_lineplot, use_container_width=True)


    st.subheader('Race Stats')

    filtered_races = df[(df.event == selected_event)&(df.athlete == selected_athlete)]['race'].unique()
    default_race = filtered_races[0]
    selected_race = st.selectbox('Select race to visualize', filtered_races, index=0)
    
    athlete_race_df = df[(df.event == selected_event)&(df.athlete == selected_athlete)&(df.race == selected_race)]

    race_summary_plot = make_special_chart(athlete_race_df)
    st.plotly_chart(race_summary_plot, use_container_width=True)

    st.dataframe(athlete_race_df,
                column_order=("split", "velocity", "strides", "interval", "hurdle_timing", "temporary_place"),
                hide_index=True,
                width=None,
                column_config={
                    "split": st.column_config.TextColumn(
                        "Split",
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
   
    st.subheader('Race Comparison (Intra-Athlete)')

    filtered_races_2 = df[(df.event == selected_event)&(df.athlete == selected_athlete)&(df.race != selected_race)]['race'].unique()
    default_race_2 = filtered_races[0]
    selected_race_2 = st.selectbox('Select race to compare', filtered_races_2, index=0)
    
    athlete_race_df_2 = df[(df.event == selected_event)&(df.athlete == selected_athlete)&(df.race == selected_race_2)]

    split_lineplot = make_athlete_velocity_lineplot(athlete_race_df,athlete_race_df_2, athlete_df,)
    st.plotly_chart(split_lineplot, use_container_width=True)

    time_lineplot = make_athlete_time_lineplot(athlete_race_df,athlete_race_df_2, athlete_df,)
    st.plotly_chart(time_lineplot, use_container_width=True)

    stride_barplot = make_athlete_stride_barplot(athlete_race_df,athlete_race_df_2, athlete_df,)
    st.plotly_chart(stride_barplot, use_container_width=True)
     
with st.container():

    dynamic_filters = DynamicFilters(df, filters=['athlete', 'race'])
    st.header("Athlete Comparison")
    dynamic_filters.display_filters(location='columns', num_columns=2, gap='medium')
    compare_df = dynamic_filters.filter_df()

    st.subheader('Splits')
    split_compare_lineplot = make_compare_split_lineplot(athlete_race_df, compare_df,)
    st.plotly_chart(split_compare_lineplot, use_container_width=True)

    split_compare_barplot = make_split_bar_chart(athlete_race_df, compare_df,)
    st.plotly_chart(split_compare_barplot, use_container_width=True)



with st.expander('About', expanded=True):
    st.write('''
        ---
        ''')
