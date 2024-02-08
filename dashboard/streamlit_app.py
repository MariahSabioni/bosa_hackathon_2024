#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from streamlit_dynamic_filters import DynamicFilters
import plotly.graph_objs as go

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
df['race'] = df['competition'].str.split('(').str[0].str.split().apply(lambda x: x[0] + '-' + ''.join(word[0] for word in x[1:]))+'-'+df['heat']
df['split'] = df['hurdle_id'].astype('str')

# Find PB
df['total_time'] = df.groupby(['athlete', 'race'])['hurdle_timing'].transform(max)
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
    default_athlete = filtered_athletes[0]
    selected_athlete = st.selectbox('Select an athlete', filtered_athletes, index=0)

    filtered_races = df[(df.event == selected_event)&(df.athlete == selected_athlete)]['race'].unique()
    default_race = filtered_races[0]
    selected_race = st.selectbox('Select a race', filtered_races, index=0)

    # dafault_athlete = df['athlete'].iloc[0]
    # selected_athlete = st.selectbox('Select an athlete', df.athlete.unique(), index=0)
    # filtered_races = df[df.athlete == selected_athlete]['race'].unique()
    # default_race = filtered_races[0]
    # selected_race = st.selectbox('Select a race', filtered_races, index=0)
    
    athlete_df = df[(df.event == selected_event)&(df.athlete == selected_athlete)]
    athlete_race_df = df[(df.event == selected_event)&(df.athlete == selected_athlete)&(df.race == selected_race)]

#######################
# Plots

# Lineplot
def make_athlete_split_lineplot(athlete_race_df, athlete_df):
    df_pb = athlete_df[athlete_df['is_pb'] == True]
    df_grouped = (
        athlete_df[['split', 'hurdle_id', 'velocity']].groupby(['split', 'hurdle_id'])
        .agg(['mean', 'std', 'count'])
    )
    df_grouped = df_grouped.droplevel(axis=1, level=0).reset_index()
    df_grouped = df_grouped.sort_values(by="hurdle_id", ascending=True)
    fig = go.Figure([
    go.Scatter(
        name=selected_athlete+' - PB @'+selected_race,
        x=df_pb['hurdle_id'],
        y=df_pb['velocity'],
        line=dict(color='yellow'),
        mode='lines'
    ),
    go.Scatter(
        name=selected_athlete+' - mean',
        x=df_grouped['hurdle_id'],
        y=round(df_grouped['mean'], 2),
        mode='lines',
        line=dict(color='rgb(31, 119, 180)'),
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
        line=dict(color='red'),
        mode='lines'
    ),
    ])
    fig.update_layout(
        xaxis_title='Split',
        yaxis_title='Velocity (km/h)',
        legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    return fig

def make_split_lineplot(df_filtered, df_compare):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            name=selected_athlete+' - '+selected_race,
            x=df_filtered['hurdle_id'],
            y=df_filtered['velocity'],
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
        yaxis_title='Velocity (km/h)',
    )
    return fig

# Donut chart
def make_donut(input_response, input_text, input_color):
    chart_color = ['#29b5e8', '#155F7A']
    
    source = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100-input_response, input_response]
    })
    source_bg = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100, 0]
    })

    plot = alt.Chart(source).mark_arc(innerRadius=45, cornerRadius=25).encode(
        theta="% value",
        color= alt.Color("Topic:N",
                        scale=alt.Scale(
                            #domain=['A', 'B'],
                            domain=[input_text, ''],
                            # range=['#29b5e8', '#155F7A']),  # 31333F
                            range=chart_color),
                        legend=None),
    ).properties(width=130, height=130)

    text = plot.mark_text(align='center', color="#29b5e8", font="Lato", fontSize=32, fontWeight=700, fontStyle="italic").encode(text=alt.value(f'{input_response} %'))
    plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=45, cornerRadius=20).encode(
        theta="% value",
        color= alt.Color("Topic:N",
                        scale=alt.Scale(
                            # domain=['A', 'B'],
                            domain=[input_text, ''],
                            range=chart_color),  # 31333F
                        legend=None),
    ).properties(width=130, height=130)
    return plot_bg + plot + text

# Bar chart
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


#######################
# Dashboard Main Panel
with st.container():

    st.header("Athlete Stats")

    col = st.columns((0.5, 0.5), gap='small')

    # with col[0]:
    #     st.markdown('#### Gains/Losses')

    #     #df_population_difference_sorted = calculate_population_difference(df_reshaped, selected_athlete)

    #     first_state_name = '-'
    #     first_state_population = '-'
    #     first_state_delta = ''
    #     st.metric(label=first_state_name, value=first_state_population, delta=first_state_delta)

        
    #     st.markdown('#### States Migration')

    #     states_migration_greater = 0
    #     states_migration_less = 0
    #     donut_chart_greater = make_donut(states_migration_greater, 'Inbound Migration', 'green')
    #     donut_chart_less = make_donut(states_migration_less, 'Outbound Migration', 'red')

    #     migrations_col = st.columns((0.2, 1, 0.2))
    #     with migrations_col[1]:
    #         st.write('Inbound')
    #         st.altair_chart(donut_chart_greater)
    #         st.write('Outbound')
    #         st.altair_chart(donut_chart_less)

    #with col[0]:
    st.subheader('Split History')
    
    split_lineplot = make_athlete_split_lineplot(athlete_race_df, athlete_df,)
    st.plotly_chart(split_lineplot, use_container_width=True)
        

    #with col[1]:
    st.subheader('Split Stats')

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
                        format="%f"+" km/h",
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

    dynamic_filters = DynamicFilters(df, filters=['athlete', 'race'])
    st.header("Athlete Comparison")
    dynamic_filters.display_filters(location='columns', num_columns=2, gap='medium')
    compare_df = dynamic_filters.filter_df()

    st.subheader('Splits')
    split_compare_lineplot = make_split_lineplot(athlete_race_df, compare_df,)
    st.plotly_chart(split_compare_lineplot, use_container_width=True)

    #col = st.columns((0.5, 0.5), gap='small')
    #with col[0]:
    #     st.subheader('Splits')
    #     split_lineplot = make_split_lineplot(athlete_race_df, compare_df, selected_race)
    #     st.plotly_chart(split_lineplot, use_container_width=True)
    # with col[1]:
    #     st.markdown('#### Athlete Comparison')
    #     split_lineplot = make_athlete_split_lineplot(athlete_race_df, athlete_df,selected_race)
    #     st.plotly_chart(split_lineplot, use_container_width=True)

    split_compare_barplot = make_split_bar_chart(athlete_race_df, compare_df,)
    st.plotly_chart(split_compare_barplot, use_container_width=True)



with st.expander('About', expanded=True):
    st.write('''
        ---
        ''')
