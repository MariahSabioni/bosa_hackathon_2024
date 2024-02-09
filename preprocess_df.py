
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

def preprocess_df(df):

    df.rename(columns={'name':'athlete', 'race_id':'race', 'heat':'round'}, inplace=True)
    pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['split'] = df['hurdle_id'].astype('str')

    # Find PB
    df['total_time'] = df.groupby(['athlete', 'race'])['hurdle_timing'].transform('max')
    pb = df.groupby(['athlete'])['total_time'].transform('min') == df['total_time']
    df['is_pb'] = pb

    # Find stride diff
    df['stride_diff'] = df.groupby(['athlete', 'race'])['strides'].transform('min')
    df['stride_diff'] = df['strides'] - df['stride_diff']

    # Calculate the first quartile (Q1) and third quartile (Q3)
    Q1 = df['total_time'].quantile(0.25)
    Q3 = df['total_time'].quantile(0.75)

    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1

    # Define the lower and upper bounds to identify outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter the DataFrame to exclude outliers
    df_filt = df[(df['total_time'] >= lower_bound) & (df['total_time'] <= upper_bound)].copy().reset_index()

    # Extract the column for clustering
    X = df_filt[['total_time']]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=7, random_state=42)
    kmeans.fit_predict(X_scaled)
    df['cluster'] = kmeans.predict(scaler.transform(df[['total_time']]))
    df_filt['cluster'] = kmeans.predict(X_scaled)

    cluster_dict = {}
    for cluster, centroid in enumerate(scaler.inverse_transform(kmeans.cluster_centers_)):
        cluster_dict[cluster] = centroid[0]
    cluster_dict

    df['cluster_centroid'] = df['cluster']
    df['cluster_centroid'] = df['cluster_centroid'].map(cluster_dict)

    df_filt['cluster_centroid'] = df_filt['cluster']
    df_filt['cluster_centroid'] = df_filt['cluster_centroid'].map(cluster_dict)

    for cluster in df.cluster.unique():
        print(df[df['cluster']==cluster]['total_time'].describe())
        
    # Group by 'cluster' and 'split', then compute statistics
    stats = df_filt.groupby(['cluster', 'split'])['interval'].agg(['mean', 'median', 'std', 'quantile'])

    # Rename columns
    stats.columns = ['mean_intervals', 'median_intervals', 'std_intervals', 'quantile_intervals']

    # Merge statistics back to the original DataFrame
    df = pd.merge(df, stats, left_on=['cluster', 'split'], right_on=['cluster', 'split'], how='left')

    # Group by 'cluster' and 'split', then compute statistics
    stats = df_filt.groupby(['cluster', 'split'])['velocity'].agg(['mean', 'median', 'std', 'quantile'])

    # Rename columns
    stats.columns = ['mean_velocity', 'median_velocity', 'std_velocity', 'quantile_velocity']

    # Merge statistics back to the original DataFrame
    df = pd.merge(df, stats, left_on=['cluster', 'split'], right_on=['cluster', 'split'], how='left')

    # Compute feature importance within clusters
    df_pivot = df.pivot(index=['athlete', 'competition', 'round', 'race', 'total_time', 'cluster'], columns='split', values='interval',).reset_index()

    # Separate data into X (features) and y (target)
    feature_importances_clusters=[]
    for cluster in df_pivot.cluster.unique():
        df_feat = df_pivot[df_pivot['cluster']==cluster]
        X = df_feat.drop(columns=['athlete','competition','round', 'race', 'total_time','cluster' ]) # Features excluding 'total_time' and 'split'
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        y = df_feat['total_time']  # Target variable

        # Initialize and fit the Random Forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_scaled, y)

        # Get feature importances
        feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)

        for split, feature in enumerate(feature_importances):
            feature_importances_clusters.append({
                'cluster':cluster,
                'split':split+1,
                'feature_importance':feature
            })

    df_importances_rf = pd.DataFrame(feature_importances_clusters, )
    df_importances_rf['split']=df_importances_rf['split'].astype('str')

    df = pd.merge(df, df_importances_rf, left_on=['cluster', 'split'], right_on=['cluster', 'split'], how='left')

    df.to_csv('data/400_m_hurdles_processed.csv')

    return df

def make_cluster_table(clusters):

    cluster_subset = clusters[['hurdle_id', 'cluster', 'cluster_centroid', 'median_intervals', 'std_intervals']]
    cluster_subset[['median_intervals', 'std_intervals']] = cluster_subset[['median_intervals', 'std_intervals']].round(2)
    cluster_subset.head()

    # Format the variables with the maximum number of decimals
    formatted_median = cluster_subset['median_intervals'].apply(lambda x: f"{x:.2f}")
    formatted_sd = cluster_subset['std_intervals'].apply(lambda x: f"{x:.2f}")

    cluster_subset['split_sd'] = formatted_median + '+/-' + formatted_sd

    cluster_subset['uniqueID'] = cluster_subset['cluster'].astype(str) + '_' + cluster_subset['hurdle_id'].astype(str)
    cluster_subset.head()

    nodups = cluster_subset[~cluster_subset['uniqueID'].duplicated()]
    nodups

    nodups_wide = pd.pivot_table(nodups,
                                values='split_sd',
                                index='cluster_centroid',
                                columns='hurdle_id',
                                aggfunc='first')  # You can choose an appropriate aggregation function

    # Reset the index if needed
    nodups_wide.reset_index(inplace=True)

    nodups_wide[['cluster_centroid',1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]] = nodups_wide[['cluster_centroid',1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]].round(2)

    # Display the wide-format DataFrame
    print(nodups_wide)

    # print(nodups_wide.columns)
    nodups_wide.rename(columns={'cluster_centroid': 'Final Time',
                                1: '[start-H1]',
                                2: '[H1-H2]',
                                3: '[H2-H3]',
                                4: '[H3-H4]',
                                5: '[H4-H5]',
                                6: '[H5-H6]',
                                7: '[H6-H7]',
                                8: '[H7-H8]',
                                9: '[H8-H9]',
                                10: '[H9-H10]',
                                11: '[H10-finish]'}, inplace=True)

    return nodups_wide
