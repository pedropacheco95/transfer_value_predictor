import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

def remove_outliers(df,column,min_value,max_value):
    z_scores = stats.zscore(df[column])
    large_outlier_mask = (z_scores > max_value)
    small_outlier_mask = (z_scores < min_value)
    outlier_mask = large_outlier_mask | small_outlier_mask
    return df[~outlier_mask]

def outlier_detection(dataframe,threshold):
    Q1 = dataframe.quantile(0.25)
    Q3 = dataframe.quantile(0.75)
    IQR = Q3 - Q1

    outlier_percentage_threshold = threshold
    columns_to_drop = []
    for column in dataframe.columns:
        outliers = ((dataframe[column] < (Q1[column] - 1.5 * IQR[column])) | 
                    (dataframe[column] > (Q3[column] + 1.5 * IQR[column])))
        outlier_percentage = outliers.mean()
        if outlier_percentage > outlier_percentage_threshold:
            columns_to_drop.append(column)
    return columns_to_drop

def remove_columns_with_to_many_outliers(dataframe):
    columns_to_drop = outlier_detection(dataframe,0.65)
    return dataframe.drop(columns_to_drop, axis=1)

def highly_correlated_columns(df, threshold=0.95,plot_graph=False):
    corr_matrix = df.corr()
    if plot_graph:
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, center=0, annot=False, fmt=".2f", cmap='coolwarm',
                    square=True, linewidths=.5, cbar_kws={"shrink": .1}, 
                    xticklabels=False, yticklabels=False) 

        plt.title("Correlation Matrix")
        output_directory = 'images/'
        output_filename = 'correlation.png'
        plt.savefig(output_directory + output_filename)
        plt.close()

    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    return to_drop

def remove_highly_correlated_columns(dataframe,threshold=0.95):
    columns_to_drop = highly_correlated_columns(dataframe,threshold=threshold,plot_graph=True)
    return dataframe.drop(columns_to_drop, axis=1)

def perform_pca(dataframe, variance_threshold=0.9, plot_graph=False):
    """
    Perform PCA on the dataset and optionally plot the graph.
    Chooses the number of components such that the cumulative explained variance is at least 'variance_threshold'.

    Parameters:
    data (pd.DataFrame): The dataset.
    variance_threshold (float, optional): Threshold for cumulative explained variance. Default is 0.9 (90%).
    plot_graph (bool, optional): If True, plot the PCA graph. Default is False.
    """

    # Standardizing the features
    x = StandardScaler().fit_transform(dataframe)

    # Performing PCA to determine the number of components
    pca_full = PCA()
    pca_full.fit(x)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.where(cumulative_variance >= variance_threshold)[0][0] + 1

    pca = PCA(n_components=n_components)
    pca.fit(x)

    # Calculate the absolute value of loadings (contribution of each feature to each PC)
    loadings = np.abs(pca.components_.T)

    # Sum the loadings for each feature across all PCs
    feature_contributions = np.sum(loadings, axis=1)

    # Sort features based on their contributions
    sorted_contributions = np.argsort(feature_contributions)

    # Identify columns to drop (those with the least contribution)
    columns_to_drop = dataframe.columns[sorted_contributions[:len(sorted_contributions) - n_components]]

    if plot_graph:
        plt.figure(figsize=(8, 5))
        sns.set(style='darkgrid')
        plt.plot(cumulative_variance)
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Scree Plot')
        plt.axhline(y=variance_threshold, color='r', linestyle='--')
        plt.axvline(x=n_components - 1, color='r', linestyle='--')
        sns.despine()
        output_directory = 'images/'
        output_filename = 'pca.png'
        plt.savefig(output_directory + output_filename)
        plt.close()

    return dataframe.drop(columns_to_drop, axis=1)

def remove_least_important_random_forest_features(df, target_column, percent_to_remove=0.1, plot_graph=False):
    """
    Remove the least important features determined by a Random Forest Regressor.

    Parameters:
    df (pd.DataFrame): The dataset.
    target_column (str): The name of the target variable column.
    percent_to_remove (float): The percentage of least important features to remove (between 0 and 1).
    """
    features = df.drop(target_column, axis=1)
    target = df[target_column]

    # Splitting into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.1, random_state=42)

    # Training the Random Forest model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Getting feature importances
    feature_importances = model.feature_importances_

    # Creating a series with feature names and their importance
    importance_series = pd.Series(feature_importances, index=features.columns)

    if plot_graph:
        importance_series_sorted = importance_series.sort_values(ascending=False)
        sns.set(style="darkgrid")
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importance_series_sorted.head(10).index, 
                    y=importance_series_sorted.head(10).values)
        plt.xticks(rotation=45)
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.title("Top Important Features in Random Forest Model")
        output_directory = 'images/'
        output_filename = 'forest_features.png'
        plt.savefig(output_directory + output_filename)
        plt.close()

    # Sorting features by importance
    importance_series = importance_series.sort_values()

    # Determining the number of features to remove
    num_features_to_remove = int(len(importance_series) * percent_to_remove)

    # Identifying the least important features
    features_to_remove = importance_series.head(num_features_to_remove).index

    # Removing the least important features from the DataFrame
    df_reduced = df.drop(features_to_remove, axis=1)

    return df_reduced

def remove_least_important_linear_regression_features(df, target_column, percent_to_remove=0.1, plot_graph=False):
    """
    Remove the least important features determined by RFE with Linear Regression and plot the most important ones.

    Parameters:
    df (pd.DataFrame): The dataset.
    target_column (str): The name of the target variable column.
    percent_to_remove (float): The percentage of least important features to remove (between 0 and 1).
    num_features_to_plot (int): The number of most important features to plot.
    """
    # Splitting the data into features and target
    features = df.drop(target_column, axis=1)
    target = df[target_column]

    # Splitting into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.1, random_state=42)

    # Creating a Linear Regression model for RFE
    model = LinearRegression()

    # Creating an RFE selector with the determined number of features
    rfe = RFE(model)
    rfe.fit(X_train, y_train)

    # Getting ranking of features
    feature_ranking = rfe.ranking_

    if plot_graph:
        features_to_plot = features.columns[(feature_ranking <= percent_to_remove) & (feature_ranking != 1)]
        top_features_to_plot = features_to_plot[:10]
        if len(top_features_to_plot) > 0:
            sns.set(style="darkgrid")
            plt.figure(figsize=(10, 6))
            top_n_importance = 1 / feature_ranking[features.columns.isin(top_features_to_plot)]
            sns.barplot(x=top_features_to_plot, y=top_n_importance)
            plt.xticks(rotation=45)
            plt.xlabel("Features")
            plt.ylabel("Relative Importance")
            plt.title(f"Important Features After RFE (Excluding Top Rank)")
            output_directory = 'images/'
            output_filename = 'liner_regression_features.png'
            plt.savefig(output_directory + output_filename)
            plt.close()

    feature_ranking = rfe.ranking_
    num_features_to_remove = int(percent_to_remove * len(feature_ranking))

    sorted_features = sorted(enumerate(features.columns), key=lambda x: feature_ranking[x[0]])
    worst_features = [feature_name for _, feature_name in sorted_features[:num_features_to_remove]]

    return df.drop(worst_features, axis=1)

def reduce_dimensionality(df):
    target_column_name = 'Transfer Value'
    df = df.sort_values(by='Transfer Date', ascending=True)
    df = df.select_dtypes(include=['number'])

    df = remove_outliers(df,target_column_name,-1,50)
    
    df = remove_least_important_random_forest_features(df, target_column_name, percent_to_remove=0.01, plot_graph=True)
    df = remove_least_important_linear_regression_features(df, target_column_name, percent_to_remove=0.01, plot_graph=True)

    target_column = df.pop(target_column_name)
    
    df = perform_pca(df,.99,plot_graph=True)

    df[target_column_name] = target_column

    return df