import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.linear_model import LinearRegression
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def load_data(data_path: str) -> pd.DataFrame:
    """
    Loads data from a CSV file into a pandas DataFrame.
     :param data_path: The path to the CSV file.
    :return: A pandas DataFrame containing the data from the CSV file.
    """
    df = pd.read_csv(data_path)
    return df
def interactive_histogram(dataframe: pd.DataFrame, column_name: str) -> None:
    """
    Displays an interactive histogram of a column in the DataFrame using Plotly.
     :param dataframe: The pandas DataFrame.
    :param column_name: The name of the column to display.
    """
    fig = px.histogram(dataframe, x=column_name, nbins=20, color_discrete_sequence=['#636EFA'])
    fig.update_layout(title=f"Distribution of {column_name}", xaxis_title=column_name, yaxis_title='Count')
    fig.show()
def seaborn_histogram(dataframe: pd.DataFrame, column_name: str) -> None:
    """
    Displays a histogram of a column in the DataFrame using Seaborn.
     :param dataframe: The pandas DataFrame.
    :param column_name: The name of the column to display.
    """
    plt.figure(figsize=(10, 6))
    sns.distplot(dataframe[column_name], kde=False, hist_kws=dict(edgecolor="k", linewidth=1))
    plt.title(f"Distribution of {column_name}")
    plt.show()
def seaborn_boxplot(dataframe: pd.DataFrame, x_column: str, y_column: str) -> None:
    """
    Displays a boxplot of a column in the DataFrame using Seaborn.
     :param dataframe: The pandas DataFrame.
    :param x_column: The name of the column to display on the x-axis.
    :param y_column: The name of the column to display on the y-axis.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=dataframe, x=x_column, y=y_column)
    plt.title(f"Distribution of {y_column} by {x_column}")
    plt.show()
def interactive_correlation_matrix(dataframe: pd.DataFrame) -> None:
    """
    Displays an interactive correlation matrix of the DataFrame using Plotly.
     :param dataframe: The pandas DataFrame.
    """
    fig = go.Figure(data=go.Heatmap(
        z=dataframe.corr(),
        x=dataframe.columns,
        y=dataframe.columns,
        colorscale='Blues',
        zmin=-1,
        zmax=1
    ))
    fig.update_layout(title="Correlation Matrix")
    fig.show()
def interactive_categorical_count(dataframe: pd.DataFrame, column_name: str) -> None:
    """
    Displays an interactive count of a categorical column in the DataFrame using Plotly.
     :param dataframe: The pandas DataFrame.
    :param column_name: The name of the categorical column.
    """
    fig = px.histogram(dataframe, x=column_name, nbins=20, color_discrete_sequence=['#636EFA'])
    fig.update_layout(title=f"{column_name} Count", xaxis_title=column_name, yaxis_title='Count')
    fig.show()
def visualize_report(dataframe: pd.DataFrame, lang: str = 'en') -> None:
    """
    Generates an interactive report for a given pandas DataFrame.
     :param dataframe: The pandas DataFrame to generate a report for.
    :param lang: The language to use for text analysis (defaults to 'en' for English).
    """
    if lang != 'en':
        stop_words = set(stopwords.words(lang))
    else:
        stop_words = set(stopwords.words('english'))
     # Iterate through columns and visualize data
    for column in dataframe.columns:
        if dataframe[column].dtype == "O":
            interactive_categorical_count(dataframe, column)
            if lang != 'en':
                text = ' '.join(dataframe[column]).lower()
                text = ' '.join([word for word in text.split() if word not in stop_words])
                freq_dist = nltk.FreqDist(text.split())
                freq_dist.plot(20, cumulative=False)
            else:
                freq_dist = nltk.FreqDist(dataframe[column])
                freq_dist.plot(20, cumulative=False)
        else:
            interactive_histogram(dataframe, column)
            seaborn_histogram(dataframe, column)
            if len(dataframe[column].unique()) <= 10:
                seaborn_boxplot(dataframe, column, dataframe.columns[-1])
     # Visualize correlation matrix
    interactive_correlation_matrix(dataframe)
     # Perform clustering and visualize data points
    if len(dataframe.columns) > 1:
        standardized_data = StandardScaler().fit_transform(dataframe.select_dtypes(include=["number"]))
        pca = PCA(n_components=2)
        pca.fit(standardized_data)
        reduced_data = pca.transform(standardized_data)
        kmeans = KMeans(n_clusters=3, random_state=0).fit(reduced_data)
        fig = px.scatter(
          reduced_data,
          x=0,
          y=1,
          color=kmeans.labels_,
          size=dataframe.iloc[:, -1],
          title="2D PCA Visualization"
        )
        fig.show()
        tukey_result = pairwise_tukeyhsd(dataframe.iloc[:, -1], kmeans.labels_)
        print(tukey_result)
     # Perform linear regression and visualize scatter plot
    if len(dataframe.columns) == 2:
        linreg = LinearRegression()
        linreg.fit(dataframe.iloc[:, :-1], dataframe.iloc[:, -1])
        fig = px.scatter(
          dataframe,
          x=dataframe.columns[0],
          y=dataframe.iloc[:, -1],
          trendline="ols",
          title=f"{dataframe.columns[0]} vs {dataframe.columns[-1]}"
        )
        fig.show()
