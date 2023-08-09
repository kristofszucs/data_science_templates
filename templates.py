from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import plotly.express as px
import pandas as pd
import numpy as np
import missingno as msno

# Set the display format for decimal places
pd.set_option('display.float_format', '{:.2f}'.format)

import warnings
warnings.filterwarnings('ignore')
        

def show_histogram(df, x, bins = None, decimals = '.1f', log_y = False, uni_color = '#00b3ff', width = 500):
    """
    Displays a histogram using the Plotly library.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - x (str): The column name from the DataFrame to plot the histogram.
    - bins (int or None, optional): The number of bins to use in the histogram. If None, an automatic bin calculation is used. Default is None.
    - decimals (str, optional): The format string for displaying the bin counts as text. Default is '.1f'.
    - log_y (bool, optional): Whether to use a logarithmic scale on the y-axis. Default is False.
    - uni_color (str, optional): The color code for the histogram bars. Default is '#00b3ff'.
    - width (int, optional): The width of the histogram plot in pixels. Default is 500.

    Returns:
    - None: This function does not return any value but prints the graph.

    Dependencies:
    - pandas
    - plotly.express (imported as px)
    """
    fig = px.histogram(df, 
                        x=x,
                        text_auto = decimals,
                        nbins = bins,
                        log_y = log_y,
                        color_discrete_sequence = [uni_color],
                        width=width,
                        height=400)
    fig.update_xaxes(title=x)
    fig.update_yaxes(title='Count')
    fig.update_traces(textposition='outside')
    fig.update_layout(plot_bgcolor='rgba(0, 0, 0, 0)',paper_bgcolor='rgba(0, 0, 0, 0)', margin=dict(l=50, r=50, t=50, b=50))
    fig.show()
#show_histogram(df, x="Sex")
#show_histogram(df, x="Age", bins=30, decimals = '.2f', log_y = False, uni_color = 'Blue', width = 700)

   
def show_model_evaluation(model, y_test,y_pred, normalize_matrix):
    """
    Displays the confusion matrix and classification report for a given model's predictions.

    Parameters:
        - model: The trained model or estimator.
        - y_test: The true labels of the test set.
        - y_pred: The predicted labels of the test set.
        - normalize_matrix: A string indicating whether to normalize the confusion matrix.
                            Possible values: 'yes' or 'no'.

    Returns:
        None

    Example usage:
        show_confusion_matrix(clf_rf, y_test, y_pred, normalize_matrix='no')

    Import necessary libraries :
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import classification_report
        import plotly.express as px
        import numpy as np
    """
    cm = []
    if normalize_matrix == 'yes':
        cm = confusion_matrix(y_test,y_pred, normalize_matrix='all')
        cm = np.around(cm, decimals=2)
    if normalize_matrix == 'no':
        cm = confusion_matrix(y_test,y_pred)
    fig = px.imshow(cm,
                    text_auto=True,
                    color_continuous_scale='Blues',
                    x = list(map(str, model.classes_)),
                    y = list(map(str, model.classes_)))
    fig.update_layout(title='Confusion Matrix <br><sup>'+str(model.estimator)+'',
                        width=450,
                        height=400)
    fig.update_xaxes(title='Predicted Labels')
    fig.update_yaxes(title='True Labels')
    fig.show()

    print("Classification report : ")
    print(classification_report(y_test, y_pred))

# Example (clf_rf needs to be trained and y_pred predicted)
#show_model_evaluation(clf_rf, y_test, y_pred, normalize_matrix = 'no')

def check_missing_values(df, matrix = None, bar = None):
    """
    Analyzes a DataFrame for missing values and displays a styled summary.

    This function calculates the count and percentage of missing values for each column in the given DataFrame.
    It then creates a summary DataFrame containing the column name, the count of missing values, and the percentage
    of missing values. The summary is styled to highlight columns with a percentage of missing values greater than 10%.

    Additionally, this function provides the option to visualize missing data using the missingno library's matrix and bar plots.

    Parameters:
    df (pandas.DataFrame): The input DataFrame to be analyzed.
    matrix (str, optional): If set to 'Yes', displays the missingno matrix plot.
    bar (str, optional): If set to 'Yes', displays the missingno bar plot.

    Returns:
    pandas.io.formats.style.Styler: A styled summary of missing values highlighting columns with high percentages.
    """
    print("The size of the dataset is : " + str(df.shape))
    d = pd.DataFrame(df.isna().sum(), columns=['nb_NA'])
    d['pct_NA'] = (d.nb_NA/df.shape[0])*100
    if matrix == 'Yes':
        print(msno.matrix(df))
    if bar == 'Yes':
        print(msno.bar(df))
    return d.style.applymap(lambda x: 'background-color: yellow' if x > 10 else '', subset=pd.IndexSlice[:, 'pct_NA'])

#check_missing_values(df, matrix = 'Yes', bar = 'No') 