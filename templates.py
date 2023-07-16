from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import plotly.express as px
import numpy as np
        

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