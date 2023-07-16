from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import plotly.express as px
import numpy as np
        
        
def show_confusion_matrix(model, y_test,y_pred, normalize_matrix):
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
#show_confusion_matrix(clf_rf, y_test, y_pred, normalize_matrix = 'no')