
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def confusion_matrix_(y_test, y_pred, category_id_df):
    conf_mat = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(conf_mat, annot=True, fmt='d',
                xticklabels=category_id_df.tag_target_class.values, yticklabels=category_id_df.tag_target_class.values)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()