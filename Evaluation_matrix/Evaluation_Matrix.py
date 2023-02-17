from sklearn import metrics

def evaluation_matrix(y_test, y_pred, target_names):
    #target_names = ['Epidemiology', 'Public_health', 'Virology', 'Influenza', 'Policies', 'Vaccines_or_Immunology', 'Ulmonary_infections']
    print(metrics.classification_report(y_test, y_pred, target_names=target_names))