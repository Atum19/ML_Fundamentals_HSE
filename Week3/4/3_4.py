import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
 f1_score, roc_auc_score, precision_recall_curve


clasf_data = pd.read_csv('classification.csv', sep=',')
scor_data = pd.read_csv('scores.csv', sep=',')

def max_prec_val(prec_rec_vals):
    prec_vals =[]
    for pos, elem in enumerate(prec_rec_vals[1]):
        if elem >= 0.7:
            prec_vals.append(prec_rec_vals[0][pos])        
    return max(prec_vals)

TP, FP, FN, TN = 0, 0, 0, 0
matr_data = clasf_data.as_matrix(columns=clasf_data.columns[0:])
for line in matr_data:
    if line[0] == 1 and line[1] == 1:
        TP += 1
    elif line[0] == 0 and line[1] == 1:
        FP += 1
    elif line[0] == 1 and line[1] == 0:
        FN += 1
    elif line[0] == 0 and line[1] == 0:
        TN += 1

print('TP: ', TP, ' FP: ', FP, ' FN: ', FN, ' TN: ', TN)
print('************************************************')

y = clasf_data['true']
x = clasf_data['pred']
print('Accuracy: ', accuracy_score(y, x))
print('Precision: ', precision_score(y, x))
print('Recall: ', recall_score(y, x))
print('F-мера: ', f1_score(y, x))
print('################################################')

ys = scor_data['true']
logreg = scor_data['score_logreg']
svm = scor_data['score_svm']
knn = scor_data['score_knn']
tree = scor_data['score_tree']
print('AUC-ROC logreg: ', roc_auc_score(ys, logreg))
print('AUC-ROC svm: ', roc_auc_score(ys, svm))
print('AUC-ROC knn: ', roc_auc_score(ys, knn))
print('AUC-ROC tree: ', roc_auc_score(ys, tree))
print('================================================')

print('max prec logreg: ', max_prec_val(precision_recall_curve(ys, logreg)))
print('max prec svm: ', max_prec_val(precision_recall_curve(ys, svm)))
print ('max prec knn: ', max_prec_val(precision_recall_curve(ys, knn)))
print ('max prec tree: ', max_prec_val(precision_recall_curve(ys, tree)))
