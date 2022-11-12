# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause


# PART: library dependencies -- sklear, torch, tensorflow, numpy, transformers

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics,tree
import pdb

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from utils import (
    preprocess_digits,
    train_dev_test_split,
    h_param_tuning,
    data_viz,
    pred_image_viz,
    get_all_h_param_comb_svm,
    get_all_h_param_comb_dt,
    tune_and_save,
    comparision_table
)
from joblib import dump, load

train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1
assert train_frac + dev_frac + test_frac == 1.0

# 1. set the ranges of hyper parameters
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

svm_params = {}
svm_params["gamma"] = gamma_list
svm_params["C"] = c_list
svm_h_param_comb = get_all_h_param_comb_svm(svm_params)

max_depth_list = [10,20,50,100]

dec_params = {}
dec_params["max_depth"] = max_depth_list
dec_h_param_comb = get_all_h_param_comb_dt(dec_params)

h_param_comb = {"svm":svm_h_param_comb, "decision_tree":dec_h_param_comb }
#h_param_comb = get_all_h_param_comb(params)


# PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()
data_viz(digits)
data, label = preprocess_digits(digits)


# define the evaluation metric
metric = metrics.accuracy_score
 
n_c = 5
results = {}
predictions_dict ={}

for n in range(n_c):    
    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
        data, label, train_frac, dev_frac
    )


    #
    # PART: Define the model
    # Create a classifier: a support vector classifier
    models_of_choice = {
        "svm":svm.SVC(), 
        "decision_tree":tree.DecisionTreeClassifier()
    }

    for clf_name in models_of_choice:
        clf = models_of_choice[clf_name]
        actual_model_path = tune_and_save(
            clf, x_train, y_train, x_dev, y_dev, metric, h_param_comb[clf_name], model_path=None
        )



        # 2. load the best_model
        best_model = load(actual_model_path)

        # PART: Get test set predictions
        # Predict the value of the digit on the test subset
        predicted = best_model.predict(x_test)
        if not clf_name in results:
            results[clf_name]=[]
        results[clf_name].append(metric(y_pred=predicted, y_true=y_test))
        

        pred_image_viz(x_test, predicted)

        if not clf_name in predictions_dict:
            predictions_dict[clf_name]=[]    
        predictions_dict[clf_name].append(predicted)

        # 4. report the test set accurancy with that best model.
        # PART: Compute evaluation metrics
        print(
            f"Classification report for classifier {clf}:\n"
            f"{metrics.classification_report(y_test, predicted)}\n"
        )


print(results)
print("-"*100)

# creating table for comarision
new_df,mean_val,std_val = comparision_table(results,n_c)
print("The comparision table with mean and standard deviation is:")
print(new_df.to_string(index=False))
print("-"*100)

# calculating confusion matrix
p_svm = predictions_dict['svm'][-1]
p_dt = predictions_dict['decision_tree'][-1]

confusion_matrix_svm = metrics.confusion_matrix(y_test, p_svm)
print("The svm confusion matrix is:\n",confusion_matrix_svm)

confusion_matrix_dt = metrics.confusion_matrix(y_test, p_dt)
print("The decision tree confusion matrix is:\n",confusion_matrix_dt)

print("-"*100)
# classification report
print(
    f"Classification report for svm classifier:\n"
    f"{metrics.classification_report(y_test, p_svm)}\n"
)
print(
    f"Classification report for decision treeclassifier:\n"
    f"{metrics.classification_report(y_test, p_dt)}\n"
)


# tsne for ploting 
tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(x_test)

target_ids = range(len(digits.target_names))

figure, axis = plt.subplots(2, 2)
figure.set_figheight(20)
figure.set_figwidth(20)

colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
for i, c, label in zip(target_ids, colors, digits.target_names):
    axis[0,0].scatter(X_2d[y_test == i, 0], X_2d[y_test == i, 1], c=c, label=label)
axis[0,0].legend()
axis[0, 0].set_title("True labels")


colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
for i, c, label in zip(target_ids, colors, digits.target_names):
    axis[0,1].scatter(X_2d[y_test == i, 0], X_2d[y_test == i, 1], c=c, label=label)
axis[0,1].legend()
axis[0,1].set_title("True labels")


colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
for i, c, label in zip(target_ids, colors, digits.target_names):
    axis[1,0].scatter(X_2d[p_svm == i, 0], X_2d[p_svm == i, 1], c=c, label=label)
axis[1,0].legend()
axis[1,0].set_title("SVM predicted labels")



colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
for i, c, label in zip(target_ids, colors, digits.target_names):
    axis[1,1].scatter(X_2d[p_dt == i, 0], X_2d[p_dt == i, 1], c=c, label=label)
axis[1,1].legend()
axis[1,1].set_title("Decision Tree predicted labels")

plt.savefig('complete_plot.png')
plt.show()
