import sys
import pdb
import os
from joblib import load


sys.path.append('.')

from utils import get_all_h_param_comb, train_save_model, preprocess_digits, data_viz
from sklearn import datasets
# test case to check if all the combinations of the hyper parameters are indeed getting created
def test_get_h_param_comb():
    gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

    params = {}
    params['gamma'] = gamma_list
    params['C'] = c_list
    h_param_comb = get_all_h_param_comb(params)

    assert len(h_param_comb) == len(gamma_list) * len(c_list)


# train/dev/test split functionality : input 200 samples, fraction is 70:15:15, then op should have 140:30:30 samples in each set

# - some test cases that will validate if models are indeed getting saved or not.
# step1: train on small data set, provide a path to save trained model
# step2: assert if the file exists at the provided path
# step3: assert if the file is indeed a scikit learn model
# step4: optionally checksome  validate the md5


# PART: load dataset -- data from csv, tsv, jsonl, pickle


# for i in range(500):
#     if i<250:
#         label[i]=0
#     else
 

def test_check_model_saving():
    model_path = 'pathh'
    digits = datasets.load_digits()
    data_viz(digits)
    data, label = preprocess_digits(digits)
    sample_data = data[:500]  
    sample_label = label[:500]

    for i in range(500):
        if i<250:
            sample_label[i] = 0
        else:
            sample_label[i] = 1
    
    #pdb.set_trace()
    gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

    params = {}
    params['gamma'] = gamma_list
    params['C'] = c_list
    h_param_comb = get_all_h_param_comb(params)
    
    actual_model_path,clf,_,_,_ = train_save_model(sample_data,sample_label,sample_data,sample_label,model_path,h_param_comb)
    assert actual_model_path == model_path
    assert os.path.exists(model_path)
    loaded_model = load(model_path)
    assert type(loaded_model) == type(clf)
      








#what more test cases should be there 
#irrespective of the changes to the refactored code.

# train/dev/test split functionality : input 200 samples, fraction is 70:15:15, then op should have 140:30:30 samples in each set


# preprocessing gives ouput that is consumable by model

# accuracy check. if acc(model) < threshold, then must not be pushed.

# hardware requirement test cases are difficult to write.
# what is possible: (model size in execution) < max_memory_you_support

# latency: tik; model(input); tok == time passed < threshold
# this is dependent on the execution environment (as close the actual prod/runtime environment)


# model variance? -- 
# bias vs variance in ML ? 
# std([model(train_1), model(train_2), ..., model(train_k)]) < threshold


# Data set we can verify, if it as desired
# dimensionality of the data --

# Verify output size, say if you want output in certain way
# assert len(prediction_y) == len(test_y)

# model persistance?
# train the model -- check perf -- write the model to disk
# is the model loaded from the disk same as what we had written?
# assert acc(loaded_model) == expected_acc 
# assert predictions (loaded_model) == expected_prediction 