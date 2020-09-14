import os
import fire
import pickle
import numpy as np
import pandas as pd
from scipy import interp
from sklearn.externals import joblib
from sklearn.preprocessing import normalize
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
from collections import defaultdict
from preprocess import load_json
from preprocess import load_benchmark_dataset

def binary_classifier_evaluation(TP,TN,FP,FN):
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN) # Recall
    # Specificity or true negative rate
    TNR = TN/(TN+FP) # Specificity
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)

    def MatthewsCorrelationCoefficient(TP,TN,FP,FN):
        a = np.multiply(TP,TN)
        b = np.multiply(FP,FN)
        c = np.multiply(TP+FP, TP+FN)
        d = np.multiply(TN+FP, TN+FN)
        mcc = (a-b)/np.sqrt(np.multiply(c,d))
        return mcc

    MCC = MatthewsCorrelationCoefficient(TP,TN,FP,FN)
    J = TPR + TNR - 1

    return (TPR, TNR, PPV, NPV, FPR, FNR, FDR, ACC, MCC, J)


class RandomForestModel():
    def __init__(self, tag="rf", min_trees=35, max_trees=200, n_jobs=20):
        self.tag = tag
        self.n_jobs = n_jobs
        self.num_tree_range = range(min_trees, max_trees)
        self.model_save_path = "classifiers/{}".format(tag)
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        else:
            os.system("rm -rf {}/*".format(self.model_save_path))
            
    def init_model(self):
        num_trees = np.random.choice(self.num_tree_range)
        model = RandomForestClassifier(class_weight = "balanced",
                                       n_estimators = num_trees,
                                       n_jobs       = self.n_jobs,
                                       max_features = "sqrt")
        return model, num_trees
    
    def train(self, dataset, num_models=1000, is_save=True):
        model_logs = "classifiers/{}/performances.csv".format(self.tag)
        model_log_dict = defaultdict(lambda: [])
        trainset, testset = dataset
        p_train, r_train, y_train = trainset
        p_test, r_test, y_test = testset
        x_train = np.concatenate([p_train, r_train], axis=-1)
        x_test  = np.concatenate([p_test, r_test], axis=-1)
        model_paths = []
        for i in range(num_models):
            model, num_trees = self.init_model()
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
                
            TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
            TPR, TNR, PPV, NPV, FPR, FNR, FDR, ACC, MCC, J = binary_classifier_evaluation(TP, TN, FP, FN)
            model_log_dict["Sensitivity"].append(TPR)
            model_log_dict["Specificity"].append(TNR)
            model_log_dict["Positive Predictive Value"].append(PPV)
            model_log_dict["Negative Predictive Value"].append(NPV)
            model_log_dict["False Positive Rate"].append(FPR)
            model_log_dict["False Negative Rate"].append(FNR)
            model_log_dict["False Discovery Rate"].append(FDR)
            model_log_dict["Accuracy"].append(ACC)
            model_log_dict["Matthews Correlation Coefficient"].append(MCC)
            model_log_dict["Youden's Index"].append(J)
            save_path = "{}/mcc{:2.3f}-ppv{:2.3f}-acc{:2.3f}-sn{:2.3f}-sp{:2.3f}-npv{:2.3f}-yd{:2.3f}-{}trees".format(self.model_save_path,MCC,PPV,ACC,TPR,TNR,NPV,J,num_trees)            
            if is_save:
                joblib.dump(model, save_path)
            model_paths.append(save_path)
            if i % 10 == 0 :print(i, save_path)
        
        model_log_df = pd.DataFrame.from_dict(model_log_dict)
        model_log_df.to_csv(model_logs)
        return model_log_df, model_paths
    
    

def main(dataset_dir, # ex. datasets/leeandhan2019 (benchmark dataset dir) 
         tag,         # ex. rf-ictf-leeandhan2019 (model name)
         min_trees,   # ex. 35  (minimum number of trees in Random Forest algorithm)
         max_trees,   # ex. 200 (maximum number of trees in Random Forest algorithm)
         n_jobs,      # ex. 10 (number of process for multiprocessing)
         num_models   # ex. 1000 (number of models to select best one)
        ):
    
    # Initialize (json format) dataset path
    train_json_path = "{}/train.json".format(dataset_dir)
    test_json_path  = "{}/test.json".format(dataset_dir)
    
    # Check files exsit
    if not os.path.exists(train_json_path): 
        raise ValueError("Dataset not exsits in, {}".format(train_json_path))
    if not os.path.exists(test_json_path): 
        raise ValueError("Dataset not exsits in, {}".format(test_json_path))
        
    # Parse sequences and preprocessing
    trainset = load_benchmark_dataset(train_json_path)
    testset  = load_benchmark_dataset(test_json_path)
    dataset  = (trainset, testset)
    
    # Run
    model = RandomForestModel(tag, min_trees, max_trees, n_jobs)
    model.train(dataset, num_models)
    
    return
    
    
if __name__ == "__main__":
    fire.Fire(main)