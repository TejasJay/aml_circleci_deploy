from xgboost import XGBClassifier
from application_logging import logger
import pandas as pd
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
import json
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix




class Model_Finder_new:
    """
    This class shall  be used to find the model with best accuracy and AUC score.
    Written By: Tejas Jay (TJ)
    Version: 1.0
    Revisions: None

    """

    def __init__(self):
        self.xgb = XGBClassifier(objective='binary:logistic')
        self.log_writer = logger.App_Logger()






    def model_for_gnb(self,train_x,train_y,test_x,test_y):

        """
        Method Name: model_for_gnb
        Description: get the parameters for Naive Bias Algorithm which give the best scores.
                    Use Hyper Parameter Tuning.
        Output: The model with the best parameters
        On Failure: Raise Exception

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None

        """
        self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.log_writer.log(self.file_object,
                               'Entered the model_for_gnb method of the Model_Finder class')
        self.file_object.close()
        try:
            # initializing the gaussianNB class
            gnb = GaussianNB()


            # applying gridsearchcv to find best parameters for the model
            param_grid = {"var_smoothing": [1e-9, 0.1, 0.001, 0.5, 0.05, 0.01, 1e-8, 1e-7, 1e-6, 1e-10, 1e-11]}
            # Creating an object of the Grid Search class
            grid = GridSearchCV(estimator=gnb, param_grid=param_grid, cv=5, verbose=3)

            # finding the best parameters
            grid.fit(train_x, train_y)

            # initializing NB class with best estimator and fitting the model with training df and predicting the results

            gnb_new =grid.best_estimator_

            pred_y_new = gnb_new.fit(train_x, train_y).predict(test_x)

            # finding cross validaion scores for 10 sample splits

            scoring = {'accuracy': make_scorer(accuracy_score),
                   'roc_auc': make_scorer(roc_auc_score),
                   'precision': make_scorer(precision_score),
                   'recall': make_scorer(recall_score),
                   'f1_score': make_scorer(f1_score)}

            gnb_cv_score = model_selection.cross_validate(gnb_new, train_x, train_y, cv=5, scoring=scoring)

            # outcome values order in sklearn
            print("Gaussian Naive Bias \n")

            tp_gnb, fn_gnb, fp_gnb, tn_gnb = confusion_matrix(test_y, pred_y_new, labels=[1, 0]).reshape(-1)
            print('Outcome values  \n', 'True positive: ' + str(tp_gnb) + '\n', 'false negetive: ' + str(fn_gnb) + '\n',
                  'false positive: ' + str(fp_gnb) + '\n', 'True negetive: ' + str(tn_gnb) + '\n')
            print('\n')

            print("=== Confusion Matrix ===")
            print(confusion_matrix(test_y, pred_y_new))
            print('\n')
            print("=== Classification Report ===")
            print(classification_report(test_y, pred_y_new))
            print('\n')

            # print("=== All Scores of gnb ===")
            # print(gnb_cv_score)
            gnb_results_df = pd.DataFrame(gnb_cv_score)
            # print(gnb_results_df)
            # print('\n')

            df_confusion_matrix_report_gnb = pd.DataFrame(
                [['True_positive: ' + str(tp_gnb), 'false_positive: ' + str(fp_gnb)],
                 ['false_negetive: ' + str(fn_gnb), 'True_negetive: ' + str(tn_gnb)]],
                index=['Predicted_Fraud', 'Predicted_Not_Fraud'],
                columns=['gnb_Actual (Fraud)', 'gnb_Actual (Not_Fraud)'])

            print(df_confusion_matrix_report_gnb)

            json_confusion_matrix_report_gnb = df_confusion_matrix_report_gnb.to_json(orient='index')

            # print(json_confusion_matrix_report)

            classification_report_gnb = classification_report(test_y, pred_y_new)
            splitting_gnb = classification_report_gnb.split('\n')
            parse_gnb = splitting_gnb[6].strip()
            parse_gnb = " ".join(parse_gnb.split())
            parse_gnb = list(parse_gnb.split(" "))

            print('\n')
            # print(parse)
            print('\n')

            precision_gnb = float(parse_gnb[2]), gnb_new, 'gnb', json_confusion_matrix_report_gnb
            recall_gnb = float(parse_gnb[3]), gnb_new, 'gnb', json_confusion_matrix_report_gnb
            F1_gnb = float(parse_gnb[4]), gnb_new, 'gnb', json_confusion_matrix_report_gnb

            print('\n')
            print("Precision: " + str(precision_gnb))
            print('\n')

            print('\n')
            print("Recall: " + str(recall_gnb))
            print('\n')

            print('\n')
            print("F1: " + str(F1_gnb))
            print('\n')

            scores_gnb = [precision_gnb, recall_gnb, F1_gnb]

            return scores_gnb

            # print(scores_gnb)


        except ZeroDivisionError as error:
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                'Exception occured in model_for_gnb method of the Model_Finder class. Exception message:  ' + str(
                                    error))
            self.log_writer.log(self.file_object,
                                'Model Selection Failed. Exited the model_for_gnb method of the Model_Finder class')
            self.file_object.close()
        except Exception as e:
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                'Exception occured in model_for_gnb method of the Model_Finder class. Exception message:  ' + str(
                                    e))
            self.log_writer.log(self.file_object,
                                'Model Selection Failed. Exited the model_for_gnb method of the Model_Finder class')
            self.file_object.close()






    def model_for_xgb(self, train_x, train_y, test_x, test_y):

        """
        Method Name: model_for_xgb
        Description: get the parameters for XGBoost Algorithm which give the best F1 score.
                    Use Hyper Parameter Tuning.
        Output: The model with the best parameters
        On Failure: Raise Exception

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None

        """
        self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.log_writer.log(self.file_object,
                            'Entered the model_for_xgb method of the Model_Finder class')
        self.file_object.close()

        try:
            # parameters for XGB gridsearchCV
            param_grid_xgboost = {"n_estimators": [50, 100, 130], "max_depth": range(3, 11, 1),
                                  "random_state": [0, 50, 100]}

            # Creating an object of the Grid Search class
            grid = GridSearchCV(XGBClassifier(objective='binary:logistic'), param_grid_xgboost, verbose=3, cv=5,
                                n_jobs=-1)

            # fitting the model with training data
            grid.fit(train_x, train_y)

            xgb_new = grid.best_estimator_

            # fitting the model with the training dataset and testing it.
            pred_y_xgb_new = xgb_new.fit(train_x, train_y).predict(test_x)

            scoring = {'accuracy': make_scorer(accuracy_score),
                       'roc_auc': make_scorer(roc_auc_score),
                       'precision': make_scorer(precision_score),
                       'recall': make_scorer(recall_score),
                       'f1_score': make_scorer(f1_score)}

            xgb_cv_score = model_selection.cross_validate(xgb_new, train_x, train_y, cv=5, scoring=scoring)

            # outcome values order in sklearn
            # outcome values order in sklearn
            print("xgb \n")

            tp_xgb, fn_xgb, fp_xgb, tn_xgb = confusion_matrix(test_y, pred_y_xgb_new, labels=[1, 0]).reshape(-1)
            print('Outcome values  \n', 'True positive: ' + str(tp_xgb) + '\n', 'false negetive: ' + str(fn_xgb) + '\n',
                  'false positive: ' + str(fp_xgb) + '\n', 'True negetive: ' + str(tn_xgb) + '\n')
            print('\n')

            print("=== Confusion Matrix ===")
            print(confusion_matrix(test_y, pred_y_xgb_new))
            print('\n')
            print("=== Classification Report ===")
            print(classification_report(test_y, pred_y_xgb_new))
            print('\n')

            # print("=== All Scores of gnb ===")
            # print(gnb_cv_score)
            xgb_results_df = pd.DataFrame(xgb_cv_score)
            # print(gnb_results_df)
            # print('\n')

            df_confusion_matrix_report_xgb = pd.DataFrame(
                [['True_positive: ' + str(tp_xgb), 'false_positive: ' + str(fp_xgb)],
                 ['false_negetive: ' + str(fn_xgb), 'True_negetive: ' + str(tn_xgb)]],
                index=['Predicted_Fraud', 'Predicted_Not_Fraud'],
                columns=['xgb_Actual (Fraud)', 'xgb_Actual (Not_Fraud)'])

            print(df_confusion_matrix_report_xgb)

            json_confusion_matrix_report_xgb = df_confusion_matrix_report_xgb.to_json(orient='index')

            # print(json_confusion_matrix_report)

            classification_report_xgb = classification_report(test_y, pred_y_xgb_new)
            splitting_xgb = classification_report_xgb.split('\n')
            parse_xgb = splitting_xgb[6].strip()
            parse_xgb = " ".join(parse_xgb.split())
            parse_xgb = list(parse_xgb.split(" "))

            print('\n')
            # print(parse)
            print('\n')

            precision_xgb = float(parse_xgb[2]), xgb_new, 'xgb', json_confusion_matrix_report_xgb
            recall_xgb = float(parse_xgb[3]), xgb_new, 'xgb', json_confusion_matrix_report_xgb
            F1_xgb = float(parse_xgb[4]), xgb_new, 'xgb', json_confusion_matrix_report_xgb

            print('\n')
            print("Precision: " + str(precision_xgb))
            print('\n')

            print('\n')
            print("Recall: " + str(recall_xgb))
            print('\n')

            print('\n')
            print("F1: " + str(F1_xgb))
            print('\n')

            scores_xgb = [precision_xgb, recall_xgb, F1_xgb]

            # print(scores_gnb)

            return scores_xgb


        except ZeroDivisionError as error:
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                'Exception occured in model_for_xgb method of the Model_Finder class. Exception message:  ' + str(
                                    error))
            self.log_writer.log(self.file_object,
                                'XGBoost Parameter tuning  failed. Exited the model_for_xgb method of the Model_Finder class')
            self.file_object.close()
            raise error
        except Exception as e:
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                   'Exception occured in model_for_xgb method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.log_writer.log(self.file_object,
                                   'XGBoost Parameter tuning  failed. Exited the model_for_xgb method of the Model_Finder class')
            self.file_object.close()
            raise e






    def model_for_rf(self, train_x, train_y, test_x, test_y):
        """
        Method Name: model_for_rf
        Description: Find out the Model which has the best score.
        Output: The best model name and the model object
        On Failure: Raise Exception

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None

        """
        self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.log_writer.log(self.file_object, 'Entered the model_for_rf method of the Model_Finder class')
        self.file_object.close()

        # create best model for XGBoost
        try:
            # parameters for RF gridsearchCV
            param_grid_rf = {
                'n_estimators': [200, 700],
                'max_features': ['auto', 'sqrt', 'log2']
            }

            # Creating an object of the Grid Search class
            grid = GridSearchCV(RandomForestClassifier(n_jobs=-1), param_grid_rf)

            # fitting the model with training data
            grid.fit(train_x, train_y)

            rf_new = grid.best_estimator_

            # fitting the model with the training dataset and testing it.
            pred_y_rf_new = rf_new.fit(train_x, train_y).predict(test_x)

            scoring = {'accuracy': make_scorer(accuracy_score),
                       'roc_auc': make_scorer(roc_auc_score),
                       'precision': make_scorer(precision_score),
                       'recall': make_scorer(recall_score),
                       'f1_score': make_scorer(f1_score)}

            rf_cv_score = model_selection.cross_validate(rf_new, train_x, train_y, cv=5, scoring=scoring)

            # outcome values order in sklearn
            # outcome values order in sklearn
            print("rf \n")

            tp_rf, fn_rf, fp_rf, tn_rf = confusion_matrix(test_y, pred_y_rf_new, labels=[1, 0]).reshape(-1)
            print('Outcome values  \n', 'True positive: ' + str(tp_rf) + '\n', 'false negetive: ' + str(fn_rf) + '\n',
                  'false positive: ' + str(fp_rf) + '\n', 'True negetive: ' + str(tn_rf) + '\n')
            print('\n')

            print("=== Confusion Matrix ===")
            print(confusion_matrix(test_y, pred_y_rf_new))
            print('\n')
            print("=== Classification Report ===")
            print(classification_report(test_y, pred_y_rf_new))
            print('\n')

            # print("=== All Scores of gnb ===")
            # print(gnb_cv_score)
            rf_results_df = pd.DataFrame(rf_cv_score)
            # print(gnb_results_df)
            # print('\n')

            df_confusion_matrix_report_rf = pd.DataFrame(
                [['True_positive: ' + str(tp_rf), 'false_positive: ' + str(fp_rf)],
                 ['false_negetive: ' + str(fn_rf), 'True_negetive: ' + str(tn_rf)]],
                index=['Predicted_Fraud', 'Predicted_Not_Fraud'],
                columns=['rf_Actual (Fraud)', 'rf_Actual (Not_Fraud)'])

            print(df_confusion_matrix_report_rf)

            json_confusion_matrix_report_rf = df_confusion_matrix_report_rf.to_json(orient='index')

            # print(json_confusion_matrix_report)

            classification_report_rf = classification_report(test_y, pred_y_rf_new)
            splitting_rf = classification_report_rf.split('\n')
            parse_rf = splitting_rf[6].strip()
            parse_rf = " ".join(parse_rf.split())
            parse_rf = list(parse_rf.split(" "))

            print('\n')
            # print(parse)
            print('\n')

            precision_rf = float(parse_rf[2]), rf_new, 'rf', json_confusion_matrix_report_rf
            recall_rf = float(parse_rf[3]), rf_new, 'rf', json_confusion_matrix_report_rf
            F1_rf = float(parse_rf[4]), rf_new, 'rf', json_confusion_matrix_report_rf

            print('\n')
            print("Precision: " + str(precision_rf))
            print('\n')

            print('\n')
            print("Recall: " + str(recall_rf))
            print('\n')

            print('\n')
            print("F1: " + str(F1_rf))
            print('\n')

            scores_rf = [precision_rf, recall_rf, F1_rf]

            # print(scores_gnb)

            return scores_rf



        except ZeroDivisionError as error:
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                'Exception occured in model_for_rf method of the Model_Finder class. Exception message:  ' + str(
                                    error))
            self.log_writer.log(self.file_object,
                                'Model Selection Failed. Exited the model_for_rf method of the Model_Finder class')
            self.file_object.close()
        except Exception as e:
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                'Exception occured in model_for_rf method of the Model_Finder class. Exception message:  ' + str(
                                    e))
            self.log_writer.log(self.file_object,
                                'Model Selection Failed. Exited the model_for_rf method of the Model_Finder class')
            self.file_object.close()







    def model_for_svc(self,train_x,train_y,test_x,test_y):
        """
        Method Name: model_for_svc
        Description: Find out the Model which has the best score.
        Output: The best model name and the model object
        On Failure: Raise Exception

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None

        """
        self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.log_writer.log(self.file_object,'Entered the model_for_svc method of the Model_Finder class')
        self.file_object.close()

        # create best model for svc
        try:
            # parameters for svc gridsearchCV
            param_grid_svc = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}

            svc = svm.SVC()

            # Creating an object of the Grid Search class
            grid = GridSearchCV(svc, param_grid_svc)

            # fitting the model with training data
            grid.fit(train_x, train_y)

            svc_new = grid.best_estimator_

            # fitting the model with the training dataset and testing it.
            pred_y_svc_new = svc_new.fit(train_x, train_y).predict(test_x)

            scoring = {'accuracy': make_scorer(accuracy_score),
                       'roc_auc': make_scorer(roc_auc_score),
                       'precision': make_scorer(precision_score),
                       'recall': make_scorer(recall_score),
                       'f1_score': make_scorer(f1_score)}

            svc_cv_score = model_selection.cross_validate(svc_new, train_x, train_y, cv=5, scoring=scoring)

            # outcome values order in sklearn
            # outcome values order in sklearn
            print("svc \n")

            tp_svc, fn_svc, fp_svc, tn_svc = confusion_matrix(test_y, pred_y_svc_new, labels=[1, 0]).reshape(-1)
            print('Outcome values  \n', 'True positive: ' + str(tp_svc) + '\n', 'false negetive: ' + str(fn_svc) + '\n',
                  'false positive: ' + str(fp_svc) + '\n', 'True negetive: ' + str(tn_svc) + '\n')
            print('\n')

            print("=== Confusion Matrix ===")
            print(confusion_matrix(test_y, pred_y_svc_new))
            print('\n')
            print("=== Classification Report ===")
            print(classification_report(test_y, pred_y_svc_new))
            print('\n')

            svc_results_df = pd.DataFrame(svc_cv_score)

            df_confusion_matrix_report_svc = pd.DataFrame(
                [['True_positive: ' + str(tp_svc), 'false_positive: ' + str(fp_svc)],
                 ['false_negetive: ' + str(fn_svc), 'True_negetive: ' + str(tn_svc)]],
                index=['Predicted_Fraud', 'Predicted_Not_Fraud'],
                columns=['svc_Actual (Fraud)', 'svc_Actual (Not_Fraud)'])

            print(df_confusion_matrix_report_svc)

            json_confusion_matrix_report_svc = df_confusion_matrix_report_svc.to_json(orient='index')

            # print(json_confusion_matrix_report)

            classification_report_svc = classification_report(test_y, pred_y_svc_new)
            splitting_svc = classification_report_svc.split('\n')
            parse_svc = splitting_svc[6].strip()
            parse_svc = " ".join(parse_svc.split())
            parse_svc = list(parse_svc.split(" "))

            print('\n')
            # print(parse)
            print('\n')

            precision_svc = float(parse_svc[2]), svc_new, 'svc', json_confusion_matrix_report_svc
            recall_svc = float(parse_svc[3]), svc_new, 'svc', json_confusion_matrix_report_svc
            F1_svc = float(parse_svc[4]), svc_new, 'svc', json_confusion_matrix_report_svc

            print('\n')
            print("Precision: " + str(precision_svc))
            print('\n')

            print('\n')
            print("Recall: " + str(recall_svc))
            print('\n')

            print('\n')
            print("F1: " + str(F1_svc))
            print('\n')

            scores_svc = [precision_svc, recall_svc, F1_svc]

            # print(scores_gnb)

            return scores_svc


        except ZeroDivisionError as error:
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                'Exception occured in model_for_svc method of the Model_Finder class. Exception message:  ' + str(
                                    error))
            self.log_writer.log(self.file_object,
                                'Model Selection Failed. Exited the model_for_svc method of the Model_Finder class')
            self.file_object.close()
        except Exception as e:
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                'Exception occured in model_for_svc method of the Model_Finder class. Exception message:  ' + str(
                                    e))
            self.log_writer.log(self.file_object,
                                'Model Selection Failed. Exited the model_for_svc method of the Model_Finder class')
            self.file_object.close()






    def model_for_lgb(self,train_x,train_y,test_x,test_y):
        """
        Method Name: model_for_lgb
        Description: Find out the Model which has the best score.
        Output: The best model name and the model object
        On Failure: Raise Exception

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None

        """
        self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.log_writer.log(self.file_object,'Entered the model_for_lgb method of the Model_Finder class')
        self.file_object.close()

        # create best model for lgb
        try:
            # parameters for lgb gridsearchCV
            param_grid_lgb = {
                'num_leaves': [31, 127],
                'reg_alpha': [0.1, 0.5],
                'min_data_in_leaf': [30, 50, 100, 300, 400],
                'lambda_l1': [0, 1, 1.5],
                'lambda_l2': [0, 1]
            }

            # Creating an object of the Grid Search class
            grid = GridSearchCV(LGBMClassifier(), param_grid_lgb)

            # fitting the gridCV with training data
            grid.fit(train_x, train_y)

            lgb_new = grid.best_estimator_

            # fitting the model with the training dataset and testing it.
            pred_y_lgb_new = lgb_new.fit(train_x, train_y).predict(test_x)

            scoring = {'accuracy': make_scorer(accuracy_score),
                       'roc_auc': make_scorer(roc_auc_score),
                       'precision': make_scorer(precision_score),
                       'recall': make_scorer(recall_score),
                       'f1_score': make_scorer(f1_score)}

            lgb_cv_score = model_selection.cross_validate(lgb_new, train_x, train_y, cv=5, scoring=scoring)

            # outcome values order in sklearn
            # outcome values order in sklearn
            print("lgb \n")

            tp_lgb, fn_lgb, fp_lgb, tn_lgb = confusion_matrix(test_y, pred_y_lgb_new, labels=[1, 0]).reshape(-1)
            print('Outcome values  \n', 'True positive: ' + str(tp_lgb) + '\n', 'false negetive: ' + str(fn_lgb) + '\n',
                  'false positive: ' + str(fp_lgb) + '\n', 'True negetive: ' + str(tn_lgb) + '\n')
            print('\n')

            print("=== Confusion Matrix ===")
            print(confusion_matrix(test_y, pred_y_lgb_new))
            print('\n')
            print("=== Classification Report ===")
            print(classification_report(test_y, pred_y_lgb_new))
            print('\n')

            lgb_results_df = pd.DataFrame(lgb_cv_score)

            df_confusion_matrix_report_lgb = pd.DataFrame(
                [['True_positive: ' + str(tp_lgb), 'false_positive: ' + str(fp_lgb)],
                 ['false_negetive: ' + str(fn_lgb), 'True_negetive: ' + str(tn_lgb)]],
                index=['Predicted_Fraud', 'Predicted_Not_Fraud'],
                columns=['lgb_Actual (Fraud)', 'lgb_Actual (Not_Fraud)'])

            print(df_confusion_matrix_report_lgb)

            json_confusion_matrix_report_lgb = df_confusion_matrix_report_lgb.to_json(orient='index')

            # print(json_confusion_matrix_report)

            classification_report_lgb = classification_report(test_y, pred_y_lgb_new)
            splitting_lgb = classification_report_lgb.split('\n')
            parse_lgb = splitting_lgb[6].strip()
            parse_lgb = " ".join(parse_lgb.split())
            parse_lgb = list(parse_lgb.split(" "))

            print('\n')
            # print(parse)
            print('\n')

            precision_lgb = float(parse_lgb[2]), lgb_new, 'lgb', json_confusion_matrix_report_lgb
            recall_lgb = float(parse_lgb[3]), lgb_new, 'lgb', json_confusion_matrix_report_lgb
            F1_lgb = float(parse_lgb[4]), lgb_new, 'lgb', json_confusion_matrix_report_lgb

            print('\n')
            print("Precision: " + str(precision_lgb))
            print('\n')

            print('\n')
            print("Recall: " + str(recall_lgb))
            print('\n')

            print('\n')
            print("F1: " + str(F1_lgb))
            print('\n')

            scores_lgb = [precision_lgb, recall_lgb, F1_lgb]

            # print(scores_gnb)

            return scores_lgb



        except ZeroDivisionError as error:
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                'Exception occured in model_for_lgb method of the Model_Finder class. Exception message:  ' + str(
                                    error))
            self.log_writer.log(self.file_object,
                                'Model Selection Failed. Exited the model_for_lgb method of the Model_Finder class')
            self.file_object.close()
        except Exception as e:
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                'Exception occured in model_for_lgb method of the Model_Finder class. Exception message:  ' + str(
                                    e))
            self.log_writer.log(self.file_object,
                                'Model Selection Failed. Exited the model_for_lgb method of the Model_Finder class')
            self.file_object.close()





    def get_best_model(self,train_x,train_y,test_x,test_y):
        """
                Method Name: get_best_model
                Description: Find out the Model which has the best score.
                Output: The best model name and the model object
                On Failure: Raise Exception

                Written By: Tejas Jay (TJ)
                Version: 1.0
                Revisions: None

                """
        self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.log_writer.log(self.file_object, 'Entered the get_best_model method of the Model_Finder class')
        self.file_object.close()

        try:
            scores_gnb = self.model_for_gnb(train_x, train_y, test_x, test_y)
            scores_xgb = self.model_for_xgb(train_x,train_y,test_x,test_y)
            scores_rf =  self.model_for_rf(train_x, train_y, test_x, test_y)
            scores_svc = self.model_for_svc(train_x, train_y, test_x, test_y)
            scores_lgb = self.model_for_lgb(train_x, train_y, test_x, test_y)

            precision_scores = [scores_gnb[0], scores_xgb[0], scores_rf[0], scores_svc[0], scores_lgb[0]]

            recall_scores = [scores_gnb[1], scores_xgb[1], scores_rf[1], scores_svc[1], scores_lgb[1]]

            F1_scores = [scores_gnb[2], scores_xgb[2], scores_rf[2], scores_svc[2], scores_lgb[2]]

            scoring = {'precision': precision_scores, 'recall': recall_scores, 'F1': F1_scores}

            return scoring


        except ZeroDivisionError as error:
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(
                                    error))
            self.log_writer.log(self.file_object,
                                'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            self.file_object.close()
        except Exception as e:
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(
                                    e))
            self.log_writer.log(self.file_object,
                                'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            self.file_object.close()





    def best_model_scoring(self, scoring, type_of_score):
        """
                Method Name: best_model_scoring
                Description: Find out the Model which has the best score.
                Output: The best model name and the model object
                On Failure: Raise Exception

                Written By: Tejas Jay (TJ)
                Version: 1.0
                Revisions: None

            """
        self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.log_writer.log(self.file_object, 'Entered the best_model_scoring method of the Model_Finder class')
        self.file_object.close()

        lst_model = []
        try:
            value = scoring[type_of_score]

            first_element = [i[0] for i in value]
            check_max = np.array(first_element).max()
            get_model = [i[1] for i in value if i[0] == check_max]
            get_model = get_model[0]


            check_max = np.array([i[0] for i in value]).max()
            first_element1 = [lst_model.append(i) for i in value if i[0] == check_max]

            return lst_model


        except ZeroDivisionError as error:
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                'Exception occured in best_model_scoring method of the Model_Finder class. Exception message:  ' + str(
                                    error))
            self.log_writer.log(self.file_object,
                                'Model Selection Failed. Exited the best_model_scoring method of the Model_Finder class')
            self.file_object.close()
        except Exception as e:
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                'Exception occured in best_model_scoring method of the Model_Finder class. Exception message:  ' + str(
                                    e))
            self.log_writer.log(self.file_object,
                                'Model Selection Failed. Exited the best_model_scoring method of the Model_Finder class')
            self.file_object.close()
































