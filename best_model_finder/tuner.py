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



class Model_Finder:
    """
    This class shall  be used to find the model with best accuracy and AUC score.
    Written By: Tejas Jay (TJ)
    Version: 1.0
    Revisions: None

    """

    def __init__(self):
        self.xgb = XGBClassifier(objective='binary:logistic')
        self.log_writer = logger.App_Logger()




    def get_best_params_for_xgboost(self,train_x,train_y):

        """
        Method Name: get_best_params_for_xgboost
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
                               'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        self.file_object.close()

        try:
            # initializing with different combination of parameters
            self.param_grid_xgboost = {

                'learning_rate': [0.5, 0.1, 0.01, 0.001],
                'max_depth': [3, 5, 10, 20],
                'n_estimators': [10, 50, 100, 200]

            }
            # Creating an object of the Grid Search class
            self.grid= GridSearchCV(XGBClassifier(objective='binary:logistic'),self.param_grid_xgboost, verbose=3,cv=5)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.learning_rate = self.grid.best_params_['learning_rate']
            self.max_depth = self.grid.best_params_['max_depth']
            self.n_estimators = self.grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            self.xgb = XGBClassifier(learning_rate=self.learning_rate, max_depth=self.max_depth, n_estimators=self.n_estimators)
            # training the mew model
            self.xgb_model = self.xgb.fit(train_x, train_y)

            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                   'XGBoost best params: ' + str(
                                       self.grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            self.file_object.close()

            return self.xgb_model

        except Exception as e:
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                   'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.log_writer.log(self.file_object,
                                   'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            self.file_object.close()
            raise Exception()



    def get_best_model_xgb(self,train_x,train_y,test_x,test_y):
        """
        Method Name: get_best_model_xgb
        Description: Find out the Model which has the best score.
        Output: The best model name and the model object
        On Failure: Raise Exception

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None

        """
        self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.log_writer.log(self.file_object,'Entered the get_best_model_xgb method of the Model_Finder class')
        self.file_object.close()

        # create best model for XGBoost
        try:
            self.xgboost = self.get_best_params_for_xgboost(train_x,train_y)

            self.prediction_xgboost = self.xgboost.predict(test_x) # Predictions using the XGBoost Model


            #self.xgboost_score = accuracy_score(test_y, self.prediction_xgboost)

            tp, fn, fp, tn = confusion_matrix(test_y, self.prediction_xgboost, labels=[1, 0]).reshape(-1)

            Precision = tp / (tp + fp)

            Recall = tp / (tp + fn)

            F1 = (2 * Precision * Recall) / (Precision + Recall)

            df_confusion_matrix_report = pd.DataFrame([['True_positive: '+str(tp),'false_positive: ' + str(fp)],['false_negetive: '+str(fn), 'True_negetive: ' + str(tn)], ['..........................','..........................'], ['precision --> '+str(Precision), 'Recall --> ' + str(Recall)]],
                  index=['Fraud (Positive)', 'Not_Fraud (Negative)','.........................', 'F1_score --> '+ str(F1)],
                  columns= ['Fraud (Positive)', 'Not_Fraud (Negative)'])

            json_confusion_matrix_report = df_confusion_matrix_report.to_json(orient='index')


            scoring = {'accuracy': make_scorer(accuracy_score),
                       'roc_auc': make_scorer(roc_auc_score),
                       'precision': make_scorer(precision_score),
                       'recall': make_scorer(recall_score),
                       'f1_score': make_scorer(f1_score)}

            xgb_cv_score = model_selection.cross_validate(self.xgboost, train_x, train_y, cv=10, scoring=scoring)


            # computing all the mean results

            # ("=== Mean roc_auc Score ===")
            xgb_mean_roc_auc = np.mean(xgb_cv_score['test_roc_auc']), self.xgboost, 'xgb', xgb_cv_score ,json_confusion_matrix_report

            #("=== Mean rf Score ===")
            xgb_mean_accuracy = np.mean(xgb_cv_score['test_accuracy']), self.xgboost, 'xgb', xgb_cv_score, json_confusion_matrix_report

            #("=== Mean precision Score ===")
            xgb_mean_precision = np.mean(xgb_cv_score['test_precision']), self.xgboost, 'xgb', xgb_cv_score, json_confusion_matrix_report

            #("=== Mean Recall Score ===")
            xgb_mean_recall = np.mean(xgb_cv_score['test_recall']), self.xgboost, 'xgb', xgb_cv_score, json_confusion_matrix_report

            #("=== Mean F1 Score ===")
            xgb_mean_F1 = np.mean(xgb_cv_score['test_f1_score']), self.xgboost, 'xgb', xgb_cv_score, json_confusion_matrix_report

            scores_xgb = [xgb_mean_roc_auc,xgb_mean_accuracy,xgb_mean_precision,xgb_mean_recall,xgb_mean_F1]

            return scores_xgb

        except Exception as e:
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                   'Exception occured in get_best_model_xgb method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.log_writer.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model_xgb method of the Model_Finder class')
            self.file_object.close()

            raise Exception()





    def get_best_params_for_gnb(self,train_x,train_y):

        """
        Method Name: get_best_params_for_gnb
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
                               'Entered the get_best_params_for_gnb method of the Model_Finder class')
        self.file_object.close()

        try:
            self.gnb = GaussianNB()

            # initializing with different combination of parameters
            self.param_grid_gnb = {"var_smoothing": [1e-9,0.1, 0.001, 0.5,0.05,0.01,1e-8,1e-7,1e-6,1e-10,1e-11]}

            # Creating an object of the Grid Search class
            self.grid= GridSearchCV(estimator=self.gnb, param_grid=self.param_grid_gnb, cv=5,  verbose=3)

            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.gnb = self.grid.best_estimator_

            # training the mew model
            self.gnb_model = self.gnb.fit(train_x, train_y)

            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                   ' Exited the get_best_params_for_gnb method of the Model_Finder class')
            self.file_object.close()

            return self.gnb_model

        except Exception as e:
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                   'Exception occured in get_best_params_for_gnb method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.log_writer.log(self.file_object,
                                   'Parameter tuning  failed. Exited the get_best_params_for_gnb method of the Model_Finder class')
            self.file_object.close()
            raise Exception()



    def get_best_model_gnb(self,train_x,train_y,test_x,test_y):
        """
        Method Name: get_best_model_gnb
        Description: Find out the Model which has the best score.
        Output: The best model name and the model object
        On Failure: Raise Exception

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None

        """
        self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.log_writer.log(self.file_object,'Entered the get_best_model_gnb method of the Model_Finder class')
        self.file_object.close()

        # create best model for gnb
        try:
            self.gnb = self.get_best_params_for_gnb(train_x,train_y)

            self.prediction_gnb = self.gnb.predict(test_x) # Predictions using the XGBoost Model


            #self.xgboost_score = accuracy_score(test_y, self.prediction_xgboost)

            tp, fn, fp, tn = confusion_matrix(test_y, self.prediction_gnb, labels=[1, 0]).reshape(-1)

            Precision = tp / (tp + fp)

            Recall = tp / (tp + fn)

            F1 = (2 * Precision * Recall) / (Precision + Recall)

            df_confusion_matrix_report = pd.DataFrame([['True_positive: '+str(tp),'false_positive: ' + str(fp)],['false_negetive: '+str(fn), 'True_negetive: ' + str(tn)], ['..........................','..........................'], ['precision --> '+str(Precision), 'Recall --> ' + str(Recall)]],
                  index=['Fraud (Positive)', 'Not_Fraud (Negative)','.........................', 'F1_score --> '+ str(F1)],
                  columns= ['Fraud (Positive)', 'Not_Fraud (Negative)'])

            json_confusion_matrix_report = df_confusion_matrix_report.to_json(orient='index')


            scoring = {'accuracy': make_scorer(accuracy_score),
                       'roc_auc': make_scorer(roc_auc_score),
                       'precision': make_scorer(precision_score),
                       'recall': make_scorer(recall_score),
                       'f1_score': make_scorer(f1_score)}

            gnb_cv_score = model_selection.cross_validate(self.gnb, train_x, train_y, cv=10, scoring=scoring)


            # computing all the mean results

            # ("=== Mean roc_auc Score ===")
            gnb_mean_roc_auc = np.mean(gnb_cv_score['test_roc_auc']), self.gnb, 'gnb', gnb_cv_score ,json_confusion_matrix_report

            #("=== Mean rf Score ===")
            gnb_mean_accuracy = np.mean(gnb_cv_score['test_accuracy']), self.gnb, 'gnb', gnb_cv_score, json_confusion_matrix_report

            #("=== Mean precision Score ===")
            gnb_mean_precision = np.mean(gnb_cv_score['test_precision']), self.gnb, 'gnb', gnb_cv_score, json_confusion_matrix_report

            #("=== Mean Recall Score ===")
            gnb_mean_recall = np.mean(gnb_cv_score['test_recall']), self.gnb, 'gnb', gnb_cv_score, json_confusion_matrix_report

            #("=== Mean F1 Score ===")
            gnb_mean_F1 = np.mean(gnb_cv_score['test_f1_score']), self.gnb, 'gnb', gnb_cv_score, json_confusion_matrix_report

            scores_gnb = [gnb_mean_roc_auc,gnb_mean_accuracy,gnb_mean_precision,gnb_mean_recall,gnb_mean_F1]

            return scores_gnb

        except Exception as e:
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                   'Exception occured in get_best_model_gnb method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.log_writer.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model_gnb method of the Model_Finder class')
            self.file_object.close()

            raise Exception()




    def get_best_params_for_rf(self,train_x,train_y):

        """
        Method Name: get_best_params_for_gnb
        Description: get the parameters for Random forest Algorithm which give the best scores.
                    Use Hyper Parameter Tuning.
        Output: The model with the best parameters
        On Failure: Raise Exception

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None

        """
        self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.log_writer.log(self.file_object,
                               'Entered the get_best_params_for_rf method of the Model_Finder class')
        self.file_object.close()

        try:
            self.rf = RandomForestClassifier(n_jobs=-1)

            # initializing with different combination of parameters
            self.param_grid_rf = {
                              'n_estimators': [200, 700],
                              'max_features': ['auto', 'sqrt', 'log2']
                                  }


            # Creating an object of the Grid Search class
            self.grid= GridSearchCV(estimator=self.rf, param_grid=self.param_grid_rf)

            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.rf = self.grid.best_estimator_

            # training the mew model
            self.rf_model = self.rf.fit(train_x, train_y)

            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                   ' Exited the get_best_params_for_rf method of the Model_Finder class')
            self.file_object.close()

            return self.rf_model

        except Exception as e:
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                   'Exception occured in get_best_params_for_rf method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.log_writer.log(self.file_object,
                                   'Parameter tuning  failed. Exited the get_best_params_for_rf method of the Model_Finder class')
            self.file_object.close()
            raise Exception()





    def get_best_model_rf(self,train_x,train_y,test_x,test_y):
        """
        Method Name: get_best_model_rf
        Description: Find out the Model which has the best score.
        Output: The best model name and the model object
        On Failure: Raise Exception

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None

        """
        self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.log_writer.log(self.file_object,'Entered the get_best_model_rf method of the Model_Finder class')
        self.file_object.close()

        # create best model for XGBoost
        try:
            self.rf = self.get_best_params_for_rf(train_x,train_y)

            self.prediction_rf = self.rf.predict(test_x) # Predictions using the XGBoost Model

            #self.xgboost_score = accuracy_score(test_y, self.prediction_xgboost)

            tp, fn, fp, tn = confusion_matrix(test_y, self.prediction_rf, labels=[1, 0]).reshape(-1)

            Precision = tp / (tp + fp)

            Recall = tp / (tp + fn)

            F1 = (2 * Precision * Recall) / (Precision + Recall)

            df_confusion_matrix_report = pd.DataFrame([['True_positive: '+str(tp),'false_positive: ' + str(fp)],['false_negetive: '+str(fn), 'True_negetive: ' + str(tn)], ['..........................','..........................'], ['precision --> '+str(Precision), 'Recall --> ' + str(Recall)]],
                  index=['Fraud (Positive)', 'Not_Fraud (Negative)','.........................', 'F1_score --> '+ str(F1)],
                  columns= ['Fraud (Positive)', 'Not_Fraud (Negative)'])

            json_confusion_matrix_report = df_confusion_matrix_report.to_json(orient='index')


            scoring = {'accuracy': make_scorer(accuracy_score),
                       'roc_auc': make_scorer(roc_auc_score),
                       'precision': make_scorer(precision_score),
                       'recall': make_scorer(recall_score),
                       'f1_score': make_scorer(f1_score)}

            rf_cv_score = model_selection.cross_validate(self.rf, train_x, train_y, cv=10, scoring=scoring)


            # computing all the mean results

            # ("=== Mean roc_auc Score ===")
            rf_mean_roc_auc = np.mean(rf_cv_score['test_roc_auc']), self.rf, 'rf', rf_cv_score ,json_confusion_matrix_report

            #("=== Mean rf Score ===")
            rf_mean_accuracy = np.mean(rf_cv_score['test_accuracy']), self.rf, 'rf', rf_cv_score, json_confusion_matrix_report

            #("=== Mean precision Score ===")
            rf_mean_precision = np.mean(rf_cv_score['test_precision']), self.rf, 'rf', rf_cv_score, json_confusion_matrix_report

            #("=== Mean Recall Score ===")
            rf_mean_recall = np.mean(rf_cv_score['test_recall']), self.rf, 'rf', rf_cv_score, json_confusion_matrix_report

            #("=== Mean F1 Score ===")
            rf_mean_F1 = np.mean(rf_cv_score['test_f1_score']), self.rf, 'rf', rf_cv_score, json_confusion_matrix_report

            scores_rf = [rf_mean_roc_auc,rf_mean_accuracy,rf_mean_precision,rf_mean_recall,rf_mean_F1]

            return scores_rf

        except Exception as e:
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                   'Exception occured in get_best_model_rf method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.log_writer.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model_rf method of the Model_Finder class')
            self.file_object.close()

            raise Exception()







    def get_best_params_for_svc(self,train_x,train_y):

        """
        Method Name: get_best_params_for_svc
        Description: get the parameters for support vector Algorithm which give the best scores.
                    Use Hyper Parameter Tuning.
        Output: The model with the best parameters
        On Failure: Raise Exception

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None

        """
        self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.log_writer.log(self.file_object,
                               'Entered the get_best_params_for_svc method of the Model_Finder class')
        self.file_object.close()

        try:
            self.svc = svm.SVC()

            # initializing with different combination of parameters
            self.param_grid_svc = {'kernel':('linear', 'rbf'), 'C':[1, 10]}


            # Creating an object of the Grid Search class
            self.grid= GridSearchCV(estimator=self.svc, param_grid=self.param_grid_svc)

            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.svc = self.grid.best_estimator_

            # training the mew model
            self.svc_model = self.svc.fit(train_x, train_y)

            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                   ' Exited the get_best_params_for_svc method of the Model_Finder class')
            self.file_object.close()

            return self.svc_model

        except Exception as e:
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                   'Exception occured in get_best_params_for_svc method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.log_writer.log(self.file_object,
                                   'Parameter tuning  failed. Exited the get_best_params_for_svc method of the Model_Finder class')
            self.file_object.close()
            raise Exception()





    def get_best_model_svc(self,train_x,train_y,test_x,test_y):
        """
        Method Name: get_best_model_svc
        Description: Find out the Model which has the best score.
        Output: The best model name and the model object
        On Failure: Raise Exception

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None

        """
        self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.log_writer.log(self.file_object,'Entered the get_best_model_svc method of the Model_Finder class')
        self.file_object.close()

        # create best model for svc
        try:
            self.svc = self.get_best_params_for_svc(train_x,train_y)

            self.prediction_svc = self.svc.predict(test_x) # Predictions using the XGBoost Model

            #self.xgboost_score = accuracy_score(test_y, self.prediction_xgboost)

            tp, fn, fp, tn = confusion_matrix(test_y, self.prediction_svc, labels=[1, 0]).reshape(-1)

            Precision = tp / (tp + fp)

            Recall = tp / (tp + fn)

            F1 = (2 * Precision * Recall) / (Precision + Recall)

            df_confusion_matrix_report = pd.DataFrame([['True_positive: '+str(tp),'false_positive: ' + str(fp)],['false_negetive: '+str(fn), 'True_negetive: ' + str(tn)], ['..........................','..........................'], ['precision --> '+str(Precision), 'Recall --> ' + str(Recall)]],
                  index=['Fraud (Positive)', 'Not_Fraud (Negative)','.........................', 'F1_score --> '+ str(F1)],
                  columns= ['Fraud (Positive)', 'Not_Fraud (Negative)'])

            json_confusion_matrix_report = df_confusion_matrix_report.to_json(orient='index')


            scoring = {'accuracy': make_scorer(accuracy_score),
                       'roc_auc': make_scorer(roc_auc_score),
                       'precision': make_scorer(precision_score),
                       'recall': make_scorer(recall_score),
                       'f1_score': make_scorer(f1_score)}

            svc_cv_score = model_selection.cross_validate(self.svc, train_x, train_y, cv=10, scoring=scoring)


            # computing all the mean results

            # ("=== Mean roc_auc Score ===")
            svc_mean_roc_auc = np.mean(svc_cv_score['test_roc_auc']), self.svc, 'svc', svc_cv_score ,json_confusion_matrix_report

            #("=== Mean rf Score ===")
            svc_mean_accuracy = np.mean(svc_cv_score['test_accuracy']), self.svc, 'svc', svc_cv_score, json_confusion_matrix_report

            #("=== Mean precision Score ===")
            svc_mean_precision = np.mean(svc_cv_score['test_precision']), self.svc, 'svc', svc_cv_score, json_confusion_matrix_report

            #("=== Mean Recall Score ===")
            svc_mean_recall = np.mean(svc_cv_score['test_recall']), self.svc, 'svc', svc_cv_score, json_confusion_matrix_report

            #("=== Mean F1 Score ===")
            svc_mean_F1 = np.mean(svc_cv_score['test_f1_score']), self.svc, 'svc', svc_cv_score, json_confusion_matrix_report

            scores_svc = [svc_mean_roc_auc,svc_mean_accuracy,svc_mean_precision,svc_mean_recall,svc_mean_F1]

            return scores_svc

        except Exception as e:
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                   'Exception occured in get_best_model_svc method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.log_writer.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model_svc method of the Model_Finder class')
            self.file_object.close()

            raise Exception()





    def get_best_params_for_lgb(self,train_x,train_y):

        """
        Method Name: get_best_params_for_lgb
        Description: get the parameters for light gradient boost Algorithm which give the best scores.
                    Use Hyper Parameter Tuning.
        Output: The model with the best parameters
        On Failure: Raise Exception

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None

        """
        self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.log_writer.log(self.file_object,
                               'Entered the get_best_params_for_lgb method of the Model_Finder class')
        self.file_object.close()

        try:
            self.lgb = LGBMClassifier()

            # initializing with different combination of parameters
            self.param_grid_lgb = {
                                           'num_leaves': [31, 127],
                                            'reg_alpha': [0.1, 0.5],
                                             'min_data_in_leaf': [30, 50, 100, 300, 400],
                                             'lambda_l1': [0, 1, 1.5],
                                               'lambda_l2': [0, 1]
                                    }



            # Creating an object of the Grid Search class
            self.grid= GridSearchCV(estimator=self.lgb, param_grid=self.param_grid_lgb)

            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.lgb = self.grid.best_estimator_

            # training the mew model
            self.lgb_model = self.lgb.fit(train_x, train_y)

            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                   ' Exited the get_best_params_for_lgb method of the Model_Finder class')
            self.file_object.close()

            return self.lgb_model

        except Exception as e:
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                   'Exception occured in get_best_params_for_lgb method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.log_writer.log(self.file_object,
                                   'Parameter tuning  failed. Exited the get_best_params_for_lgb method of the Model_Finder class')
            self.file_object.close()
            raise Exception()





    def get_best_model_lgb(self,train_x,train_y,test_x,test_y):
        """
        Method Name: get_best_model_lgb
        Description: Find out the Model which has the best score.
        Output: The best model name and the model object
        On Failure: Raise Exception

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None

        """
        self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.log_writer.log(self.file_object,'Entered the get_best_model_lgb method of the Model_Finder class')
        self.file_object.close()

        # create best model for lgb
        try:
            self.lgb = self.get_best_params_for_lgb(train_x,train_y)

            self.prediction_lgb = self.lgb.predict(test_x) # Predictions using the XGBoost Model

            #self.xgboost_score = accuracy_score(test_y, self.prediction_xgboost)

            tp, fn, fp, tn = confusion_matrix(test_y, self.prediction_lgb, labels=[1, 0]).reshape(-1)

            Precision = tp / (tp + fp)

            Recall = tp / (tp + fn)

            F1 = (2 * Precision * Recall) / (Precision + Recall)

            df_confusion_matrix_report = pd.DataFrame([['True_positive: '+str(tp),'false_positive: ' + str(fp)],['false_negetive: '+str(fn), 'True_negetive: ' + str(tn)], ['..........................','..........................'], ['precision --> '+str(Precision), 'Recall --> ' + str(Recall)]],
                  index=['Fraud (Positive)', 'Not_Fraud (Negative)','.........................', 'F1_score --> '+ str(F1)],
                  columns= ['Fraud (Positive)', 'Not_Fraud (Negative)'])

            json_confusion_matrix_report = df_confusion_matrix_report.to_json(orient='index')


            scoring = {'accuracy': make_scorer(accuracy_score),
                       'roc_auc': make_scorer(roc_auc_score),
                       'precision': make_scorer(precision_score),
                       'recall': make_scorer(recall_score),
                       'f1_score': make_scorer(f1_score)}

            lgb_cv_score = model_selection.cross_validate(self.lgb, train_x, train_y, cv=10, scoring=scoring)


            # computing all the mean results

            # ("=== Mean roc_auc Score ===")
            lgb_mean_roc_auc = np.mean(lgb_cv_score['test_roc_auc']), self.lgb, 'lgb', lgb_cv_score ,json_confusion_matrix_report

            #("=== Mean rf Score ===")
            lgb_mean_accuracy = np.mean(lgb_cv_score['test_accuracy']), self.lgb, 'lgb', lgb_cv_score, json_confusion_matrix_report

            #("=== Mean precision Score ===")
            lgb_mean_precision = np.mean(lgb_cv_score['test_precision']), self.lgb, 'lgb', lgb_cv_score, json_confusion_matrix_report

            #("=== Mean Recall Score ===")
            lgb_mean_recall = np.mean(lgb_cv_score['test_recall']), self.lgb, 'lgb', lgb_cv_score, json_confusion_matrix_report

            #("=== Mean F1 Score ===")
            lgb_mean_F1 = np.mean(lgb_cv_score['test_f1_score']), self.lgb, 'lgb', lgb_cv_score, json_confusion_matrix_report

            scores_lgb = [lgb_mean_roc_auc,lgb_mean_accuracy,lgb_mean_precision,lgb_mean_recall,lgb_mean_F1]

            return scores_lgb

        except Exception as e:
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                   'Exception occured in get_best_model_lgb method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.log_writer.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model_lgb method of the Model_Finder class')
            self.file_object.close()

            raise Exception()







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
            scores_xgb = self.get_best_model_xgb(train_x,train_y,test_x,test_y)
            scores_gnb = self.get_best_model_gnb(train_x, train_y, test_x, test_y)
            scores_rf =  self.get_best_model_rf(train_x, train_y, test_x, test_y)
            scores_svc = self.get_best_model_svc(train_x, train_y, test_x, test_y)
            scores_lgb = self.get_best_model_lgb(train_x, train_y, test_x, test_y)

            roc_auc_scores = [scores_gnb[0], scores_xgb[0], scores_rf[0], scores_svc[0], scores_lgb[0]]
            accuracy_scores = [scores_gnb[1], scores_xgb[1], scores_rf[1], scores_svc[1], scores_lgb[1]]
            precision_scores = [scores_gnb[2], scores_xgb[2], scores_rf[2], scores_svc[2], scores_lgb[2]]
            recall_scores = [scores_gnb[3], scores_xgb[3], scores_rf[3], scores_svc[3], scores_lgb[3]]
            F1_scores = [scores_gnb[4], scores_xgb[4], scores_rf[4], scores_svc[4], scores_lgb[4]]

            scoring = {'roc_auc': roc_auc_scores, 'accuracy': accuracy_scores, 'precision': precision_scores, 'recall':recall_scores, 'F1':F1_scores}

            return scoring

        except Exception as e:
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                   'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.log_writer.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            self.file_object.close()

            raise Exception()






    def select_best_model_scoring_new(self, scoring, type_of_score):

        self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.log_writer.log(self.file_object, 'Entered the select_best_model_scoring_new method of the Model_Finder class')
        self.file_object.close()

        try:

            value = scoring[type_of_score]

            first_element = [i[0] for i in value]
            check_max = np.array(first_element).max()
            get_model = [i[1] for i in value if i[0] == check_max]
            get_model= get_model[0]

            get_file_name = [i[2] for i in value if i[0] == check_max]
            get_file_name = get_file_name[0]

            get_report = [i[3] for i in value if i[0] == check_max]
            get_report_json = get_report[0]
            json_response = pd.DataFrame(get_report_json)
            result_json = json_response.to_json(orient="index")
            parsed = json.loads(result_json)
            json_output = json.dumps(parsed)
            get_report_json = json.loads(json_output)

            get_confusion_matrix_report = [i[4] for i in value if i[0] == check_max]
            get_confusion_matrix_report = get_confusion_matrix_report[0]
            confusion_matrix_report_json = json.loads(get_confusion_matrix_report)

            return get_model, get_file_name, get_report_json, confusion_matrix_report_json



        except Exception as e:
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                   'Exception occured in select_best_model_scoring_new method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.log_writer.log(self.file_object,
                                   'Model Selection Failed. Exited the select_best_model_scoring_new method of the Model_Finder class')
            self.file_object.close()

            raise Exception()


