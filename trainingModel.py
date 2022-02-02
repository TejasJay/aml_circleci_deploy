"""
This is the Entry point for Training the Machine Learning Model.

Written By: Tejas Jay
Version: 1.0
Revisions: None

"""



# Doing the necessary imports
from sklearn.model_selection import train_test_split
from data_ingestion import data_loader
from data_preprocessing import preprocessing
from best_model_finder import tuner
from file_operations import file_methods
from application_logging import logger
from best_model_finder import tuner_new
import pandas as pd
import json
import csv
from training_Validation_Insertion import train_validation


class trainModel:

    def __init__(self, path, type_of_score):
        self.log_writer = logger.App_Logger()
        self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.train_data_val = train_validation(path)
        self.type_of_score = type_of_score




    def trainingModel(self):
        # Logging the start of Training
        self.log_writer.log(self.file_object, 'Start of Training')
        try:
            # Getting the data from the source
            data_getter = data_loader.Data_Getter()

            data = data_getter.get_data()

            """doing the data preprocessing"""

            preprocessor = preprocessing.Preprocessor()

            new_data = preprocessor.remove_columns(data, columns = 'unique_id')

            # create separate features and labels
            Y, X = preprocessor.separate_label_feature(new_data, label_column_name='sar_fraud')


            # splitting the data into training and test set for each cluster one by one
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1 / 3)

            # Proceeding with more data pre-processing steps by scaling the features
            train_x = preprocessor.scale_numerical_columns(x_train)
            test_x = preprocessor.scale_numerical_columns(x_test)

            # finding the best parameters and model
            model_finder = tuner_new.Model_Finder_new()  # object initialization

            scoring = model_finder.get_best_model(train_x,y_train,test_x,y_test)

            type_of_score = self.type_of_score

            lst_model = model_finder.best_model_scoring(scoring, type_of_score)

            confusion_matrices = []

            all_model_names = []

            for number in range(len(lst_model)):
                model_score = lst_model[number][0]
                print(model_score)
                model = lst_model[number][1]
                print(model)
                model_name = [i[2] for i in lst_model][number]
                print(model_name)
                all_model_names.append(model_name)
                cm = lst_model[number][3]
                matrix_resl = json.loads(cm)
                df_matrix_resl = pd.DataFrame(matrix_resl)

                confusion_matrices.append(df_matrix_resl)

                # saving the model
                file_op = file_methods.File_Operation()
                file_op.save_model(model, model_name)

                # logging the successful Training
                self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
                self.log_writer.log(self.file_object, model_name + ' Successfully trained and saved in project directory')
                self.file_object.close()


            full_df = pd.concat(confusion_matrices, axis=0)

            full_df = full_df.reset_index()

            full_df.rename(columns={"index": "Actual"}, inplace=True)

            confusion_matrix_data = full_df.to_csv("csv_output_files\confusion_matrix.csv", index=False)

            final_json_str = full_df.to_json(orient='index')

            final_json = json.loads(final_json_str)

            jsonString = json.dumps(final_json)

            jsonFile = open("data.json", "w")

            jsonFile.write(jsonString)

            jsonFile.close()

            # logging the successful Training
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object, 'Successful End of Training, sending json')
            self.file_object.close()

            return final_json, confusion_matrix_data


        except ZeroDivisionError as error:
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                'Exception occured in trainingModel method of the trainModel class. Exception message:  ' + str(
                                    error))
            self.log_writer.log(self.file_object,
                                'Model Selection Failed. Exited the trainingModel method of the trainModel class')
            self.file_object.close()
        except Exception as e:
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                'Exception occured in trainingModel method of the trainModel class. Exception message:  ' + str(
                                    e))
            self.log_writer.log(self.file_object,
                                'Model Selection Failed. Exited the trainingModel method of the trainModel class')
            self.file_object.close()
















