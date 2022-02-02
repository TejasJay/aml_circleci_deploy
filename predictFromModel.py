"""
This is the Entry point for predicting the results from the Machine Learning Model.

Written By: Tejas Jay
Version: 1.0
Revisions: None

"""


from file_operations import file_methods
from data_preprocessing import preprocessing
from data_ingestion import data_loader_prediction
from application_logging import logger
from Prediction_Raw_Data_Validation.predictionDataValidation import Prediction_Data_validation



class prediction:

    def __init__(self,path, modelname):
        self.file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
        self.log_writer = logger.App_Logger()
        self.pred_data_val = Prediction_Data_validation(path)
        self.modelname = modelname



    def predictionFromModel(self):
        try:
            self.pred_data_val.deletePredictionFile() #deletes the existing prediction file from last run!

            self.log_writer.log(self.file_object,'Start of Prediction')


            data_getter=data_loader_prediction.Data_Getter_Pred()

            data=data_getter.get_data()

            preprocessor = preprocessing.Preprocessor()

            new_data= preprocessor.remove_columns(data, columns='unique_id')

            # Proceeding with more data pre-processing steps
            X = preprocessor.scale_numerical_columns(new_data)

            file_loader = file_methods.File_Operation()

            model = file_loader.load_model(self.modelname)

            result = (model.predict(X))

            result_list = list(result)

            model_prediction_output = data

            model_prediction_output['Fraud_Predictions'] = result_list

            path_csv = "Prediction_Output_File/Predictions.csv"

            model_prediction_output.to_csv(path_csv, index=False)

            #sample_json_prediction_output = model_prediction_output.sample(n=10)

            #json_prediction_output = sample_json_prediction_output.to_json(orient='index')

            json_prediction_output = model_prediction_output.to_json(orient='index')

            self.log_writer.log(self.file_object, 'End of Prediction')

            return json_prediction_output

        except Exception as e:
            self.log_writer.log(self.file_object, 'Error occured while running the prediction!! Error:: %s' %e)
            raise e




