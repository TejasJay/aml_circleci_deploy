import pickle
import os
import shutil
from application_logging import logger


class File_Operation:
    """
    This class shall be used to save the model after training
    and load the saved model for prediction.

    Written By: Tejas Jay (TJ)
    Version: 1.0
    Revisions: None

    """
    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.model_directory = 'models/'





    def save_model(self,model,filename):
        """
        Method Name: save_model
        Description: Save the model file to directory
        Outcome: File gets saved
        On Failure: Raise Exception

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None
        """
        self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.log_writer.log(self.file_object, 'Entered the save_model method of the File_Operation class')
        self.file_object.close()
        try:
            path = os.path.join(self.model_directory,filename) #create seperate directory for each cluster
            if os.path.isdir(path): #remove previously existing models for each clusters
                shutil.rmtree(self.model_directory)
                os.makedirs(path)
            else:
                os.makedirs(path) #
            with open(path +'/' + filename+'.pickle','wb') as f:
                pickle.dump(model, f) # save the model to file

            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                   'Model File '+filename+' saved. Exited the save_model method of the Model_Finder class')
            self.file_object.close()

            return 'success'
        except Exception as e:
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,'Exception occured in save_model method of the Model_Finder class. Exception message:  ' + str(e))
            self.log_writer.log(self.file_object,
                                   'Model File '+filename+' could not be saved. Exited the save_model method of the Model_Finder class')
            self.file_object.close()

            raise Exception()











    def load_model(self,filename):
        """
        Method Name: load_model
        Description: load the model file to memory
        Output: The Model file loaded in memory
        On Failure: Raise Exception

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None
        """
        self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.log_writer.log(self.file_object, 'Entered the load_model method of the File_Operation class')
        self.file_object.close()

        try:
            with open(self.model_directory + filename + '/' + filename + '.pickle',
                      'rb') as f:
                self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
                self.log_writer.log(self.file_object,
                                       'Model File ' + filename + ' loaded. Exited the load_model method of the Model_Finder class')
                self.file_object.close()

                return pickle.load(f)
        except Exception as e:
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                   'Exception occured in load_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.log_writer.log(self.file_object,
                                   'Model File ' + filename + ' could not be saved. Exited the load_model method of the Model_Finder class')
            self.file_object.close()

            raise Exception()







    def delete_log_Records(self):
        """
        Method Name: deleteRecords
        Description: This method is used to delete all the data from the table in cassandra database.
        Output: None
        On Failure: Raise Exception

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None
        """

        try:
            path = 'Training_Logs/Training_Main_Log.txt'
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            raise e




    def delete_log_pred_Records(self):
        """
        Method Name: deleteRecords
        Description: This method is used to delete all the data from the table in cassandra database.
        Output: None
        On Failure: Raise Exception

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None
        """

        try:
            path = 'Prediction_Logs/Prediction_Log.txt'
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            raise e

