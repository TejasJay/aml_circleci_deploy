import pandas as pd
from application_logging import logger




class Data_Getter:
    """
    This class shall  be used for obtaining the data from the source for training.

    Written By: Tejas Jay (TJ)
    Version: 1.0
    Revisions: None

    """
    def __init__(self):
        self.training_file='Training_FileFromDB/InputFile.csv'
        self.log_writer = logger.App_Logger()




    def get_data(self):
        """
        Method Name: get_data
        Description: This method reads the data from source.
        Output: A pandas DataFrame.
        On Failure: Raise Exception

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None

        """
        self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.log_writer.log(self.file_object,'Entered the get_data method of the Data_Getter class')
        self.file_object.close()
        try:
            self.data = pd.read_csv(self.training_file) # reading the data file
            #self.data = self.data.drop(columns='unique_ID', axis=1, inplace=True)
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,'Data Load Successful.Exited the get_data method of the Data_Getter class')
            self.file_object.close()

            return self.data
        except Exception as e:
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,'Exception occured in get_data method of the Data_Getter class. Exception message: '+str(e))
            self.log_writer.log(self.file_object,
                                   'Data Load Unsuccessful.Exited the get_data method of the Data_Getter class')
            self.file_object.close()

            raise e