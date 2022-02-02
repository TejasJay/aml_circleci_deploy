import pandas as pd
from sklearn.preprocessing import StandardScaler
from application_logging import logger




class Preprocessor:
    """
    This class shall be used to clean and transform the data before training.

    Written By: Tejas Jay (TJ)
    Version: 1.0
    Revisions: None

    """

    def __init__(self):
        self.log_writer = logger.App_Logger()




    def remove_columns(self,data,columns):
        """
        Method Name: remove_columns
        Description: This method removes the given columns from a pandas dataframe.
        Output: A pandas DataFrame after removing the specified columns.
        On Failure: Raise Exception

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None

        """
        self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.log_writer.log(self.file_object, 'Entered the remove_columns method of the Preprocessor class')
        self.file_object.close()
        self.data= data
        self.columns= columns
        try:
            self.useful_data = self.data.drop(labels=self.columns, axis=1) # drop the labels specified in the columns
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                   'Column removal Successful.Exited the remove_columns method of the Preprocessor class')
            self.file_object.close()

            return self.useful_data

        except Exception as e:
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,'Exception occured in remove_columns method of the Preprocessor class. Exception message:  '+str(e))
            self.log_writer.log(self.file_object,'Column removal Unsuccessful. Exited the remove_columns method of the Preprocessor class')
            self.file_object.close()

            raise e




    def separate_label_feature(self, data, label_column_name):
        """
        Method Name: separate_label_feature
        Description: This method separates the features and a Label Coulmns.
        Output: Returns two separate Dataframes, one containing features and the other containing Labels .
        On Failure: Raise Exception

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None

        """
        self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.log_writer.log(self.file_object, 'Entered the separate_label_feature method of the Preprocessor class')
        self.file_object.close()

        try:
            self.Y = data[label_column_name]  # Filter the Label columns
            self.X = data.drop(labels=label_column_name, axis=1) # drop the columns specified and separate the feature columns
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,
                                   'Label Separation Successful. Exited the separate_label_feature method of the Preprocessor class')
            self.file_object.close()

            return self.Y, self.X
        except Exception as e:
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,'Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  ' + str(e))
            self.log_writer.log(self.file_object, 'Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
            self.file_object.close()
            raise Exception()





    def scale_numerical_columns(self,data):
        """
        Method Name: scale_numerical_columns
        Description: This method scales the numerical values using the Standard scaler.
        Output: A dataframe with scaled
        On Failure: Raise Exception

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None
        """
        self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.log_writer.log(self.file_object,'Entered the scale_numerical_columns method of the Preprocessor class')
        self.file_object.close()

        self.data=data
        try:
            self.num_df = self.data.select_dtypes(include=['int64','float64']).copy()
            self.scaler = StandardScaler()
            self.scaled_data = self.scaler.fit_transform(self.num_df)
            self.scaled_num_df = pd.DataFrame(data=self.scaled_data, columns=self.num_df.columns)
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object, 'scaling for numerical values successful. Exited the scale_numerical_columns method of the Preprocessor class')
            self.file_object.close()

            return self.scaled_num_df
        except Exception as e:
            self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.log_writer.log(self.file_object,'Exception occured in scale_numerical_columns method of the Preprocessor class. Exception message:  ' + str(e))
            self.log_writer.log(self.file_object, 'scaling for numerical columns Failed. Exited the scale_numerical_columns method of the Preprocessor class')
            self.file_object.close()

            raise e


