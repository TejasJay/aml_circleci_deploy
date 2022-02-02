from os import listdir
import pandas
from application_logging.logger import App_Logger


class dataTransform:
    """
    This class shall be used for transforming the Good Raw Training Data before loading it in Database!!.

    Written By: Tejas Jay (TJ)
    Version: 1.0
    Revisions: None

    """

    def __init__(self):
        self.goodDataPath = "Training_Raw_files_validated/Good_Raw"
        self.logger = App_Logger()





    def datatransformation(self):
        """
          Method Name: datatransformation
          Description: This method replaces the missing values in columns with "NULL" to
                  store in the table. All the object type columns are removed.

          Written By: Tejas Jay (TJ)
          Version: 1.0
          Revisions: None

        """
        self.file = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.logger.log(self.file, 'Entered datatransformation method of dataTransform class')
        self.file.close()
        try:
            onlyfiles = [f for f in listdir(self.goodDataPath)]

            for file in onlyfiles:
                data = pandas.read_csv(self.goodDataPath + "/" + file)
                data.fillna('NULL', inplace=True)
                # Our interest is only on the +ve alerts. Hence seperating them from the whole data.
                positive_alerts_df = data.loc[data['alerts_generated'] == 1]
                # we drop the alerts_generated feature from the data, alerts_generated feature does not contribute any value, as we seperted it.
                positive_alerts_df.drop(columns='alerts_generated', inplace=True)
                positive_alerts_df.reset_index(drop=True, inplace=True)
                # we find that the day number and total_hrs_from_first_trans to be corelated, which makes sense, hence dropping day_number feature
                # Also we drop the feature step as it is same as total_hrs_from_first_trans.
                positive_alerts_df.drop(columns=['day_number', 'step'], inplace=True)
                # new_data is the proessed data
                new_data = positive_alerts_df
                # dropping all the features with type as object, it does not contribute to model training
                new_data.drop(columns=['type', 'Originator_acc_num', 'Destination_acc_num', 'street', 'country', 'state','Day_time'], axis=1, inplace=True)
                new_data.to_csv(self.goodDataPath + "/" + file, index=None, header=True)
                #self.logger.log(self.file, " %s: Quotes added successfully!!" % file)
        except Exception as e:
            self.file = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.logger.log(self.file, 'Unsuccessful in executing datatransformation method of dataTransform class : ' + str(e))
            self.file.close()
            raise e
        #self.file.close()











