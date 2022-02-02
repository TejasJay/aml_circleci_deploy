import shutil
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import pandas as pd
from os import listdir
import csv
from application_logging.logger import App_Logger








class dBOperation:
    """
    This class shall be used for handling all the cassandra operations.

    Written By: Tejas Jay (TJ)
    Version: 1.0
    Revisions: None

    """

    def __init__(self, path_secure, user_id, secure_key, key_space, table_name):
        self.path = 'Training_Database/'
        self.badFilePath = "Training_Raw_files_validated/Bad_Raw"
        self.goodFilePath = "Training_Raw_files_validated/Good_Raw"
        self.path_secure = path_secure
        self.user_id = user_id
        self.secure_key = secure_key
        self.key_space = key_space
        self.table_name = table_name
        self.logging = App_Logger()







    def cassandra_connection(self):
        """
        Method Name: cassandra_connection
        Description: This method is used to connect to the cassandra database.
        Output: None
        On Failure: Raise Exception

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None
        """
        self.log_file = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.logging.log(self.log_file, 'Entered the cassandra_connection method of dBOperation class')
        self.log_file.close()
        try:
            cloud_config = {'secure_connect_bundle': self.path_secure}
            auth_provider = PlainTextAuthProvider(self.user_id, self.secure_key)
            cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
            session = cluster.connect()
            return session
        except Exception as e:
            self.log_file = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.logging.log(self.log_file,
                             'Unsuccessful in executing cassandra_connection method of the dBOperation class: error message is: ' + str(
                                 e))
            self.log_file.close()

            raise e






    def isconnectionestablished(self):
        """
        Method Name: isconnectionestablished
        Description: This method is used to check if the connection to the cassandra database is established.
        Output: None
        On Failure: Raise Exception

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None
        """
        self.log_file = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.logging.log(self.log_file, 'Entered the isconnectionestablished method of dBOperation class')
        self.log_file.close()

        try:
            if self.cassandra_connection():
                return True
        except Exception as e:
            self.log_file = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.logging.log(self.log_file,
                             'Unsuccessful in executing isconnectionestablished method of the dBOperation class: error message is: ' + str(
                                 e))
            self.log_file.close()

            raise e







    def get_key_space(self):
        """
        Method Name: get_key_space
        Description: This method is used to connect to the key_space of the cassandra database.
        Output: None
        On Failure: Raise Exception

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None
        """
        self.log_file = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.logging.log(self.log_file, 'Entered the get_key_space method of dBOperation class')
        self.log_file.close()

        try:
            session = self.cassandra_connection()
            row = session.execute("use {}".format(self.key_space))
            return row
        except Exception as e:
            self.log_file = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.logging.log(self.log_file,
                             'Unsuccessful in executing get_key_space method of the dBOperation class: error message is: ' + str(
                                 e))
            self.log_file.close()

            raise e






    def create_table(self):
        """
        Method Name: create_table
        Description: This method is used to create a table in cassandra database.
        Output: creates table
        On Failure: Raise Exception

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None
        """
        self.log_file = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.logging.log(self.log_file, 'Entered the create_table method of dBOperation class')
        self.log_file.close()

        try:
            session = self.cassandra_connection()
            row = session.execute(
                "CREATE TABLE IF NOT EXISTS {}.{table}(unique_ID UUID PRIMARY KEY, Transacted_amount float , Orig_OLD_acc_balance float, Orig_NEW_acc_balance float, Dest_OLD_acc_balance float,Dest_NEW_acc_balance float, latitude float, longitude float, total_hrs_from_first_trans float, transaction_type int, SAR_Fraud int);".format(
                    self.key_space, table=self.table_name)).one()
            return row
        except Exception as e:
            self.log_file = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.logging.log(self.log_file,
                             'In exception block of create_table method of dBOperation class: error message: ' + str(e))
            self.log_file.close()

            raise e







    def insertIntoTableGoodData(self):

        """
        Method Name: insertIntoTableGoodData
        Description: This method is used to insert rows into the table in cassandra database.
        Output: None
        On Failure: Raise Exception

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None
        """
        self.log_file = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.logging.log(self.log_file, 'Entered the insertIntoTableGoodData method of dBOperation class')
        self.log_file.close()

        session = self.cassandra_connection()
        goodFilePath = self.goodFilePath
        badFilePath = self.badFilePath
        onlyfiles = [f for f in listdir(goodFilePath)]

        for file in onlyfiles:
            try:
                with open(goodFilePath + '/' + file, "r") as f:
                    next(f)
                    reader = csv.reader(f, delimiter="\n")
                    for line in enumerate(reader):
                        for list_ in (line[1]):
                            try:
                                session.execute(
                                    "insert into {}.{table} (unique_ID,Transacted_amount,Orig_OLD_acc_balance,Orig_NEW_acc_balance,Dest_OLD_acc_balance,Dest_NEW_acc_balance,latitude,longitude,total_hrs_from_first_trans,transaction_type,SAR_Fraud) values({values});".format(
                                        self.key_space, table=self.table_name, values=list_))
                                #self.logging.log(self.log_file, " %s: File loaded successfully!!" % file)
                            except Exception as e:
                                raise e

            except Exception as e:
                self.log_file = open("Training_Logs/Training_Main_Log.txt", 'a+')
                self.logging.log(self.log_file, "Error while creating table: %s " % e)
                shutil.move(goodFilePath + '/' + file, badFilePath)
                self.logging.log(self.log_file, "File Moved Successfully %s" % file)
                self.log_file.close()
        #self.log_file.close()






    def selectingDatafromtableintocsv(self):
        """
        Method Name: selectingDatafromtableintocsv
        Description: This method is used to get all data from the table in cassandra database.
        Output: None
        On Failure: Raise Exception

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None
        """
        self.log_file = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.logging.log(self.log_file, 'Entered the selectingDatafromtableintocsv method of dBOperation class')
        self.log_file.close()

        self.fileFromDb = 'Training_FileFromDB/'
        self.fileName = 'InputFile.csv'
        try:
            session = self.cassandra_connection()

            results = session.execute("select * from  {}.{table}  ;".format(self.key_space, table=self.table_name))

            dataframe = pd.DataFrame(results)

            dataframe.to_csv(self.fileFromDb + self.fileName, index=False)

            self.log_file = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.logging.log(self.log_file, "File exported successfully!!!")
            self.log_file.close()

        except Exception as e:
            self.log_file = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.logging.log(self.log_file,
                             'In exception block of create_table method of dBOperation class: error message: ' + str(e))
            self.log_file.close()

            raise e




    def deleteRecords(self):
        """
        Method Name: deleteRecords
        Description: This method is used to delete all the data from the table in cassandra database.
        Output: None
        On Failure: Raise Exception

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None
        """
        self.log_file = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.logging.log(self.log_file,'Entered the deleteRecords method of dBOperation class')
        self.log_file.close()

        try:
            session = self.cassandra_connection()
            session.execute("DROP TABLE IF EXISTS {}.{table};".format(self.key_space, table=self.table_name))
        except Exception as e:
            self.log_file = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.logging.log(self.log_file,'Unsuccessful in executing deleteRecords method of the dBOperation class: error message is: ' + str(e))
            self.log_file.close()

            raise e






