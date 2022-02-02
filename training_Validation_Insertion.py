from Training_Raw_data_validation.rawValidation import Raw_Data_validation
from DataTypeValidation_Insertion_Training.DataTypeValidation import dBOperation
from DataTransform_Training.DataTransformation import dataTransform
from application_logging import logger
import os





class train_validation:
    def __init__(self, path):
        self.raw_data = Raw_Data_validation(path)
        self.dataTransform = dataTransform()
        self.cwd = os.getcwd()
        self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.log_writer = logger.App_Logger()
        self.path_secure = 'secure-connect-aml-sar.zip'
        self.user_id = 'cHjCrxnuAUNswjKsaTZhbjOZ'
        self.secure_key = '-8rSI4yaDAGhXrzIomt,b4tiqpQ8hyvcU77s6a8a+SkK05fOIJDcnBs7M15-_x5ZG_3_LlO,ssqCZMmn.JLQCrdZbRfnmG,x+rHrIQzaOsfP.jPEUmg74nA4,M,N1Nko'
        self.key_space = 'training'
        self.table_name = 'amltrain'




    def train_validation(self):
        try:
            #self.log_writer.log(self.file_object, 'Start of Validation on files for Training')
            # extracting values from prediction schema
            LengthOfDateStampInFile, column_names, noofcolumns = self.raw_data.valuesFromSchema()
            # getting the regex defined to validate filename
            regex = self.raw_data.manualRegexCreation()
            # validating filename of prediction files
            self.raw_data.validationFileNameRaw(regex, LengthOfDateStampInFile)
            # validating column length in the file
            self.raw_data.validateColumnLength(noofcolumns)
            # validating if any column has all values missing
            self.raw_data.validateMissingValuesInWholeColumn()
            #self.log_writer.log(self.file_object, "Raw Data Validation Complete!!")

            #self.log_writer.log(self.file_object, "Starting Data Transforamtion!!")
            # replacing blanks in the csv file with "Null" values and transforming the data to insert in table
            self.dataTransform.datatransformation()
            #self.log_writer.log(self.file_object, "DataTransformation Completed!!!")

            #self.log_writer.log(self.file_object,"Creating Training_Database and tables on the basis of given schema!!!")
            # create database with given name, if present open the connection! Create table with columns given in schema
            dBOperation_obj = dBOperation(self.path_secure, self.user_id, self.secure_key, self.key_space, self.table_name)
            # obtaining DB connection
            dBOperation_obj.cassandra_connection()
            # obtaining keyspace for the DB connection
            dBOperation_obj.get_key_space()
            #delete existing table
            dBOperation_obj.deleteRecords()
            # creating table
            dBOperation_obj.create_table()
            # inserting data into the table
            dBOperation_obj.insertIntoTableGoodData()
            # exporting the input csv for model training
            dBOperation_obj.selectingDatafromtableintocsv()

        except Exception as e:
            raise e








