from datetime import datetime
from os import listdir
import os
import re
import json
import shutil
import pandas as pd
from application_logging.logger import App_Logger



class Raw_Data_validation:
    """
    This class shall be used for handling all the validation done on the Raw Training Data.
    Written By: Tejas Jay (TJ)
    Version: 1.0
    Revisions: None

    """
    def __init__(self,path):
        self.Batch_Directory = path
        self.schema_path = 'schema_training.json'
        self.logger = App_Logger()




    def valuesFromSchema(self):
        """
        Method Name: valuesFromSchema
        Description: This method extracts all the relevant information from the pre-defined "Schema" file.
        Output: LengthOfDateStampInFile, LengthOfTimeStampInFile, column_names, Number of Columns
        On Failure: Raise ValueError,KeyError,Exception

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None

        """
        self.file = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.logger.log(self.file, 'Entered valuesFromSchema method of Raw_Data_validation class')
        self.file.close()
        try:
            with open(self.schema_path, 'r') as f:
                dic = json.load(f)
                f.close()
            pattern = dic['SampleFileName']
            LengthOfDateStampInFile = dic['LengthOfDateStampInFile']
            column_names = dic['ColName']
            NumberofColumns = dic['NumberofColumns']

            message ="LengthOfDateStampInFile:: %s" %LengthOfDateStampInFile + "\t" +  "NumberofColumns:: %s" % NumberofColumns + "\n"
            self.file = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.logger.log(self.file, message)
            self.file.close()

        except Exception as e:
            self.file = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.logger.log(self.file, 'Unsuccessful in executing valuesFromSchema method of Raw_Data_validation class : ' + str(e))
            self.file.close()
            raise e

        return LengthOfDateStampInFile, column_names, NumberofColumns




    def manualRegexCreation(self):
        """
        Method Name: manualRegexCreation
        Description: This method contains a manually defined regex based on the "FileName" given in "Schema" file.
                    This Regex is used to validate the filename of the training data.
        Output: Regex pattern
        On Failure: None

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None

        """
        regex = "['AmlSarScreening']+['\_'']+[\d]+\.csv"
        return regex




    def createDirectoryForGoodBadRawData(self):
        """
        Method Name: createDirectoryForGoodBadRawData
        Description: This method creates directories to store the Good Data and Bad Data
                    after validating the training data.
        Output: None
        On Failure: OSError

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None

        """
        self.file = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.logger.log(self.file, 'Entered createDirectoryForGoodBadRawData method of Raw_Data_validation class')
        self.file.close()

        try:
            path = os.path.join("Training_Raw_files_validated/", "Good_Raw/")
            if not os.path.isdir(path):
                os.makedirs(path)
            path = os.path.join("Training_Raw_files_validated/", "Bad_Raw/")
            if not os.path.isdir(path):
                os.makedirs(path)

        except Exception as e:
            self.file = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.logger.log(self.file, 'Unsuccessful in executing createDirectoryForGoodBadRawData method of Raw_Data_validation class : ' + str(e))
            self.file.close()
            raise e




    def deleteExistingGoodDataTrainingFolder(self):

        """
        Method Name: deleteExistingGoodDataTrainingFolder
        Description: This method deletes the directory made  to store the Good Data
                    after loading the data in the table. Once the good files are
                    loaded in the DB,deleting the directory ensures space optimization.
        Output: None
        On Failure: OSError

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None

        """
        self.file = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.logger.log(self.file, 'Entered deleteExistingGoodDataTrainingFolder method of Raw_Data_validation class')
        self.file.close()

        try:
            path = 'Training_Raw_files_validated/'
            if os.path.isdir(path + 'Good_Raw/'):
                shutil.rmtree(path + 'Good_Raw/')
                self.file = open("Training_Logs/Training_Main_Log.txt", 'a+')
                self.logger.log(self.file,"GoodRaw directory deleted successfully!!!")
                self.file.close()
        except Exception as e:
            self.file = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.logger.log(self.file, 'Unsuccessful in executing deleteExistingGoodDataTrainingFolder method of Raw_Data_validation class : ' + str(e))
            self.file.close()
            raise e




    def deleteExistingBadDataTrainingFolder(self):
        """
        Method Name: deleteExistingBadDataTrainingFolder
        Description: This method deletes the directory made  to store the Bad Data
        Output: None
        On Failure: OSError

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None

        """
        self.file = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.logger.log(self.file, 'Entered deleteExistingBadDataTrainingFolder method of Raw_Data_validation class')
        self.file.close()

        try:
            path = 'Training_Raw_files_validated/'
            if os.path.isdir(path + 'Bad_Raw/'):
                shutil.rmtree(path + 'Bad_Raw/')
                self.file = open("Training_Logs/Training_Main_Log.txt", 'a+')
                self.logger.log(self.file,"BadRaw directory deleted before starting validation!!!")
                self.file.close()
        except Exception as e:
            self.file = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.logger.log(self.file, 'Unsuccessful in executing deleteExistingBadDataTrainingFolder method of Raw_Data_validation class : ' + str(e))
            self.file.close()
            raise e





    def moveBadFilesToArchiveBad(self):
        """
        Method Name: moveBadFilesToArchiveBad
        Description: This method deletes the directory made  to store the Bad Data
                     after moving the data in an archive folder. We archive the bad
                      files to send them back to the client for invalid data issue.
        Output: None
        On Failure: OSError

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None

        """
        now = datetime.now()
        date = now.date()
        time = now.strftime("%H%M%S")
        self.file = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.logger.log(self.file, 'Entered moveBadFilesToArchiveBad method of Raw_Data_validation class')
        self.file.close()

        try:
            source = 'Training_Raw_files_validated/Bad_Raw/'
            if os.path.isdir(source):
                path = "TrainingArchiveBadData"
                if not os.path.isdir(path):
                    os.makedirs(path)
                dest = 'TrainingArchiveBadData/BadData_' + str(date)+"_"+str(time)
                if not os.path.isdir(dest):
                    os.makedirs(dest)
                files = os.listdir(source)
                for f in files:
                    if f not in os.listdir(dest):
                        shutil.move(source + f, dest)
                self.file = open("Training_Logs/Training_Main_Log.txt", 'a+')
                self.logger.log(self.file,"Bad files moved to archive")
                self.file.close()

                path = 'Training_Raw_files_validated/'
                if os.path.isdir(path + 'Bad_Raw/'):
                    shutil.rmtree(path + 'Bad_Raw/')
                self.file = open("Training_Logs/Training_Main_Log.txt", 'a+')
                self.logger.log(self.file,"Bad Raw Data Folder Deleted successfully!!")
                self.file.close()
        except Exception as e:
            self.file = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.logger.log(self.file, 'Unsuccessful in executing moveBadFilesToArchiveBad method of Raw_Data_validation class : ' + str(e))
            self.file.close()
            raise e





    def validationFileNameRaw(self,regex,LengthOfDateStampInFile):
        """
        Method Name: validationFileNameRaw
        Description: This function validates the name of the training csv files as per given name in the schema!
                    Regex pattern is used to do the validation.If name format do not match the file is moved
                    to Bad Raw Data folder else in Good raw data.
        Output: None
        On Failure: Exception

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None

        """
        #pattern = "['AmlSarScreening']+['\_'']+[\d]+\.csv"
        # delete the directories for good and bad data in case last run was unsuccessful and folders were not deleted.
        self.deleteExistingBadDataTrainingFolder()
        self.deleteExistingGoodDataTrainingFolder()
        #create new directories
        self.createDirectoryForGoodBadRawData()
        onlyfiles = [f for f in listdir(self.Batch_Directory)]

        self.file = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.logger.log(self.file, 'Entered validationFileNameRaw method of Raw_Data_validation class')
        self.file.close()

        try:
            for filename in onlyfiles:
                if (re.match(regex, filename)):
                    splitAtDot = re.split('.csv', filename)
                    splitAtDot = (re.split('_', splitAtDot[0]))
                    if len(splitAtDot[1]) == LengthOfDateStampInFile:
                        shutil.copy("Training_Batch_Files/" + filename, "Training_Raw_files_validated/Good_Raw")
                        #self.logger.log(self.file,"Valid File name!! File moved to GoodRaw Folder :: %s" % filename)

                    else:
                        shutil.copy("Training_Batch_Files/" + filename, "Training_Raw_files_validated/Bad_Raw")
                        #self.logger.log(self.file,"Invalid File Name!! File moved to Bad Raw Folder :: %s" % filename)
                else:
                    shutil.copy("Training_Batch_Files/" + filename, "Training_Raw_files_validated/Bad_Raw")
                    #self.logger.log(self.file, "Invalid File Name!! File moved to Bad Raw Folder :: %s" % filename)

            #self.file.close()

        except Exception as e:
            self.file = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.logger.log(self.file, 'Unsuccessful in executing validationFileNameRaw method of Raw_Data_validation class : ' + str(e))
            self.file.close()
            raise e





    def validateColumnLength(self,NumberofColumns):
        """
        Method Name: validateColumnLength
        Description: This function validates the number of columns in the csv files.
                    It is should be same as given in the schema file.
                    If not same file is not suitable for processing and thus is moved to Bad Raw Data folder.
                    If the column number matches, file is kept in Good Raw Data for processing.
                    The csv file is missing the first column name, this function changes the missing name to "Wafer".
        Output: None
        On Failure: Exception

        Written By: Tejas Jay
        Version: 1.0
        Revisions: None

        """
        self.file = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.logger.log(self.file, 'Entered validateColumnLength method of Raw_Data_validation class')
        self.file.close()

        try:
            self.file = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.logger.log(self.file,"Column Length Validation Started!!")
            self.file.close()

            for file in listdir('Training_Raw_files_validated/Good_Raw/'):
                csv = pd.read_csv("Training_Raw_files_validated/Good_Raw/" + file)
                if csv.shape[1] == NumberofColumns:
                    pass
                else:
                    shutil.move("Training_Raw_files_validated/Good_Raw/" + file, "Training_Raw_files_validated/Bad_Raw")
                    self.file = open("Training_Logs/Training_Main_Log.txt", 'a+')
                    self.logger.log(self.file, "Invalid Column Length for the file!! File moved to Bad Raw Folder :: %s" % file)
                    self.file.close()

            self.file = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.logger.log(self.file, "Column Length Validation Completed!!")
            self.file.close()

        except Exception as e:
            self.file = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.logger.log(self.file, 'Unsuccessful in executing validateColumnLength method of Raw_Data_validation class : ' + str(e))
            self.file.close()
            raise e





    def validateMissingValuesInWholeColumn(self):
        """
        Method Name: validateMissingValuesInWholeColumn
        Description: This function validates if any column in the csv file has all values missing.
                    If all the values are missing, the file is not suitable for processing.
                    SUch files are moved to bad raw data.
        Output: None
        On Failure: Exception

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None

        """
        self.file = open("Training_Logs/Training_Main_Log.txt", 'a+')
        self.logger.log(self.file, 'Entered validateMissingValuesInWholeColumn method of Raw_Data_validation class')
        self.file.close()

        try:
            self.file = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.logger.log(self.file,"Missing Values Validation Started!!")
            self.file.close()

            for file in listdir('Training_Raw_files_validated/Good_Raw/'):
                csv = pd.read_csv("Training_Raw_files_validated/Good_Raw/" + file)
                count = 0
                for columns in csv:
                    if (len(csv[columns]) - csv[columns].count()) == len(csv[columns]):
                        count+=1
                        shutil.move("Training_Raw_files_validated/Good_Raw/" + file,
                                    "Training_Raw_files_validated/Bad_Raw")
                        #self.logger.log(self.file,"Invalid Column Length for the file!! File moved to Bad Raw Folder :: %s" % file)
                        break
                if count==0:
                    csv.to_csv("Training_Raw_files_validated/Good_Raw/" + file, index=None, header=True)

        except Exception as e:
            self.file = open("Training_Logs/Training_Main_Log.txt", 'a+')
            self.logger.log(self.file, 'Unsuccessful in executing validateMissingValuesInWholeColumn method of Raw_Data_validation class : ' + str(e))
            self.file.close()
            raise e














