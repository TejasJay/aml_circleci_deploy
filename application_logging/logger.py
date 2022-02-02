from datetime import datetime


class App_Logger:
    """
        This class shall  be used for logging.

        Written By: Tejas Jay (TJ)
        Version: 1.0
        Revisions: None

    """
    def __init__(self):
        pass

    def log(self, file_object, log_message):
        """
                Method Name: log
                Description: This method logs the message to the file_object file
                Output: text message.
                On Failure: None

                Written By: Tejas Jay (TJ)
                Version: 1.0
                Revisions: None

        """
        self.now = datetime.now()
        self.date = self.now.date()
        self.current_time = self.now.strftime("%H:%M:%S")
        file_object.write(
            str(self.date) + "\t\t\t" + str(self.current_time) + "\t\t\t" + log_message +"\n")
