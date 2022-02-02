from wsgiref import simple_server
from flask import Flask, request, render_template
from flask import Response
import os
from flask_cors import CORS, cross_origin
from prediction_Validation_Insertion import pred_validation
from trainingModel import trainModel
from training_Validation_Insertion import train_validation
import flask_monitoringdashboard as dashboard
from predictFromModel import prediction
from data_preprocessing.preprocessing import Preprocessor
import json
import numpy as np
import pandas as pd
import pickle



os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')




app = Flask(__name__)
dashboard.bind(app)
CORS(app)



@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('predict_train.html')



@app.route("/predict_page", methods=['GET'])
@cross_origin()
def predict_page():
    return render_template('predict.html')


@app.route("/train_page", methods=['GET'])
@cross_origin()
def train_page():
    return render_template('train.html')



@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRouteClient():
    try:
        if request.json is not None:

            path = request.json['folderpath']

            pred_val = pred_validation(path) #object initialization

            pred_val.prediction_validation() #calling the prediction_validation function

            pred = prediction(path, "svc") #object initialization

            # predicting for dataset present in database
            json_prediction_output = pred.predictionFromModel()

            return json_prediction_output


        elif request.form is not None:

            path = request.form['folderpath']

            model_name = request.form['modelname']

            print(path)

            print(model_name)

            pred_val = pred_validation(path) #object initialization

            pred_val.prediction_validation() #calling the prediction_validation function

            pred = prediction(path, model_name) #object initialization

            # predicting for dataset present in database
            pred.predictionFromModel()


        else:
            print('Nothing Matched')

    except ValueError:
        return Response("Error Occurred! %s" %ValueError)
    except KeyError:
        return Response("Error Occurred! %s" %KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" %e)

    return Response("Prediction successfull!")





@app.route("/viewtrainedfile", methods=['GET'])
@cross_origin()
def viewTraining():
    df2 = pd.read_csv("Training_Logs\Training_Main_Log.txt", sep="\t\t\t")
    df2.columns = ['Date', 'time', 'log_message']

    #table1 = df1.to_html(index=False )
    df1 = pd.read_csv("csv_output_files\confusion_matrix.csv")
    #table2 = df2.to_html(index=False, columns= ['Actual', 'predicted', 'predicted_n'])
    return render_template('training_results.html', column_names1=df1.columns.values, row_data1=list(df1.values.tolist()), zip1=zip,
                           column_names2=df2.columns, row_data2=list(df2.values.tolist()), zip2=zip)





@app.route("/viewpredictedfile", methods=['GET'])
@cross_origin()
def viewprediction():
    df2 = pd.read_csv("Prediction_Logs\Prediction_Log.txt", sep="\t\t\t")
    df2.columns = ['Date', 'time', 'log_message']

    #table1 = df1.to_html(index=False )
    df1 = pd.read_csv("Prediction_Output_File\Predictions.csv")
    #table2 = df2.to_html(index=False, columns= ['Actual', 'predicted', 'predicted_n'])
    return render_template('prediction_results.html', column_names1=df1.columns.values, row_data1=list(df1.values.tolist()), zip1=zip,
                           column_names2=df2.columns, row_data2=list(df2.values.tolist()), zip2=zip)







@app.route('/train', methods=['POST'])
@cross_origin()
def trainRouteClient():

    try:
        if request.json is not None:

            path = request.json['folderpath']

            train_valObj = train_validation(path)  # object initialization

            train_valObj.train_validation()  # calling the prediction_validation function

            trainModelObj = trainModel(path, "F1")  # object initialization

            # predicting for dataset present in database
            final_json = trainModelObj.trainingModel()

            return final_json



        elif request.form is not None:

            path = request.form['folderpath']

            metrix_name = request.form['metrix']

            print(path)

            print(metrix_name)

            train_valObj = train_validation(path)  # object initialization

            train_valObj.train_validation()  # calling the prediction_validation function

            trainModelObj = trainModel(path, metrix_name)  # object initialization

            # predicting for dataset present in database
            trainModelObj.trainingModel()


        else:
            print('Nothing Matched')


    except ValueError:

        return Response("Error Occurred! %s" % ValueError)

    except KeyError:

        return Response("Error Occurred! %s" % KeyError)

    except Exception as e:

        return Response("Error Occurred! %s" % e)

    return Response("Training successfull!")








port = int(os.getenv("PORT",5000))
if __name__ == "__main__":
    host = '0.0.0.0'
    #port = 5000
    httpd = simple_server.make_server(host, port, app)
    # print("Serving on %s %d" % (host, port))
    httpd.serve_forever()
