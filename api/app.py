from flask import Flask
from flask import request
from joblib import load

app = Flask(__name__)
model_path = "svm_gamma=0.0005_C=2.joblib"
model = load(model_path)

@app.route("/")
def hello_world():
    return "<!-- hello --> <b> Hello, World!</b>"


# get x and y somehow    
#     - query parameter
#     - get call / methods
#     - post call / methods ** 

# @app.route("/sum", methods=['POST'])
# def sum():
#     x = request.json['x']
#     y = request.json['y']
#     z = x + y 
#     return {'sum':z}



# @app.route("/predict", methods=['POST'])
# def predict_digit():
#     image = request.json['image']
#     predicted = model.predict([image])
#     return {"y_predicted":int(predicted[0])}

# code for comparing the predictions of two images
@app.route("/compare_predict", methods=['POST'])
def compare_prediction():
    image1 = request.json['image1']
    image2 = request.json['image2']
    predicted1 = model.predict([image1])
    predicted2 = model.predict([image2])
    if int(predicted1[0]) == int(predicted2[0]):
        return {"y_predicted1":int(predicted1[0]) , "y_predicted2":int(predicted2[0]) , "Comparision result":"Same"}
    else:
        return {"y_predicted1":int(predicted1[0]) , "y_predicted2":int(predicted2[0]) , "Comparision result":"Not Same"}