from flask import Flask,request,jsonify
import pickle
import pandas as pd
import numpy as np

with open("loan_approval.pkl","rb") as file_reading_obj:
    train_model = pickle.load(file_reading_obj)

loan_approval_app = Flask(__name__)
@loan_approval_app.route("/loan_approval_status", methods = ["POST"])
def loan_approval_status():
    # Get the data posted as a JSON object
    processes = {'loan_intent':'le','onehot','minmax'}
    data = request.get_json()
    user_input = data.get("user_ip")
    if not user_input or not isinstance(user_input,list):
        return jsonify({"error":"input should be in list format"}),400

    user_input_array = np.array(list(data.values())).reshape(1, -1)
    prediction = train_model.predict(user_input_array)
    return jsonify({"predicted Output": prediction.tolist()})
if __name__ == "__main__":
    loan_approval_app.run(debug = True)