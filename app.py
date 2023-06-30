from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods = ['POST','GET'])
def predict():
	# Getting input values from user
	gender = int(request.form['gender'])
	branch = int(request.form['branch'])
	ssc_cgpa = float(request.form['ssc_cgpa'])
	hsc_cgpa = float(request.form['hsc_cgpa'])
	ug_cgpa = float(request.form['ug_cgpa'])
	workexp = int(request.form['workexp'])
	num_proj = int(request.form['num_proj'])
	num_cert = int(request.form['num_cert'])
	git_acc = int(request.form['git_acc'])
	git_repo = int(request.form['git_repo'])

	# Creating input data array
	input_data = np.array([[gender, branch, ssc_cgpa, hsc_cgpa, ug_cgpa,workexp, num_proj, num_cert, git_acc, git_repo]])

	# Predicting whether the status using pre-trained model
	pred = model.predict(input_data)[0]

	# Rendering the result page with predicted value of status
	# return render_template('index.html', pred = pred)

	if pred == 1:
		return render_template('output.html', pred = "The student is PLACED")
	else:
		return render_template('output.html', pred = "The student is NOT PLACED")



if __name__ == "__main__":
	app.run(debug = True)