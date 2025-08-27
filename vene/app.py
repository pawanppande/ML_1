import pickle
from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import pandas as pd 

from sklearn.preprocessing import StandardScaler, OneHotEncoder

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')


