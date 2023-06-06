from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

app=Flask(__name__)
cors=CORS(app)
model=pickle.load(open('LinearRegressionModel.pkl', 'rb'))
df=pd.read_csv('Cleaned Art.csv')

@app.route('/',methods=['GET','POST'])
def index():
    auction_location=sorted(df['auction_location'].unique())
    auction_weekday = sorted(df['auction_weekday'].unique())


    return render_template('index.html',auction_location=auction_location, auction_weekday=auction_weekday)


@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():

    auction_location=int(request.form.get('auction_location'))
    auction_weekday=request.form.get('auction_weekday')
    estimate_range = int(request.form.get('estimate_range'))
    surface=int(request.form.get('surface'))
    auction_year=int(request.form.get('auction_year'))
    auction_month = int(request.form.get('auction_month'))

    prediction=model.predict(pd.DataFrame(columns=['auction_location','auction_weekday', 'estimate_range','surface', 'auction_year','auction_month'],
                              data=np.array([[auction_location,auction_weekday,estimate_range,surface,auction_year,auction_month]])))
    print(prediction)

    return str(np.round(prediction[0],2))



if __name__=='__main__':


    app.run()
