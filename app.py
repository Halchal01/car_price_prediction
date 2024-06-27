from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__) 
url ='https://raw.githubusercontent.com/Halchal01/car_price_prediction/main/car_price_prediction.csv'

# Load and preprocess the dataset
df = pd.read_csv(url)

# Extract unique values for mapping
mf = df['Manufacturer'].unique()
cat = df['Category'].unique()
li = df['Leather interior'].unique()
ft = df['Fuel type'].unique()
gbt = df['Gear box type'].unique()
dw = df['Drive wheels'].unique()
wh = df['Wheel'].unique()
c = df['Color'].unique()

# Data cleaning and preprocessing
df['Levy'] = df['Levy'].str.replace('-', '')
df['Levy'] = pd.to_numeric(df['Levy'], errors='coerce')
df['Levy'] = df['Levy'].fillna(method='ffill')
df['Prod. year'] = pd.to_datetime(df['Prod. year'], format='%Y')
df['Prod. year'] = df['Prod. year'].dt.year
df['Engine volume'] = pd.to_numeric(df['Engine volume'], errors='coerce')
df['Engine volume'] = df['Engine volume'].fillna(method='ffill')
df['Mileage'] = df['Mileage'].str.replace('km', '')
df['Mileage'] = pd.to_numeric(df['Mileage'], errors='coerce')
df['Mileage'] = df['Mileage'].fillna(method='ffill')
df['Doors'] = df['Doors'].str.replace('-May', '')
df['Doors'] = df['Doors'].str.replace('-Mar', '')
df['Doors'] = df['Doors'].str.replace('>', '')
df['Doors'] = pd.to_numeric(df['Doors'], errors='coerce')

# Drop unnecessary columns
df.drop(columns=['ID', 'Model'], inplace=True)

# Encode categorical features
le = LabelEncoder()
df['Manufacturer'] = le.fit_transform(df['Manufacturer'])
df['Category'] = le.fit_transform(df['Category'])
df['Leather interior'] = le.fit_transform(df['Leather interior'])
df['Fuel type'] = le.fit_transform(df['Fuel type'])
df['Gear box type'] = le.fit_transform(df['Gear box type'])
df['Drive wheels'] = le.fit_transform(df['Drive wheels'])
df['Wheel'] = le.fit_transform(df['Wheel'])
df['Color'] = le.fit_transform(df['Color'])

# Create dictionaries for label encoding
mfd = {mf[i]: df['Manufacturer'].unique()[i] for i in range(len(mf))}
catd = {cat[i]: df['Category'].unique()[i] for i in range(len(cat))}
lid = {li[i]: df['Leather interior'].unique()[i] for i in range(len(li))}
ftd = {ft[i]: df['Fuel type'].unique()[i] for i in range(len(ft))}
gbtd = {gbt[i]: df['Gear box type'].unique()[i] for i in range(len(gbt))}
dwd = {dw[i]: df['Drive wheels'].unique()[i] for i in range(len(dw))}
whd = {wh[i]: df['Wheel'].unique()[i] for i in range(len(wh))}
cd = {c[i]: df['Color'].unique()[i] for i in range(len(c))}

# Define features (X) and target (y)
X = df.drop('Price', axis=1)  # Make sure 'Price' is the correct column name for car prices
y = df['Price']

# Train the model
model = DecisionTreeRegressor()
model.fit(X, y)

@app.route('/') 
def home(): 
    return render_template('a.html')

  
  
@app.route('/rec', methods=['POST']) 
def processdata(): 
    try:
        l = float(request.form.get('l'))
        m = mfd[request.form.get('m')]
        p = float(request.form.get('p'))
        c = catd[request.form.get('c')]
        ll = lid[request.form.get('ll')]
        f = ftd[request.form.get('f')]
        e = float(request.form.get('e'))
        mi = float(request.form.get('mi'))
        cy = int(request.form.get('cy'))
        gb = gbtd[request.form.get('gb')]
        di = dwd[request.form.get('di')]
        do = int(request.form.get('do'))  # Verify if 'do' corresponds to any form input
        w = whd[request.form.get('w')]
        co = cd[request.form.get('co')]
        a = float(request.form.get('a'))

        arr1 = np.array([[l, m, p, c, ll, f, e, mi, cy, gb, di, do, w, co, a]])
        res = model.predict(arr1)
        return render_template('a.html', prediction=str(res[0]))
    except Exception as e:
        return str(e)

      
      
if __name__ == '__main__': 
    app.run()

