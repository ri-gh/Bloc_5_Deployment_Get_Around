import uvicorn
import pandas as pd 
import numpy as np
from pydantic import BaseModel
from typing import Literal, Union
from fastapi import FastAPI
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

description = """
Get Around API helps you to estimate the rental price per day for your car 🤑. 
 

## Preview

Where you can: 
* get a `/preview` of a few rows of your dataset
* get `/unique-values` of a given column in your dataset
* get `/max` value of a given column iny your dataset
* get `/min` value of a given column iny your dataset
* get `/price_per_car_model` mean of price per day for each car model_key

## Machine-Learning

* `/predict` you can estimate the rental price per day for your car


Check out documentation for more information on each endpoint. 
"""

tags_metadata = [
    {
        "name": "Preview",
        "description": "Endpoints that deal with general datas",
    },
    {
        "name": "Machine-Learning",
        "description": "Endpoint that uses our Machine Learning model to estimate rental price"
    }

]

#on fait une instance 'app' de FastApi
app = FastAPI(
    title="🚗 Get Around API",
    description=description,
    version="0.1",
    contact={
        "name": "Get Around API - by R.G."
    },
    openapi_tags=tags_metadata
)

class PredictionFeatures(BaseModel):
    model_key: Literal['Citroën', 'Peugeot', 'PGO', 'Renault', 'Audi', 'BMW', 'Ford',
       'Mercedes', 'Opel', 'Porsche', 'Volkswagen', 'KIA Motors',
       'Alfa Romeo', 'Ferrari', 'Fiat', 'Lamborghini', 'Maserati',
       'Lexus', 'Honda', 'Mazda', 'Mini', 'Mitsubishi', 'Nissan', 'SEAT',
       'Subaru', 'Suzuki', 'Toyota', 'Yamaha'] 
    mileage: Union[int , float]
    engine_power: Union[int, float ]
    fuel: Literal['diesel', 'petrol', 'hybrid_petrol', 'electro']
    paint_color: Literal['black', 'grey', 'white', 'red', 'silver', 'blue', 'orange',
       'beige', 'brown', 'green']
    car_type: Literal['convertible', 'coupe', 'estate', 'hatchback', 'sedan',
       'subcompact', 'suv', 'van']
    private_parking_available: Literal[True, False]
    has_gps: Literal[True, False]
    has_air_conditioning: Literal[True, False]
    automatic_car: Literal[True, False]
    has_getaround_connect: Literal[True, False]
    has_speed_regulator : Literal[True, False]
    winter_tires : Literal[True, False]
    

#on définit nos Endpoints
# les decorateurs (@app.get par ex.)sont des fonctions qui appellent d'autres fonctions

@app.get("/", tags=["Preview"])
async def index():
    """
    Just saying Hello to you 
    """
    message = "Hello World 😁, welcome to the 'Get Around' predictions price API!"
    return message


@app.get("/preview", tags=["Preview"])
async def random_rental(rows: int=10):
    """
    Get a sample of your whole dataset. 
    You can specify how many rows you want by specifying a value for `rows`, default is `10`

    """
    df = pd.read_csv("get_around_pricing_project.csv")
    sample = df.sample(rows)
    return sample.to_json()


@app.get("/unique-values", tags=["Preview"])
async def unique_values(column: str = "fuel"):
    """
    Get unique values from a given column

    You have to choose between one of those column names:

    ['model_key', 'mileage', 'engine_power', 'fuel', 'paint_color',
        'car_type', 'private_parking_available', 'has_gps',
        'has_air_conditioning', 'automatic_car', 'has_getaround_connect',
        'has_speed_regulator', 'winter_tires', 'rental_price_per_day']
    """
    
    df = pd.read_csv("get_around_pricing_project.csv")
    df = pd.Series(df[column].unique())
    return df.to_json()
          
@app.get("/max", tags=["Preview"])
async def max_values(column: str = "rental_price_per_day"):
    """
    Get max values from a given column 

    You have to choose between one of those column names:

    ['model_key', 'mileage', 'engine_power', 'fuel', 'paint_color',
       'car_type', 'private_parking_available', 'has_gps',
       'has_air_conditioning', 'automatic_car', 'has_getaround_connect',
       'has_speed_regulator', 'winter_tires', 'rental_price_per_day']
    """
    
    df = pd.read_csv("get_around_pricing_project.csv")
    df = pd.Series(df[column].max())
    return df.to_json()   


@app.get("/min", tags=["Preview"])
async def min_values(column: str = "rental_price_per_day"):
    """
    Get min values from a given column 

    You have to choose between one of those column names:

    ['model_key', 'mileage', 'engine_power', 'fuel', 'paint_color',
       'car_type', 'private_parking_available', 'has_gps',
       'has_air_conditioning', 'automatic_car', 'has_getaround_connect',
       'has_speed_regulator', 'winter_tires', 'rental_price_per_day']
    """
    
    df = pd.read_csv("get_around_pricing_project.csv")
    df = pd.Series(df[column].min())
    return df.to_json()

@app.get("/price_per_car_model", tags=["Preview"])
async def price_per_car_mean(car_brand: str = "Citroën"):
    """
    Get mean rental price per day per car model key

    For model_key price you have to choose one of this value:

    model_key: Literal['Citroën', 'Peugeot', 'PGO', 'Renault', 'Audi', 'BMW', 'Ford',
       'Mercedes', 'Opel', 'Porsche', 'Volkswagen', 'KIA Motors',
       'Alfa Romeo', 'Ferrari', 'Fiat', 'Lamborghini', 'Maserati',
       'Lexus', 'Honda', 'Mazda', 'Mini', 'Mitsubishi', 'Nissan', 'SEAT',
       'Subaru', 'Suzuki', 'Toyota', 'Yamaha'] 
    """
    
    df = pd.read_csv("get_around_pricing_project.csv")
    data_f = df[df['model_key']  == car_brand]
    data_f = pd.Series(np.round((data_f['rental_price_per_day'].mean()),2))
    
    return data_f.to_json()

@app.post("/predict",tags=["Machine-Learning"])
async def predict(predictionFeatures: PredictionFeatures):
    """
    For each feature to predict price you have to choose one of this value or of same type:

    model_key: Literal['Citroën', 'Peugeot', 'PGO', 'Renault', 'Audi', 'BMW', 'Ford',
       'Mercedes', 'Opel', 'Porsche', 'Volkswagen', 'KIA Motors',
       'Alfa Romeo', 'Ferrari', 'Fiat', 'Lamborghini', 'Maserati',
       'Lexus', 'Honda', 'Mazda', 'Mini', 'Mitsubishi', 'Nissan', 'SEAT',
       'Subaru', 'Suzuki', 'Toyota', 'Yamaha'] 

    mileage: Union[int , float]
    engine_power: Union[int, float ]

    fuel: Literal['diesel', 'petrol', 'hybrid_petrol', 'electro']

    paint_color: Literal['black', 'grey', 'white', 'red', 'silver', 'blue', 'orange',
       'beige', 'brown', 'green']

    car_type: Literal['convertible', 'coupe', 'estate', 'hatchback', 'sedan',
       'subcompact', 'suv', 'van']

    private_parking_available: Literal[True, False]

    has_gps: Literal[True, False]

    has_air_conditioning: Literal[True, False]

    automatic_car: Literal[True, False]

    has_getaround_connect: Literal[True, False]

    has_speed_regulator : Literal[True, False]

    winter_tires : Literal[True, False]

    """
  
#we train the model & use his predictions to the predict endpoint of our API
    data = pd.read_csv("get_around_pricing_project.csv")
    data = data.drop(labels = ['Unnamed: 0'], axis = 1)
    data = data[data['mileage'] >= 0]
    data[['model_key','fuel','paint_color','car_type']] = data[['model_key','fuel','paint_color','car_type']].astype('string')

#separate features with target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1:]

#finding numerical and categorical value to encode or normalize them
    numeric_features = []
    categorical_features = []
    for i,t in X.dtypes.iteritems():
        if ('float' in str(t)) or ('int' in str(t)) :
            numeric_features.append(i)
        else :
            categorical_features.append(i)

#make the train /test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, y,test_size=0.2,
                                                        random_state=0)
    

                
    # Create pipeline for numeric features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')), # missing values will be replaced by columns' mean
        ('scaler', StandardScaler())
    ])

    # Create pipeline for categorical features
    categorical_transformer = Pipeline(
    steps=[
    ('encoder',OneHotEncoder(drop='first',handle_unknown='ignore')) # first column will be dropped to avoid creating correlations between features
    ])
    # Use ColumnTransformer to make a preprocessor object that describes all the treatments to be done
    preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

#we train the model
    regressor = LinearRegression()
    model = regressor.fit(X_train, Y_train)
   
 #we apply the same preprocessing on datas that would be submit to the API
 #in a POST request to predict a rental price per day
     
    df = pd.DataFrame(dict(predictionFeatures), index=[0])
    numeric_features = []
    categorical_features = []
    for i,t in df.dtypes.iteritems():
            if ('float' in str(t)) or ('int' in str(t)) :
                numeric_features.append(i)
            else :
                categorical_features.append(i)

    df = preprocessor.transform(df)

    # Predictions on df set
    prediction = np.round(model.predict(df),2)
    
    # Format response
    response = {"prediction rental price per day": prediction.tolist()}
    return response


#on utilise le serveur uvicorn pour faire tourner notre instance de FastApi
#heroku n'accepte que les serveurs gunicorn

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000, debug=True, reload=True)