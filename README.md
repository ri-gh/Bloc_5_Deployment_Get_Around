# Bloc_5_Deployment_Get_Around

## Deployment of a Dashboard &amp; an API on the web

GetAround is the Airbnb for cars. You can rent cars from any person for a few hours to a few days!
The checkin and checkout of our rentals can be done with three distinct flows:

* Mobile rental agreement on native apps: driver and owner meet and both sign the rental agreement on the owner’s smartphone

* Connect: the driver doesn’t meet the owner and opens the car with his smartphone

* Paper contract (negligible)

Late returns at checkout can generate high friction for the next driver if the car was supposed to be rented again on the same day : 

Customer service often reports users unsatisfied because they had to wait for the car to come back from the previous rental or users that even had to cancel their rental because the car wasn’t returned on time

- threshold: how long should the minimum delay between 2 rentals be?

- scope: should we enable the feature for all cars?, only Connect cars?

## Deliverable :

1/ First build a dashboard that will help the product Management team with the above questions,
here it is : https://streamfast.herokuapp.com/

2/ In addition to the above question, the Data Science team is working on pricing optimization,
you should provide at least one endpoint /predict , here it is again : https://fastapilast.herokuapp.com/

3/ You need to provide the users with a documentation about your API

Example of POST request to test my API:

url = "https://fastapilast.herokuapp.com/predict"


data ={"model_key": "BMW",
  "mileage": 10000,
  "engine_power": 150,
  "fuel": "petrol",
  "paint_color": "black",
  "car_type": "convertible",
  "private_parking_available": False,
  "has_gps": True,
  "has_air_conditioning": False,
  "automatic_car": False,
  "has_getaround_connect": True,
  "has_speed_regulator":True,
  "winter_tires": False
}

r = requests.post(url = url, json = data)

print(r.json())


=> The 'DockerfileAPI' , 'requirementsapi.txt' & the 'appapi.py' files are for the FastApi app (feel free to rename them)

the other ones (DockerFile, requirements.txt & app.py) are for the Streamlit Dashboard.

