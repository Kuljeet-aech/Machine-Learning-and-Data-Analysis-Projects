import numpy as np
import pandas as pd
import plotly.express as px
<<<<<<< HEAD
data = pd.read_csv("D:\Kuljeet\Projects\Data-Analysis-Projects\DataSets\Delivery Time Optimization\deliverytime.txt")
=======
data = pd.read_csv("DataSets\Delivery time\deliverytime.txt")
>>>>>>> 851163cae2234ba05cfade6db97f1950df1cb03a

R = 6371

def deg_to_rad(degrees):
    return degrees * (np.pi/180)

def distcalculate(lat1, lon1, lat2, lon2):
    d_lat = deg_to_rad(lat2-lat1)
    d_lon = deg_to_rad(lon2-lon1)
    a = np.sin(d_lat/2)**2 + np.cos(deg_to_rad(lat1)) * np.cos(deg_to_rad(lat2)) * np.sin(d_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

data['Distance'] = np.nan

for i in range(len(data)):
    data.loc[i, 'Distance'] = distcalculate(data.loc[i, 'Restaurant_latitude'], 
                                        data.loc[i, 'Restaurant_longitude'], 
                                        data.loc[i, 'Delivery_location_latitude'], 
                                        data.loc[i, 'Delivery_location_longitude'])
    
print(data.head())
    
#figure1 = px.scatter(data_frame = data,
#                   x = "distance",
#                   y = "Time_taken(min)",
#                   size = "Time_taken(min)",
#                   trendline="ols",
#                   title="Relationship between Distance and Time taken")
#figure1.show()

#figure2 = px.scatter(data_frame = data, 
#                    x="Delivery_person_Age",
#                    y="Time_taken(min)", 
#                    size="Time_taken(min)", 
#                    color = "Distance",
#                    trendline="ols", 
#                    title = "Relationship Between Time Taken and Age")
<<<<<<< HEAD
#figure2.show()
=======
#figure2.show()
>>>>>>> 851163cae2234ba05cfade6db97f1950df1cb03a
