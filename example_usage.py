import pandas as pd 
from custommodel import RandomForestRegressor as RF

data = pd.DataFrame(data={
    'year':[2026],
    'ndvi': [4197.05],
    'nir': [1264.16],
    'red': [0.547683],
})
data['year'] = pd.to_datetime(data['year'], format='%Y')

rf = RF.load_model('./rfmodel.pkl')

[area] = rf.predict(data)
print(area)
