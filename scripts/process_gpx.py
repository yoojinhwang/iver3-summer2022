
import numpy as np
import gpxpy 
import gpxpy.gpx 
from datetime import datetime
from utils import to_cartesian
import matplotlib.pyplot as plt
import pandas as pd

filepath = '/Users/Declan/Desktop/test.gpx'
savepath = '/Users/Declan/Desktop/result.csv'
gpx_file = open(filepath, 'r') 

gpx = gpxpy.parse(gpx_file)

time = []
lat = []
lon = []

origin = np.array([10.773861464112997, -85.34960218705237])

for track in gpx.tracks: 
    for segment in track.segments: 
        for point in segment.points: 
            time.append(point.time)
            lat.append(point.latitude)
            lon.append(point.longitude)

cartesian = []

for i in range(len(lat)):
    cartesian.append(to_cartesian(np.array([lat[i], lon[i]]), origin))

x = [coord[0] for coord in cartesian]
y = [coord[1] for coord in cartesian]

plt.plot(x, y)
plt.xlabel('x position (m)')
plt.ylabel('y position (m)')
plt.axis('scaled')
plt.show()

data = {'datetime': time,
        'latitude': lat,
        'longitude': lon}

df = pd.DataFrame(data, columns = ['datetime', 'latitude', 'longitude'])

df.to_csv(savepath, index=False)
