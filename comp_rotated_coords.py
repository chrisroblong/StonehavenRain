"""
Compute co-ords on **rotated** grid using pyproj library
"""

import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# co-ords of the pole.
pole_long = 177.5
pole_lat = 37.5

coords=dict(Edinburgh=(-3.1833,55.95),
            Rothampsted=(-0.35, 51.8),
            Malvern=(-2.317, 52.117),
            Squires_Gate=(-3.033, 53.767),
            Ringway=(-2.267, 53.35),
            Stonehaven=(-2.32, 56.95)
            )

original_crs = ccrs.PlateCarree()
target_crs = ccrs.RotatedPole(pole_longitude=pole_long, pole_latitude=pole_lat)

coords_rotated = dict()

# Define the original coordinates (lon, lat)
lon = np.array([-3.1833, -0.35, -2.317, -3.033, -2.267, -2.32])
lat = np.array([55.95, 51.8, 52.117, 53.767, 53.35, 56.95])

# Transform the coordinates
lon_rot, lat_rot, _ = target_crs.transform_points(original_crs, lon, lat).T

lon_rot = np.add(lon_rot, 360)

# Print the rotated coordinates
for lon_r, lat_r in zip(lon_rot, lat_rot):
    print(f"Rotated: {lon_r:.2f}, {lat_r:.2f}")

#for name,c in coords.items():
#    lon, lat = c[0], c[1]
#    rot_lon += 360.
#    print(f"{name}: {rot_lon:4.2f} {rot_lat:4.2f}")
#    # need co-ords around 360 for matching the model..
#    coords_rotated[name]=[round(t,2) for t in (rot_lon,rot_lat)] # store to 2sf

# now plot things so can check..
projRot=ccrs.RotatedPole(pole_longitude=pole_long,pole_latitude=pole_lat)
proj=ccrs.PlateCarree()

fig=plt.figure(num='test_rotasted',clear=True)
ax=fig.add_subplot(111,projection=proj)
ax.set_extent([-5,2, 50,57],crs=proj)
# plot in rotated coords
for key,coords in coords.items():
    ax.plot(coords[0],coords[1],marker='x',color='red')
for key,coords in coords_rotated.items():
    ax.plot(coords[0]+0.05,coords[1],marker='x',color='blue',transform=projRot)#offset by 0.05 degree
ax.coastlines()
fig.show()
