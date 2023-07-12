"""
Compute co-ords on **rotated** grid using cordex library
"""
import cordex
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

coords_rotated = dict()

for name,c in coords.items():
    rot_lon,rot_lat = cordex.rotated_coord_transform(c[0],c[1],pole_long,pole_lat,direction='geo2rot')
    rot_lon += 360.
    print(f"{name}: {rot_lon:4.2f} {rot_lat:4.2f}")
    # need co-ords around 360 for matching the model..
    coords_rotated[name]=[round(t,2) for t in (rot_lon,rot_lat)] # store to 2sf

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
