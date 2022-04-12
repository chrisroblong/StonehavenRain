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
            Ringway=(-2.267, 53.35)
            )

coords_rotated = dict()

for name,c in coords.items():
    transform = cordex.rotated_coord_transform(c[0],c[1],pole_long,pole_lat,direction='geo2rot')
    print(f"{name}: {transform[0]:4.2f} {transform[1]:4.2f}")
    coords_rotated[name]=[round(t,2) for t in transform] # store to 2sf

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
