"""
Compute seasonal-mean CET from monthly-mean temperatures
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
coords_raw = dict(Rothampsted=(-0.35,51.8),
                  Malvern=(-2.317,52.117),
                  Squires_Gate=(-3.033,53.767),
                  Ringway=(-2.267,53.35))

coords_rotated= dict(
    Rothampsted=(1.33,-0.68),
    Malvern=(0.11,-0.38),
    Squires_Gate=(359.68,1.27),
    Ringway=(0.14,0.85)
)

projRot=ccrs.RotatedPole(pole_longitude=177.5,pole_latitude=37.5)
proj=ccrs.PlateCarree()

fig=plt.figure(num='test_cet',clear=True)
ax=fig.add_subplot(111,projection=proj)
ax.set_extent([-5,2, 50,56],crs=proj)
# plot in rotated coords
for key,coords in coords_raw.items():
    ax.plot(coords[0],coords[1],marker='x',color='red')
for key,coords in coords_rotated.items():
    ax.plot(coords[0]+0.05,coords[1],marker='x',color='blue',transform=projRot)
ax.coastlines()
fig.show()

## now to read the ensemble data.