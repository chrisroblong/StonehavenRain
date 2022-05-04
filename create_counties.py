"""
Create county boundaries from OS data so can have them at about 1:10,000
"""
import matplotlib.pyplot as plt
import platform
import pathlib
#code from edinburghLib as uses the county data!
machine = platform.node()
if ('jasmin.ac.uk' in machine) or ('jc.rl.ac.uk' in machine):
    # specials for Jasmin cluster or LOTUS cluster
    dataDir = pathlib.Path.cwd()
    nimrodRootDir = pathlib.Path("/badc/ukmo-nimrod/data/composite") # where the nimrod data lives
    cpmDir = pathlib.Path("/badc/ukcp18/data/land-cpm/uk/2.2km/rcp85") # where the CPM data lives
    outdir = pathlib.Path(".")  #writing locally. Really need a workspace..
elif 'GEOS-' in machine.upper() :
    dataDir = pathlib.Path(r'C:\Users\stett2\OneDrive - University of Edinburgh\data\EdinburghRainfall')
    nimrodRootDir = dataDir/'nimrod_data'
    outdir = dataDir/'output_data'
else: # don't know what to do so raise an error.
    raise Exception(f"On platform {machine} no idea where data lives")
# create the outdir
outdir.mkdir(parents=True,exist_ok=True)
import fiona
import shapely.ops
import shapely.geometry
import cartopy
import cartopy.feature
import cartopy.crs as ccrs
import itertools
# get in the OS data. Basically wards so need to merge them

resoln = 10 # resolution to use when simplifying merged geometries
groupfn = lambda rec: rec['properties']['FILE_NAME'].split("-")[0].replace("_"," ").title()

counties=dict()
for file in ['greater_london_const_region.shx','county_electoral_division_region.shx','unitary_electoral_division_region.shx','district_borough_unitary_region.shx']:
    fname = "zip://"+(dataDir/f'bdline_essh_gb.zip/Data/GB/{file}').as_posix()
    with fiona.open(fname) as records:
        metaData = records.meta
        print(f"Dealing with {fname} {metaData}")
        records=sorted(iter(records),key=groupfn)
        for key, group in itertools.groupby(records, key=groupfn):
            merge = shapely.ops.unary_union([shapely.geometry.shape(s['geometry']) for s in group]).simplify(resoln)
            counties[key] = merge
            print("Dealing with", key," area ",merge.area/1e6," km^2")

GB = shapely.ops.unary_union(counties.values())
## then remove GB from the counties...
#for key,shape in counties.items():
#    counties[key]= shape.difference(GB)
#    print(f"removed coastline from {key}")
## write the data out to make it fast
props = metaData['schema']
metaData['schema']['properties']=dict(Name='str:100')
filename=dataDir/'GB_OS_boundaries'/f'counties_{resoln:04d}.shp'
filename.parent.mkdir(exist_ok=True,parents=True) # make directory
with fiona.open(filename, 'w', driver='ESRI Shapefile', schema=metaData['schema']) as file:
    # write out GB shape info.
    #file.write({'geometry': shapely.geometry.mapping(GB), 'properties': {'Name': 'GB_coastline'}})
    for name,geom in counties.items():
        print(name)
        file.write({'geometry': shapely.geometry.mapping(geom),'properties': {'Name': name} })


##  plot
fig=plt.figure(num='test',clear=True)
ax=fig.add_subplot(111,projection=ccrs.OSGB())
gb = cartopy.feature.ShapelyFeature([GB],crs=ccrs.OSGB(),facecolor='green')
ax.add_feature(gb)
for k,g in counties.items():

    f=cartopy.feature.ShapelyFeature([g], crs=ccrs.OSGB(),
                                   edgecolor='black', facecolor='none')
    ax.add_feature(f)
fig.show()


