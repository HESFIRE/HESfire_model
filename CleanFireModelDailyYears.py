#!/usr/local/bin/Python
### Fire model

# Developed by Yannick Le Page et al. (REF) - Contact: Yannick.LePage@pnnl.gov
import os
import csv
from numpy import *
from copy import deepcopy
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from yancolbar import *
from yanutil import *
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.mlab import griddata
from matplotlib import cm
import matplotlib.cm
from scipy.stats import binom
import random as rnd
import scipy.interpolate
import time
import shutil
from scipy.stats import pearsonr
import random
import pylab
#import gc

# FUNCTION: Projection experiment
# transforms input file types to grab them from the climate model folder instead of 
# observations when using the model for fire projections.
def projpath(projection,obspath,projfilename):
	if projection == 'no':
		newpath = obspath # In case we use observations
	else: # in case we use cimate projections
		rootpath=obspath.split('/')[0:len(obspath.split('/'))-2]
		newpath = './'
		for i in range(len(rootpath)):
			newpath += rootpath[i] + '/'
	newpath += projection + '/'+ obspath.split('/')[-2] +'/'+ projfilename
	return obspath, newpath

# FUNCTION: Full experiment 
def isfullexp(expname):
	if expname[-5:-1]=='Full':
		relpath='../../../../'
	else:
		relpath='./'
	outfolder=relpath+'Outputs/Optimization/'
	return outfolder, relpath

# FUNCTION: Saves source code
def savesourcode(outpath,outfolder):
	try:
		os.mkdir(outfolder)
	except OSError:
		'do nothing'
	try:
		os.mkdir(outpath)
		print "created output folder"
	except OSError:
		# !!!
		print " WARNING: experiment folder already exists, overwriting (at end of fire cycle)"
	try:
		shutil.copy('./'+os.path.basename(__file__), outpath+'SourceCodeFireModel.py')
	except:
		print "WARNING: could not save source code file"

# FUNCTION: Random parameter generator
# a: lowest value acceptable for the parameter
# b: highest value acceptable
# paramtype: exponent ('exp') or quantitative ('num')
def randparam(a,b,paramtype): 
	if paramtype == 'exp': # shape parameters, typically from 0.05 to 15, but we want same probability from 0.05-1 than 1-15  
		paramtest = rnd.randint(1,2)
		if paramtest == 1: # we go for a value between a and 1
			param = rnd.randint(a*100,100)/100.
		elif paramtest == 2: # we go for a value between 1 an b
			param=rnd.randint(1,b)/1.
	if paramtype == 'num': # typically 0.0001 to 0.1, but we want same probability to be 0.0001-0.001 than 0.01-0.1
		magnrange = b/a
		paramtest = rnd.randint(1,log10(magnrange))
		param = rnd.randint(1,10)/1.
		param = param / (pow(10,paramtest) * 1. / b)
	return param


# Computational time analysis
start_time = time.time()

##########################################################################################
##################################### 1. User inputs #####################################
##########################################################################################


###################################################
### 1.0. Model name for climate projection runs ###
###################################################
projection = 'ACCESS1-3/RCP4.5'#'ACCESS1-3/RCP4.5'#'ACCESS1-3/RCP4.5'#'ACCESS1-3/RCP4.5'#'ACCESS1-3'
yearproj = 2090 
landusedata = 'Globcover/'#'RCP4.5/2090/'#'Globcover/'
GDPdata = 'GDPCIA'


###############################
### 1.a. Filenames, outputs ###
###############################
expname='testday6projnolightning/' ### Experiment name (folder name for outputs)
saveresults=1 
saveannual=0 # To save annual summaries, 20% more time to run...
# Paths to data inputs, relative to location of the model file.
pathdailyrh, pathrhproj = projpath(projection,'./Data/RH/interpdailyrh_','hurs_Amon_')
pathnightlyrh, pathrhproj = projpath(projection,'./Data/RH/interpnightlyrh_', 'hurs_Amon_')
pathdailytemp, pathtempproj = projpath(projection,'./Data/Temp/interpdailytemp_','tas_Amon_')
pathnightlytemp, pathtempproj = projpath(projection, './Data/Temp/interpnightlytemp_','tas_Amon_')
pathdailymoist, pathmoistproj = projpath(projection,'./Data/SoilMoisture/interpdailymoist_','mrsos_Lmon_')
pathnightlymoist, pathmoistproj = projpath(projection, './Data/SoilMoisture/interpnightlymoist_','mrsos_Lmon_')
pathdailywind, pathwindproj = projpath(projection, './Data/Wind/interpdailywind_','sfcWind_Amon_')
pathnightlywind, pathwindproj = projpath(projection,'./Data/Wind/interpnightlywind_','sfcWind_Amon_')
pathdailylight, pathlightdailyproj = projpath(projection,'./Data/Convprecip/interpdailylight_','interpdailylight_')
pathnightlylight, pathlightnightlyproj = projpath(projection,'./Data/Convprecip/interpnightlylight_','interpnightlylight_')
pathprecip = './Data/Precip/mprecip_'
bouh, pathprecipproj = projpath(projection,'./Data/Precip/mprecip_1','pr_Amon_')

#############################
### 1.b. Spatial/temporal ###
#############################
### Spatial/temporal input
# Maskfile is Lat/lon map with desired grid-cells as 1 and everything else not 1 (any value, or nan).
# Use a map with all grid-cells as 1 to do global runs.
res=1 # Resolution
maskfile='no'#'ProjAccess1.0'#'ProjAccess1.0'#'testlight1.0' # Spatial subset, "no" if you want to use all the data in your inputs.
yearB=2001 # first year of modeling time range
yearE=2002 # last year of modeling time range
numyears=yearE-yearB+1
leapyears = 1 # If the temporal data (e.g. climate) have leap years

#############################
### 1.c. Parameterization ###
#############################
# NOTE: if the optimization procedure is turned on, parameter values set below
# are irrelevant for the parameters being optimized

### stochasticity of fire ignitions and termination
# Note: if turned off, average fire the expected number of ignitions and terminations is directly used, 
# these are generally decimals (and <1), thus single fire statistics (e.g. fire size and duration) cannot be computed.
stochastic=1 

### Fuel load
precipfactor=[0.5,3.] # first number, minimum average precip (mm/day) for enough fuel load, second number: maximum (no more limitation beyond that).
precipdelay=3 # Months before fuel actually builds up in arid regions after substantial rainfall
dryexp=1.72 # Shape parameter

### GDP
gdpfactor=[0,60.] # Range of influence of GDP on fire ignitions.
gdpsuppression='yes' # if activated, suppression is a function of landuse*gdp, if not of landuse only. Note that GDP also has an effect on ignitions so when that is activated, GDP has two driving impacts.
gdpsupfactor=[0, 60.] # Range of influence of GDP on fire suppression.
gdpexp=1.28 # Shape parameter
gdpsupexp=1.28 # Shape parameter

### Land use
variableluignfactor=[0.,0.1] # Range of influence of land use fraction on fire ignitions.
landuselimit=0.1 # upper land use fraction for fire suppression (no suppression beyond that point)
supexp=4.08 # Shape parameter
luignexp=14.9 # Shape parameter
luign=0.00228 # 0.001 Ignitions per day per km2 of landuse (before applying luignexp
bgign=0. # Background ignitions (accidental, criminal, independent of GDP)


### Fragmentation
fragexp=1.81 # shape parameter

### Climate
# Lightning
uselight = 1
natign=0.069 # Probability of Cloud-to-ground lightning strikes to ignite a fire.

# !!! 
speedfuel=0
usewind = 1  # To activate wind influence on fire spread rate

# Range of influence for climate parameters
usetemp = 1
Trange=[0.,30.] # Range of influence of temperature on fire spread.
userh = 1
Rhrange=[30., 80.] # Range of influence of RH on fire spread.
usemoist = 1
Moistrange=[0.20, 0.35] # Range of influence of soil moisture on fire spread.

# Shape parameters
rhexp=1.18
moistexp=1.21
tempexp=1.78

### Fire intensity
# if turned on, suppression is less likely to be successful under more intense fires.
# Fire intensity is a function of climate and fuel conditions.
intensitysuppression=1 
# !!!
meanintensity=0

### Fire spread
partialellispse = 0 # Fragmentation influences the area of the fire ellipse that actually burns.
forestspread=0.28 # Fire speed in forests, m/s
shrubspread=1.12 # Fire speed in shrubs, m/s
grassspread=2.79 # Fire speed in grass, m/s
kmpersecperdayperms=3600*24/1000
if usewind==0:
	lbratio=5 # Standard length to breadth ratio for fire spread if wind influence not considered
mbamemory=8 # number of month after which a burned patch is able to burn again


#########################
### 1.d. Optimization ###
#########################
metropolis='no' # 'yes' to activate metropolis optimization
randinitstate = 0 # if 1, will randomly compute initial parameters (given upper/lower boundaries in paramub and paramlb).
params=['dryexp','gdpexp','supexp','gdpsupexp','fragexp','rhexp','moistexp','tempexp','luign','luignexp','natign']
paramstype=['exp','exp','exp','exp','exp','exp','exp','exp','num','exp','num']
paramsnumber=['um','dois','tres','quatro','cinco','seis','sete','oito','nove','dez','onze']
paramub=[15,15,15,15,15,15,15,15,0.1,20,0.2] # Upper bound
paramlb=[0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.000001,0.05,0.000002] # Lower bound
paramdelta=[1,1,1,1,1,1,1,1,1,1,1] # maximum step parameter
testbool=[1,1,1,0,1,1,1,1,1,1,1] # 1 to optimize, 0 otherwise

#######################
### 1.e. Evaluation ###
#######################
powereval=2
# Boundaries of Fire fraction classes for optimization/evaluation
classbound=array([0,1,5,10,20,35,50,200])

#######################
### 1.f. Input data ###
#######################
GFEDdata='GFED_BA_natural_Fireviz'

###################
### 1.g. Inputs ###
###################




##########################################################################################
#################### 2. Run setup (reads data, select space/time, etc) ###################
##########################################################################################

### Checking Whether is is a global application of a previously optimized parameterization
outfolder, relpath = isfullexp(expname)
outpath=outfolder+expname

### Output folder creation and saving Source code
savesourcode(outpath,outfolder)

		

#############################
### 2.a. Parameterization ###
#############################
print '2.a. parameterization'

### Optimization run
if metropolis == 'yes':
	if randinitstate == 1:
		# Random computation of initial parameter values (given upper and lower bounds in paramlb and paramub)
		for param in range(len(params)):
			exec(params[param] + '= randparam(paramlb[param],paramub[param],paramstype[param])')

### Post-optimization run
# If this is a full run from after optimization has been done, we get the final set 
# of parameter values from the optimization run folder.
if expname[-5:-1]=='Full':
	paramfin = genfromtxt('./Evaluation.csv', skip_header=0, missing_values='-9999', filling_values=0,delimiter=',')
	evalcol = paramfin[:,0]
	evalcol[evalcol==0]=40
	bestind=argmin(evalcol)
	bestparams=paramfin[bestind,:]
	for param in range(len(params)):
		eval(params[param] + '= bestparams[-12+param]')

#######################
### 2.b. Evaluation ###
#######################
print '2.b. Evaluation'

# !!!
totcombs=1
paramatrix=zeros(shape=(10000,len(params)+6))
evalmatrix=zeros(shape=(10000,4))*nan
globindold=40 # initialize model error for first loop
evalindold=40

###########################################
### 2.c. Spatio-temporal data selection ###
###########################################
print '2.c. Spatio-temporal data selection'



	
#########################
### 2.d. Reading data ###
#########################
print '2.d. Reading data'

### Land cover/land use
print 'Land cover/use'
forest=genfromtxt(relpath+'./Data/'+landusedata+'forests_'+str(res)+'.csv', skip_header=0, missing_values='-9999', filling_values=0,delimiter=',')
shrubs=genfromtxt(relpath+'./Data/'+landusedata+'shrubsonly_'+str(res)+'.csv', skip_header=0, missing_values='-9999', filling_values=0,delimiter=',')
grass=genfromtxt(relpath+'./Data/'+landusedata+'herbaceous_'+str(res)+'.csv', skip_header=0, missing_values='-9999', filling_values=0,delimiter=',')
landuse=genfromtxt(relpath+'./Data/'+landusedata+'landuse_'+str(res)+'.csv', skip_header=0, missing_values='-9999', filling_values=0,delimiter=',')
nonland=genfromtxt(relpath+'./Data/'+landusedata+'deserts_'+str(res)+'.csv', skip_header=0, missing_values='-9999', filling_values=0,delimiter=',')
# NOTE: nonland is water+bare+ice/snow, considering that they wont participate in the fragmentation index.
water=genfromtxt(relpath+'./Data/'+landusedata+'water_'+str(res)+'.csv', skip_header=0, missing_values='-9999', filling_values=0,delimiter=',')
# NOTE: we do not model fires in grid-cells > 0.5 water, considering the fragmentation index is not appropriate given most 
# of these grid-cells have continuous water (lakes, oceans)

# Spatial selection
print 'Spatial domain'
numgrid=len(forest[:,0])-2 # number of grid-cells in landcover input data
if maskfile != 'no':
	maskmap=genfromtxt(relpath+'./Data/Masks/'+maskfile+'.csv', skip_header=0, missing_values='-9999', filling_values=0,delimiter=',')
	latmask,lonmask=where(maskmap==1)
	indmask=[]
	for i in range(len(latmask)):
		indorig=where((forest[2:,0]==latmask[i]) & (forest[2:,1]==lonmask[i]))
		if len(indorig[0])==0 or len(indorig[0])>1:
			'nothing'
			# !!! print len(indorig[0])
		else:
			indmask.append(indorig[0][0])
	indmask=array(indmask)		
	numgrid=len(indmask)
	selgrids=indmask
else:
	selgrids=arange(0,numgrid,1)

print 'Total number of gridcells: ' + str(numgrid)

# Computing grid-cell sizes
gridsize=cos(radians(forest[selgrids+2,2]))*(111.320*111.320)*res*res
gridsizesel=cos(radians(forest[selgrids+2,2]))*(111.320*111.320)*res*res	

### GDP
print 'GDP'
if gdpfactor[0]>=0 or gdpsupfactor[0]>=0:
	gdp=genfromtxt(relpath+'./Data/GDP/'+GDPdata+str(res)+'.csv', skip_header=0, missing_values='-9999', filling_values=0,delimiter=',')

### GFED
print 'GFED'
gfedorig=genfromtxt(relpath+'../../GFED/'+GFEDdata+str(res)+'.csv', skip_header=0, missing_values='-9999', filling_values=0,delimiter=',')
gfedba=zeros(shape=(len(forest[:,0]),len(gfedorig[0,:])))
gfedba[0:2,:]=gfedorig[0:2,:]
ind=-1
# Matching coordinates with land use input
for i in range(len(forest[2:,0])):
	ind+=1
	pix1=where(gfedorig[2:,0]==forest[i+2,0])
	pix2=where(gfedorig[2:,1]==forest[i+2,1])
	pix=intersect1d(pix1[0],pix2[0])
	if len(pix)==1:
		gfedba[ind+2,:]=gfedorig[pix+2,:]
	if len(pix)==0: # if grid-cell is not in observation-derived data, we assume that's because it doesn't have any fire.
		gfedba[ind+2,4:]=0
		gfedba[ind+2,0:4]=forest[i+2,0:4]
	if len(pix)>1:
		print 'ERROR: duplicate coordinates in GFED data'
del(gfedorig)

# Spatial subsets
gfedsel=gfedba[selgrids+2,:]

# common timespan with GFED
if yearB > gfedba[0,4]:
	yearcb=yearB
else:
	yearcb=gfedba[0,4]
if yearE > gfedba[0,-1]:
	yearce=gfedba[0,-1]
else:
	yearce= yearE
	
indbeg=where(gfedba[0,:]==yearcb)[0][0]
indend=where(gfedba[0,:]==yearce)[0][-1]
gfedc=gfedba
gfedc=gfedc[[0,1]+list(selgrids+2),:]
gfedc=gfedc[:,[0,1,2,3]+list(arange(indbeg,indend+1,1))]
#gfedcsel=gfedba[[0,1]+(selgrids+2),[0,1,2,3]+arange(indbeg,indend+1,1)]
del(gfedba)


### Climate
print 'Climate'

### Precip in average mm/day for each month, as a fuel proxy
precip=genfromtxt(relpath+pathprecip+str(res)+'.csv', skip_header=0, missing_values='-9999', filling_values=0,delimiter=',')
if projection != 'no':
	totprecipproj=loadtxt(relpath+pathprecipproj + str(int(yearproj)) + '_' + str(res) + '.csv', delimiter=',')

### lightnings
if uselight == 1:
	lightflash=genfromtxt(relpath+'./Data/Lightning/lightning_clim_'+str(res)+'.csv', skip_header=0, missing_values='-9999', filling_values=0,delimiter=',')

### Climate
print 'Reading bi-daily climate data, can take a while'

# !!! shorten ! 
for cv in ['rh','temp','moist','wind','light']:
	if eval('use' + cv + '==1'):  
		exec('alldaily' + cv + '=zeros(shape=(yearE-yearB+1,ceil(numgrid/2.)+2,370),dtype=float32)')
		exec('allnightly' + cv + '=zeros(shape=(yearE-yearB+1,ceil(numgrid/2.)+2,370),dtype=float32)')
		if (projection != 'no') and (cv != 'light'):
			exec('tot' + cv + 'proj=loadtxt(relpath+path' + cv + 'proj + str(int(yearproj)) + "'"_"'" + str(res) + "'".csv"'" , delimiter="'","'")')
			exec('all' + cv + 'proj=zeros(shape=(ceil(numgrid/2.)+2,12+4),dtype=float32)')
			exec('all' + cv + 'proj=tot' + cv + 'proj[0:ceil(numgrid/2.)+2:]')		
	for y in range(yearE-yearB+1):
		### Reading annual input files
		print 'Extracting ' + cv + ' for year ' + str(int(yearB+y))
		exec('ydaily' + cv + '=genfromtxt(relpath+pathdaily' + cv + '+ str(int(yearB+y)) + "'"_"'" + str(res) + "'".csv"'", skip_header=0, missing_values=''-9999'', filling_values=0,delimiter="'","'")')
		exec('alldaily' + cv + '[int(y),2:,:]=ydaily' + cv + '[selgrids[0:ceil(numgrid/2.)]+2,:]')
		exec('ynightly' + cv + '=genfromtxt(relpath+pathnightly' + cv + '+ str(int(yearB+y)) + "'"_"'" + str(res) + "'".csv"'", skip_header=0, missing_values=''-9999'', filling_values=0,delimiter="'","'")')
		exec('allnightly' + cv + '[int(y),2:,:]=ynightly' + cv + '[selgrids[0:ceil(numgrid/2.)]+2,:]')
		exec('del(ynightly' + cv + ')')
		
# For lightning projection based on convective precip, we use daily projected file
# if (projection != 'no'):
# 	cv = 'light'
# 	for y in range(yearE-yearB+1):
# 		### Reading annual input files
# 		print 'Extracting ' + cv + ' for year ' + str(int(yearB+y))
# 		exec('ydaily' + cv + '=genfromtxt(relpath+path' + cv + 'dailyproj+ str(int(yearproj+y)) + "'"_"'" + str(res) + "'".csv"'", skip_header=0, missing_values=''-9999'', filling_values=0,delimiter="'","'")')
# 		exec('alldaily' + cv + '[int(y),2:,:]=ydaily' + cv + '[selgrids[0:ceil(numgrid/2.)]+2,:]')
# 		exec('del(ydaily' + cv + ')')
# 		exec('ynightly' + cv + '=genfromtxt(relpath+path' + cv + 'nightlyproj+ str(int(yearproj+y)) + "'"_"'" + str(res) + "'".csv"'", skip_header=0, missing_values=''-9999'', filling_values=0,delimiter="'","'")')
# 		exec('allnightly' + cv + '[int(y),2:,:]=ynightly' + cv + '[selgrids[0:ceil(numgrid/2.)]+2,:]')
# 		exec('del(ynightly' + cv + ')')
		




#######################################
### 2.e. Model data Initialization  ###
#######################################
print '2.e. Model data Initialization'

burneddailyforestorig=zeros(shape=(numgrid,mbamemory*30)) # 240=8 months
burneddailyshrubsorig=zeros(shape=(numgrid,mbamemory*30)) # 240=8 months
burneddailygrassorig=zeros(shape=(numgrid,mbamemory*30)) # 240=8 months

###  Extracting yearB-1 data from GFED to initialize the burned area memory before fire can return 
# !!! (TO ADD, if before 1997, initialize with the average)
ratiofs=zeros(shape=(numgrid,1))
ratiofs[:,0]=divide(forest[selgrids+2,4],(forest[selgrids+2,4]+shrubs[selgrids+2,4]+grass[selgrids+2,4]))
ratiofs[isnan(ratiofs)]=0
ratioss=zeros(shape=(numgrid,1))
ratioss[:,0]=divide(shrubs[selgrids+2,4],(forest[selgrids+2,4]+shrubs[selgrids+2,4]+grass[selgrids+2,4]))
ratioss[isnan(ratioss)]=0
ratiogs=zeros(shape=(numgrid,1))
ratiogs[:,0]=divide(grass[selgrids+2,4],(forest[selgrids+2,4]+shrubs[selgrids+2,4]+grass[selgrids+2,4]))
ratiogs[isnan(ratiogs)]=0
for m in arange(12-mbamemory+1,12+1,1):
	daysindex=arange((m-(12-mbamemory+1))*30,(m-(12-mbamemory+1)+1)*30,1)
	burneddailyforestorig[:,daysindex] = tile(multiply(gfedsel[:,(yearB-1997)*12+(m-1)+4],ratiofs[:,0]), (30, 1)).T * (1/30) * 1/1000000.
	burneddailyshrubsorig[:,daysindex] = tile(multiply(gfedsel[:,(yearB-1997)*12+(m-1)+4],ratioss[:,0]), (30, 1)).T * (1/30) * 1/1000000.
	burneddailygrassorig[:,daysindex] = tile(multiply(gfedsel[:,(yearB-1997)*12+(m-1)+4],ratiogs[:,0]), (30, 1)).T * (1/30) * 1/1000000.
burneddailyforestorig[isnan(burneddailyforestorig)]=0
burneddailyshrubsorig[isnan(burneddailyshrubsorig)]=0
burneddailygrassorig[isnan(burneddailygrassorig)]=0

### Fire classes
optba=zeros(shape=shape(gfedc[:,0:5]))
optba[2:,0:4]=gfedc[2:,0:4]
optba[0:2,:]=gfedc[0:2,0:5]
optba[2:,4]=nansum(gfedc[2:,4:],axis=1)/(indend-indbeg+1)*12
optfrac=yanspatialunit(optba,2,1)
optclass=yanfireclasscontdome(optfrac[2:,4],classbound)

### Peak fire month
optseas=yanpeakmonth(gfedc[:,4:])
optseas=argmax(optseas,axis=1)+1
optseas=where(nansum(gfedc[2:,4:],axis=1)<=0.00000001,0,optseas)

### Inter-annual variability (here computing annual burned areas from GFED)
optiav=zeros(shape=(len(gfedc[2:,0]),yearce-yearcb+1))
ind=-1
for y in arange(yearcb,yearce+0.01,1):
	ind+=1
	indy=where(gfedc[0,:]==y)[0][:]
	optiav[:,ind]=nansum(gfedc[2:,indy],axis=1)

### Precipitation tracking for fuel accumulation
seasontrackerorig=zeros(shape=(numgrid,12+precipdelay))
indprecip=where(precip[0,:]==yearB-1)
seasontrackerorig[:,:]=precip[selgrids+2,indprecip[0][0]-precipdelay:indprecip[0][-1]+1]
if projection != 'no':
	seasontrackerorig[:,0:precipdelay] += totprecipproj[selgrids+2,-precipdelay:]
	seasontrackerorig[:,precipdelay:] += totprecipproj[selgrids+2,4:]

# Now selecting precip from yearB to yearE so that we dont have to call the function where() later
preciporig=precip[selgrids+2,indprecip[0][-1]+1:].copy()

### Optimization performance through iterations
perfintime=[]

### Computational performance
# !!! simplify !
elapsed_11=0
elapsed_12=0
elapsed_wet=0
elapsed_head=0
elapsed_start=0
elapsed_spread=0
elapsed_stoch=0
elapsed_ba=0
elapsed_out=0
elapsed_fragm=0
elapsed_up=0
elapsed_out1=0
elapsed_out2=0
elapsed_out3=0
elapsed_out4=0
elapsed_perf=0
elapsed_gridcells=0
elapsed_wind=0
elapsed_exec=0

### Optimization
metropend=0
metropstep=0

########################################
### 2.f. Model output Initialization ###
########################################
print '2.f. Model output Initialization'

# Monthly outputs
for outname in ['mfires','mbashrubs','mbagrass','mbaforest','mba']:
	exec(outname+'=zeros(shape=(numgrid+2,numyears*12+4))')
	exec(outname+'[2:,0:4]=forest[selgrids+2,0:4]')
	exec(outname+'[0,0:2]=[res,2]')
	for y in arange(yearB,yearE+0.1,1):
		indstimevec=arange((y-yearB)*12+4,(y+1-yearB)*12+4,1)
		exec(outname+'[0,int_(list(indstimevec))]=y')
		exec(outname+'[1,int_(list(indstimevec))]=[1,2,3,4,5,6,7,8,9,10,11,12]')

for outname in ['atotbaforest','atotbashrubs','atotbagrass','atotba','amaxsize','afivemaxsize','amaxdur','afivemaxdur','aavdur']:
	exec(outname+'=zeros(shape=(numgrid+2,numyears+4))')
	exec(outname+'[2:,0:4]=forest[selgrids+2,0:4]')
	exec(outname+'[0,4:]=array([int(i) for i in arange(yearB,yearE+0.1,1)])')
	exec(outname+'[1,4:]=6')
	exec(outname+'[0,0:2]=[res,2]')

for outname in ['hignout','acignout','gdpout','suppout']:
	exec(outname+'=zeros(shape=(numgrid+2,5))')
	exec(outname+'[2:,0:4]=forest[selgrids+2,0:4]')
	exec(outname+'[0,4:]=2000')
	exec(outname+'[1,4:]=6')
	exec(outname+'[0,0:2]=[res,2]')

afivemaxsize=zeros(shape=(numgrid+2,numyears+4,5))
for imerde in range(5):
	afivemaxsize[2:,0:4,imerde]=forest[selgrids+2,0:4]
	afivemaxsize[0,4:,imerde]=array([int(i) for i in arange(yearB,yearE+0.1,1)])
	afivemaxsize[1,4:,imerde]=6
	afivemaxsize[0,0:2,imerde]=[res,2]
afivemaxdur=zeros(shape=(numgrid+2,numyears+4,5))
for imerde in range(5):
	afivemaxdur[2:,0:4,imerde]=forest[selgrids+2,0:4]
	afivemaxdur[0,4:,imerde]=array([int(i) for i in arange(yearB,yearE+0.1,1)])
	afivemaxdur[1,4:,imerde]=6
	afivemaxdur[0,0:2,imerde]=[res,2]													

######################
### 2.h. Utilities ###
######################
print '2.h. Utilities'

### Time utilities
monthnoleapcal=array([ 31,  61,  92, 122, 153, 183, 214, 245, 275, 306, 336, 366])
monthleapcal=array([ 31,  62,  93, 123, 154, 184, 215, 246, 276, 307, 337, 367])
numyears=yearE-yearB+1

### Lookup tables
# Exponent lookup table (see details in Data/Lookup folder)
lookup=genfromtxt(relpath+'./Data/Lookup/lookuptablebig.csv', skip_header=0, missing_values='-9999', filling_values=0,delimiter=',')

# Wind lookup tables
lookupLB=genfromtxt(relpath+'./Data/Lookup/lookuptableLB.csv', skip_header=0, missing_values='-9999', filling_values=0,delimiter=',')
lookupHB=genfromtxt(relpath+'./Data/Lookup/lookuptableHB.csv', skip_header=0, missing_values='-9999', filling_values=0,delimiter=',')
lookupGW=genfromtxt(relpath+'./Data/Lookup/lookuptableGW.csv', skip_header=0, missing_values='-9999', filling_values=0,delimiter=',')





##########################
#########################
########################
#######################
##### FIRE MODEL #####
#####################
####################
###################
##################
print "Now running fire model"

combind=-1

#########################
#########################							
#########################
### Optimization loop ###
#########################
#########################
#########################

while metropend==0:
	combind+=1
	if metropolis!='yes':
		metropend=1 # one run through only if no optimization
		
	start_11 = time.time()
	
	#######################
	### Parameter trial ###
	#######################
	# NOTE: only in case of optimization run
	if metropolis=='yes':
		if metropstep!=0:
			# Randomly selects a parameter to be tested
			ptest=random.randint(0,len(params)-1)
			while testbool[ptest]==0:
				ptest=random.randint(0,len(params)-1)
			# Retrieve value
			exec('pold='+params[ptest])
			# Change its value
			padd=((random.randint(0,1000)/1000.0)-0.5)*paramdelta[ptest]
			pnew=pold*exp(padd)
			# Again until we're within the specified bounds
			while pnew<paramlb[ptest] or pnew>paramub[ptest]:
				padd=(random.randint(0,1000)/1000.0)-0.5*paramdelta[ptest]
				pnew=pold*exp(padd)
			# Attribute that new value to the parameter
			exec(params[ptest]+'=pnew')
			exec(paramsnumber[ptest]+'=pnew')
		else:
			pnew='pnew'
			pold='pold'
			ptest='ptest'
			for param in range(len(params)):
				exec('paramatrix[combind,6+param]='+params[param])
	else:
		for param in range(len(params)):
			exec('paramatrix[combind,6+param]='+params[param])

	######################
	### Initialization ###
	###################### 
	### consecutive burning days -- start at 0
	consday=zeros(len(selgrids))
						
	### Burned area tracker, for memory of burned area as fragmentation patches
	burneddailyforest=burneddailyforestorig
	burneddailyshrubs=burneddailyshrubsorig
	burneddailygrass=burneddailygrassorig
				
	### Seasonality tracker
	seasontracker=seasontrackerorig
			
	### Precip tracker
	precip=preciporig.copy()
	
	##########################
	##########################							
	##########################
	### Loop on grid-cells ###
	##########################
	##########################
	##########################
	for il in range(len(selgrids)):
		start_gridcells = time.time()
		i=selgrids[il]+2
		if il%100==0:
			print 'Grid-cell ' + str(il)+'/'+str(len(selgrids))
		inonland=nonland[i,4]	
		# Cases of grid-cells we dont want to model (eg water, covered in ice, no wildland)
		if water[i,4]>0.5 or isnan(water[i,4]) or inonland>0.9 or (forest[i,4]+shrubs[i,4]+grass[i,4])<=0.:
			"Do nothing"
		##############################################
		### Case of grid-cells we do want to model ###
		##############################################
		else:
			# In case we split the climate data for data management, we check whether we're at the point we should read new ones.
			ilclim=il
			if il >= ceil(len(selgrids)/2.):
				ilclim=il-ceil(len(selgrids)/2.)	
				if il == ceil(len(selgrids)/2.):	
					for cv in ['rh','temp','moist','wind','light']:
						if eval('use' + cv + '==1'):  
							exec('alldaily' + cv + '=zeros(shape=(yearE-yearB+1,numgrid-ceil(numgrid/2.)+2,370),dtype=float32)')
							exec('allnightly' + cv + '=zeros(shape=(yearE-yearB+1,numgrid-ceil(numgrid/2.)+2,370),dtype=float32)')
							if (projection != 'no') and (cv != 'light'):
								exec('tot' + cv + 'proj=loadtxt(relpath+path' + cv + 'proj + str(int(yearproj)) + "'"_"'" + str(res) + "'".csv"'",  delimiter="'","'")')
								exec('all' + cv + 'proj=zeros(shape=(numgrid-ceil(numgrid/2.)+1+2,12+4),dtype=float32)')
								exec('all' + cv + 'proj=tot' + cv +'proj[numgrid-ceil(numgrid/2.)+2:,:]')
						for y in range(yearE-yearB+1):
							### Reading annual input files
							print 'Extracting ' + cv + ' for year ' + str(int(yearB+y))
							exec('ydaily' + cv + '=genfromtxt(relpath+pathdaily' + cv + ' + str(int(yearB+y)) + "'"_"'" + str(res) + "'".csv"'", skip_header=0, missing_values=''-9999'', filling_values=0,delimiter="'","'")')
							exec('alldaily' + cv + '[int(y),2:,:]=ydaily' + cv + '[selgrids[numgrid-floor(numgrid/2.):]+2,:]')
							exec('del(ydaily' + cv + ')')
							exec('ynightly' + cv + '=genfromtxt(relpath+pathnightly' + cv + ' + str(int(yearB+y)) + "'"_"'" + str(res) + "'".csv"'", skip_header=0, missing_values=''-9999'', filling_values=0,delimiter="'","'")')
							exec('allnightly' + cv + '[int(y),2:,:]=ynightly' + cv + '[selgrids[numgrid-floor(numgrid/2.):]+2,:]')
							exec('del(ynightly' + cv + ')')							
					# For lightning projection based on convective precip, we use daily projected file
# 					if (projection != 'no'):
# 						cv = 'light'
# 						for y in range(yearE-yearB+1):
# 							### Reading annual input files
# 							print 'Extracting ' + cv + ' for year ' + str(int(yearB+y))
# 							exec('ydaily' + cv + '=genfromtxt(relpath+path' + cv + 'dailyproj+ str(int(yearproj+y)) + "'"_"'" + str(res) + "'".csv"'", skip_header=0, missing_values=''-9999'', filling_values=0,delimiter="'","'")')
# 							exec('alldaily' + cv + '[int(y),2:,:]=ydaily' + cv + '[selgrids[numgrid-ceil(numgrid/2.)+1:]+2,:]')
# 							exec('del(ydaily' + cv + ')')
# 							exec('ynightly' + cv + '=genfromtxt(relpath+path' + cv + 'nightlyproj+ str(int(yearproj+y)) + "'"_"'" + str(res) + "'".csv"'", skip_header=0, missing_values=''-9999'', filling_values=0,delimiter="'","'")')
# 							exec('allnightly' + cv + '[int(y),2:,:]=ynightly' + cv + '[selgrids[numgrid-ceil(numgrid/2.)+1:]+2,:]')
# 							exec('del(ynightly' + cv + ')')

			dailyrh=alldailyrh[:,ilclim+2,:]
			nightlyrh=allnightlyrh[:,ilclim+2,:]
			dailytemp=alldailytemp[:,ilclim+2,:]
			nightlytemp=allnightlytemp[:,ilclim+2,:]
			if usemoist>=0:
				dailymoist=alldailymoist[:,ilclim+2,:]
				nightlymoist=allnightlymoist[:,ilclim+2,:]
			if usewind==1:
				dailywind=alldailywind[:,ilclim+2,:]
				nightlywind=allnightlywind[:,ilclim+2,:]
			if uselight == 1:
				dailylight=alldailylight[:,ilclim+2,:]
				nightlylight=allnightlylight[:,ilclim+2,:]																	
						

			################################
			### Grid-cell initialization ###
			################################
			### Initializing a bunch of variables for that grid cell that are then updated through time

			### Coordinates
			ilat=forest[i,2]
			ilon=forest[i,3]
			ilatind=forest[i,0]	
			ilonind=forest[i,1]
			igridsize = gridsizesel[il]
			
			### Burned area for each land cover over the last 8 months (contributes to fragmentation)
			# NOTE: we make sure the burned area is not more than the land cover area in the landcover data used
			for lcname in ['forest','shrubs','grass']:
				if eval('nansum(burneddaily' + lcname +'[il,:])/igridsize>' + lcname + '[i,4]'):
					exec('burneddaily' + lcname + '[il,:]=burneddaily' + lcname + '[il,:]*(igridsize*' + lcname + '[i,4])/(nansum(burneddaily' + lcname + '[il,:]+1))')
				exec('i' + lcname + '=' + lcname + '[i,4]-nansum(burneddaily' + lcname + '[il,:])/igridsize')
				exec('i' + lcname + '=nanmax([0,i' + lcname + '])')
					
			### GDP
			# GDP Influence on ignitions
			if gdpfactor[0]>=0:
				igdp=gdp[i,4]
				# Computing the normalized GDP factor and it's influence on fire ignitions through the shape parameter gdpexp
				gdpeffect=(1-nanmin([(gdpfactor[1]-gdpfactor[0]),nanmax([0,igdp-gdpfactor[0]])])/(gdpfactor[1]-gdpfactor[0]))				
				if gdpexp>1:
					gdpeffect=lookup[gdpeffect*1000,(gdpexp-1)*10+95]
				elif gdpexp<1:
					gdpeffect=lookup[gdpeffect*1000,(gdpexp-0.05)*100]
			else: 
				gdpeffect=1.
			# GDP Influence on suppression	
			if gdpsupfactor[0]>-0.1:
				igdp=gdp[i,4]
				# Computing the normalized GDP factor and it's influence on fire ignitions through the shape parameter gdpsupexp
				gdpsupeffect=(1-nanmin([(gdpsupfactor[1]-gdpsupfactor[0]),nanmax([0,igdp-gdpsupfactor[0]])])/(gdpsupfactor[1]-gdpsupfactor[0]))				
				if gdpsupexp>1:
					gdpsupeffect=lookup[gdpsupeffect*1000,(gdpsupexp-1)*10+95]
				elif gdpsupexp<1:
					gdpsupeffect=lookup[gdpsupeffect*1000,(gdpsupexp-0.05)*100]
			else:
				gdpsupeffect=1.
			# Saving suppression effort metric as a model output
			gdpout[il+2,4]=gdpeffect
			
			### Fuel
			# Precipitation proxy 
			seastrack=seasontracker[il,:]
			# Retrieving the precip over 12 months (dependent on precip delay for fuel accumulation).
			itotprecip=sum(seastrack[0:-precipdelay])/12.
			# Computing the fuel influence on fire termination through the shape parameter dryexp
			if itotprecip>precipfactor[0] and itotprecip<precipfactor[1]:
				idryseasoneffect=(itotprecip-precipfactor[0])/(precipfactor[1]-precipfactor[0]) # some fuel limitation 
			elif itotprecip<precipfactor[0]:
				idryseasoneffect=0 # no fuel limitation
			elif itotprecip>precipfactor[1]:
				idryseasoneffect=1 # too dry for fuel
			if dryexp>1:
				idryseasoneffect=lookup[idryseasoneffect*1000,(dryexp-1)*10+95]
			elif dryexp<1:
				idryseasoneffect=lookup[idryseasoneffect*1000,(dryexp-0.05)*100]
			iprecipgrid=precip[il,:]
			if projection != 'no':
				iprecipproj = totprecipproj[il+2,4:]
			
			### Land use and land use ignitions
			ilanduse=landuse[i,4]
			# Land use Ignitions
			# Case of decreasing ignition frequency per area with additional land use settlements
			if variableluignfactor[0]>=0:
				lubgignitions=0
				if ilanduse>0:
					luarea=ilanduse*igridsize
					maxluarea=variableluignfactor[1]*igridsize
					# Computing land use ignitions based on land use area, ignitions per area (luign), GDP (gdpeffect) and the factor to decrease additional ignitions with denser land use (variablelueffect, based on luignexp)
					# !!! replace with integral
					for incrlu in arange(0.,nanmin([luarea,maxluarea]),maxluarea/1000.):
						# Computing the normalized land use factor and it's influence on fire ignitions through the shape parameter gdpexp
						variablelueffect=1-(incrlu/igridsize-variableluignfactor[0])/(variableluignfactor[1]-variableluignfactor[0])
						if luignexp>1:
							lubgignitions+=(nanmin([incrlu+maxluarea/1000.,luarea])-incrlu)*(luign*gdpeffect+bgign)*lookup[variablelueffect*1000,(luignexp-1)*10+95]
						elif luignexp<1:
							lubgignitions+=(nanmin([incrlu+maxluarea/1000.,luarea])-incrlu)*(luign*gdpeffect+bgign)*lookup[variablelueffect*1000,(luignexp-0.05)*100]
						elif luignexp==1:
							lubgignitions+=(nanmin([incrlu+maxluarea/1000.,luarea])-incrlu)*(luign*gdpeffect+bgign)*variablelueffect
				hignout[il+2,4]=lubgignitions
			# Case of no impact of land use density on land use ignitions per area
			else:
				lubgignitions=(luign*ilanduse*gdpeffect+ilanduse*bgign)*(igridsize)
				hignout[il+2,4]=lubgignitions
			elapsed_gridcells += time.time() - start_gridcells
			
			### Lightnings climatology (if convection precip derived, done later)
			if uselight == 'yes':
				ilight=lightflash[i,4:]
				
			### Climate projection
			# each grid cell has monthly climate anomalies (absolute value), for one average year.
			# We retrieve the 12 monthly values for the grid cell considered
			if projection != 'no':
				for cv in ['rh','temp','moist','wind']:
					if eval('use' + cv + '==1'):
						exec('i' + cv + 'proj=all' + cv + 'proj[ilclim+2,4:]')

			##########################
			##########################							
			##########################
			##### Loop on years ######
			##########################
			##########################
			##########################

			for y in arange(yearB,yearE+0.1,1):
				# tracking program performances
				elapsed_time = time.time() - start_time
				start_12 = time.time()
				y=int(y)
				yearind=y-yearB+4
				month=1
				day=-1
				
				### Leap years
				if leapyears == 1:
					if y%4==0: 
						monthcal=monthleapcal
						daysinyear=366
					else:
						monthcal=monthnoleapcal
						daysinyear=365		
				elapsed_12 += time.time() - start_12
				
				### Fragmentation index
				# Total natural vegetation (sum of forests, shrubs, grasslands)
				iwildland=zeros(shape=(1,daysinyear+1))*0
				iwildland[0,0]=(iforest+ishrubs+igrass)*igridsize
				# Total non-fire prone areas (sum of landuse, recently burned, bare, water)
				ibarriers=zeros(shape=(1,daysinyear+1))*0
				ibarriers[0,0]=(ilanduse+inonland)*igridsize+nansum(burneddailyforest[il,:])+nansum(burneddailyshrubs[il,:])+nansum(burneddailygrass[il,:])
				# Fragmentation index (ratio of non-fire prone over the whole area)
				ifragm=zeros(shape=(1,daysinyear+1))*0
				ifragm[0,0]=ibarriers[0,0]/(iwildland[0,0]+ibarriers[0,0])

				### Ignition tracker
				# Actual daily ignitions
				actualign=zeros(shape=(1,daysinyear+1))
				# Total number of fires for model output
				atotfires=0
				

				##########################
				##########################							
				##########################
				###### Loop on days ######
				##########################
				##########################
				##########################
				for id in array(range(daysinyear)):
					day+=1
					
					
					################################
					################################					
					################################
					###### Wet day -> no fire ######
					################################
					################################
					################################
					### Selecting grid-cell data for that day
					if uselight == 1:
						idaylight=dailylight[y-yearB,day+4]
						inightlight=nightlylight[y-yearB,day+4]
					else:
						idaylight=0
						inightlight=0
					if userh==1:
						idayrh=nanmax([dailyrh[y-yearB,day+4],0]) # just in case there are non-valid data (<0)
						inightrh=nanmax([nightlyrh[y-yearB,day+4],0])
						if projection != 'no':
							idayrh += irhproj[month]
							inightrh += irhproj[month]
							idayrh = nanmax([idayrh,0])
							inightrh = nanmax([inightrh,0])					
					if usemoist==1:
						idaymoist=nanmax([dailymoist[y-yearB,day+4],0])
						inightmoist=nanmax([nightlymoist[y-yearB,day+4],0])
						if projection != 'no':
							idaymoist += imoistproj[month]
							inightmoist += imoistproj[month]
							idaymoist = nanmax([idaymoist,0])
							inightmoist = nanmax([inightmoist,0])														
					if usetemp==1:						
						idaytemp=dailytemp[y-yearB,day+4]
						inighttemp=nightlytemp[y-yearB,day+4]
						if projection != 'no':
							idaytemp += itempproj[month]
							inighttemp += itempproj[month]
							#idaytemp = nanmax([idaytemp,0])
							#inighttemp = nanmax([inighttemp,0])							
					if usewind==1: # Case wind is accounted for
						idaywind=dailywind[y-yearB,day+4]
						inightwind=nightlywind[y-yearB,day+4]
						if projection != 'no':
							idaywind += iwindproj[month]
							inightwind += iwindproj[month]							
							idaywind = nanmax([idaywind,0])
							inightwind = nanmax([inightwind,0])													
	
					### Checking whether day is potentially conducive to fires
					# Wet, no fuel period or too cold -> no fires, we just update the day counting and put back land burned more than bamemory ago
					fireprone = 1	
					if usemoist==1: # Model accounts for soil moisture 
						if (idaymoist>Moistrange[1] and inightmoist>Moistrange[1]):
							fireprone = 0
					if userh==1: # Model accounts for soil moisture 
						if (idayrh>Rhrange[1] and inightrh>Rhrange[1]):
							fireprone = 0
					if usetemp==1: # Model accounts for soil moisture 
						if (idaytemp<Trange[0] and inighttemp<Trange[0]):
							fireprone = 0
					if ((fireprone == 0) or (itotprecip < precipfactor[0])):
						# Case of a day not prone to fires: re-sets fire counters to zero
						consday[il]=0
						start_wet = time.time()
						monthold=month
						month=where(monthcal>day)[0][0]
						monthind=(y-yearB)*12+(month)+4
						# Updating fragmentation and land cover data from previous fires (need to do that every day)
						iwildland[0,day+1]=iwildland[0,day]+burneddailyforest[il,0]+burneddailyshrubs[il,0]+burneddailygrass[il,0]
						ibarriers[0,day+1]=ibarriers[0,day]-burneddailyforest[il,0]-burneddailyshrubs[il,0]-burneddailygrass[il,0]
						ifragm[0,day+1]=ibarriers[0,day+1]/(iwildland[0,day+1]+ibarriers[0,day+1])
						ishrubs=((ishrubs*igridsize)+burneddailyshrubs[il,0])/igridsize
						iforest=((iforest*igridsize)+burneddailyforest[il,0])/igridsize
						igrass=((igrass*igridsize)+burneddailygrass[il,0])/igridsize
						burneddailyforest[il,0:-1]=burneddailyforest[il,1:]
						burneddailyforest[il,-1]=0
						burneddailyshrubs[il,0:-1]=burneddailyshrubs[il,1:]
						burneddailyshrubs[il,-1]=0
						burneddailygrass[il,0:-1]=burneddailygrass[il,1:]
						burneddailygrass[il,-1]=0
						# Updating fuel tracker: we take out the oldest month (if changing), and add up current month as most recent
						### If we change month, updating the precip index for fuel availability
						if monthold != month:
							seasontracker[il,0:-1]=seasontracker[il,1:]
							seasontracker[il,-1]=iprecipgrid[month]
							if projection != 'no':
								seasontracker[il,-1] += iprecipproj[month]
								seasontracker[il,-1] = nanmax([seasontracker[il,-1],0])
							seastrack=seasontracker[il,:]
							itotprecip=sum(seastrack[0:-precipdelay])/12.
							# Recomputing new fuel proxy index
							if itotprecip>precipfactor[0] and itotprecip<precipfactor[1]:
								idryseasoneffect=(itotprecip-precipfactor[0])/(precipfactor[1]-precipfactor[0]) # some fuel limitation 
							elif itotprecip<precipfactor[0]:
								idryseasoneffect=0 # no fuel limitation
							elif itotprecip>precipfactor[1]:
								idryseasoneffect=1 # too dry for fuel
							if dryexp>1:
								idryseasoneffect=lookup[idryseasoneffect*1000,(dryexp-1)*10+95]
							elif dryexp<1:
								idryseasoneffect=lookup[idryseasoneffect*1000,(dryexp-0.05)*100]																
						elapsed_wet += time.time() - start_wet
						
							
					################################
					################################					
					################################
					###### Dry day -> fire ######
					################################
					################################
					################################
					else:
						start_head = time.time()
						# Updating consecutive fire prone consecutive days
						consday[il]+=1
						d=consday[il]-1
						
						# Fire duration limit
						# !!! reverse
						if d>101:
							d=101
						
						# checking whether firespread is gonna stop sometime in that day
						dayend = 0 # will be == 1 if conditions are fire prone for only part of the day
						if usemoist==1: # Model accounts for soil moisture 
							if (idaymoist>Moistrange[1] or inightmoist>Moistrange[1]):
								dayend = 1
						if userh==1: # Model accounts for soil moisture 
							if (idayrh>Rhrange[1] or inightrh>Rhrange[1]):
								dayend = 1
						if usetemp==1: # Model accounts for soil moisture 
							if (idaytemp<Trange[0] or inighttemp<Trange[0]):
								dayend = 1																		

						##########################################
						############ DATA INITIALIZATION #########
						##########################################
						start_exec = time.time()
						if d==0: # first day of burning
							for dmat in ['activefires','bashrubs','bagrass','baforest','balength','bawidth']:
								exec(dmat +'=zeros(shape=(1,1))')
						elif d<=100: # more than one day
							for dmat in ['activefires','bashrubs','bagrass','baforest','balength','bawidth']:
								exec(dmat + '=concatenate((' + dmat + ',zeros(shape=(1,d))),axis=0)')
								exec(dmat + '=concatenate((' + dmat + ',zeros(shape=(d+1,1))),axis=1)')
						else: # Limit of 100 days max per fire
							for dmat in ['activefires','bashrubs','bagrass','baforest','balength','bawidth']:							
								exec(dmat + '=concatenate((' + dmat + '[1:,:],zeros(shape=(1,d))),axis=0)')
								exec(dmat + '=concatenate((activefires[:,1:],zeros(shape=(d,1))),axis=1)')
							d=100
						elapsed_exec += time.time() - start_exec
							
							
						################################
						############ IGNITIONS #########
						################################
						### Total potential ignitions
						ignitionsday=lubgignitions+idaylight*natign*igridsize
						ignitionsnight=lubgignitions+inightlight*natign*igridsize
					
						##################################
						############ SUPPRESSION #########
						##################################
						### Depends on landuse on the grid-cell.
						# !!! move to grid-cell level computation
						suppression=1-(ilanduse/(1-inonland)) 
						suppression=(ilanduse/(1-inonland))/(landuselimit) 
						if suppression>1:
							suppression=1
						suppression=1-suppression
						if supexp>1:
							suppression=lookup[suppression*1000,(supexp-1)*10+95]
						elif supexp<1:
							suppression=lookup[suppression*1000,(supexp-0.05)*100]
																													
						elapsed_head = time.time() - start_head				
						start_start = time.time()
						
						### Month
						# !!! move to month computation above ?
						monthold=month
						month=where(monthcal>day)[0][0]
						monthind=(y-yearB)*12+(month)+4


						####################################
						############ FRAGMENTATION #########
						####################################
						### Fragmentation influence on fire termination using the shape parameter fragexp
						fragmeffect=1-ifragm[0,day]
						if fragmeffect<0:
							fragmeffect=0
						if fragexp>1:
							fragmeffect=lookup[fragmeffect*1000,(fragexp-1)*10+95]
						elif fragexp<1:
							fragmeffect=lookup[fragmeffect*1000,(fragexp-0.05)*100]
						
						elapsed_start += time.time() - start_start
						
						##################################
						############ SPREAD RATE #########
						##################################
						### FINAL IMPACT OF HUMIDITY AND TEMP ON SPEED (WEIGHTED BY LANDCOVER FRACTION)
						start_spread = time.time()

						### RH				
						if userh == 1:		
							if idayrh>Rhrange[0] and idayrh<Rhrange[1]:
								dayrhspeedfactor=1-(idayrh-Rhrange[0])/(Rhrange[1]-Rhrange[0])
							elif idayrh<=Rhrange[0]:
								dayrhspeedfactor=1 # no limitation
							elif idayrh>=Rhrange[1]:
								dayrhspeedfactor=0 # too wet to burn
							if inightrh>Rhrange[0] and inightrh<Rhrange[1]:
								nightrhspeedfactor=1-(inightrh-Rhrange[0])/(Rhrange[1]-Rhrange[0])
							elif inightrh<=Rhrange[0]:
								nightrhspeedfactor=1 # no limitation
							elif inightrh>=Rhrange[1]:
								nightrhspeedfactor=0 # too wet to burn
							# Now computing final factor using look up table and rhexp.
							if rhexp>1:
								dayrhspeedfactor=lookup[dayrhspeedfactor*1000,(rhexp-1)*10+95]
								nightrhspeedfactor=lookup[nightrhspeedfactor*1000,(rhexp-1)*10+95]
							elif rhexp<1:
								dayrhspeedfactor=lookup[dayrhspeedfactor*1000,(rhexp-0.05)*100]
								nightrhspeedfactor=lookup[nightrhspeedfactor*1000,(rhexp-0.05)*100]
						else: # (Case RH has no impact)
							dayrhspeedfactor=1.
							nightrhspeedfactor=1.	
						
						### Soil moisture (Case of soil moisture having an influence)
						if usemoist==1:
							if idaymoist>Moistrange[0] and idaymoist<Moistrange[1]:
								daymoistspeedfactor=1-(idaymoist-Moistrange[0])/(Moistrange[1]-Moistrange[0])
							elif idaymoist<=Moistrange[0]:
								daymoistspeedfactor=1 # no limitation
							elif idaymoist>=Moistrange[1]:
								daymoistspeedfactor=0 # too wet to burn
							if inightmoist>Moistrange[0] and inightmoist<Moistrange[1]:
								nightmoistspeedfactor=1-(inightmoist-Moistrange[0])/(Moistrange[1]-Moistrange[0])
							elif inightmoist<=Moistrange[0]:
								nightmoistspeedfactor=1 # no limitation
							elif inightmoist>=Moistrange[1]:
								nightmoistspeedfactor=0 # too wet to burn	
							# Now computing final factor using look up table and moistexp.
							if moistexp>1:
								daymoistspeedfactor=lookup[daymoistspeedfactor*1000,(moistexp-1)*10+95]
								nightmoistspeedfactor=lookup[nightmoistspeedfactor*1000,(moistexp-1)*10+95]
							elif moistexp<1:
								daymoistspeedfactor=lookup[daymoistspeedfactor*1000,(moistexp-0.05)*100]
								nightmoistspeedfactor=lookup[nightmoistspeedfactor*1000,(moistexp-0.05)*100]							
						else: # (Case soit moisture has no impact)
							daymoistspeedfactor=1.
							nightmoistspeedfactor=1.
						
						### Temperature
						if usetemp==1:						
							if idaytemp>Trange[0] and idaytemp<Trange[1]:
								daytempspeedfactor=(idaytemp-Trange[0])/(Trange[1]-Trange[0])
							elif idaytemp<=Trange[0]:
								daytempspeedfactor=0 
							elif idaytemp>=Trange[1]:
								daytempspeedfactor=1 
							if inighttemp>Trange[0] and inighttemp<Trange[1]:
								nighttempspeedfactor=(inighttemp-Trange[0])/(Trange[1]-Trange[0])
							elif inighttemp<=Trange[0]:
								nighttempspeedfactor=0 
							elif inighttemp>=Trange[1]:
								nighttempspeedfactor=1 							
																						
							# Now computing final factor using look up table and tempexp.																												
							if tempexp>1:
								daytempspeedfactor=lookup[daytempspeedfactor*1000,(tempexp-1)*10+95]
								nighttempspeedfactor=lookup[nighttempspeedfactor*1000,(tempexp-1)*10+95]
							elif tempexp<1:
								daytempspeedfactor=lookup[daytempspeedfactor*1000,(tempexp-0.05)*100]
								nighttempspeedfactor=lookup[nighttempspeedfactor*1000,(tempexp-0.05)*100]															
						else: # (Case temp has no impact)
							daytempspeedfactor=1.
							nighttempspeedfactor=1.
							
							
						### Fuel influence on spread rate ?
						if speedfuel==1:
							speedfact=idryseasoneffect #
						else:
							speedfact=1.
						
						### Computation of spread rate after RH, soilw and temp are accounted for (prior to wind computations) 
						dayshrubspeed=shrubspread*dayrhspeedfactor*daytempspeedfactor*daymoistspeedfactor*speedfact
						daygrassspeed=grassspread*dayrhspeedfactor*daytempspeedfactor*daymoistspeedfactor*speedfact
						dayforestspeed=forestspread*dayrhspeedfactor*daytempspeedfactor*daymoistspeedfactor
						nightforestspeed=forestspread*nightrhspeedfactor*nighttempspeedfactor*nightmoistspeedfactor
						nightshrubspeed=shrubspread*nightrhspeedfactor*nighttempspeedfactor*nightmoistspeedfactor*speedfact
						nightgrassspeed=grassspread*nightrhspeedfactor*nighttempspeedfactor*nightmoistspeedfactor*speedfact
						elapsed_spread += time.time() - start_spread
						
						### Wind: influence on spread rate and L/B ratio
						start_wind = time.time()
						if usewind==1: # Case wind is accounted for
							LBday=lookupLB[idaywind*10.]
							LBnight=lookupLB[inightwind*10.]
							LB=(LBday+LBnight)/2.
							HBday=lookupHB[idaywind*10.]
							HBnight=lookupHB[inightwind*10.]
							HB=(HBday+HBnight)/2.
							GWday=lookupGW[idaywind*10.]
							GWnight=lookupGW[inightwind*10.]
							GW=(GWday+GWnight)/2.
						else: # no influence of wind
							LB=lbratio # Default user-input value
							HB=10.
							GWday=1.
							GWnight=1.
						
						### Final average spread rate
						# !!! reverse to bi-daily spread ?
						avspeed=((dayshrubspeed*GWday+nightshrubspeed*GWnight)/2.*ishrubs+(dayforestspeed*GWday+nightforestspeed*GWnight)/2.*iforest)+(daygrassspeed*GWday+nightshrubspeed*GWnight)/2.*igrass/(ishrubs+iforest+igrass)
						avspeedback=avspeed/HB
						avspeedperp=(avspeed+avspeedback)/(2*LB)
						elapsed_wind = time.time()-start_wind
						
						#####################################
						############ FIRE INTENSITY #########
						#####################################
						# times
						# Case of interaction (rhfactor of 0.5 and soilw of 0.1 result in 0.05)
						if meanintensity==0:
							fireintensity=idryseasoneffect*(dayrhspeedfactor*daytempspeedfactor*daymoistspeedfactor*GWday+nightrhspeedfactor*nighttempspeedfactor*nightmoistspeedfactor*GWnight)/2.
						# Case of the average (rhfactor of 0.5 and soilw of 0.1 result in 0.3)
						if meanintensity==1:
							fireintensity=(idryseasoneffect+idryseasoneffect+dayrhspeedfactor+daytempspeedfactor+daymoistspeedfactor+GWday+nightrhspeedfactor+nighttempspeedfactor+nightmoistspeedfactor+GWnight)/10.																							
						# Influence on suppression (linear, no shape parameter)
						if intensitysuppression==1:
							intsupp=1.-fireintensity
						else:
							intsupp=1.
						
						##################################
						############ SUPPRESSION #########
						##################################
						if gdpsuppression=='yes':
							gdpsupp=gdpsupeffect
						else:
							gdpsupp=1.
						# Suppression effort model output
						suppout[il+2,4]=(1-(1-suppression)*(1-gdpsupp))
														
						
						#######################################
						############ FUEL TERMINATION #########
						#######################################							
						if speedfuel==1:
							fuelsupfact=1 # when lack of fuel acts on speed
						else:
							fuelsupfact=idryseasoneffect
						
						#######################################
						############ ACTUAL IGNITIONS #########
						#######################################
						actualign[0,day]=(ignitionsday+ignitionsnight)*(1-(1-suppression)*(1-gdpsupp)*intsupp)*fragmeffect*fuelsupfact					
						
						####################################
						############ STOCHASTICITY #########
						####################################	
						### Based on binomial distribution
						start_stoch = time.time()
						# Case of no ignitions
						if actualign[0,day]<=0:
							actualign[0,day]=0
						# Case of potential ignitions
						else:
							if stochastic == 1:
								actualign[0,day] = binom.rvs(10000., actualign[0,day]/10000., size=1)				
							else:
								"do nothing"
						### Final number of fires for that day
						activefires[d,d]=actualign[0,day]
						elapsed_stoch += time.time() - start_stoch
						
						####################################
						############ BURNED AREA ###########
						####################################	
						### Ellipsoid function, partitioned into the land cover types
						# !!! reverse to bi-daily ?
						start_ba = time.time()
						if (ishrubs+iforest+igrass)>0:
							# length * width * pi/4 (12 for 12 hours, ignitions can happen anytime of the day, will burn for 12 hours on average)
							bashrubs[d,d]=(avspeed+avspeedback) * avspeedperp * pi/4. * kmpersecperdayperms/2. * ishrubs/(ishrubs+iforest+igrass)
							baforest[d,d]=(avspeed+avspeedback) * avspeedperp * pi/4. * kmpersecperdayperms/2. * iforest/(ishrubs+iforest+igrass)
							bagrass[d,d]=(avspeed+avspeedback) * avspeedperp * pi/4. * kmpersecperdayperms/2. * igrass/(ishrubs+iforest+igrass)																				
							# Length and breadth of ellipse)
							balength[0,d]=(avspeed+avspeedback) * kmpersecperdayperms/2.
							bawidth[0,d]=avspeedperp * kmpersecperdayperms/2.
							# Fractions of each landcover (for later computations of partitionned burned area)
							fracshrubs=ishrubs/(ishrubs+iforest+igrass)
							fracforest=iforest/(ishrubs+iforest+igrass)
							fracgrass=igrass/(ishrubs+iforest+igrass)
							
							# Case of full day of burning (not last climate-fire day -- unless termination).
							# Tracks ongoing fires fate from day-1 -- computes additional burned area that day.
							if d>0 and dayend==0:
								# Termination of ongoing fires due to suppression or fragmentation
								activefires[0:d,d]=activefires[0:d,d-1]*(1-(1-suppression)*(1-gdpsupp)*intsupp)*fragmeffect*fuelsupfact
								if stochastic==1:
									meandiff=activefires[0:d,d-1]-activefires[0:d,d-1]*(1-(1-suppression)*(1-gdpsupp)*intsupp)*fragmeffect*fuelsupfact
									takeout=binom.rvs(10000, meandiff/10000., size=d)
									activefires[0:d,d]=activefires[0:d,d-1]-takeout
									activefires[0:d,d] = where(activefires[0:d,d]<0,0,activefires[0:d,d])																					
								else:
									activefires[0:d,d]=activefires[0:d,d-1]*(1-(1-suppression)*(1-gdpsupp)*intsupp)*fragmeffect*fuelsupfact																					

								# Computing spread (newb is the new burned area that day)
								newb=((balength[0,0:d]+(avspeed+avspeedback)*kmpersecperdayperms) * (bawidth[0,0:d]+(avspeedperp)*kmpersecperdayperms) * pi/4) - (balength[0,0:d] * bawidth[0,0:d] * pi/4)
								# Influence of fragmentation and other factors on the area of the ellipse that actually burned
								if partialellispse == 1:
									newb = newb * fragmeffect
								bashrubs[0:d,d]= newb * fracshrubs
								baforest[0:d,d]= newb * fracforest
								bagrass[0:d,d]= newb * fracgrass		
								balength[0,0:d]+=(avspeed+avspeedback) * kmpersecperdayperms
								bawidth[0,d]+=avspeedperp * kmpersecperdayperms
							
							# Case of last climate-fire day -- due to climate/moisture conditions.
							# Tracks ongoing fires fate from day-1 -- computes additional burned area that day but just for 12 hours (termination can happen any time).								
							if d>0 and dayend==1: # Last fire spread day
								activefires[0:d,d]=activefires[0:d,d-1]*(1-(1-suppression)*(1-gdpsupp)*intsupp)*fragmeffect*fuelsupfact
								if stochastic==1:
									meandiff=activefires[0:d,d-1]-activefires[0:d,d-1]*(1-(1-suppression)*(1-gdpsupp)*intsupp)*fragmeffect*fuelsupfact
									takeout=binom.rvs(10000, meandiff/10000., size=d)
									activefires[0:d,d]=activefires[0:d,d-1]-takeout
									activefires[0:d,d] = where(activefires[0:d,d]<0,0,activefires[0:d,d])																					
								else:
									activefires[0:d,d]=activefires[0:d,d-1]*(1-(1-suppression)*(1-gdpsupp)*intsupp)*fragmeffect*fuelsupfact
								# Computing spread (newb is the new burned area that day)
								newb=((balength[0,0:d]+(avspeed+avspeedback)*kmpersecperdayperms/2.) * (bawidth[0,0:d]+(avspeedperp)*kmpersecperdayperms/2.) * pi/4) - (balength[0,0:d] * bawidth[0,0:d] * pi/4)
								# Influence of fragmentation and other factors on the area of the ellipse that actually burned
								if partialellispse == 1:
									newb = newb * fragmeffect								
								bashrubs[0:d,d]= newb * fracshrubs			
								baforest[0:d,d]= newb * fracforest	
								bagrass[0:d,d]= newb * fracgrass		
								balength[0,0:d]+=(avspeed+avspeedback) * kmpersecperdayperms/2.
								bawidth[0,d]+=avspeedperp * kmpersecperdayperms/2.	
						# Case of no natural land cover to burn
						else:
							bashrubs[0:d+1,d]=0
							baforest[0:d+1,d]=0
							bagrass[0:d+1,d]=0
						elapsed_ba += time.time() - start_ba
						

						#################################################
						############ MONTHLY & ANNUAL OUTPUTS ###########
						#################################################
						start_out = time.time()
						start_out1 = time.time()
						
						### Burned area for that day (number of active fires in each track and their additional burned area)
						totfires=sum(activefires[:,d])
						for lcname in ['forest','shrubs','grass']:
							exec('totba' + lcname +'=sum(multiply(ba' + lcname + '[:,d],activefires[:,d]))')
							if eval('totba' + lcname +'>i' + lcname +'*igridsize'):
								exec('totba' + lcname +'=i' + lcname +'*igridsize')
						totba=totbaforest+totbashrubs+totbagrass
						elapsed_out1 += time.time() - start_out1
						
						start_out3 = time.time()
						### Monthly outputs
						mfires[il+2,monthind]+=actualign[0,day]#sum(activefires[:,d-1]-activefires[:,d])
						mbashrubs[il+2,monthind]+=totbashrubs
						mbaforest[il+2,monthind]+=totbaforest
						mbagrass[il+2,monthind]+=totbagrass
						mba[il+2,monthind]+=totba
# 						if sum(dailylight[y-yearB,0:4]==[1.07000e+02,3.07000e+02,-1.75000e+01,1.27500e+02])==4:
# 							if d==6:
# 								bleeeeeee
						elapsed_out3 += time.time() - start_out3
						
						start_out4 = time.time()
						### Annual outputs
						if saveannual==1: # pretty computationally demanding (~20% of the whole code !)
							if int(consday[il])==0 and dayend==1:
								aavdur[il+2,yearind]=nansum(activefires[:,d])/2. # (assuming half day)
								atotfires=nansum(activefires[:,d])
							elif int(consday[il])>0 and dayend==0:
								aavdur[il+2,yearind]+=nansum(multiply((activefires[0:d+1,d-1]-activefires[0:d+1,d]),arange(d,-1,-1)))
								atotfires+=nansum(activefires[:,d-1]-activefires[:,d])
							elif int(consday[il])>0 and dayend==1:
								firesize=nansum((bashrubs+baforest+bagrass)*where(activefires>0,1,0),1)
								firedur=nansum(activefires>0,1)
								aavdur[il+2,yearind]+=nansum(multiply(activefires[0:d+1,d]>0,arange(d,-1,-1)))
								atotfires+=nansum(activefires[:,d])
								# Max size and duration	
								amaxsize[il+2,yearind]=nanmax([amaxsize[il+2,yearind],nanmax(firesize)])
								amaxdur[il+2,yearind]= nanmax([amaxdur[il+2,yearind],nanmax(nansum(where(activefires>0,1,0),1))])
								# 5 biggest/longest fires
								afivemaxsize[il+2,yearind,:] = sort(concatenate((afivemaxsize[il+2,yearind,:],firesize),axis=0))[-5:]
								afivemaxdur[il+2,yearind,:] = sort(concatenate((afivemaxdur[il+2,yearind,:],firedur),axis=0))[-5:]
						elapsed_out4 += time.time() - start_out4
						elapsed_out += time.time() - start_out											
						
						############################################################
						############ UPDATING FRAGMENTATION / LAND COVER ###########
						############################################################
						# To account for new burned areas having no fuel and old burned areas coming back with fuel
						start_fragm= time.time()
						if totba<0:
							dtiuhwpijw
						if day<daysinyear:
							ishrubs=((ishrubs*igridsize)-totbashrubs+burneddailyshrubs[il,0])/igridsize
							iforest=((iforest*igridsize)-totbaforest+burneddailyforest[il,0])/igridsize							
							igrass=((igrass*igridsize)-totbagrass+burneddailygrass[il,0])/igridsize
							iwildland[0,day+1]=iwildland[0,day]-totba+burneddailyforest[il,0]+burneddailyshrubs[il,0]+burneddailygrass[il,0]
							ibarriers[0,day+1]=ibarriers[0,day]+totba-burneddailyforest[il,0]-burneddailyshrubs[il,0]-burneddailygrass[il,0]
							ifragm[0,day+1]=ibarriers[0,day+1]/(iwildland[0,day+1]+ibarriers[0,day+1])
						
							### Inconsistency check
							if ibarriers[0,day+1]<0:
								if ibarriers[0,day+1]<-0.1:
									print "ibarriers < 0"
									print il
									print ibarriers[0,day+1]
									evjinvf
								ibarriers[0,day+1]=0
								ifragm[0,day+1]=ibarriers[0,day+1]/(iwildland[0,day+1]+ibarriers[0,day+1])

						
						### Taking out burned area that just went beyond the memory delay (mbamemory) from the fragmentation index
						burneddailyforest[il,0:-1]=burneddailyforest[il,1:]
						burneddailyforest[il,-1]=totbaforest
						burneddailyshrubs[il,0:-1]=burneddailyshrubs[il,1:]
						burneddailyshrubs[il,-1]=totbashrubs
						burneddailygrass[il,0:-1]=burneddailygrass[il,1:]
						burneddailygrass[il,-1]=totbagrass
						elapsed_fragm += time.time() - start_fragm
						
						### Updating fuel accumulation tracker (precip)
						# Not considering zeros
						start_up = time.time()
						### Case based on precip
						if precipfactor[0]>0:
							### If we change month, updating the precip data in seasontracker, and computing new idryseasoneffect
							if monthold != month:
								seasontracker[il,0:-1]=seasontracker[il,1:]
								seasontracker[il,-1]=iprecipgrid[month]
								if projection != 'no':
									seasontracker[il,-1] += iprecipproj[month]
									seasontracker[il,-1] = nanmax([seasontracker[il,-1],0])								
								seastrack=seasontracker[il,:]
								itotprecip=sum(seastrack[0:-precipdelay])/12.
								if itotprecip>precipfactor[0] and itotprecip<precipfactor[1]:
									idryseasoneffect=(itotprecip-precipfactor[0])/(precipfactor[1]-precipfactor[0])
								elif itotprecip<precipfactor[0]:
									idryseasoneffect=0 # no fuel limitation
								elif itotprecip>precipfactor[1]:
									idryseasoneffect=1 # too dry for fuel
								if dryexp>1:
									idryseasoneffect=lookup[idryseasoneffect*1000,(dryexp-1)*10+95]
								elif dryexp<1:
									idryseasoneffect=lookup[idryseasoneffect*1000,(dryexp-0.05)*100]															
						elapsed_up += time.time() - start_up
						
						### Updating consecutive dry days counter (back to zero if fires end that day).
						if dayend==1:
							consday[il]=0
							
					###################################
					###################################					
					###################################
					###### ENDOF Dry day -> fire ######
					###################################
					###################################
					###################################

				################################
				################################							
				################################
				###### ENDOF Loop on days ######
				################################
				################################
				################################

				### At the end of the year, now computing average fire duration
				if saveannual==1:
					aavdur[il+2,yearind]=aavdur[il+2,yearind]/atotfires
					atotbaforest[il+2,yearind]=nansum(mbaforest[il+2,monthind-11:monthind+1])
					atotbashrubs[il+2,yearind]=nansum(mbashrubs[il+2,monthind-11:monthind+1])
					atotbagrass[il+2,yearind]=nansum(mbagrass[il+2,monthind-11:monthind+1])
					atotba[il+2,yearind]=nansum(mba[il+2,monthind-11:monthind+1])
					
				### At the end of one year, we update the precip matrix to get rid of that years data. 
				### This way we know the first 12 columns are always for the current year and we dont need to call the 
				### where() function (costly)
				if precipfactor[0]>0:
					iprecipgrid=iprecipgrid[12:]

			################################
			################################							
			################################
			##### ENDOF Loop on years ######
			################################
			################################
			################################				

		####################################################
		### ENDOF Case of grid-cells we do want to model ###
		####################################################	

	################################
	################################							
	################################
	### ENDOF Loop on grid-cells ###
	################################
	################################
	################################

	
	#####################################
	###### Performance assessment #######
	#####################################
	print expname
	### Av. Burned area, seasonality and interannual variability
	start_perf = time.time()

	### Av. Burned area (obs: opt---)
	modba=mba[:,0:5].copy()
	modba[2:,4]=nansum(mba[2:,4:],axis=1)/len(mba[0,4:])*12
	modfrac=yanspatialunit(modba,2,1) # needs full matrix
	modclass=yanfireclasscontdome(modfrac[2:,4], classbound) # needs reduced matrix
	
	diffclass=modclass-optclass
	#diffclass=where((modclass==0) | (optclass==0),0,diffclass)
	globind=nansum(power(abs(diffclass),powereval))/sum(isnan(diffclass)==False)
	globbias=nansum(diffclass)/sum(isnan(diffclass)==False)
	
	evalmatrix[combind,0]=globind
	paramatrix[combind,0]=globind
	evalmatrix[combind,1]=globbias
	paramatrix[combind,1]=globbias
	
	### Peak fire month (obs: optseas)
	modseas=yanpeakmonth(mba)
	modseas=argmax(modseas,axis=1)+1
	modseas=where(nansum(mba[2:,4:],axis=1)<=0.00000001,0,modseas)
	difftemp=modseas-optseas
	diffseas=difftemp.copy()
	diffseas=where(difftemp>6,(12-difftemp)*-1,diffseas)
	diffseas=where(difftemp<-6,12-abs(difftemp),diffseas)
	diffseas=where(difftemp<-6,12-abs(difftemp),diffseas)
	diffseas=where((modseas==0) | (optseas==0),nan,diffseas)
	globseasind=nansum(abs(diffseas))/sum(isnan(diffseas)==False)
	evalmatrix[combind,3]=globseasind
	paramatrix[combind,3]=globseasind
	
	# Total burned area
	paramatrix[combind,4]=nansum(mba[2:,4:])
	observationtotba=nansum(optba[2:,4:])
	ratioba=paramatrix[combind,4]/observationtotba
	print 'ratioba'
	print ratioba
	
	### length fire season (obs: optlength)
# 								modlength=yanpeakmonth(mba)
# 								avmonth=nansum(modlength,axis=1)/12
# 								DO LENGTH FIRE SEASON
# 								modseas=nansum(where(modlength>catavmonth,axis=1)+1
# 								modseas=where(nansum(mba[2:,4:],axis=1)<=0.00000001,0,modseas)
# 								difftemp=modseas-optseas
# 								diffseas=difftemp.copy()
# 								diffseas=where(difftemp>6,(12-difftemp)*-1,diffseas)
# 								diffseas=where(difftemp<-6,12-abs(difftemp),diffseas)
# 								diffseas=where(difftemp<-6,12-abs(difftemp),diffseas)
# 								diffseas=where((modseas==0) | (optseas==0),nan,diffseas)
# 								globseasind=nansum(abs(diffseas))/sum(isnan(diffseas)==False)
# 								evalmatrix[combind,2]=globseasind
# 								paramatrix[combind,2]=globseasind								
	
	### Inter-annual variability (obs: optiav)
	modiav=zeros(shape=(len(gfedc[2:,0]),yearce-yearcb+1))
	ind=-1
	for y in arange(yearcb,yearce+0.01,1):
		ind+=1
		indy=where(mba[0,:]==y)[0][:]
		modiav[:,ind]=nansum(mba[2:,indy],axis=1)
	# Correlation
	iavcorrmatrix=zeros(shape=(numgrid,1))
	for imod in range(len(modiav[:,0])):
		iavcorrmatrix[imod,0]=pearsonr(optiav[imod,:],modiav[imod,:])[0]
	
	# Taking out cells with < 1/3 of years with fire in GFED 
	badbool=where(optiav==0,0,1)
	badsum=where(sum(badbool,axis=1)<(yearE-yearB)/3.)
	iavcorrmatrix[badsum,0]=nan
	
	# In model
	badbool=where(modiav==0,0,1)
	badsum=where(sum(badbool,axis=1)<(yearE-yearB)/3.)
	iavcorrmatrix[badsum,0]=nan
	
	# Taking out regions with ++ fires (>10% burned/year), because there interrannual variability is not driven by climate.
	badind=where(optclass>3)
	iavcorrmatrix[badind,0]=nan
	
	# Taking out regions with -- landuse fires (<0.001), because there interrannual variability is mostly driven by lightning.
	landusefrac=landuse[list(selgrids+2),4]
	badind=where(landusefrac<0.001)
	iavcorrmatrix[badind,0]=nan
	
	
	globiavind=nansum(iavcorrmatrix)/sum(isnan(iavcorrmatrix)==False)
	evalmatrix[combind,2]=globiavind
	paramatrix[combind,2]=globiavind
	
	# Printing prep
	paramstring=''
	indstring=-1
	for coisa in params:
		indstring+=1
		paramstring = paramstring + coisa + '=' + '%.3g' % paramatrix[combind,6+indstring] + ' ; '
		
	# Evaluation index
	evalind=globind#(1-globiavind)
	paramatrix[combind,5]=evalind
	if globind<globindold:
		globindold=globind
	print 'BEST FIT EVER: ' +str(globindold)

	
	if evalind<=evalindold:
		print "----------------------------"
		print "--- NEW OPTIMUM SOLUTION ---"
		print "----------------------------"
		print 'index = ' + str(evalind)
		print paramstring
		print ['%.3g' % val for val in paramatrix[combind,0:6]]
		mapoptclass,latmap,lonmap=yanbuildmap(optclass,atotba[2:,0:4],res)
		mapmodclass,latmap,lonmap=yanbuildmap(modclass,atotba[2:,0:4],res)
		mapclassdiff,latmap,lonmap=yanbuildmap(diffclass,atotba[2:,0:4],res)
		mapiav,latmap,lonmap=yanbuildmap(iavcorrmatrix[:,0],atotba[2:,0:4],res)
		mapseasdiff,latmap,lonmap=yanbuildmap(diffseas,atotba[2:,0:4],res)
		mfiresg=mfires.copy()
		mbashrubsg=mbashrubs.copy()
		mbagrassg=mbagrass.copy()
		mbaforestg=mbaforest.copy()
		mbag=mba.copy()
		amaxsizeg=amaxsize.copy()
		afivemaxsizeg=afivemaxsize.copy()
		afivemaxdurg=afivemaxdur.copy()
		amaxdurg=amaxdur.copy()
		aavdurg=aavdur.copy()
		atotbaforestg=atotbaforest.copy()
		atotbagrassg=atotbagrass.copy()
		atotbashrubsg=atotbashrubs.copy()
		atotbag=atotba.copy()
									
	else:
		print "Not good solution: "
		print 'index = ' + str(evalind) + ' vs ' +str(evalindold)
		print paramstring
		print ['%.3g' % val for val in paramatrix[combind,0:6]]
		
		
	# Saving Evaluation
	sortedeval=paramatrix[paramatrix[:,5].argsort()]
	savetxt(outpath+'Evaluation.csv',sortedeval,fmt='%2.5e',delimiter=',')									
	elapsed_perf += time.time() - start_perf

	#print "Run time for year "+str(y-1)+": "+ str(time.time() - start_time)
	
	
	
	if metropolis=='yes':
		if (evalind<evalindold):
			accept=1;
		elif (random.randint(0,10000) < 1/(evalind-evalindold)) and 1/(evalind-evalindold)>0.0001:
			# Save results as that wasnt done before
			accept=1;
		else:
			accept=0;
		
		if accept==1:
			print 'METROPOLIS: Solution was accepted'
			print 'Step ' +str(metropstep)
			print str(evalind) + ' vs ' + str(evalindold)
			if metropstep!=0:
				exec(params[ptest]+'=pnew')
				print 'param='+params[ptest]
				print 'pold='+str(pold)
				print 'pnew='+str(pnew)
				savetxt(outpath+'mfires_'+str(res)+'.csv',mfiresg,fmt='%2.5e',delimiter=',')
				savetxt(outpath+'mbashrubs_'+str(res)+'.csv',mbashrubsg,fmt='%2.5e',delimiter=',')
				savetxt(outpath+'mbagrass_'+str(res)+'.csv',mbagrassg,fmt='%2.5e',delimiter=',')
				savetxt(outpath+'mbaforest_'+str(res)+'.csv',mbaforestg,fmt='%2.5e',delimiter=',')
				savetxt(outpath+'mba_'+str(res)+'.csv',mbag,fmt='%2.5e',delimiter=',')
				
				# Annual
				afivemaxsizeg=nansum(afivemaxsizeg,axis=2)/5.
				afivemaxdurg=nansum(afivemaxdurg,axis=2)/5.
				savetxt(outpath+'amaxsize_'+str(res)+'.csv',amaxsizeg,fmt='%2.5e',delimiter=',')
				savetxt(outpath+'afiveaxsize_'+str(res)+'.csv',afivemaxsizeg,fmt='%2.5e',delimiter=',')
				savetxt(outpath+'afiveaxdur_'+str(res)+'.csv',afivemaxdurg,fmt='%2.5e',delimiter=',')
				savetxt(outpath+'amaxdur_'+str(res)+'.csv',amaxdurg,fmt='%2.5e',delimiter=',')
				savetxt(outpath+'aavdur_'+str(res)+'.csv',aavdurg,fmt='%2.5e',delimiter=',')
				savetxt(outpath+'atotbaforest_'+str(res)+'.csv',atotbaforestg,fmt='%2.5e',delimiter=',')
				savetxt(outpath+'atotbashrubs_'+str(res)+'.csv',atotbashrubsg,fmt='%2.5e',delimiter=',')
				savetxt(outpath+'atotbagrass_'+str(res)+'.csv',atotbagrassg,fmt='%2.5e',delimiter=',')
				savetxt(outpath+'atotba_'+str(res)+'.csv',atotbag,fmt='%2.5e',delimiter=',')


				savetxt(outpath+'hignout'+str(res)+'.csv',hignout,fmt='%2.5e',delimiter=',')
				savetxt(outpath+'gdpout'+str(res)+'.csv',gdpout,fmt='%2.5e',delimiter=',')
				savetxt(outpath+'suppout'+str(res)+'.csv',suppout,fmt='%2.5e',delimiter=',')

			
			if metropstep!=0:
				# Increase temperature
				paramdelta[ptest]=paramdelta[ptest]*1.01
				# Dont let temperature go too big
				if paramdelta[ptest]>10:
					paramdelta[ptest]=10
				evalindold=evalind
					
			mapoptclass,latmap,lonmap=yanbuildmap(optclass,atotba[2:,0:4],res)
			mapmodclass,latmap,lonmap=yanbuildmap(modclass,atotba[2:,0:4],res)
			mapclassdiff,latmap,lonmap=yanbuildmap(diffclass,atotba[2:,0:4],res)
			mapiav,latmap,lonmap=yanbuildmap(iavcorrmatrix[:,0],atotba[2:,0:4],res)
			mapseasdiff,latmap,lonmap=yanbuildmap(diffseas,atotba[2:,0:4],res)
			mfiresg=mfires.copy()
			mbashrubsg=mbashrubs.copy()
			mbagrassg=mbagrass.copy()
			mbaforestg=mbaforest.copy()
			mbag=mba.copy()
			amaxsizeg=amaxsize.copy()
			afivemaxsizeg=afivemaxsize.copy()
			afivemaxdurg=afivemaxdur.copy()
			amaxdurg=amaxdur.copy()
			aavdurg=aavdur.copy()
			atotbaforestg=atotbaforest.copy()
			atotbashrubsg=atotbashrubs.copy()
			atotbagrassg=atotbagrass.copy()
			atotbag=atotba.copy()
			
		else: #(accept==0)
			print 'param='+params[ptest]
			print 'pold='+str(pold)
			print 'pnew='+str(pnew)										
			if metropstep!=0:
				# Reverse the tested parameter to old value
				exec(params[ptest]+'=pold')
				# Decrease temperature
				paramdelta[ptest]=paramdelta[ptest]*0.99
				# Dont let temperature go too low
				if paramdelta[ptest]<0.01:
					paramdelta[ptest]=0.01
		metropstep+=1
		elapsed_11 += time.time() - start_11
		print "Elapsed time: " + str(elapsed_11)
		perfintime.append(evalindold)
		savetxt(outpath+'perfintime.csv',perfintime,fmt='%2.5e',delimiter=',')
		#plotit(perfintime,max(metropstep,500),outpath)

		
		
###############################
###############################							
###############################
### ENDOF Optimization loop ###
###############################
###############################
###############################
												
												
			


#############################
###### SAVING OUTPUTS
#############################
if saveresults==1:
	print "Saving outputs"

	# Averaging the 5 biggest fires of each year to a single value
	afivemaxsizeg=nansum(afivemaxsizeg,axis=2)/5.
	afivemaxdurg=nansum(afivemaxdurg,axis=2)/5.
	savetxt(outpath+'afivemaxsize_'+str(res)+'.csv',afivemaxsizeg,fmt='%2.5e',delimiter=',')
	savetxt(outpath+'afivemaxdur_'+str(res)+'.csv',afivemaxdurg,fmt='%2.5e',delimiter=',')
	
	# Monthly
	savetxt(outpath+'mfires_'+str(res)+'.csv',mfiresg,fmt='%2.5e',delimiter=',')
	savetxt(outpath+'mbashrubs_'+str(res)+'.csv',mbashrubsg,fmt='%2.5e',delimiter=',')
	savetxt(outpath+'mbagrass_'+str(res)+'.csv',mbagrassg,fmt='%2.5e',delimiter=',')
	savetxt(outpath+'mbaforest_'+str(res)+'.csv',mbaforestg,fmt='%2.5e',delimiter=',')
	savetxt(outpath+'mba_'+str(res)+'.csv',mbag,fmt='%2.5e',delimiter=',')
	
	# Annual
	savetxt(outpath+'amaxsize_'+str(res)+'.csv',amaxsizeg,fmt='%2.5e',delimiter=',')
	savetxt(outpath+'amaxdur_'+str(res)+'.csv',amaxdurg,fmt='%2.5e',delimiter=',')
	savetxt(outpath+'aavdur_'+str(res)+'.csv',aavdurg,fmt='%2.5e',delimiter=',')
	savetxt(outpath+'atotbaforest_'+str(res)+'.csv',atotbaforestg,fmt='%2.5e',delimiter=',')
	savetxt(outpath+'atotbashrubs_'+str(res)+'.csv',atotbashrubsg,fmt='%2.5e',delimiter=',')
	savetxt(outpath+'atotbagrass_'+str(res)+'.csv',atotbagrassg,fmt='%2.5e',delimiter=',')
	savetxt(outpath+'atotba_'+str(res)+'.csv',atotbag,fmt='%2.5e',delimiter=',')
	
	savetxt(outpath+'hignout'+str(res)+'.csv',hignout,fmt='%2.5e',delimiter=',')
	savetxt(outpath+'gdpout'+str(res)+'.csv',gdpout,fmt='%2.5e',delimiter=',')
	savetxt(outpath+'suppout'+str(res)+'.csv',suppout,fmt='%2.5e',delimiter=',')	

else:
	print 'Not saving results, as requested'

#######################################
###### MAPPING FOR VISUAL INSPECTION
#######################################
mapavdur,latmap,lonmap=yanbuildmap(aavdur[2:,4],aavdur[2:,0:4],res)
mapmaxsize,latmap,lonmap=yanbuildmap(amaxsize[2:,4],amaxsize[2:,0:4],res)
mapmaxdur,latmap,lonmap=yanbuildmap(amaxdur[2:,4],amaxdur[2:,0:4],res)
mapforest,latmap,lonmap=yanbuildmap(atotbaforest[2:,4],atotbaforest[2:,0:4],res)
mapshrubs,latmap,lonmap=yanbuildmap(atotbashrubs[2:,4],atotbashrubs[2:,0:4],res)
mapgrass,latmap,lonmap=yanbuildmap(atotbagrass[2:,4],atotbagrass[2:,0:4],res)
maptotba,latmap,lonmap=yanbuildmap(divide(atotba[2:,4],gridsizesel),atotba[2:,0:4],res)

# Evaluation								
borders=yanborders('Regions')
latspan=[min(mba[2:,2])-res/2,max(mba[2:,2])+res/2]
lonspan=[min(mba[2:,3])-res/2,max(mba[2:,3])+res/2]
mapnames=['Observation','Modeled','Diffclass','IAV Corr','Diffseas', 'burnedarea',]
maps=[mapoptclass,mapmodclass,mapclassdiff,mapiav,mapseasdiff, mapforest]
setbad=[0,0,nan,nan,nan,nan]
vminval=[0,0,-len(classbound),-1,-6,0]
vmaxval=[len(classbound),len(classbound),len(classbound),1,6,40000]
barmax, barmin, cmapmirror = yancolim(-len(classbound),len(classbound),-len(classbound),len(classbound))
barmax, barmin, cmapnorm = yancolim(0,len(classbound),0,len(classbound))
colormaptouse=[cmapnorm,cmapnorm,cmapmirror,cmapmirror,cmapmirror,cmapnorm]
								
								
for fignum in range(len(mapnames)):
	fig=plt.figure(figsize=(18,6))
	a1 = plt.subplot2grid((1,20),(0,0),colspan=19)
	b1 = plt.subplot2grid((1,20),(0,19))
	cmap = matplotlib.cm.BrBG
	cmap.set_bad('grey',setbad[fignum])
	ax1 = a1.imshow(maps[fignum],vmin = vminval[fignum], vmax=vmaxval[fignum],extent=[lonspan[0],lonspan[1],latspan[0],latspan[1]],cmap=colormaptouse[fignum],interpolation='nearest',origin='upper', aspect='equal')
	ax2 = a1.plot(borders[0,:], borders[1,:],color='black',linestyle='-',linewidth=0.4)
	a1.axis([lonspan[0],lonspan[1],latspan[0],latspan[1]])
	a1.set_title(mapnames[fignum])
	a1.grid(True)
	colbar1=fig.colorbar(ax1,cax=b1,orientation='vertical')
	plt.savefig(outpath+mapnames[fignum]+'.png',format='png')
																	

# fig=plt.figure(figsize=(10,6))
# a1 = plt.subplot2grid((2,10),(0,0),colspan=9)
# b1 = plt.subplot2grid((2,10),(0,9))
# a2 = plt.subplot2grid((2,10),(1,0),colspan=9)
# b2 = plt.subplot2grid((2,10),(1,9))
# cmap = matplotlib.cm.BrBG
# cmap.set_bad('w',1.)
# ax1 = a1.imshow(mapavdur,interpolation='nearest',origin='upper', aspect='auto')
# a1.set_title('Average duration')
# #a.plot(axborders,ayborders,'k-')
# colbar1=fig.colorbar(ax1,cax=b1,orientation='vertical')
# 
# ax2 = a2.imshow(mapmaxsize,interpolation='nearest', aspect='auto',origin='upper')
# a2.set_title('Max size')
# #a.plot(axborders,ayborders,'k-')
# colbar2=fig.colorbar(ax2,cax=b2,orientation='vertical')
# 
# fig=plt.figure(figsize=(10,6))
# a1 = plt.subplot2grid((2,10),(0,0),colspan=9)
# b1 = plt.subplot2grid((2,10),(0,9))
# a2 = plt.subplot2grid((2,10),(1,0),colspan=9)
# b2 = plt.subplot2grid((2,10),(1,9))
# cmap = matplotlib.cm.BrBG
# cmap.set_bad('w',1.)
# ax1 = a1.imshow(mapmaxdur,interpolation='nearest',origin='upper', aspect='auto')
# a1.set_title('Max duration')
# #a.plot(axborders,ayborders,'k-')
# colbar1=fig.colorbar(ax1,cax=b1,orientation='vertical')
# 
# ax2 = a2.imshow(maptotba,interpolation='nearest', aspect='auto',origin='upper')
# a2.set_title('BA tot')
# #a.plot(axborders,ayborders,'k-')
# colbar2=fig.colorbar(ax2,cax=b2,orientation='vertical')
# 
# fig=plt.figure(figsize=(10,6))
# a1 = plt.subplot2grid((2,10),(0,0),colspan=9)
# b1 = plt.subplot2grid((2,10),(0,9))
# a2 = plt.subplot2grid((2,10),(1,0),colspan=9)
# b2 = plt.subplot2grid((2,10),(1,9))
# cmap = matplotlib.cm.BrBG
# cmap.set_bad('w',1.)
# ax1 = a1.imshow(mapforest,interpolation='nearest',origin='upper', aspect='auto')
# a1.set_title('BA forests')
# #a.plot(axborders,ayborders,'k-')
# colbar1=fig.colorbar(ax1,cax=b1,orientation='vertical')
# 
# ax2 = a2.imshow(mapshrubs,interpolation='nearest', aspect='auto',origin='upper')
# a2.set_title('BA shrubs')
# #a.plot(axborders,ayborders,'k-')
# colbar2=fig.colorbar(ax2,cax=b2,orientation='vertical')



plt.show()

print '11 12 wet head start spread stoch ba out fragm up out1 out2 out3 out4 perf gridells wind'
print ceil(array([elapsed_11,elapsed_12,elapsed_wet,elapsed_head,elapsed_start,elapsed_spread,elapsed_stoch,elapsed_ba,elapsed_out,elapsed_fragm,elapsed_up,elapsed_out1,elapsed_out2,elapsed_out3,elapsed_out4,elapsed_perf,elapsed_gridcells,elapsed_wind])/((yearE-yearB+1)*totcombs))
					
