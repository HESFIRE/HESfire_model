#!/usr/local/bin/Python
### Fire model

# Developed by Yannick Le Page et al. (REF) - Contact: Yannick.LePage@pnnl.gov
import os
import csv
from numpy import *
from copy import deepcopy
from scipy.stats import binom
import scipy.interpolate
import shutil
from scipy.stats import pearsonr
import random
from xlrd import *


### FUNCTION: Reading climate data
def readclim(cv,yearE,yearB,slicebeg,sliceend,currentslice,relpath,pathdaily,pathnightly,selgrids,projection):
	exec('alldaily=zeros(shape=(yearE-yearB+1,sliceend[currentslice]-slicebeg[currentslice]+2,370),dtype=float32)')
	exec('allnightly=zeros(shape=(yearE-yearB+1,sliceend[currentslice]-slicebeg[currentslice]+2,370),dtype=float32)')
	for y in range(yearE-yearB+1):
		### Reading annual input files
		print 'Extracting ' + cv + ' for year ' + str(int(yearB+y))
		exec('ydaily=genfromtxt(relpath+pathdaily' + cv + ' + str(int(yearB+y)) + "'"_"'" + str(res) + "'".csv"'", skip_header=0, missing_values=''-9999'', filling_values=0,delimiter="'","'")')
		exec('alldaily[int(y),2:,:]=ydaily[selgrids[slicebeg[currentslice]:sliceend[currentslice]]+2,:]')
		exec('ynightly=genfromtxt(relpath+pathnightly' + cv + ' + str(int(yearB+y)) + "'"_"'" + str(res) + "'".csv"'", skip_header=0, missing_values=''-9999'', filling_values=0,delimiter="'","'")')
		exec('allnightly[int(y),2:,:]=ynightly[selgrids[slicebeg[currentslice]:sliceend[currentslice]]+2,:]')
	return alldaily, allnightly	



### FUNCTION: climate projections file path
def projpath(projection,obspath,projfilename):
	if projection == 'no/':
		newpath = obspath # In case we use observations
	else: # in case we use climate projections
		rootpath=obspath.split('/')[1:len(obspath.split('/'))-3]
		newpath = ''
		for i in range(len(rootpath)):
			newpath += rootpath[i] + '/'
	newpath += projection + obspath.split('/')[-3] +'/'+ projfilename
	return obspath, newpath
	
### FUNCTION: Adapts output path in case of "full" experiment (all grid-cells) with parameters from a given optimization run
def isfullexp(expname):
	if expname[-5:-1]=='Full':
		relpath='../../../'
	else:
		relpath='./'
	outfolder=relpath+'Outputs/InputUncertainties/'
	return outfolder, relpath

### FUNCTION: Saves source code and parameter file in the experiment folder
def savesourcode(outpath,outfolder):
	try:
		os.mkdir(outfolder)
	except OSError:
		'do nothing'
	try:
		os.mkdir(outpath)
		print "created output folder"
		print 'Expname: ' + outfolder
	except OSError:
		# !!!
		print " WARNING: experiment folder already exists, overwriting (at end of fire cycle)"
		print 'Expname: ' + outfolder
	try:
		shutil.copy('./'+os.path.basename(__file__), outpath+'SourceCodeHESFIRE.py')
		shutil.copy('./'+'HESfire_params.xls', outpath+'SourceParameters.xls')
	except:
		print "WARNING: could not save source code file"
		print oupath
		print outfolder

### FUNCTION: Random parameter generator
# a: lowest value acceptable for the parameter
# b: highest value acceptable
# paramtype: exponent ('exp') or quantitative ('num')
def randparam(a,b,paramtype): 
	if paramtype == 'exp': # shape parameters, typically from 0.033 to 30, but we want same probability from 0.033-1 than 1-30  
		paramtest = random.randint(1,2)
		if paramtest == 1: # we go for a value between a and 1
			param = random.randint(a*100,100)/100.
		elif paramtest == 2: # we go for a value between 1 an b
			param=random.randint(1,b)/1.
	if paramtype == 'num': # typically 0.0001 to 0.1, but we want same probability to be 0.0001-0.001 than 0.01-0.1
		magnrange = b/a
		paramtest = random.randint(1,log10(magnrange))
		param = random.randint(1,10)/1.
		param = param / (pow(10,paramtest) * 1. / b)
	return param


##########################################################################################
##################################### 1. User inputs #####################################
##########################################################################################
parameterfile = 'HESfire_params.xls'
wb = open_workbook(filename=parameterfile)
sh = wb.sheet_by_index(0)
paramnames = [str(sh.col_values(0)[i+1]) for i in range(len(sh.col_values(0))-1)]
paramvalues = sh.col_values(1)[1:]
paramtypes = [str(sh.col_values(2)[i+1]) for i in range(len(sh.col_values(2))-1)]

for i in range(len(paramnames)):
	if paramtypes[i]=='range':
		exec(paramnames[i] +'=str(paramvalues[' + str(i) + '])')
		exec(paramnames[i] +'= '+paramvalues[i])
	else:
		exec(paramnames[i] +'=' + paramtypes[i] + '(paramvalues[' + str(i) + '])')


###############################
### 1.a. Filenames, outputs ###
###############################
pathdailyrh, pathrhproj = projpath(projection,'./Data/RH/'+climdata+'/interpdailyrh_','hurs_')
pathnightlyrh, pathrhproj = projpath(projection,'./Data/RH/'+climdata+'/interpnightlyrh_', 'hurs_')
pathdailytemp, pathtempproj = projpath(projection,'./Data/Temp/'+climdata+'/interpdailytemp_','tas_')
pathnightlytemp, pathtempproj = projpath(projection, './Data/Temp/'+climdata+'/interpnightlytemp_','tas_')
pathdailymoist, pathmoistproj = projpath(projection,'./Data/SoilMoisture/'+climdata+'/interpdailymoist_','mrsos_')
pathnightlymoist, pathmoistproj = projpath(projection, './Data/SoilMoisture/'+climdata+'/interpnightlymoist_','mrsos_')
pathdailywind, pathwindproj = projpath(projection, './Data/Wind/'+climdata+'/interpdailywind_','sfcWind_')
pathnightlywind, pathwindproj = projpath(projection,'./Data/Wind/'+climdata+'/interpnightlywind_','sfcWind_')
pathdailylight, pathlightdailyproj = projpath(projection,'./Data/Convprecip/'+climdata+'/interpdailylight_','interpdailylight_')
pathnightlylight, pathlightnightlyproj = projpath(projection,'./Data/Convprecip/'+climdata+'/interpnightlylight_','interpnightlylight_')
pathprecip = './Data/Precip/GPCP/mprecip_'
bouh, pathprecipproj = projpath(projection,'./Data/Precip/GPCP/mprecip_1','pr_')

############################
### 1.b. Constant and temporal extent ###
############################
kmpersecperdayperms=3600*24/1000
numyears=yearE-yearB+1

#########################
### 1.c. Optimization ###
#########################
params=['dryexp','gdpexp','supexp','gdpsupexp','fragexp','rhexp','moistexp','tempexp','luign','luignexp','natign']
paramstype=['exp','exp','exp','exp','exp','exp','exp','exp','num','exp','num']
paramsnumber=['um','dois','tres','quatro','cinco','seis','sete','oito','nove','dez','onze']
paramub=[30,30,30,30,30,30,30,30,0.1,30,0.5] # Upper bound
paramlb=[0.033,0.033,0.033,0.033,0.033,0.033,0.033,0.033,0.000001,0.033,0.00001] # Lower bound
paramdelta=[1,1,1,1,1,1,1,1,1,1,1] # maximum step parameter
testbool=[1,1,1,0,1,1,1,1,1,1,1] # 1 to optimize, 0 otherwise
acceptparam = [0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001]# Threshold of minimum metric improvement for acceptance. 


#######################
### 1.d. Evaluation ###
#######################
# Exponent for RMSE computation
powereval=2
# Observation data
GFEDdata='GFED_BA_natural'



##########################################################################################
#################### 2. Run setup (reads data, select space/time, etc) ###################
##########################################################################################

### Checking Whether is is a global application of a previously optimized parameterization
outfolder, relpath = isfullexp(expname)
outpath=outfolder+expname

### Output folder creation and saving Source code
savesourcode(outpath,outfolder)
print 'RUN: '+ outpath

		

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
	metropolis = 'no'
	paramfin = genfromtxt('./Evaluation.csv', skip_header=0, missing_values='-9999', filling_values=0,delimiter=',')
	evalcol = paramfin[:,0]
	evalcol[evalcol==0]=40
	bestind=argmin(evalcol)
	bestparams=paramfin[bestind,:]
	for param in range(len(params)):
		exec(params[param] + '= bestparams[-1*len(params)+param]')

#######################
### 2.b. Evaluation ###
#######################
print '2.b. Evaluation'

# Initializing evaluation analysis
totcombs=1
paramatrix=zeros(shape=(10000,len(params)+6))
evalmatrix=zeros(shape=(10000,4))*nan
globindold=40 # initialize model error for first loop
evalindold=40

	
#########################
### 2.c. Reading data ###
#########################
print '2.c. Reading data'

### Land cover/land use
print 'Land cover/use'
forest=genfromtxt(relpath+'./Data/'+landusedata+'forests_'+str(res)+'.csv', skip_header=0, missing_values='-9999', filling_values=0,delimiter=',')
shrubs=genfromtxt(relpath+'./Data/'+landusedata+'shrubs_'+str(res)+'.csv', skip_header=0, missing_values='-9999', filling_values=0,delimiter=',')
grass=genfromtxt(relpath+'./Data/'+landusedata+'herbaceous_'+str(res)+'.csv', skip_header=0, missing_values='-9999', filling_values=0,delimiter=',')
landuse=genfromtxt(relpath+'./Data/'+landusedata+'landuse_'+str(res)+'.csv', skip_header=0, missing_values='-9999', filling_values=0,delimiter=',')
nonland=genfromtxt(relpath+'./Data/'+landusedata+'deserts_'+str(res)+'.csv', skip_header=0, missing_values='-9999', filling_values=0,delimiter=',')
water=genfromtxt(relpath+'./Data/'+landusedata+'water_'+str(res)+'.csv', skip_header=0, missing_values='-9999', filling_values=0,delimiter=',')
# NOTE: we do not model fires in grid-cells > 0.5 water, considering the fragmentation index is not appropriate given most 
# of these grid-cells have continuous water (lakes, oceans)

### Spatial selection
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
		else:
			indmask.append(indorig[0][0])
	indmask=array(indmask)		
	numgrid=len(indmask)
	selgrids=indmask
else:
	selgrids=arange(0,numgrid,1)

print 'Total number of gridcells: ' + str(numgrid)
# Saving subset data inputs if required to make later runs more efficient
if saveinputs==1:
	selgrids2=concatenate(([0,1],selgrids+2),axis=0)
	savetxt(outpath+'forests_1.csv',forest[selgrids2,:],delimiter=',')
	savetxt(outpath+'shrubs_1.csv',shrubs[selgrids2,:],delimiter=',')
	savetxt(outpath+'herbaceous_1.csv',grass[selgrids2,:],delimiter=',')
	savetxt(outpath+'landuse_1.csv',landuse[selgrids2,:],delimiter=',')
	savetxt(outpath+'deserts_1.csv',nonland[selgrids2,:],delimiter=',')
	savetxt(outpath+'water_1.csv',water[selgrids2,:],delimiter=',')

# Computing grid-cell sizes
gridsize=cos(radians(forest[selgrids+2,2]))*(111.320*111.320)*res*res
gridsizesel=cos(radians(forest[selgrids+2,2]))*(111.320*111.320)*res*res	

### GDP
print 'GDP'
if gdpfactor[0]>=0 or gdpsupfactor[0]>=0:
	gdp=genfromtxt(relpath+'./Data/GDP/'+GDPdata+str(res)+'.csv', skip_header=0, missing_values='-9999', filling_values=0,delimiter=',')
	if saveinputs==1:
		savetxt(outpath+'GDPCIA'+str(int_(res))+'.csv',gdp[selgrids2,:],delimiter=',')
	
### GFED
print 'GFED'
gfedorig=genfromtxt(relpath+'./Data/Obs/GFED/'+GFEDdata+str(res)+'.csv', skip_header=0, missing_values='-9999', filling_values=0,delimiter=',')
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
if saveinputs==1:
	savetxt(outpath+'GFED_BurnedFraction'+str(int_(res))+'.csv',gfedba[selgrids2,:],delimiter=',')

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
del(gfedba)


### Climate
print 'Climate'

### Precip in average mm/day for each month, as a fuel proxy
precip=genfromtxt(relpath+pathprecip+str(res)+'.csv', skip_header=0, missing_values='-9999', filling_values=0,delimiter=',')
if saveinputs==1:
	savetxt(outpath+'mprecip_1.csv',precip[selgrids2,:],delimiter=',')
if projection != 'no/':
	totprecipproj=loadtxt(relpath+pathprecipproj + str(int(yearproj)) + '_' + str(res) + '.csv', delimiter=',')

### Climate
print 'Reading bi-daily climate data, can take a while'
slicebeg = zeros(climatesplits)
sliceend = zeros(climatesplits)
cursor=0
for i in range(climatesplits):
	slicebeg[i]=cursor
	sliceend[i]=cursor+ceil(numgrid/float(climatesplits))
	cursor+=ceil(numgrid/float(climatesplits))
sliceend[-1]=numgrid
currentslice=0
for cv in ['rh','temp','moist','wind','light']:
	if eval('use' + cv + '==1'):  
		exec('alldaily' + cv + '=zeros(shape=(yearE-yearB+1,sliceend[0]-slicebeg[0]+2,370),dtype=float32)')
		exec('allnightly' + cv + '=zeros(shape=(yearE-yearB+1,sliceend[0]-slicebeg[0]+2,370),dtype=float32)')
		if (projection != 'no/') and (cv != 'light'):
			exec('tot' + cv + 'proj=loadtxt(relpath+path' + cv + 'proj + str(int(yearproj)) + "'"_"'" + str(res) + "'".csv"'" , delimiter="'","'")')
			exec('all' + cv + 'proj=zeros(shape=(sliceend[0]-slicebeg[0]+2,12+4),dtype=float32)')
			exec('all' + cv + 'proj=tot' + cv + 'proj[selgrids[slicebeg[0]:sliceend[0]]+2,:]')		
	for y in range(yearE-yearB+1):
		### Reading annual input files
		print 'Extracting ' + cv + ' for year ' + str(int(yearB+y))
		exec('ydaily' + cv + '=genfromtxt(relpath+pathdaily' + cv + '+ str(int(yearB+y)) + "'"_"'" + str(res) + "'".csv"'", skip_header=0, missing_values=''-9999'', filling_values=0,delimiter="'","'")')
		if saveinputs==1:
			exec('savetxt(outpath+pathdaily'+cv+'.split("'"/"'")[-1] + str(int(yearB+y)) + "'"_"'" + str(res) + "'".csv"'",ydaily'+cv+'[selgrids2,:],delimiter="'","'")')
		exec('alldaily' + cv + '[int(y),2:,:]=ydaily' + cv + '[selgrids[slicebeg[0]:sliceend[0]]+2,:]')
		exec('del(ydaily' + cv + ')')
		exec('ynightly' + cv + '=genfromtxt(relpath+pathnightly' + cv + '+ str(int(yearB+y)) + "'"_"'" + str(res) + "'".csv"'", skip_header=0, missing_values=''-9999'', filling_values=0,delimiter="'","'")')
		if saveinputs==1:
			exec('savetxt(outpath+pathnightly'+cv+'.split("'"/"'")[-1] + str(int(yearB+y)) + "'"_"'" + str(res) + "'".csv"'",ynightly'+cv+'[selgrids2,:],delimiter="'","'")')
		exec('allnightly' + cv + '[int(y),2:,:]=ynightly' + cv + '[selgrids[slicebeg[0]:sliceend[0]]+2,:]')
		exec('del(ynightly' + cv + ')')



#######################################
### 2.e. Model data Initialization  ###
#######################################
print '2.e. Model data Initialization'

burneddailyforestorig=zeros(shape=(numgrid,mbamemory*30)) # 240=8 months
burneddailyshrubsorig=zeros(shape=(numgrid,mbamemory*30)) # 240=8 months
burneddailygrassorig=zeros(shape=(numgrid,mbamemory*30)) # 240=8 months

###  Extracting yearB-1 data from GFED to initialize the burned area memory before fire can return 
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
coslat=cos(radians(optba[2:,2]))
optfrac=multiply(optba[2:,4],coslat*111.320*111.320*res*res)
optclass=zeros(shape=(shape(optfrac)))
for b in arange(1,len(classbound),1):
	optclass=where((optfrac>classbound[b-1]*0.01) & (optfrac<=classbound[b]*0.01),b-1+(optfrac-classbound[b-1]*0.01)/(classbound[b]*0.01-classbound[b-1]*0.01),optclass)
optclass=where((optfrac>classbound[b]*0.01),b+1,optclass)


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
if projection != 'no/':
	seasontrackerorig[:,0:precipdelay] += totprecipproj[selgrids+2,-precipdelay:]
	seasontrackerorig[:,precipdelay:] += totprecipproj[selgrids+2,4:]

### Now selecting precip from yearB to yearE so that we dont have to call the function where() later
preciporig=precip[selgrids+2,indprecip[0][-1]+1:].copy()

### Optimization performance through iterations
perfintime=[]

### Optimization
metropend=0
metropstep=0


######################
### 2.f. Utilities ###
######################
print '2.f. Utilities'

### Time utilities
monthnoleapcal=array([ 31,  61,  92, 122, 153, 183, 214, 245, 275, 306, 336, 366])
monthleapcal=array([ 31,  62,  93, 123, 154, 184, 215, 246, 276, 307, 337, 367])
numyears=yearE-yearB+1

### Lookup tables
# Exponent lookup table (see details in Data/Lookup folder)
lookup=genfromtxt(relpath+'./Data/Lookup/lookuptablebig.csv', skip_header=0, missing_values='-9999', filling_values=0,delimiter=',')
#savetxt(outpath+'explookuptable.csv',lookup,delimiter=',')

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
	###################################
	### Model output Initialization ###
	###################################
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

	#######################
	### Parameter trial ###
	#######################
	for param in range(len(params)):
		exec('paramatrix[combind,6+param]='+params[param])
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
			paramatrix[combind,6+ptest]=pnew
		else:
			pnew='pnew'
			pold='pold'
			ptest='ptest'
		if GDPmerge == 1:
			gdpsupexp = gdpexp
	
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
		i=selgrids[il]+2
		if il%100==0:
			print 'Grid-cell ' + str(il)+'/'+str(len(selgrids))
		inonland=nonland[i,4]	

		# In case we split the climate data for memory management, we check whether we're at the point we should read new ones.
		ilclim = il-slicebeg[currentslice]
		if il == sliceend[currentslice]:	
			currentslice+=1
			ilclim = il-slicebeg[currentslice]
			for cv in ['rh','temp','moist','wind','light']:
				if eval('use' + cv + '==1'): 
					exec('del(alldaily' + cv +')')
					exec('del(allnightly' + cv +')')
					exec('del(daily' + cv +')')
					exec('del(nightly' + cv +')')
					exec('alldaily' + cv +',allnightly' + cv +' = readclim(cv,yearE,yearB,slicebeg,sliceend,currentslice,relpath,pathdaily' + cv +',pathnightly' + cv +',selgrids,projection)')
 					if (projection != 'no/') and (cv != 'light'):
 						exec('tot' + cv + 'proj=loadtxt(relpath+path' + cv + 'proj + str(int(yearproj)) + "'"_"'" + str(res) + "'".csv"'",  delimiter="'","'")')
 						exec('all' + cv + 'proj=zeros(shape=(sliceend[currentslice]-slicebeg[currentslice]+2,12+4),dtype=float32)')
 						exec('all' + cv + 'proj=tot' + cv +'proj[sliceend[currentslice:slicebeg[currentslice]+2,:]')
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
						
		### Cases of grid-cells we dont want to model (eg water, covered in ice, no wild land)
		if water[i,4]>0.5 or isnan(water[i,4]) or inonland>0.9 or (forest[i,4]+shrubs[i,4]+grass[i,4])<=0.:
			"Do nothing"
		else: 
			##############################################
			### Case of grid-cells we do want to model ###
			##############################################
			
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
			if projection != 'no/':
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
					# !!! replace with integral ? (not necessarily faster computationally, cause requires calling power function).
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

	
			### Climate projection
			# each grid cell has monthly climate anomalies (absolute value), for one average year.
			# We retrieve the 12 monthly values for the grid cell considered
			if projection != 'no/':
				for cv in ['rh','temp','moist','wind']:
					if eval('use' + cv + '==1'):
						exec('i' + cv + 'proj=all' + cv + 'proj[ilclim,4:]')

			##########################
			##########################							
			##########################
			##### Loop on years ######
			##########################
			##########################
			##########################
			for y in arange(yearB,yearE+0.1,1):
				# tracking program performances
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
						if projection != 'no/':
							idayrh += irhproj[month]
							inightrh += irhproj[month]
							idayrh = nanmax([idayrh,0])
							inightrh = nanmax([inightrh,0])					
					if usemoist==1:
						idaymoist=nanmax([dailymoist[y-yearB,day+4],0])
						inightmoist=nanmax([nightlymoist[y-yearB,day+4],0])
						if projection != 'no/':
							idaymoist += imoistproj[month]
							inightmoist += imoistproj[month]
							idaymoist = nanmax([idaymoist,0])
							inightmoist = nanmax([inightmoist,0])														
					if usetemp==1:						
						idaytemp=dailytemp[y-yearB,day+4]
						inighttemp=nightlytemp[y-yearB,day+4]
						if projection != 'no/':
							idaytemp += itempproj[month]
							inighttemp += itempproj[month]
							#idaytemp = nanmax([idaytemp,0])
							#inighttemp = nanmax([inighttemp,0])							
					if usewind==1: # Case wind is accounted for
						idaywind=dailywind[y-yearB,day+4]
						inightwind=nightlywind[y-yearB,day+4]
						if projection != 'no/':
							idaywind += iwindproj[month]
							inightwind += iwindproj[month]							
							idaywind = nanmax([idaywind,0])
							inightwind = nanmax([inightwind,0])													
	
					### Checking whether day is potentially conducive to fires
					### Wet, no fuel period or too cold -> no fires, we just update the day counting and put back land burned more than bamemory ago
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
						################################
						################################					
						################################
						###### non fire-prone day ######
						################################
						################################
						################################
						# Case of a day not prone to fires: re-sets fire counters to zero
						consday[il]=0
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
							if projection != 'no/':
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
	
							
					################################
					################################					
					################################
					######   Fire-prone day   ######
					################################
					################################
					################################
					else:
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
						
						##################################
						############ SPREAD RATE #########
						##################################
						### FINAL IMPACT OF HUMIDITY AND TEMP ON SPEED (WEIGHTED BY LANDCOVER FRACTION)
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
						
						### Wind: influence on spread rate and L/B ratio
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
						
						####################################
						############ BURNED AREA ###########
						####################################	
						### Ellipsoid function, partitioned into the land cover types
						# !!! reverse to bi-daily ?
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

						#################################################
						############ MONTHLY & ANNUAL OUTPUTS ###########
						#################################################
						
						### Burned area for that day (number of active fires in each track and their additional burned area)
						totfires=sum(activefires[:,d])
						for lcname in ['forest','shrubs','grass']:
							exec('totba' + lcname +'=sum(multiply(ba' + lcname + '[:,d],activefires[:,d]))')
							if eval('totba' + lcname +'>i' + lcname +'*igridsize'):
								exec('totba' + lcname +'=i' + lcname +'*igridsize')
						totba=totbaforest+totbashrubs+totbagrass
						
						### Monthly outputs
						mfires[il+2,monthind]+=actualign[0,day]#sum(activefires[:,d-1]-activefires[:,d])
						mbashrubs[il+2,monthind]+=totbashrubs
						mbaforest[il+2,monthind]+=totbaforest
						mbagrass[il+2,monthind]+=totbagrass
						mba[il+2,monthind]+=totba

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
						
						############################################################
						############ UPDATING FRAGMENTATION / LAND COVER ###########
						############################################################
						# To account for new burned areas having no fuel and old burned areas coming back with fuel
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
						
						### Updating fuel accumulation tracker (precip)
						# Not considering zeros
						### Case based on precip
						if precipfactor[0]>0:
							### If we change month, updating the precip data in seasontracker, and computing new idryseasoneffect
							if monthold != month:
								seasontracker[il,0:-1]=seasontracker[il,1:]
								seasontracker[il,-1]=iprecipgrid[month]
								if projection != 'no/':
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

	### Av. Burned area (obs: opt---)
	modba=mba[:,0:5].copy()
	modba[2:,4]=nansum(mba[2:,4:],axis=1)/len(mba[0,4:])*12
	coslat=cos(radians(modba[2:,2]))
	modfrac=multiply(modba[2:,4],coslat*111.320*111.320*res*res)
	modclass=zeros(shape=(shape(modfrac)))
	for b in arange(1,len(classbound),1):
		modclass=where((modfrac>classbound[b-1]*0.01) & (modfrac<=classbound[b]*0.01),b-1+(modfrac-classbound[b-1]*0.01)/(classbound[b]*0.01-classbound[b-1]*0.01),modclass)
	modclass=where((modfrac>classbound[b]*0.01),b+1,modclass)	
	
	
	diffclass=modclass-optclass
	#diffclass=where((modclass==0) | (optclass==0),0,diffclass)
	globind=nansum(power(abs(diffclass),powereval))/sum(isnan(diffclass)==False)
	globbias=nansum(diffclass)/sum(isnan(diffclass)==False)
	
	evalmatrix[combind,0]=globind
	paramatrix[combind,0]=globind
	evalmatrix[combind,1]=globbias
	paramatrix[combind,1]=globbias
	

	
	# Total burned area
	paramatrix[combind,4]=nansum(mba[2:,4:])
	observationtotba=nansum(optba[2:,4:])
	ratioba=paramatrix[combind,4]/observationtotba
	print 'ratioba'
	print ratioba
								
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
	

	# Taking out regions with -- landuse fires (<0.001), because there interrannual variability is mostly driven by lightning.
	#landusefrac=landuse[list(selgrids+2),4]
	#badind=where(landusefrac<0.001)
	#iavcorrmatrix[badind,0]=nan
	
	
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
	evalind=globind+(1-globiavind)
	paramatrix[combind,5]=evalind
	if globind<globindold:
		globindold=globind
	print 'BEST FIT EVER: ' +str(globindold)
	print 'step: ' + str(metropstep)
	
	if metropstep>0:
		acceptthresh = acceptparam[ptest]
	else:
		acceptthresh = 0.001
	print 'evalindold - evalind: ' + str(evalindold - evalind)
	print 'acceptthresh: ' + str(acceptthresh)
	
	if evalindold - evalind > acceptthresh:
		if GDPmerge == 1:
			gdpsupexp = gdpexp
		print "----------------------------"
		print "--- NEW OPTIMUM SOLUTION ---"
		print "----------------------------"
		print 'index = ' + str(evalind)
		print paramstring
		print ['%.3g' % val for val in paramatrix[combind,0:6]]
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
	
	if metropolis=='yes':
		if (evalindold - evalind > acceptthresh):
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

			
				# Increase temperature
				paramdelta[ptest]=paramdelta[ptest]*1.01
				# Dont let temperature go too big
				if paramdelta[ptest]>10:
					paramdelta[ptest]=10
			evalindold=evalind
					
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
		print "Elapsed time: " + str(elapsed_11)
		perfintime.append(evalindold)
		savetxt(outpath+'perfintime.csv',perfintime,fmt='%2.5e',delimiter=',')
		
		
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