library(TSA)
library(fUnitRoots)
library(forecast)
l=0
pref='continuous_hist_data_'
suf='.csv'
filename=paste(pref, 1, suf, sep='')
p=read.csv(filename,header=F)
p=ts(p[,2], frequency=5760)
p=p[1:10000]
f=auto.arima(p)
f

