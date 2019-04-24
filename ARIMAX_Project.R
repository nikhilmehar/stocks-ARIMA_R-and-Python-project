library(tseries)
library(forecast)
library(lmtest)
library(plyr)
library(tree)
library(caret)
library(ggplot2)
library(pls)
library(psych)

#library(foreign)
library(psych)
library(ggplot2)
library(Hmisc)	#Only load this after using psych; it overrides psych
library(car)	#Used for Durbin-Watson test and VIF scores
#library(fmsb)	#Alternative to calculate the VIF scores

layout(1)
options(digits = 4)

workingdirectory = "D:\\Bijoy\\Padhai\\Courses\\Obj Oriented Programming\\Project\\Data"
setwd(workingdirectory)

##Read data
bio = read.csv("NYSE Arca Biotechnology Index.csv")
pharma = read.csv("NYSE Arca Pharmaceutical Index.csv")
tech = read.csv("NYSE Arca Technology 100 Index.csv")
energy = read.csv("NYSE Energy Index.csv")
finance = read.csv("NYSE Financial Index.csv")
health = read.csv("NYSE Healthcare Index.csv")
airline = read.csv("NYSE_ARCA_AIRLINE_INDEX.csv")
snp500 = read.csv("S_P 500.csv")
gold = read.csv("WGC-GOLD_DAILY_USD.csv")
oil = read.csv("OPEC-ORB.csv")

##Select only relevant data
bio = bio[,c("Date","Close")]
pharma = pharma[,c("Date","Close")]
tech = tech[,c("Date","Close")]
energy = energy[,c("Date","Close")]
finance = finance[,c("Date","Close")]
health = health[,c("Date","Close")]
airline = airline[,c("Date","Adj.Close")]
snp500 = snp500[,c("Date","Adj.Close")]
gold = gold[,c("Date","Value")]
oil = oil[,c("Date","Values")]

##Change column names
names(bio)[2]="bio_close"
names(pharma)[2]="pharma_close"
names(tech)[2]="tech_close"
names(energy)[2]="energy_close"
names(finance)[2]="finance_close"
names(health)[2]="health_close"
names(airline)[2]="airline_close"
names(snp500)[2]="snp500_close"
names(gold)[2]="gold_close"
names(oil)[2]="oil_close"

##Change date column to date format
bio$Date = as.Date(bio$Date, "%m/%d/%y")
pharma$Date = as.Date(pharma$Date, "%m/%d/%y")
tech$Date = as.Date(tech$Date, "%m/%d/%y")
energy$Date = as.Date(energy$Date, "%m/%d/%y")
finance$Date = as.Date(finance$Date, "%m/%d/%y")
health$Date = as.Date(health$Date, "%m/%d/%y")
airline$Date = as.Date(airline$Date)
snp500$Date = as.Date(snp500$Date)
gold$Date = as.Date(gold$Date)
oil$Date = as.Date(oil$Date)

##Merge into single dataset
data = merge(bio, pharma, by = c("Date"), sort = TRUE)
data = merge(data, tech, by = c("Date"), sort = TRUE)
data = merge(data, energy, by = c("Date"), sort = TRUE)
data = merge(data, finance, by = c("Date"), sort = TRUE)
data = merge(data, health, by = c("Date"), sort = TRUE)
data = merge(data, airline, by = c("Date"), sort = TRUE)
data = merge(data, snp500, by = c("Date"), sort = TRUE)
data = merge(gold, data, by = c("Date"), sort = TRUE)
data = merge(oil, data, by = c("Date"), sort = TRUE)

summary(data_scale)

##Choose only 10 years worth data for analysis
##between 1-Jan-2007 and 31-Dec-2016
data = data[data$Date >= "2007-01-01" & data$Date <= "2016-12-31",]
nrow(data) #2511

##Remove any rows with missing values
data = data[complete.cases(data),]
nrow(data) 
#2511

data1 = data[,2:11]
names(data1)
##Export dataframe to use in Python
write.table(data, file="USStocks_v3.csv", sep=",", col.names=TRUE, quote=FALSE, row.names=FALSE)

layout(1)
##Decision trees
bio_tree = tree(data[,c("bio_close","oil_close","gold_close")])
plot(bio_tree, main="Decision Tree for Bio Close")
text(bio_tree)
?plot
pharma_tree = tree(data[,c("pharma_close","oil_close","gold_close")])
plot(pharma_tree)
text(pharma_tree)

tech_tree = tree(data[,c("tech_close","oil_close","gold_close")])
plot(tech_tree)
text(tech_tree)

energy_tree = tree(data[,c("energy_close","oil_close","gold_close")])
plot(energy_tree)
text(energy_tree)

finance_tree = tree(data[,c("finance_close","oil_close","gold_close")])
plot(finance_tree)
text(finance_tree)

health_tree = tree(data[,c("health_close","oil_close","gold_close")])
plot(health_tree)
text(health_tree)

airline_tree = tree(data[,c("airline_close","oil_close","gold_close")])
plot(airline_tree)
text(airline_tree)

snp500_tree = tree(data[,c("snp500_close","oil_close","gold_close")])
plot(snp500_tree)
text(snp500_tree)

data_scale = scale(data[,2:11], center = TRUE, scale = TRUE)
data_scale
summary(data_scale)

##Creating time series data
bio_ts=ts(data_scale[,c("bio_close")],frequency=252,start=c(2007,1))
pharma_ts=ts(data_scale[,c("pharma_close")],frequency=252,start=c(2007,1))
tech_ts=ts(data_scale[,c("tech_close")],frequency=252,start=c(2007,1))
energy_ts=ts(data_scale[,c("energy_close")],frequency=252,start=c(2007,1))
finance_ts=ts(data_scale[,c("finance_close")],frequency=252,start=c(2007,1))
health_ts=ts(data_scale[,c("health_close")],frequency=252,start=c(2007,1))
airline_ts=ts(data_scale[,c("airline_close")],frequency=252,start=c(2007,1))
snp500_ts=ts(data_scale[,c("snp500_close")],frequency=252,start=c(2007,1))
oil_ts=ts(data_scale[,c("oil_close")],frequency=252,start=c(2007,1))
gold_ts=ts(data_scale[,c("gold_close")],frequency=252,start=c(2007,1))


bio_ts_dc = decompose(bio_ts)
pharma_ts_dc = decompose(pharma_ts)
tech_ts_dc = decompose(tech_ts)
energy_ts_dc = decompose(energy_ts)
finance_ts_dc = decompose(finance_ts)
health_ts_dc = decompose(health_ts)
airline_ts_dc = decompose(airline_ts)
snp500_ts_dc = decompose(snp500_ts)
oil_ts_dc = decompose(oil_ts)
gold_ts_dc = decompose(gold_ts)


##Extract trend data
bio_ts_trend = bio_ts - bio_ts_dc$seasonal
pharma_ts_trend = pharma_ts - pharma_ts_dc$seasonal
tech_ts_trend = tech_ts - tech_ts_dc$seasonal
energy_ts_trend = energy_ts - energy_ts_dc$seasonal
finance_ts_trend = finance_ts - finance_ts_dc$seasonal
health_ts_trend = health_ts - health_ts_dc$seasonal
airline_ts_trend = airline_ts - airline_ts_dc$seasonal
snp500_ts_trend = snp500_ts - snp500_ts_dc$seasonal
oil_ts_trend = oil_ts - oil_ts_dc$seasonal
gold_ts_trend = gold_ts - gold_ts_dc$seasonal

#=======================================
# Augmented Dickey–Fuller (ADF) t-test
#=======================================
adf.test(bio_ts_trend, alternative = "stationary")
kpss.test(bio_ts_trend)
bio_ts_diff1 = diff(bio_ts_trend, differences = 1)
adf.test(bio_ts_diff1, alternative = "stationary")
kpss.test(bio_ts_diff1)

adf.test(pharma_ts_trend, alternative = "stationary")
kpss.test(bio_ts_trend)
pharma_ts_diff1 = diff(pharma_ts_trend, differences = 1)
adf.test(pharma_ts_diff1, alternative = "stationary")
kpss.test(bio_ts_diff1)

adf.test(tech_ts_trend, alternative = "stationary")
kpss.test(tech_ts_trend)
tech_ts_diff1 = diff(tech_ts_trend, differences = 1)
adf.test(tech_ts_diff1, alternative = "stationary")
kpss.test(tech_ts_diff1)

adf.test(energy_ts_trend, alternative = "stationary")
kpss.test(energy_ts_trend)
energy_ts_diff1 = diff(energy_ts_trend, differences = 1)
adf.test(energy_ts_diff1, alternative = "stationary")
kpss.test(energy_ts_diff1)

adf.test(finance_ts_trend, alternative = "stationary")
kpss.test(finance_ts_trend)
finance_ts_diff1 = diff(finance_ts_trend, differences = 1)
adf.test(finance_ts_diff1, alternative = "stationary")
kpss.test(finance_ts_diff1)
finance_ts_diff2 = diff(finance_ts_trend, differences = 2)
adf.test(finance_ts_diff2, alternative = "stationary")
kpss.test(finance_ts_diff2)

adf.test(health_ts_trend, alternative = "stationary")
kpss.test(health_ts_trend)
health_ts_diff1 = diff(health_ts_trend, differences = 1)
adf.test(health_ts_diff1, alternative = "stationary")
kpss.test(health_ts_diff1)

adf.test(airline_ts_trend, alternative = "stationary")
kpss.test(airline_ts_trend)
airline_ts_diff1 = diff(airline_ts_trend, differences = 1)
adf.test(airline_ts_diff1, alternative = "stationary")
kpss.test(airline_ts_diff1)
airline_ts_diff2 = diff(airline_ts_trend, differences = 2)
adf.test(airline_ts_diff2, alternative = "stationary")
kpss.test(airline_ts_diff2)

adf.test(snp500_ts_trend, alternative = "stationary")
kpss.test(snp500_ts_trend)
snp500_ts_diff1 = diff(snp500_ts_trend, differences = 1)
adf.test(snp500_ts_diff1, alternative = "stationary")
kpss.test(snp500_ts_diff1)

adf.test(oil_ts_trend, alternative = "stationary")
kpss.test(oil_ts_trend)
oil_ts_diff1 = diff(oil_ts_trend, differences = 1)
adf.test(oil_ts_diff1, alternative = "stationary")
kpss.test(oil_ts_diff1)

adf.test(gold_ts_trend, alternative = "stationary")
kpss.test(gold_ts_trend)
gold_ts_diff1 = diff(gold_ts_trend, differences = 1)
adf.test(gold_ts_diff1, alternative = "stationary")
kpss.test(gold_ts_diff1)

##Finding lag components
layout(1)

##Data split indexes
train = round(nrow(data_scale)*0.5)
test = round(nrow(data_scale)*0.3)
validation = round(nrow(data_scale)*0.2)

#energy_train = energy_ts_trend[1:train]
#energy_test = energy_ts_trend[1:(train+test)]

oil_gold_ts = ts.union(oil_ts_trend,gold_ts_trend)
str(oil_gold_ts)
oil_gold_matrix = matrix(oil_gold_ts, ncol = 2)
str(oil_gold_matrix)
layout(1:2)
acf(energy_ts_diff1, lag.max = 50)
pacf(energy_ts_diff1, lag.max = 50)

energy_arima1 = arima(energy_ts_trend[1:train], order = c(0, 1, 1), xreg=oil_gold_matrix[c(1:train),])
energy_arima2 = arima(energy_ts_trend[1:train], order = c(1, 1, 1), xreg=oil_gold_matrix[c(1:train),])
energy_arima3 = arima(energy_ts_trend[1:train], order = c(1, 1, 2), xreg=oil_gold_matrix[c(1:train),])
energy_arima4 = arima(energy_ts_trend[1:train], order = c(1, 1, 15), xreg=oil_gold_matrix[c(1:train),])
energy_arima5 = auto.arima(energy_ts_trend[1:train], xreg=oil_gold_matrix[c(1:train),])


energy_arima1 = arima(energy_ts_trend[1:train], order = c(0, 1, 1), xreg=oil_gold_matrix[c(1:train),])
energy_arima1_fore = forecast(energy_arima1, h = train, xreg=oil_gold_matrix[c((train+1):(train+test)),])
layout(1:2)
plot(energy_arima1_fore)
plot(energy_ts_trend[1:(train+test)])
energy_arima1
coeftest(energy_arima1)
accuracy(energy_arima1_fore,energy_ts_trend[train+1:train+test])

energy_arima2 = arima(energy_ts_trend[1:train], order = c(1, 1, 1), xreg=oil_gold_matrix[c(1:train),])
energy_arima2_fore = forecast(energy_arima2, h = train, xreg=oil_gold_matrix[c((train+1):(train+test)),])
layout(1:2)
plot(energy_arima2_fore)
plot(energy_ts_trend[1:(train+test)])
plot(energy_arima2_fore)
energy_arima2
coeftest(energy_arima2)
accuracy(energy_arima2_fore,energy_ts_trend[train+1:train+test])

energy_arima3 = arima(energy_ts_trend[1:train], order = c(1, 1, 2), xreg=oil_gold_matrix[c(1:train),])
energy_arima3_fore = forecast(energy_arima3, h = train, xreg=oil_gold_matrix[c((train+1):(train+test)),])
layout(1:2)
plot(energy_arima3_fore)
plot(energy_ts_trend[1:(train+test)])
energy_arima3
coeftest(energy_arima3)
accuracy(energy_arima3_fore,energy_ts_trend[train+1:train+test])

energy_arima4 = arima(energy_ts_trend[1:train], order = c(1, 1, 15), xreg=oil_gold_matrix[c(1:train),])
energy_arima4_fore = forecast(energy_arima4, h = train, xreg=oil_gold_matrix[c((train+1):(train+test)),])
layout(1:2)
plot(energy_arima4_fore)
plot(energy_ts_trend[1:(train+test)])
energy_arima4
coeftest(energy_arima4)
accuracy(energy_arima4_fore,energy_ts_trend[train+1:train+test])
accuracy(energy_arima4_fore)

energy_arima4_vali = arima(energy_ts_trend[1:train+test], order = c(1, 1, 15), xreg=oil_gold_matrix[c(1:train+test),])
energy_arima4_fore_vali = forecast(energy_arima4_vali, xreg=oil_gold_matrix[c((train+test+1):(train+test+validation)),])
plot(energy_arima4_fore_vali)
plot(energy_ts_trend[1:(train+test+validation)])
accuracy(energy_arima4_fore_vali)

energy_arima5 = auto.arima(energy_ts_trend[1:train], xreg=oil_gold_matrix[c(1:train),])
energy_arima5_fore = forecast(energy_arima5, h = train, xreg=oil_gold_matrix[c((train+1):(train+test)),])
layout(1:2)
plot(energy_arima5_fore)
plot(energy_ts_trend[1:(train+test)])
energy_arima5
coeftest(energy_arima5)
#accuracy(energy_arima5_fore,energy_ts_trend[train+1:train+test],energy_ts_trend[train+test+1:train+test+validation])
accuracy(energy_arima5_fore,energy_ts_trend[train+1:train+test])

energy_arima6 = auto.arima(energy_ts_trend[1:(train+test)], xreg=oil_gold_matrix[c(1:(train+test)),])
energy_arima6_fore = forecast(energy_arima6, h = validation, xreg=oil_gold_matrix[c((train+test+1):(train+test+validation)),])
layout(1:2)
plot(energy_arima6_fore)
plot(energy_ts_trend[1:(train+test+validation)])
energy_arima6
coeftest(energy_arima6)
#accuracy(energy_arima6_fore,energy_ts_trend[train+1:train+test],energy_ts_trend[train+test+1:train+test+validation])
accuracy(energy_arima6_fore,energy_ts_trend[train+1:train+test])

energy_arima3
energy_arima5
energy_arima6

energy_arima1_bic = AIC(energy_arima1, k = log(length(energy_ts_trend[1:train])))
energy_arima2_bic = AIC(energy_arima2, k = log(length(energy_ts_trend[1:train])))
energy_arima3_bic = AIC(energy_arima3, k = log(length(energy_ts_trend[1:train])))
energy_arima4_bic = AIC(energy_arima4, k = log(length(energy_ts_trend[1:train])))
energy_arima5_bic = AIC(energy_arima5, k = log(length(energy_ts_trend[1:train])))

energy_arima1_bic
energy_arima2_bic
energy_arima3_bic
energy_arima4_bic
energy_arima5_bic

energy_arima1$aic
energy_arima2$aic
energy_arima3$aic
energy_arima4$aic
energy_arima5$aic

layout(1:2)
acf(finance_ts_diff2, lag.max = 50) 
pacf(finance_ts_diff2, lag.max = 50) 

finance_arima1 = arima(finance_ts_trend[1:train], order = c(1, 2, 1), xreg=oil_gold_matrix[c(1:train),])
finance_arima2 = arima(finance_ts_trend[1:train], order = c(2, 2, 1), xreg=oil_gold_matrix[c(1:train),])
finance_arima3 = arima(finance_ts_trend[1:train], order = c(2, 2, 10), xreg=oil_gold_matrix[c(1:train),])
finance_arima4 = arima(finance_ts_trend[1:train], order = c(2, 2, 13), xreg=oil_gold_matrix[c(1:train),])
finance_arima5 = auto.arima(finance_ts_trend[1:train], xreg=oil_gold_matrix[c(1:train),])


finance_arima1 = arima(finance_ts_trend[1:train], order = c(1, 2, 1), xreg=oil_gold_matrix[c(1:train),])
finance_arima1_fore = forecast(finance_arima1, h = train, xreg=oil_gold_matrix[c((train+1):(train+test)),])
layout(1:2)
plot(finance_arima1_fore)
plot(finance_ts_trend[1:(train+test)])
finance_arima1
coeftest(finance_arima1)
accuracy(finance_arima1_fore,finance_ts_trend[train+1:train+test])

finance_arima2 = arima(finance_ts_trend[1:train], order = c(2, 2, 1), xreg=oil_gold_matrix[c(1:train),])
finance_arima2_fore = forecast(finance_arima2, h = train, xreg=oil_gold_matrix[c((train+1):(train+test)),])
layout(1:2)
plot(finance_arima2_fore)
plot(finance_ts_trend[1:(train+test)])
finance_arima2
coeftest(finance_arima2)
accuracy(finance_arima2_fore,finance_ts_trend[train+1:train+test])

finance_arima3 = arima(finance_ts_trend[1:train], order = c(2, 2, 10), xreg=oil_gold_matrix[c(1:train),])
finance_arima3_fore = forecast(finance_arima3, h = train, xreg=oil_gold_matrix[c((train+1):(train+test)),])
layout(1:2)
plot(finance_arima3_fore)
plot(finance_ts_trend[1:(train+test)])
finance_arima3
coeftest(finance_arima3)
accuracy(finance_arima3_fore,finance_ts_trend[train+1:train+test])

finance_arima4 = arima(finance_ts_trend[1:train], order = c(2, 2, 13), xreg=oil_gold_matrix[c(1:train),])
finance_arima4_fore = forecast(finance_arima4, h = train, xreg=oil_gold_matrix[c((train+1):(train+test)),])
layout(1:2)
plot(finance_arima4_fore)
plot(finance_ts_trend[1:(train+test)])
finance_arima4
coeftest(finance_arima4)
accuracy(finance_arima4_fore,finance_ts_trend[train+1:train+test])

finance_arima5 = auto.arima(finance_ts_trend[1:train], xreg=oil_gold_matrix[c(1:train),])
finance_arima5_fore = forecast(finance_arima5, h = train, xreg=oil_gold_matrix[c((train+1):(train+test)),])
layout(1:2)
plot(finance_arima5_fore)
plot(finance_ts_trend[1:(train+test)])
finance_arima5
coeftest(finance_arima5)
accuracy(finance_arima5_fore,finance_ts_trend[train+1:train+test])
accuracy(finance_arima4_fore)

finance_arima1_bic = AIC(finance_arima1, k = log(length(finance_ts_trend[1:train])))
finance_arima2_bic = AIC(finance_arima2, k = log(length(finance_ts_trend[1:train])))
finance_arima3_bic = AIC(finance_arima3, k = log(length(finance_ts_trend[1:train])))
finance_arima4_bic = AIC(finance_arima4, k = log(length(finance_ts_trend[1:train])))
finance_arima5_bic = AIC(finance_arima5, k = log(length(finance_ts_trend[1:train])))

finance_arima1$aic
finance_arima2$aic
finance_arima3$aic
finance_arima4$aic
finance_arima5$aic

finance_arima1_bic
finance_arima2_bic
finance_arima3_bic
finance_arima4_bic
finance_arima5_bic

finance_arima4_vali = arima(finance_ts_trend[1:train+test], order = c(2, 2, 13), xreg=oil_gold_matrix[c(1:train+test),])
finance_arima4_fore_vali = forecast(finance_arima4_vali, xreg=oil_gold_matrix[c((train+test+1):(train+test+validation)),])
layout(1:2)
plot(finance_arima4_fore_vali)
plot(finance_ts_trend[1:(train+test+validation)])
accuracy(finance_arima4_fore_vali)

acf(pharma_ts_diff1, lag.max = 20)
pacf(pharma_ts_diff1, lag.max = 20)

acf(tech_ts_diff1, lag.max = 20)
pacf(tech_ts_diff1, lag.max = 20)

acf(energy_ts_diff1, lag.max = 20)
pacf(energy_ts_diff1, lag.max = 20)

acf(finance_ts_diff2, lag.max = 20)
pacf(finance_ts_diff2, lag.max = 20)

acf(health_ts_diff1, lag.max = 20)
pacf(health_ts_diff1, lag.max = 20)

acf(airline_ts_diff2, lag.max = 20)
pacf(airline_ts_diff2, lag.max = 20)

acf(snp500_ts_diff1, lag.max = 20)
pacf(snp500_ts_diff1, lag.max = 20)

acf(oil_ts_diff1, lag.max = 20)
pacf(oil_ts_diff1, lag.max = 20)

acf(gold_ts_diff1, lag.max = 20)
pacf(gold_ts_diff1, lag.max = 20)
#The number of hidden neurons should be between the size of the input layer and the size of the output layer.
#The number of hidden neurons should be 2/3 the size of the input layer plus the size of the output layer.
#The number of hidden neurons should be less than twice the size of the input layer.

head(data)
summary(data)
str(data)

?scale
#Select only numeric data before scaling
data2=data[,2:11]
str(data2)
#Scaling function
data_scale=scale(data2, center = TRUE, scale = TRUE)
summary(data_scale)

bio_ts=ts(data_scale[,c("bio_close")],frequency=252,start=c(2007,1))
pharma_ts=ts(data[,c("pharma_close")],frequency=252,start=c(2007,1))
tech_ts=ts(data[,c("tech_close")],frequency=252,start=c(2007,1))
energy_ts=ts(data_scale[,c("energy_close")],frequency=252,start=c(2007,1))
finance_ts=ts(data[,c("finance_close")],frequency=252,start=c(2007,1))
health_ts=ts(data[,c("health_close")],frequency=252,start=c(2007,1))
airline_ts=ts(data[,c("airline_close")],frequency=252,start=c(2007,1))
snp500_ts=ts(data[,c("snp500_close")],frequency=252,start=c(2007,1))
oil_ts=ts(data[,c("oil_close")],frequency=252,start=c(2007,1))
gold_ts=ts(data[,c("gold_close")],frequency=252,start=c(2007,1))


bio_ts_dc = decompose(bio_ts)
pharma_ts_dc = decompose(pharma_ts)
tech_ts_dc = decompose(tech_ts)
energy_ts_dc = decompose(energy_ts)
finance_ts_dc = decompose(finance_ts)
health_ts_dc = decompose(health_ts)
airline_ts_dc = decompose(airline_ts)
snp500_ts_dc = decompose(snp500_ts)
oil_ts_dc = decompose(oil_ts)
gold_ts_dc = decompose(gold_ts)


##Extract trend data
bio_ts_trend = bio_ts - bio_ts_dc$seasonal
pharma_ts_trend = pharma_ts - pharma_ts_dc$seasonal
tech_ts_trend = tech_ts - tech_ts_dc$seasonal
energy_ts_trend = energy_ts - energy_ts_dc$seasonal
finance_ts_trend = finance_ts - finance_ts_dc$seasonal
health_ts_trend = health_ts - health_ts_dc$seasonal
airline_ts_trend = airline_ts - airline_ts_dc$seasonal
snp500_ts_trend = snp500_ts - snp500_ts_dc$seasonal
oil_ts_trend = oil_ts - oil_ts_dc$seasonal
gold_ts_trend = gold_ts - gold_ts_dc$seasonal

adf.test(energy_ts_trend, alternative = "stationary")
kpss.test(energy_ts_trend)
energy_ts_diff1 = diff(energy_ts_trend, differences = 1)
adf.test(energy_ts_diff1, alternative = "stationary")
kpss.test(energy_ts_diff1)

energy_arima1 = arima(energy_ts_trend[1:train], order = c(0, 1, 1), xreg=oil_gold_matrix[c(1:train),])
energy_arima1_fore = forecast(energy_arima1, h = train, xreg=oil_gold_matrix[c((train+1):(train+test)),])
plot(energy_arima1_fore)
energy_arima1
coeftest(energy_arima1)
accuracy(energy_arima1_fore,energy_ts_trend[train+1:train+test])

summary(data$energy)
plot(data_scale)

plot(stocks_data$Date,stocks_data$SP500_Adj_Close, type = 'l', main = 'SP500_Adj_Close')
plot(stocks_data$Date,stocks_data$Airline_Adj_Close, type = 'l', main = 'Airline_Adj_Close')
plot(stocks_data$Date,stocks_data$Health_Close, type = 'l', main = 'Health_Close')
plot(stocks_data$Date,stocks_data$Fin_Close, type = 'l', main = 'Fin_Close')
plot(stocks_data$Date,stocks_data$Energy_Close, type = 'l', main = 'Energy_Close')
plot(stocks_data$Date,stocks_data$Tech_Close, type = 'l', main = 'Tech_Close')
plot(stocks_data$Date,stocks_data$Pharma_Close, type = 'l', main = 'Pharma_Close')
plot(stocks_data$Date,stocks_data$Bio_Close, type = 'l', main = 'Bio_Close')

library(psych)
library()
rcorr(as.matrix(na.omit(data))
?rcorr
