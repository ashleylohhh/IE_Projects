library(fBasics)
library(forecast) 

# Set the folder (path) that contains this R file as the working directory
dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(dir)

# *** Get data ***
datos<-read.csv("../coca_cola_earnings.csv",header=TRUE,sep=";",dec=",")
z<-datos[,2]
y<-datos[1:(length(z)),2]

ts.plot(y)

par(mar=c(5,5,5,5))

nlags=40    # play with this parameter..
par(mfrow=c(2,1))
acf(y,nlags)
pacf(y,nlags)  

s=4       # seasonal parameter FOR THIS DATA SET


# log y for variance
r=log(y)

## Creation of frame to afterwards evaluate the performance of our 4 models

Errorsframe <- as.data.frame(matrix(data = 0, nrow=4,ncol = 20))
colnames(Errorsframe) <- c("MAPE_rec_M1","MSFE_rec_M1","MAPE_rol_M1","MSFE_rol_M1","MAPE_rec_M2","MSFE_rec_M2","MAPE_rol_M2","MSFE_rol_M2","MAPE_rec_M3","MSFE_rec_M3","MAPE_rol_M3","MSFE_rol_M3","MAPE_rec_M4","MSFE_rec_M4","MAPE_rol_M4","MSFE_rol_M4","MAPE_rec_M5","MSFE_rec_M5","MAPE_rol_M5","MSFE_rol_M5")

############  Model 1 LOG AR(3,1,0)(1,1,0)4  ################
fit<-arima(r,order=c(3,1,0),seasonal=list(order=c(1,1,0),period=s)) 
fit


ts.plot(fit$residuals)
par(mfrow=c(2,1))
acf(fit$residuals,nlags)
pacf(fit$residuals,nlags)    

# Check for White noise residuals
Box.test(fit$residuals,lag=30)

Box.test(fit$residuals,lag=20)

Box.test(fit$residuals,lag=15)

Box.test(fit$residuals,lag=10)

# Check for normality
shapiro.test(fit$residuals)

# Graphical normality test
hist(fit$residuals,prob=T,ylim=c(0,15),xlim=c(mean(fit$residuals)-3*sd(fit$residuals),mean(fit$residuals)+3*sd(fit$residuals)),col="red")
lines(density(fit$residuals),lwd=2)
mu<-mean(fit$residuals)
sigma<-sd(fit$residuals)
x<-seq(mu-3*sigma,mu+3*sigma,length=100)
yy<-dnorm(x,mu,sigma)
lines(x,yy,lwd=2,col="blue")

# point predictions and standard errors

n<-length(y)
n.forecasting<-25 # 25 forecasts
n.estimation<-n-n.forecasting 
horizontes<-4 # number of periods ahead

predicc<-matrix(0,nrow=n.forecasting,ncol=horizontes)
real<-matrix(0,nrow=n.forecasting,ncol=1)
real<-y[(n.estimation+1):length(y)] 
MSFE<-matrix(0,nrow=horizontes,ncol=1)
MAPE<-matrix(0,nrow=horizontes,ncol=1)

# Recursive

for (Periods_ahead in 1:horizontes) {
  for (i in 1:n.forecasting) {
    aux.y<-y[1:(n.estimation-Periods_ahead+i)];
    fit2<-arima(log(aux.y),order=c(3,1,0),seasonal=list(order=c(1,1,0),period=s));
    y.pred<-predict(fit2,n.ahead=Periods_ahead);
    predicc[i,Periods_ahead]<- exp(y.pred$pred[Periods_ahead]);
  }
  error<-real-predicc[,Periods_ahead];
  MSFE[Periods_ahead]<-mean(error^2);
  MAPE[Periods_ahead]<-mean(abs(error/real)) *100;
}

# Model 1 Recursive Scores
MSFE;
MAPE;


# Save Errors in Matrix to afterwards plot them
Errorsframe[,1] <- MAPE
Errorsframe[,2] <- MSFE

# Plot predictions vs. actual data
library(ggplot2)
x <- 25
# Recursive Predictions Plots
new <- c(z[1:(length(z)-x)],predicc[,1]) # real data + predicted values  We need as many as horizons
new2 <- c(z[1:(length(z)-x)],predicc[,2]) # real data + predicted values
new3 <- c(z[1:(length(z)-x)],predicc[,3]) # real data + predicted values
new4 <- c(z[1:(length(z)-x)],predicc[,4]) # real data + predicted values
df0 <- as.data.frame(strptime(datos[,1],format="%Y%m%d"))
df1 <- as.data.frame(z)
df2 <- as.data.frame(new)
df3 <- as.data.frame(new2)
df4 <- as.data.frame(new3)
df5 <- as.data.frame(new4)

DF <- cbind(df0,df1,df2,df3,df4,df4)
colnames(DF) <- c("date","real","pred_h1","pred_h2","pred_h3","pred_h4")

par(mfrow=c(1,1))

jpeg("Model 1 : 1 Horizons (Recursive)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h1),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 1 : 1 Horizon (Recursive)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[1],3), ' MSFE: ',round(MSFE[1],3)), caption="SARIMA (3,1,0)x(1,0,0) s = 4");
dev.off()

jpeg("Model 1 : 2 Horizons (Recursive)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h2),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 1 : 2 Horizons (Recursive)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[2],3), ' MSFE: ',round(MSFE[2],3)), caption="SARIMA (3,1,0)x(1,0,0) s = 4");
dev.off()

jpeg("Model 1 : 3 Horizons (Recursive)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h2),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 1 : 3 Horizons (Recursive)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[3],3), ' MSFE: ',round(MSFE[3],3)), caption="SARIMA (3,1,0)x(1,0,0) s = 4");
dev.off()

jpeg("Model 1 : 4 Horizons (Recursive)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h2),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 1 : 4 Horizons (Recursive)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[4],3), ' MSFE: ',round(MSFE[4],3)), caption="SARIMA (3,1,0)x(1,0,0) s = 4");
dev.off()

# Rolling

for (Periods_ahead in 1:horizontes) {
  for (i in 1:n.forecasting) {
    aux.y<-y[i:(n.estimation-Periods_ahead+i)];
    fit<-arima(log(aux.y),order=c(3,1,0),seasonal=list(order=c(1,1,0),period=s));
    y.pred<-predict(fit,n.ahead=Periods_ahead);
    predicc[i,Periods_ahead]<- exp(y.pred$pred[Periods_ahead]);
  }
  error<-real-predicc[,Periods_ahead];
  MSFE[Periods_ahead]<-mean(error^2);
  MAPE[Periods_ahead]<-mean(abs(error/real)) *100;
}

# Model 1 Rolling Scores
MSFE;
MAPE;

Errorsframe[,3] <- MAPE
Errorsframe[,4] <- MSFE
# Plot predictions vs. actual data
library(ggplot)

# Rolling Predictions Plots
new <- c(z[1:(length(z)-x)],predicc[,1]) # real data + predicted values  We need as many as horizons
new2 <- c(z[1:(length(z)-x)],predicc[,2]) # real data + predicted values
new3 <- c(z[1:(length(z)-x)],predicc[,3]) # real data + predicted values
new4 <- c(z[1:(length(z)-x)],predicc[,4]) # real data + predicted values
df0 <- as.data.frame(strptime(datos[,1],format="%Y%m%d"))
df1 <- as.data.frame(z)
df2 <- as.data.frame(new)
df3 <- as.data.frame(new2)
df4 <- as.data.frame(new3)
df5 <- as.data.frame(new4)

DF <- cbind(df0,df1,df2,df3,df4,df4)
colnames(DF) <- c("date","real","pred_h1","pred_h2","pred_h3","pred_h4")

par(mfrow=c(1,1))

jpeg("Model 1 : 1 Horizons (Rolling)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h1),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 1 : 1 Horizon (Rolling)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[1],3), ' MSFE: ',round(MSFE[1],3)), caption="SARIMA (3,1,0)x(1,0,0) s = 4");
dev.off()

jpeg("Model 1 : 2 Horizons (Rolling)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h2),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 1 : 2 Horizons (Rolling)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[2],3), ' MSFE: ',round(MSFE[2],3)), caption="SARIMA (3,1,0)x(1,0,0) s = 4");
dev.off()

jpeg("Model 1 : 3 Horizons (Rolling)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h2),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 1 : 3 Horizons (Rolling)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[3],3), ' MSFE: ',round(MSFE[3],3)), caption="SARIMA (3,1,0)x(1,0,0) s = 4");
dev.off()

jpeg("Model 1 : 4 Horizons (Rolling)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h2),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 1 : 4 Horizons (Rolling)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[4],3), ' MSFE: ',round(MSFE[4],3)), caption="SARIMA (3,1,0)x(1,0,0) s = 4");
dev.off()

############  Model 2 LOG AR(0,1,3)(0,1,1)4  ################
y<-datos[1:(length(z)),2]
r=log(y)

fit<-arima(r,order=c(0,1,3),seasonal=list(order=c(0,1,1),period=s)) 
fit


ts.plot(fit$residuals)
par(mfrow=c(2,1))
acf(fit$residuals,nlags)
pacf(fit$residuals,nlags)    

# Check for White noise residuals
Box.test(fit$residuals,lag=30)

Box.test(fit$residuals,lag=20)

Box.test(fit$residuals,lag=15)

Box.test(fit$residuals,lag=10)

# Check for normality
shapiro.test(fit$residuals)

# Graphical normality test
hist(fit$residuals,prob=T,ylim=c(0,15),xlim=c(mean(fit$residuals)-3*sd(fit$residuals),mean(fit$residuals)+3*sd(fit$residuals)),col="red")
lines(density(fit$residuals),lwd=2)
mu<-mean(fit$residuals)
sigma<-sd(fit$residuals)
x<-seq(mu-3*sigma,mu+3*sigma,length=100)
yy<-dnorm(x,mu,sigma)
lines(x,yy,lwd=2,col="blue")

# point predictions and standard errors

n<-length(y)
n.forecasting<-25 # 25 forecasts
n.estimation<-n-n.forecasting 
horizontes<-4 # number of periods ahead

predicc<-matrix(0,nrow=n.forecasting,ncol=horizontes)
real<-matrix(0,nrow=n.forecasting,ncol=1)
real<-y[(n.estimation+1):length(y)] 
MSFE<-matrix(0,nrow=horizontes,ncol=1)
MAPE<-matrix(0,nrow=horizontes,ncol=1)

# Recursive

for (Periods_ahead in 1:horizontes) {
  for (i in 1:n.forecasting) {
    aux.y<-y[1:(n.estimation-Periods_ahead+i)];
    fit2<-arima(log(aux.y),order=c(0,1,3),seasonal=list(order=c(0,1,1),period=s));
    y.pred<-predict(fit2,n.ahead=Periods_ahead);
    predicc[i,Periods_ahead]<- exp(y.pred$pred[Periods_ahead]);
  }
  error<-real-predicc[,Periods_ahead];
  MSFE[Periods_ahead]<-mean(error^2);
  MAPE[Periods_ahead]<-mean(abs(error/real)) *100;
}

# Model 2 Recursive Scores
MSFE;
MAPE;

Errorsframe[,5] <- MAPE
Errorsframe[,6] <- MSFE
# Plot predictions vs. actual data
library(ggplot)

# Recursive Predictions Plots
new <- c(z[1:(length(z)-x)],predicc[,1]) # real data + predicted values  We need as many as horizons
new2 <- c(z[1:(length(z)-x)],predicc[,2]) # real data + predicted values
new3 <- c(z[1:(length(z)-x)],predicc[,3]) # real data + predicted values
new4 <- c(z[1:(length(z)-x)],predicc[,4]) # real data + predicted values
df0 <- as.data.frame(strptime(datos[,1],format="%Y%m%d"))
df1 <- as.data.frame(z)
df2 <- as.data.frame(new)
df3 <- as.data.frame(new2)
df4 <- as.data.frame(new3)
df5 <- as.data.frame(new4)

DF <- cbind(df0,df1,df2,df3,df4,df4)
colnames(DF) <- c("date","real","pred_h1","pred_h2","pred_h3","pred_h4")

par(mfrow=c(1,1))

jpeg("Model 2 : 1 Horizons (Recursive)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h1),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 2 : 1 Horizon (Recursive)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[1],3), ' MSFE: ',round(MSFE[1],3)), caption="SARIMA (0,1,3)x(0,1,1) s = 4");
dev.off()

jpeg("Model 2 : 2 Horizons (Recursive)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h2),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 2 : 2 Horizons (Recursive)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[2],3), ' MSFE: ',round(MSFE[2],3)), caption="SARIMA (0,1,3)x(0,1,1) s = 4");
dev.off()

jpeg("Model 2 : 3 Horizons (Recursive)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h2),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 2 : 3 Horizons (Recursive)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[3],3), ' MSFE: ',round(MSFE[3],3)), caption="SARIMA (0,1,3)x(0,1,1) s = 4");
dev.off()

jpeg("Model 2 : 4 Horizons (Recursive)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h2),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 2 : 4 Horizons (Recursive)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[4],3), ' MSFE: ',round(MSFE[4],3)), caption="SARIMA (0,1,3)x(0,1,1) s = 4");
dev.off()

# Rolling

for (Periods_ahead in 1:horizontes) {
  for (i in 1:n.forecasting) {
    aux.y<-y[i:(n.estimation-Periods_ahead+i)];
    fit<-arima(log(aux.y),order=c(0,1,3),seasonal=list(order=c(0,1,1),period=s));
    y.pred<-predict(fit,n.ahead=Periods_ahead);
    predicc[i,Periods_ahead]<- exp(y.pred$pred[Periods_ahead]);
  }
  error<-real-predicc[,Periods_ahead];
  MSFE[Periods_ahead]<-mean(error^2);
  MAPE[Periods_ahead]<-mean(abs(error/real)) *100;
}

# Model 2 Rolling Scores
MSFE;
MAPE;

Errorsframe[,7] <- MAPE
Errorsframe[,8] <- MSFE
# Plot predictions vs. actual data
library(ggplot)

# Rolling Predictions Plots
new <- c(z[1:(length(z)-x)],predicc[,1]) # real data + predicted values  We need as many as horizons
new2 <- c(z[1:(length(z)-x)],predicc[,2]) # real data + predicted values
new3 <- c(z[1:(length(z)-x)],predicc[,3]) # real data + predicted values
new4 <- c(z[1:(length(z)-x)],predicc[,4]) # real data + predicted values
df0 <- as.data.frame(strptime(datos[,1],format="%Y%m%d"))
df1 <- as.data.frame(z)
df2 <- as.data.frame(new)
df3 <- as.data.frame(new2)
df4 <- as.data.frame(new3)
df5 <- as.data.frame(new4)

DF <- cbind(df0,df1,df2,df3,df4,df4)
colnames(DF) <- c("date","real","pred_h1","pred_h2","pred_h3","pred_h4")

par(mfrow=c(1,1))

jpeg("Model 2 : 1 Horizons (Rolling)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h1),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 2 : 1 Horizon (Rolling)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[1],3), ' MSFE: ',round(MSFE[1],3)), caption="SARIMA (0,1,3)x(0,1,1) s = 4");
dev.off()

jpeg("Model 2 : 2 Horizons (Rolling)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h2),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 2 : 2 Horizons (Rolling)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[2],3), ' MSFE: ',round(MSFE[2],3)), caption="SARIMA (0,1,3)x(0,1,1) s = 4");
dev.off()

jpeg("Model 2 : 3 Horizons (Rolling)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h2),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 2 : 3 Horizons (Rolling)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[3],3), ' MSFE: ',round(MSFE[3],3)), caption="SARIMA (0,1,3)x(0,1,1) s = 4");
dev.off()

jpeg("Model 2 : 4 Horizons (Rolling)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h2),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 2 : 4 Horizons (Rolling)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[4],3), ' MSFE: ',round(MSFE[4],3)), caption="SARIMA (0,1,3)x(0,1,1) s = 4");
dev.off()

############  Model 3 LOG AR(0,0,2)(0,2,1)4  ################
y<-datos[1:(length(z)),2]
r=log(y)

fit<-arima(r,order=c(0,0,2),seasonal=list(order=c(0,2,1),period=s)) 
fit

ts.plot(fit$residuals)
par(mfrow=c(2,1))
acf(fit$residuals,nlags)
pacf(fit$residuals,nlags)    

# Check for White noise residuals
Box.test(fit$residuals,lag=30)

Box.test(fit$residuals,lag=20)

Box.test(fit$residuals,lag=15)

Box.test(fit$residuals,lag=10)

# Check for normality
shapiro.test(fit$residuals)

# Graphical normality test
hist(fit$residuals,prob=T,ylim=c(0,15),xlim=c(mean(fit$residuals)-3*sd(fit$residuals),mean(fit$residuals)+3*sd(fit$residuals)),col="red")
lines(density(fit$residuals),lwd=2)
mu<-mean(fit$residuals)
sigma<-sd(fit$residuals)
x<-seq(mu-3*sigma,mu+3*sigma,length=100)
yy<-dnorm(x,mu,sigma)
lines(x,yy,lwd=2,col="blue")

# point predictions and standard errors

n<-length(y)
n.forecasting<-25 # 25 forecasts
n.estimation<-n-n.forecasting 
horizontes<-4 # number of periods ahead

predicc<-matrix(0,nrow=n.forecasting,ncol=horizontes)
real<-matrix(0,nrow=n.forecasting,ncol=1)
real<-y[(n.estimation+1):length(y)] 
MSFE<-matrix(0,nrow=horizontes,ncol=1)
MAPE<-matrix(0,nrow=horizontes,ncol=1)

# Recursive

for (Periods_ahead in 1:horizontes) {
  for (i in 1:n.forecasting) {
    aux.y<-y[1:(n.estimation-Periods_ahead+i)];
    fit2<-arima(log(aux.y),order=c(0,0,2),seasonal=list(order=c(0,2,1),period=s));
    y.pred<-predict(fit2,n.ahead=Periods_ahead);
    predicc[i,Periods_ahead]<- exp(y.pred$pred[Periods_ahead]);
  }
  error<-real-predicc[,Periods_ahead];
  MSFE[Periods_ahead]<-mean(error^2);
  MAPE[Periods_ahead]<-mean(abs(error/real)) *100;
}

# Model 3 Recursive Scores
MSFE;
MAPE;

Errorsframe[,9] <- MAPE
Errorsframe[,10] <- MSFE
# Plot predictions vs. actual data
library(ggplot)

# Recursive Predictions Plots
new <- c(z[1:(length(z)-x)],predicc[,1]) # real data + predicted values  We need as many as horizons
new2 <- c(z[1:(length(z)-x)],predicc[,2]) # real data + predicted values
new3 <- c(z[1:(length(z)-x)],predicc[,3]) # real data + predicted values
new4 <- c(z[1:(length(z)-x)],predicc[,4]) # real data + predicted values
df0 <- as.data.frame(strptime(datos[,1],format="%Y%m%d"))
df1 <- as.data.frame(z)
df2 <- as.data.frame(new)
df3 <- as.data.frame(new2)
df4 <- as.data.frame(new3)
df5 <- as.data.frame(new4)

DF <- cbind(df0,df1,df2,df3,df4,df4)
colnames(DF) <- c("date","real","pred_h1","pred_h2","pred_h3","pred_h4")

par(mfrow=c(1,1))

jpeg("Model 3 : 1 Horizons (Recursive)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h1),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 3 : 1 Horizon (Recursive)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[1],3), ' MSFE: ',round(MSFE[1],3)), caption="SARIMA (0,0,2)x(0,2,1) s = 4");
dev.off()

jpeg("Model 3 : 2 Horizons (Recursive)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h2),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 3 : 2 Horizons (Recursive)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[2],3), ' MSFE: ',round(MSFE[2],3)), caption="SARIMA (0,0,2)x(0,2,1) s = 4");
dev.off()

jpeg("Model 3 : 3 Horizons (Recursive)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h2),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 3 : 3 Horizons (Recursive)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[3],3), ' MSFE: ',round(MSFE[3],3)), caption="SARIMA (0,0,2)x(0,2,1) s = 4");
dev.off()

jpeg("Model 3 : 4 Horizons (Recursive)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h2),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 3 : 4 Horizons (Recursive)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[4],3), ' MSFE: ',round(MSFE[4],3)), caption="SARIMA (0,0,2)x(0,2,1) s = 4");
dev.off()

# Rolling

for (Periods_ahead in 1:horizontes) {
  for (i in 1:n.forecasting) {
    aux.y<-y[i:(n.estimation-Periods_ahead+i)];
    fit<-arima(log(aux.y),order=c(0,0,2),seasonal=list(order=c(0,2,1),period=s));
    y.pred<-predict(fit,n.ahead=Periods_ahead);
    predicc[i,Periods_ahead]<- exp(y.pred$pred[Periods_ahead]);
  }
  error<-real-predicc[,Periods_ahead];
  MSFE[Periods_ahead]<-mean(error^2);
  MAPE[Periods_ahead]<-mean(abs(error/real)) *100;
}

# Model 3 Rolling Scores
MSFE;
MAPE;

Errorsframe[,11] <- MAPE
Errorsframe[,12] <- MSFE
# Plot predictions vs. actual data
library(ggplot)

# Rolling Predictions Plots
new <- c(z[1:(length(z)-x)],predicc[,1]) # real data + predicted values  We need as many as horizons
new2 <- c(z[1:(length(z)-x)],predicc[,2]) # real data + predicted values
new3 <- c(z[1:(length(z)-x)],predicc[,3]) # real data + predicted values
new4 <- c(z[1:(length(z)-x)],predicc[,4]) # real data + predicted values
df0 <- as.data.frame(strptime(datos[,1],format="%Y%m%d"))
df1 <- as.data.frame(z)
df2 <- as.data.frame(new)
df3 <- as.data.frame(new2)
df4 <- as.data.frame(new3)
df5 <- as.data.frame(new4)

DF <- cbind(df0,df1,df2,df3,df4,df4)
colnames(DF) <- c("date","real","pred_h1","pred_h2","pred_h3","pred_h4")

par(mfrow=c(1,1))

jpeg("Model 3 : 1 Horizons (Rolling)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h1),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 3 : 1 Horizon (Rolling)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[1],3), ' MSFE: ',round(MSFE[1],3)), caption="SARIMA (0,0,2)x(0,2,1) s = 4");
dev.off()

jpeg("Model 3 : 2 Horizons (Rolling)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h2),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 3 : 2 Horizons (Rolling)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[2],3), ' MSFE: ',round(MSFE[2],3)), caption="SARIMA (0,0,2)x(0,2,1) s = 4");
dev.off()

jpeg("Model 3 : 3 Horizons (Rolling)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h2),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 3 : 3 Horizons (Rolling)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[3],3), ' MSFE: ',round(MSFE[3],3)), caption="SARIMA (0,0,2)x(0,2,1) s = 4");
dev.off()

jpeg("Model 3 : 4 Horizons (Rolling)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h2),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 3 : 4 Horizons (Rolling)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[4],3), ' MSFE: ',round(MSFE[4],3)), caption="SARIMA (0,0,2)x(0,2,1) s = 4");
dev.off()

############  Model 4 AR(0,1,0)(1,1,0)4  ################
y<-datos[1:(length(z)),2]
r=log(y)

par(mfrow=c(3,1))
ts.plot(y)
acf(y,nlags)
pacf(y,nlags)

fit<-arima(y,order=c(0,1,0),seasonal=list(order=c(1,1,0),period=s)) 
fit

ts.plot(fit$residuals)
par(mfrow=c(2,1))
acf(fit$residuals,nlags)
pacf(fit$residuals,nlags)    

# Check for White noise residuals
Box.test(fit$residuals,lag=30)

Box.test(fit$residuals,lag=20)

Box.test(fit$residuals,lag=15)

Box.test(fit$residuals,lag=10)

# Check for normality
shapiro.test(fit$residuals)

# Graphical normality test
hist(fit$residuals,prob=T,ylim=c(0,15),xlim=c(mean(fit$residuals)-3*sd(fit$residuals),mean(fit$residuals)+3*sd(fit$residuals)),col="red")
lines(density(fit$residuals),lwd=2)
mu<-mean(fit$residuals)
sigma<-sd(fit$residuals)
x<-seq(mu-3*sigma,mu+3*sigma,length=100)
yy<-dnorm(x,mu,sigma)
lines(x,yy,lwd=2,col="blue")

# point predictions and standard errors

n<-length(y)
n.forecasting<-25 # 25 forecasts
n.estimation<-n-n.forecasting 
horizontes<-4 # number of periods ahead

predicc<-matrix(0,nrow=n.forecasting,ncol=horizontes)
real<-matrix(0,nrow=n.forecasting,ncol=1)
real<-y[(n.estimation+1):length(y)] 
MSFE<-matrix(0,nrow=horizontes,ncol=1)
MAPE<-matrix(0,nrow=horizontes,ncol=1)

# Recursive

for (Periods_ahead in 1:horizontes) {
  for (i in 1:n.forecasting) {
    aux.y<-y[1:(n.estimation-Periods_ahead+i)];
    fit2<-arima(aux.y,order=c(0,1,0),seasonal=list(order=c(1,1,0),period=s));
    y.pred<-predict(fit2,n.ahead=Periods_ahead);
    predicc[i,Periods_ahead]<- y.pred$pred[Periods_ahead];
  }
  error<-real-predicc[,Periods_ahead];
  MSFE[Periods_ahead]<-mean(error^2);
  MAPE[Periods_ahead]<-mean(abs(error/real)) *100;
}

# Model 4 Recursive Scores
MSFE;
MAPE;

Errorsframe[,13] <- MAPE
Errorsframe[,14] <- MSFE
# Plot predictions vs. actual data
library(ggplot)

# Recursive Predictions Plots
new <- c(z[1:(length(z)-x)],predicc[,1]) # real data + predicted values  We need as many as horizons
new2 <- c(z[1:(length(z)-x)],predicc[,2]) # real data + predicted values
new3 <- c(z[1:(length(z)-x)],predicc[,3]) # real data + predicted values
new4 <- c(z[1:(length(z)-x)],predicc[,4]) # real data + predicted values
df0 <- as.data.frame(strptime(datos[,1],format="%Y%m%d"))
df1 <- as.data.frame(z)
df2 <- as.data.frame(new)
df3 <- as.data.frame(new2)
df4 <- as.data.frame(new3)
df5 <- as.data.frame(new4)

DF <- cbind(df0,df1,df2,df3,df4,df4)
colnames(DF) <- c("date","real","pred_h1","pred_h2","pred_h3","pred_h4")

par(mfrow=c(1,1))

jpeg("Model 4 : 1 Horizons (Recursive)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h1),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 4 : 1 Horizon (Recursive)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[1],3), ' MSFE: ',round(MSFE[1],3)), caption="SARIMA (0,1,0)x(1,1,0) s = 4");
dev.off()

jpeg("Model 4 : 2 Horizons (Recursive)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h2),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 4 : 2 Horizons (Recursive)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[2],3), ' MSFE: ',round(MSFE[2],3)), caption="SARIMA (0,1,0)x(1,1,0) s = 4");
dev.off()

jpeg("Model 4 : 3 Horizons (Recursive)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h2),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 4 : 3 Horizons (Recursive)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[3],3), ' MSFE: ',round(MSFE[3],3)), caption="SARIMA (0,1,0)x(1,1,0) s = 4");
dev.off()

jpeg("Model 4 : 4 Horizons (Recursive)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h2),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 4 : 4 Horizons (Recursive)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[4],3), ' MSFE: ',round(MSFE[4],3)), caption="SARIMA (0,1,0)x(1,1,0) s = 4");
dev.off()

# Rolling

for (Periods_ahead in 1:horizontes) {
  for (i in 1:n.forecasting) {
    aux.y<-y[i:(n.estimation-Periods_ahead+i)];
    fit<-arima(aux.y,order=c(0,1,0),seasonal=list(order=c(1,1,0),period=s));
    y.pred<-predict(fit,n.ahead=Periods_ahead);
    predicc[i,Periods_ahead]<- y.pred$pred[Periods_ahead];
  }
  error<-real-predicc[,Periods_ahead];
  MSFE[Periods_ahead]<-mean(error^2);
  MAPE[Periods_ahead]<-mean(abs(error/real)) *100;
}

# Model 4 Rolling Scores
MSFE;
MAPE;

Errorsframe[,15] <- MAPE
Errorsframe[,16] <- MSFE
# Plot predictions vs. actual data
library(ggplot)

# Rolling Predictions Plots
new <- c(z[1:(length(z)-x)],predicc[,1]) # real data + predicted values  We need as many as horizons
new2 <- c(z[1:(length(z)-x)],predicc[,2]) # real data + predicted values
new3 <- c(z[1:(length(z)-x)],predicc[,3]) # real data + predicted values
new4 <- c(z[1:(length(z)-x)],predicc[,4]) # real data + predicted values
df0 <- as.data.frame(strptime(datos[,1],format="%Y%m%d"))
df1 <- as.data.frame(z)
df2 <- as.data.frame(new)
df3 <- as.data.frame(new2)
df4 <- as.data.frame(new3)
df5 <- as.data.frame(new4)

DF <- cbind(df0,df1,df2,df3,df4,df4)
colnames(DF) <- c("date","real","pred_h1","pred_h2","pred_h3","pred_h4")

par(mfrow=c(1,1))

jpeg("Model 4 : 1 Horizons (Rolling)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h1),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 4 : 1 Horizon (Rolling)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[1],3), ' MSFE: ',round(MSFE[1],3)), caption="SARIMA (0,1,0)x(1,1,0) s = 4");
dev.off()

jpeg("Model 4 : 2 Horizons (Rolling)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h2),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 4 : 2 Horizons (Rolling)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[2],3), ' MSFE: ',round(MSFE[2],3)), caption="SARIMA (0,1,0)x(1,1,0) s = 4");
dev.off()

jpeg("Model 4 : 3 Horizons (Rolling)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h2),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 4 : 3 Horizons (Rolling)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[3],3), ' MSFE: ',round(MSFE[3],3)), caption="SARIMA (0,1,0)x(1,1,0) s = 4");
dev.off()

jpeg("Model 4 : 4 Horizons (Rolling)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h2),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 4 : 4 Horizons (Rolling)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[4],3), ' MSFE: ',round(MSFE[4],3)), caption="SARIMA (0,1,0)x(1,1,0) s = 4");
dev.off()

############  Model 5 LOG AR(1,1,0)(2,1,0)4  ################
y<-datos[1:(length(z)),2]
r=log(y)

par(mfrow=c(3,1))
ts.plot(y)
acf(y,nlags)
pacf(y,nlags)

fit<-arima(r,order=c(1,1,0),seasonal=list(order=c(2,1,0),period=s)) 
fit

ts.plot(fit$residuals)
par(mfrow=c(2,1))
acf(fit$residuals,nlags)
pacf(fit$residuals,nlags)    

# Check for White noise residuals
Box.test(fit$residuals,lag=30)

Box.test(fit$residuals,lag=20)

Box.test(fit$residuals,lag=15)

Box.test(fit$residuals,lag=10)

# Check for normality
shapiro.test(fit$residuals)

# Graphical normality test
hist(fit$residuals,prob=T,ylim=c(0,15),xlim=c(mean(fit$residuals)-3*sd(fit$residuals),mean(fit$residuals)+3*sd(fit$residuals)),col="red")
lines(density(fit$residuals),lwd=2)
mu<-mean(fit$residuals)
sigma<-sd(fit$residuals)
x<-seq(mu-3*sigma,mu+3*sigma,length=100)
yy<-dnorm(x,mu,sigma)
lines(x,yy,lwd=2,col="blue")

# point predictions and standard errors

n<-length(y)
n.forecasting<-25 # 25 forecasts
n.estimation<-n-n.forecasting 
horizontes<-4 # number of periods ahead

predicc<-matrix(0,nrow=n.forecasting,ncol=horizontes)
real<-matrix(0,nrow=n.forecasting,ncol=1)
real<-y[(n.estimation+1):length(y)] 
MSFE<-matrix(0,nrow=horizontes,ncol=1)
MAPE<-matrix(0,nrow=horizontes,ncol=1)

# Recursive

for (Periods_ahead in 1:horizontes) {
  for (i in 1:n.forecasting) {
    aux.y<-y[1:(n.estimation-Periods_ahead+i)];
    fit2<-arima(log(aux.y),order=c(1,1,0),seasonal=list(order=c(2,1,0),period=s));
    y.pred<-predict(fit2,n.ahead=Periods_ahead);
    predicc[i,Periods_ahead]<- exp(y.pred$pred[Periods_ahead]);
  }
  error<-real-predicc[,Periods_ahead];
  MSFE[Periods_ahead]<-mean(error^2);
  MAPE[Periods_ahead]<-mean(abs(error/real)) *100;
}

# Model 5 Recursive Scores
MSFE;
MAPE;

Errorsframe[,17] <- MAPE
Errorsframe[,18] <- MSFE
# Plot predictions vs. actual data
library(ggplot)

# Recursive Predictions Plots
new <- c(z[1:(length(z)-x)],predicc[,1]) # real data + predicted values  We need as many as horizons
new2 <- c(z[1:(length(z)-x)],predicc[,2]) # real data + predicted values
new3 <- c(z[1:(length(z)-x)],predicc[,3]) # real data + predicted values
new4 <- c(z[1:(length(z)-x)],predicc[,4]) # real data + predicted values
df0 <- as.data.frame(strptime(datos[,1],format="%Y%m%d"))
df1 <- as.data.frame(z)
df2 <- as.data.frame(new)
df3 <- as.data.frame(new2)
df4 <- as.data.frame(new3)
df5 <- as.data.frame(new4)

DF <- cbind(df0,df1,df2,df3,df4,df4)
colnames(DF) <- c("date","real","pred_h1","pred_h2","pred_h3","pred_h4")

par(mfrow=c(1,1))

jpeg("Model 5 : 1 Horizons (Recursive)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h1),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 5 : 1 Horizon (Recursive)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[1],3), ' MSFE: ',round(MSFE[1],3)), caption="SARIMA (1,1,0)x(2,1,0) s = 4");
dev.off()

jpeg("Model 5 : 2 Horizons (Recursive)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h2),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 5 : 2 Horizons (Recursive)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[2],3), ' MSFE: ',round(MSFE[2],3)), caption="SARIMA (1,1,0)x(2,1,0) s = 4");
dev.off()

jpeg("Model 5 : 3 Horizons (Recursive)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h2),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 5 : 3 Horizons (Recursive)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[3],3), ' MSFE: ',round(MSFE[3],3)), caption="SARIMA (1,1,0)x(2,1,0) s = 4");
dev.off()

jpeg("Model 5 : 4 Horizons (Recursive)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h2),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 5 : 4 Horizons (Recursive)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[4],3), ' MSFE: ',round(MSFE[4],3)), caption="SARIMA (1,1,0)x(2,1,0) s = 4");
dev.off()

# Rolling

for (Periods_ahead in 1:horizontes) {
  for (i in 1:n.forecasting) {
    aux.y<-y[i:(n.estimation-Periods_ahead+i)];
    fit<-arima(log(aux.y),order=c(0,1,0),seasonal=list(order=c(2,1,0),period=s));
    y.pred<-predict(fit,n.ahead=Periods_ahead);
    predicc[i,Periods_ahead]<- exp(y.pred$pred[Periods_ahead]);
  }
  error<-real-predicc[,Periods_ahead];
  MSFE[Periods_ahead]<-mean(error^2);
  MAPE[Periods_ahead]<-mean(abs(error/real)) *100;
}

# Model 4 Rolling Scores
MSFE;
MAPE;

Errorsframe[,19] <- MAPE
Errorsframe[,20] <- MSFE
# Plot predictions vs. actual data
library(ggplot)

# Rolling Predictions Plots
new <- c(z[1:(length(z)-x)],predicc[,1]) # real data + predicted values  We need as many as horizons
new2 <- c(z[1:(length(z)-x)],predicc[,2]) # real data + predicted values
new3 <- c(z[1:(length(z)-x)],predicc[,3]) # real data + predicted values
new4 <- c(z[1:(length(z)-x)],predicc[,4]) # real data + predicted values
df0 <- as.data.frame(strptime(datos[,1],format="%Y%m%d"))
df1 <- as.data.frame(z)
df2 <- as.data.frame(new)
df3 <- as.data.frame(new2)
df4 <- as.data.frame(new3)
df5 <- as.data.frame(new4)

DF <- cbind(df0,df1,df2,df3,df4,df4)
colnames(DF) <- c("date","real","pred_h1","pred_h2","pred_h3","pred_h4")

par(mfrow=c(1,1))

jpeg("Model 5 : 1 Horizons (Rolling)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h1),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 5 : 1 Horizon (Rolling)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[1],3), ' MSFE: ',round(MSFE[1],3)), caption="SARIMA (1,1,0)x(2,1,0) s = 4");
dev.off()

jpeg("Model 5 : 2 Horizons (Rolling)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h2),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 5 : 2 Horizons (Rolling)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[2],3), ' MSFE: ',round(MSFE[2],3)), caption="SARIMA (1,1,0)x(2,1,0) s = 4");
dev.off()

jpeg("Model 5 : 3 Horizons (Rolling)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h2),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 5 : 3 Horizons (Rolling)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[3],3), ' MSFE: ',round(MSFE[3],3)), caption="SARIMA (1,1,0)x(2,1,0) s = 4");
dev.off()

jpeg("Model 5 : 4 Horizons (Rolling)")
ggplot(DF) + geom_line(aes(x=date,y=pred_h2),color='red') + 
  geom_line(aes(x=date,y=real),color='blue') + 
  ylab('Values')+xlab('date') + ggtitle("Model 5 : 4 Horizons (Rolling)") +
  labs(subtitle=paste('MAPE: ',round(MAPE[4],3), ' MSFE: ',round(MSFE[4],3)), caption="SARIMA (1,1,0)x(2,1,0) s = 4");
dev.off()