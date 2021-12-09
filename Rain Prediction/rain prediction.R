# Name: Tashi Phuntsho
# Registration Number: 19BDS0169.
# Rain Prediction Data collection.
# Last updated on : 10.11.2021
# Programming for Data Science.
# Target Location : Vellore.

# Task: Given the current weather information of Vellore City we will predict whether it will rain in the next
# few moments or not using Decisioin tree classifier. 

# Methodology for data collection:
# Using open weatather map API, the current conditions informations are collected in JSON format. 
#The json data collected are then converted into a R data frame and written into a csv file.
# Using windows task scheduler the r script to collect data will be executed in every 10 - 15 minutes interval.

# Data Description: 
# As of now the data set consists of 28 columns and 200 rows(may change during modelling)
# The data set is collected over a period of 2 weeks from Open Weather Map, Current Conditions API for Vellore City.
# It contains the measures of weather condition features such as temperature, pressure, humidity, wind speed , etc. 


# Column Description:
# 1. 

# Data Source.
# open weather map API.
# since the API returns all the necessary information for rain prediction like temperature, wind,humidity and other
# weather infromation, we will use this API to collect data for our prediction.


# Column Description:
# 1.coord.lon - Longitude.

# 2.coord.lat - Latitude.

# 3.weather.id - code to describe various weather conditions(eg. 800 for clear sky, 804 for clouds, 500 for light 
# rain, 504 for heavy intense rainfall) 

# 4.weather.main - scenario/main weather condition eg. rainy, cloudy

# 5.weather.description - description of weather.main in terms of magnitude. i.e. if weather.main is rainy 
# weather.description may be heavy rainfall or light rainfall.

# 6.weather.icon - weather icon i.e. image(emote) of sun to describe sunny weather.

# 7.base 

# 8. main.temp - current temperature in farenheit

# 9.main.temp_min , 10. main.temp_max = temp_min and temp_max reflect deviations in measurement for the given city. 
# If they are equal to temp then either all weather stations are perfect or there is just one for the whole city.

# 11.main.pressure - pressure

# 12.main.humidity - humidity.

# 13.main.sea_level - Atmospheric pressure on the sea level, hPa

# 14.main.grnd_level - Atmospheric pressure on the ground level, hPa

# 15.main.visibility - farthest distance at which an object can be seen clearly (measured in metres)

# 16.wind.speed -  Wind speed. Unit: meter/sec

# 17.wind.deg - Wind direction, degrees (meteorological).

# 18.wind.gust - Wind gust. Unit: meter/sec.

# 19.dt - data collection date and time utc.

# 20.date_time - data collection time system(local time)

# 21.sys.country - country code.

# 22.sys.sunrise , sys.sunset - sunrise and sun set time utc.

# 24.name- city name.

# 25.id - city id.

# 26.main.feels_like - emperature. This temperature parameter accounts for the human perception of weather. 
# Unit: Kelvin.


# Data Collection Script: 
#library(jsonlite)
#url_api <- "https://api.openweathermap.org/data/2.5/weather?q=Vellore&appid=06e0f53091ffbec4f4ce8f53eb4a7b32"
#content <- fromJSON(url_api)
#date_time <- Sys.time()
#df <- as.data.frame(content)
#df1 <-cbind(df,date_time)
#write.table(df1,'19BDS0169.csv',sep = ',',append = T,row.names = F,col.names = F)
#cat("Successfully executed")




# importing all the necessary libraries:
library(base)
library(stats)
library(utils)
library(graphics)
library(dplyr)
library(caret)
library(caTools)
library(rpart)


# DATA PREPROCESSING AND CLEANING R SCRIPT ON TEST DATASET:


#data <- read.csv('https://raw.githubusercontent.com/anthoniraj/dsr_datasets_basic/main/19BDS0169.csv')
#head(data)
# handling the misplaced columns.
# The rows in which the value of X1h is missing lead to misplaced dateset. 

#data1 <- data %>% filter(data$X1h >= 30)
#data2 <- data%>% filter(data$X1h <30)

# renaming the columns of df1

#names(data1) <- c("X","lon","lat","weather_id","main","weather_description","weather_icon","base","temp","feels_like","temp_min","temp_max","pressure","humidity","sea_level","ground_level","visibility","wind_speed","wind_deg","wind_gust","all","dt","sunrise_time","country","sunset_time","timezone","id","name","cod","date_time","X1h")

#head(data1)

# dropping the X1h column both from df1 and df2

#data1 <- data1 %>% select( subset = -c(X1h))
#data2 <- select(data2,subset = -c(X1h))
# renaming the columns of df2.
#names(data2) <- c('X','lon','lat','weather_id','main','weather_description','weather_icon','base','temp','feels_like','temp_min','temp_max','pressure','humidity','sea_level','ground_level','visibility','wind_speed','wind_deg','wind_gust','all','dt','country','sunrise_time','sunset_time','timezone','id','name','cod','date_time')
#head(data1)
#head(data2)
# making the arrangement of columns in data1 and data2 same
#data2 <- data2[,names(data1)]
# merging the two data frames.
#df <- rbind(data1,data2) 

# converting the date time column to R date time object.

#df$date_time <- as.POSIXct(df$date_time, format = '%Y-%m-%D %H:%M:%S')


# sorting the data based on date time.
#df <- df[order(df$date_time),]

# removing duplicate values based on time of data collection.
#df$minute <- format(df$date_time, format = "%Y-%m-%D %H:%M")

#df <- distinct(df,minute,.keep_all = TRUE) 

# Assigning class label to the data frame.
#df$class_label <- 0
#for (i in 1:nrow(df)){
#  if(i < nrow(df)){
#    label <- df$main[i+1]

#  }
#  else{
#    label <- df$main[i]
#  }
#  if(label =='Rain'){
#    df$class_label[i] <- "YES"
#  }
#  else{
#    df$class_label[i] <- "NO"
#  }

#}



# MODEL DEVELOPMENT ON THE FINAL DATASET
df <- read.csv('https://raw.githubusercontent.com/anthoniraj/dsr_datasets_final/main/19BDS0169.csv')


reqd_features <- c('weather_description','temp','feels_like','pressure','humidity','visibility','wind_speed','wind_deg','wind_gust','class_label')
df1 = df[reqd_features]
df1$class_label = as.factor(df1$class_label)
# handling imbalanced dataset using upsampling 

df2 = upSample(df1, df1$class_label)
df2  = subset(df2, select= -c(class_label))
# encoding the weather description column using label encoding.
df2$weather_description<- factor(df2$weather_description, levels = c('overcast clouds','scattered clouds','broken clouds','light rain','moderate rain','heavy intensity rain','very heavy rain') , labels = c(0,1,2,3,4,5,6))

(sum(is.na(df2)))
# the dataset has no missing values. 
# splitting the dataset into training and testing set. 
split = sample.split(df2, 0.8)
training_set = subset(df2, split == TRUE)
testing_set = subset(df2, split == FALSE)

# Fitting decision tree classifier on the training set. 

model = rpart(Class ~ ., data = training_set , method = "class")

test_samples = subset(testing_set, select = -c(Class))
predicted_labels = predict(model,test_samples, type = 'class')

confusion_matrix = table(testing_set$Class,predicted_labels)
print("Confusion Matrix : ")
(confusion_matrix)

model_evaluation <- function(conf_matrix){
  TN <- conf_matrix[1][1]
  TP <- diag(conf_matrix)[[2]]
  FN <- conf_matrix[2][1]
  FP <- sum(conf_matrix) - TN- TP- FN
  accuracy = (TP + TN)/(TP + TN + FP + FN)
  precision = TP/(TP + FP)
  recall = TP/(TP + FN)
  f1_measure = (2 * precision* recall) / (precision + recall)
  
  
  return(data.frame(TN,TP,FN,FP, accuracy, precision, recall, f1_measure))
}

mod_ev <- model_evaluation(conf_matrix = confusion_matrix)

print("Model evaluation metrics: ")
(mod_ev)


# Data Visualization. 
> library(ggplot2)
# Histogram of temperature
hist(df$temp, main = "Histogram of Temperature",xlab = "Temperature in Kelvin",ylab = "Frequency")

# plotting pie chart of weather description.
pie(c(64,42,29,16,233,12,10),labels = c('broken clouds','heavy intensity rain','light rain','moderate rain','overcast clouds','scattered clouds','very heavy rain'), col = c('violet','coral','blue','green','yellow','orange','red'),)

# Box and whister plot of pressure column.
box(df$pressure,main = "Box Plot of pressure", xlab = 'pressure in mmHg', horizontal = TRUE)

# Count plot of the output variable from the original dataset (before upsampling)
barplot(c(309,97), names.arg = c("NO","YES"), col =c("brown","blue"),main = "Count plot of class label",xlab = "Rain",ylab = "Frequency")
# plotting heatmap of the confusion matrix.

TClass <- factor(c("No",'No','Yes','Yes'))
> PClass <- factor(c('No','Yes','No','Yes'))
> Y      <- c(50, 4,12, 58)
> dframe <- data.frame(TClass, PClass, Y)
> ggplot(data =  dframe, mapping = aes(x = TClass, y = PClass)) +
+   geom_tile(aes(fill = Y), colour = "white") +
+   geom_text(aes(label = sprintf("%1.0f", Y)), vjust = 1) +
+   scale_fill_gradient(low = "blue", high = "red") +
+   theme_bw() + theme(legend.position = "none")


