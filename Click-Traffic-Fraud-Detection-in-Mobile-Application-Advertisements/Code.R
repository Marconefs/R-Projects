setwd("C:/Projects")
set.seed(42)

# Import necessary libraries
library(e1071)
require(lubridate)
library(readr)
library(dplyr)
library(plyr)
library(caret)
library(ROCR) 
library(ggplot2)
library(gridExtra)
library(randomForest)
library(e1071)

# Read .csv file from zipped folder
folder <- "C:/CursoDSA/BigDataAzure/Projetos/dataset_project1.zip"
con <- unz(folder,"dataset_project1/train_sample.csv")
train <- read_csv(con)
View(head(train))

# Convert time to seconds
train$click_time <- as.numeric(train$click_time)

# Check if there is NA values on each colum
for(i in 1:length(train)){if(any(is.na(train[, i]))){print(i)}}

# Drop attributed time column as it is not going to be used 
train <- train[,-7]

# Transform column to factor data type
train$is_attributed <- as.factor(train$is_attributed)

#================================= DATA EXPLORATION ================================= 

# Function that plots 2 graphs: Donwload rate by the total of clicks (upper graph), and total clicks (bellow graph)
plot_graph <- function(feature,limit_n){
  item <- train%>%filter(is_attributed==1)%>%group_by_at(feature)%>%group_keys()
  df1 <- data.frame(item=item[[1]] ,
                  downloads=train%>%filter(is_attributed==1)%>%group_by_at(feature)%>%group_size())
  item <- train%>%filter(is_attributed==0)%>%group_by_at(feature)%>%group_keys()
  df2 <- data.frame(item=item[[1]],
                  no.downloads=train%>%filter(is_attributed==0)%>%group_by_at(feature)%>%group_size())
  df <- merge.data.frame(df1, df2, by.y=0)
  df <- df%>%mutate(total=no.downloads+downloads)%>%mutate(rate=100*downloads/total)
  p1 <- ggplot(df,aes(x=item, y=rate)) + geom_bar(stat="identity") + xlim(0, limit_n) + ylab("% down/total") + theme(axis.title.x=element_blank(),
                                                                                                             axis.text.x=element_blank())
  p2 <- ggplot(df,aes(x=item, y=total)) + geom_bar(stat="identity") + xlim(0, limit_n) + xlab(feature)
  grid.arrange(p1, p2, ncol = 1)
}
# Plot function for app
plot_graph("app",150)
# Plot function for os
plot_graph("os",70)
# Plot function for channel
plot_graph("channel",150)

# Labels proportion
prop.table(table(train$is_attributed))

#================================= FEATURE ENGINEERING ================================= 

variables = c("ip","app","device","os","channel")

# create colums with the time since the previous click by groups
for(variable in variables){
  train <- arrange_at(train,c(variable,"click_time"))
  train[[paste("delay_last_",variable,sep="")]] <- c(0,sapply(seq(2,length(train$click_time)), 
       function(i){ifelse(train$ip[i]==train$ip[i-1],train$click_time[i]-train$click_time[i-1],0)}))
}

# create colums with the runned time since the first and last click by groups
for(variable in variables){
  train <- train %>% group_by_at(variable) %>% mutate(time_first = min(click_time))
  train[[paste("delay_first_",variable,sep="")]] <- train$click_time - train$time_first
}

# Drop time_first colum
train <- train%>%subset(select=-time_first)

#=================================  NORMALIZATION ================================= 

train[-7] <- sapply(train%>%select(-is_attributed), function(x){return ((x - min(x)) / (max(x) - min(x)))})
View(head(train))

#=================================  FEATURE SELECTION ================================= 

train$is_attributed <- as.numeric(train$is_attributed)
control <- rfeControl(functions = rfFuncs, method = "cv", 
                      verbose = FALSE, returnResamp = "all", 
                      number = 6)
results.rfe <- rfe(x = train%>%subset(select=-is_attributed), 
                   y = as.matrix(train%>%subset(select=is_attributed)), 
                   sizes = 1:10, 
                   rfeControl = control)
results.rfe
varImp((results.rfe))

# Select 10 top features
train_sel <- train%>%ungroup()%>%select(-c(delay_last_device,delay_last_channel,delay_last_app,
                                   delay_last_os,ip,delay_last_ip))
View(head(train_sel))

#=================================  TRAINING ================================= 

train_sel$is_attributed <- as.factor(train_sel$is_attributed)

# Split data
index <- createDataPartition(train_sel$is_attributed,p=0.7,list=FALSE)
train_data <- train_sel[c(index),]
test_data <- data.frame(train_sel)[-index,]

# Train models with up sampling method for data balance
# Random Forest model:
ctrl <- trainControl(method = "repeatedcv", 
                     number = 5, 
                     repeats = 5, 
                     verboseIter = FALSE,
                     sampling = "up")
model_rf <- caret::train(is_attributed ~ .,
                      data = train_data,
                      method = "rf",
                      trControl = ctrl)
final_rf <- data.frame(actual = test_data$is_attributed,
                    predict(model_rf, newdata = test_data))

cm_rf_test <- confusionMatrix(final_rf$predict, test_data$is_attributed)
cm_rf_test
cm_rf_test <- confusionMatrix(final_rf$predict, test_data$is_attributed)
cm_rf_test

# Knn model:
model_knn <- caret::train(is_attributed ~ .,
                         data = train_data,
                         method = "knn",
                         trControl = ctrl)
final_knn <- data.frame(actual = test_data$is_attributed,
                       predict(model_knn, newdata = test_data))

cm_knn_test <- confusionMatrix(final_knn$predict, test_data$is_attributed)
cm_knn_test