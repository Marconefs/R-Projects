setwd("C:/CursoDSA/BigDataAzure/Projetos/projeto2")
set.seed(42)

# Import necessary libraries
library(e1071)
require(lubridate)
library(readr)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(caret) 
library(randomForest) 
library(arules)

# Read .csv data from zipped folder
folder2 <- "cliente_tabla.zip"
folder3 <- "producto_tabla.zip"
df <- read_csv("sampled_train.csv")
client <- read_csv(unz(folder2,"cliente_tabla.csv"))  
product <- read_csv(unz(folder3,"producto_tabla.csv"))
View(head(df))
View(head(client))
View(head(product))
length(df$Semana)

# Drop NA values presented in the Data Frame
df <- na.omit(df)

# Chane type of data for each feature/column
feat <- colnames(df)
df[feat[7:11]] <- mapply(as.numeric,df[feat[7:11]])
df[feat[1:6]] <- mapply(as.factor,df[feat[1:6]])

# Merge data frames in order to get the names of each client and product presented in the data
df <- df %>% merge(client, by = "Cliente_ID", all.x = T)
df <- df %>% merge(product, by = "Producto_ID", all.x = T)

#----------------  Data exploration -----------------
summary(df)

# Generate the following graph: Selling demand vs lost vs weeks
df_sum <- merge(df %>% group_by(Semana) %>% summarise(sold=sum(Demanda_uni_equil)) ,
                     df %>% group_by(Semana) %>% summarise(returned=sum(Dev_uni_proxima)))
                
p1 <- ggplot(data=df_sum) + geom_bar(aes(x=Semana, y=sold),stat="identity",fill = "blue", width = 0.3, position = "dodge") + 
                            theme(axis.title.x=element_blank(),axis.text.x=element_blank())
p2 <- ggplot(data=df_sum) + geom_bar(aes(x=Semana, y=returned),fill = "red", width = 0.3,stat="identity")+ labs(x="Week")
grid.arrange(p1, p2, ncol = 1)

# Generate the following graph: weeks vs (units sold)/(total units)
df_sum <- df_sum %>% mutate(rate = 100*returned/(sold+returned))
ggplot(data=df_sum, aes(x=Semana,y=rate)) + geom_line(colour="red",aes(group = 1)) + labs(y="% units Returned/Sold", x="Week")

# Show the 10 Top Clients (customers who had more products sold)
top_clients <- df %>% slice_max(n=10,order_by=Demanda_uni_equil)
ggplot( df %>% subset(Cliente_ID %in% top_clients$Cliente_ID) ) +
        geom_boxplot( aes(NombreCliente,Demanda_uni_equil), varwidth=F, fill="plum") +
        theme(axis.text.x = element_text(angle=65, vjust=0.6)) + labs( x="Client", y="Units sold")

# Show 10 top selling products
top_products <- df %>% slice_max(n=10,order_by=Demanda_uni_equil)
ggplot( df %>% subset(Producto_ID %in% top_products$Producto_ID) ) +
        geom_boxplot( aes(NombreProducto,Demanda_uni_equil), varwidth=F, fill="plum") +
        theme(axis.text.x = element_text(angle=65, vjust=0.6)) + labs( x="Product", y="Units sold")

#----------------  Feature Engineering -----------------

df_new <- df

# Add information of the last 4 weeks:
# 1 week ago
to_merge <- df_new %>% select(Producto_ID,Semana,Demanda_uni_equil)
to_merge$Semana <- to_merge$Semana %>% sapply(function(i){ as.numeric(i)+1 })
to_merge <- to_merge %>% group_by(Producto_ID,Semana) %>% summarise("previous_demanda_1" = sum(Demanda_uni_equil)) %>% subset(Semana<9)
to_merge$Semana <- to_merge$Semana %>% as.character()
df_new <- df_new %>% merge(to_merge, by = c("Producto_ID","Semana"), all.x = T)

# 2 weeks ago
to_merge <- df_new %>% select(Producto_ID,Semana,Demanda_uni_equil)
to_merge$Semana <- to_merge$Semana %>% sapply(function(i){ as.numeric(i)+2 })
to_merge <- to_merge %>% group_by(Producto_ID,Semana) %>% summarise("previous_demanda_2" = sum(Demanda_uni_equil)) %>% subset(Semana<9)
to_merge$Semana <- to_merge$Semana %>% as.character()
df_new <- df_new %>% merge(to_merge, by = c("Producto_ID","Semana"), all.x = T)

# 3 weeks ago
to_merge <- df_new %>% select(Producto_ID,Semana,Demanda_uni_equil)
to_merge$Semana <- to_merge$Semana %>% sapply(function(i){ as.numeric(i)+3 })
to_merge <- to_merge %>% group_by(Producto_ID,Semana) %>% summarise("previous_demanda_3" = sum(Demanda_uni_equil)) %>% subset(Semana<9)
to_merge$Semana <- to_merge$Semana %>% as.character()
df_new <- df_new %>% merge(to_merge, by = c("Producto_ID","Semana"), all.x = T)

# 4 weeks ago
to_merge <- df_new %>% select(Producto_ID,Semana,Demanda_uni_equil)
to_merge$Semana <- to_merge$Semana %>% sapply(function(i){ as.numeric(i)+4 })
to_merge <- to_merge %>% group_by(Producto_ID,Semana) %>% summarise("previous_demanda_4" = sum(Demanda_uni_equil)) %>% subset(Semana<9)
to_merge$Semana <- to_merge$Semana %>% as.character()
df_new <- df_new %>% merge(to_merge, by = c("Producto_ID","Semana"), all.x = T)

# Add product weight column
df_new['weight'] <- df_new$Venta_hoy /df_new$Venta_uni_hoy 

# Replace all NA values by zero in the data frame
df_new <- df_new %>% mutate(previous_demanda_1 = ifelse(is.na(previous_demanda_1),0,previous_demanda_1)) %>% 
           mutate(previous_demanda_2 = ifelse(is.na(previous_demanda_2),0,previous_demanda_2)) %>% 
           mutate(previous_demanda_3 = ifelse(is.na(previous_demanda_3),0,previous_demanda_3)) %>% 
           mutate(previous_demanda_4 = ifelse(is.na(previous_demanda_4),0,previous_demanda_4)) %>%
           mutate(weight = ifelse(is.na(weight),0,weight))

#-------------------- Normalization ----------------------------                  
       
# Change data type to Numeric for specific features and normalize them
df_norm <- df_new 
numeric.vars <- c("Venta_uni_hoy", "Venta_hoy", "Dev_uni_proxima", "Dev_proxima", "Demanda_uni_equil",
                  "previous_demanda_1", "previous_demanda_2", "previous_demanda_3", "previous_demanda_4", "weight")
for (variable in numeric.vars){
  df_norm[[variable]] <- scale(df_norm[[variable]], center=T, scale=T)
}

# --------------------- Feature Selection -------------------------------

# Drop unnecessary columns
df_norm <- df_norm%>%subset(select=-c(Dev_proxima,Dev_uni_proxima,Venta_hoy,Venta_uni_hoy,NombreCliente,NombreProducto,X1 ))

# Drop NA values
df_norm <- na.omit(df_norm)

# Change data type of specific columns to factor type
df[feat[1:6]] <- mapply(as.factor,df[feat[1:6]])

# Find best features to be used in the trainning step
summarise(df_norm)
control <- rfeControl(functions = rfFuncs, method = "cv", 
                      verbose = FALSE, returnResamp = "all", 
                      number = 5)
results.rfe <- rfe(x = df_norm%>%subset(select=-Demanda_uni_equil), 
                   y = as.matrix(df_norm%>%subset(select=Demanda_uni_equil)), 
                   sizes = 1:10, 
                   rfeControl = control)
results.rfe
varImp((results.rfe))

# Select 8 best features
df_sel <- df_norm%>%subset(select=c(Canal_ID,Producto_ID,Ruta_SAK,weight,previous_demanda_3,previous_demanda_1,previous_demanda_2,
                                    Demanda_uni_equil))

#-------------------------- Training Model ------------------------------------------

str(df_sel)

# Split data into train and test data
index <- createDataPartition(df_sel$Demanda_uni_equil,p=0.7,list=FALSE)
train_data <- df_sel[c(index),]
test_data <-  df_sel[-c(index),]

# Drop test data which its feature factors are not in train data
features = c(1,2,3)
for(f in features){
  test_data <- test_data %>% subset(test_data[,f] %in% levels(as.factor(train_data[,f])))
}

# Function that display results
show_results <- function(df_final){
  colnames(df_final) <- c("true","pred")
  df_final['resid'] <- df_final$true - df_final$pred
  
  ggplot(df_final) + geom_point(aes(x=true, y=pred))
  
  b <- c(2,1.5,1,0.5,0,-0.5,-1,-1.5,-2)
  pred <- discretize(x=df_final$pred, method="fixed", breaks=b)
  true <- discretize(x=df_final$true, method="fixed", breaks=b)
  cm <- confusionMatrix(pred,true) 
  ggplot(data = as.data.frame(cm$table), mapping = aes(x = Reference, y = Prediction)) + 
    geom_tile(aes(fill = Freq), colour = "white") +
    scale_fill_gradient(low = "white", high = "steelblue") +
    geom_text(aes(label = Freq), colour = "black") +
    ggtitle("Plot")  +
    theme_bw()
}


# Multiple Linear Regression
ctrl <- trainControl(method = "repeatedcv", 
                     number = 5, 
                     repeats = 5, 
                     verboseIter = FALSE)
model_lm <- caret::train(Demanda_uni_equil ~ .,
                         data = train_data,
                         method = "lm",
                         trControl = ctrl)
final_lm <- data.frame(actual = test_data$Demanda_uni_equil,
                       predict(model_lm, newdata = test_data))

model_lm
show_results(final_lm)


# SVM with Linear Kernel
model_svm <- caret::train(Demanda_uni_equil ~ .,
                         data = train_data,
                         method = 'svmLinear3',
                         trControl = ctrl)
final_svm <- data.frame(actual = test_data$Demanda_uni_equil,
                       predict(model_svm, newdata = test_data))

model_svm
show_results(final_svm)


# Random Forest Regression
model_rf <- caret::train(Demanda_uni_equil ~ .,
                         data = train_data,
                         method = 'cforest',
                         trControl = ctrl)
final_rf <- data.frame(actual = test_data$Demanda_uni_equil,
                       predict(model_rf, newdata = test_data))

model_rf
show_results(final_rf)
