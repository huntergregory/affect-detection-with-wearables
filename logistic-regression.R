setwd('/Users/huntergregory/OneDrive - Duke University/Documents/Duke/7th semester/case studies/affect-detection-with-wearables/')

library(ggplot2)
library(finalfit)

set.seed(57)

## Data Prep
data = read.csv('./WESAD-one-sec-shift.csv')
good_columns = c('affect', 'subject_id', 
                 'chest_EDA_slope', 
                 'wrist_EDA_slope', 'chest_RESP_breath_rate',
                 'chest_RESP_volume', 'chest_SCR_num_segments', 'wrist_SCR_num_segments',
                 'chest_ACC_magnitude_mean', 'wrist_ACC_magnitude_mean', 'chest_TEMP_mean', 'wrist_TEMP_mean', 'chest_ECG_HRV_mean', 'wrist_BVP_HRV_mean')
final_data = data[good_columns]
# ff_glimpse(final_data)
contrasts(factor(final_data$affect))

## Modeling
percent_train = 0.8
train = data.frame()
test = data.frame()
for (id in unique(final_data$subject_id)) {
  subject_df =  final_data[final_data$subject_id == id,]
  train_indices = sample(nrow(subject_df), floor(percent_train * nrow(subject_df)))
  train = rbind(train, subject_df[train_indices,])
  test = rbind(test, subject_df[-train_indices,])
}
full_logistic_model = glm(affect ~ . - subject_id, data=train, family="binomial", maxit=100)
summary(full_logistic_model)
probabilities = predict(full_logistic_model, test, type='response')
predictions = ifelse(probabilities > 0.5, 'stress', 'amusement')
mean(predictions == test$affect) # accuracy

model_with_subject_id = glm(affect ~ ., data=train, family="binomial", maxit=100)
summary(model_with_subject_id)
probabilities = predict(model_with_subject_id, test, type='response')
predictions = ifelse(probabilities > 0.5, 'stress', 'amusement')
mean(predictions == test$affect) # accuracy
