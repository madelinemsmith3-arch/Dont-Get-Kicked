library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)
library(DataExplorer)
library(dplyr)
library(glmnet)
library(rpart)
library(ranger)
library(bonsai)
library(lightgbm)
library(agua)
library(ggplot2)
library(embed)
library(recipes)
library(discrim)
library(themis)
library(workflows)
library(kernlab)
library(themis)
library(lme4)

####### score > 0.235


# setwd("C:\\Users\\madel\\OneDrive\\Documents\\Stat 348\\DontGetKicked")

##################################################################

# read in the data
train <- vroom("./train.csv")
test <- vroom("./test.csv")

train <- train %>%
  mutate(IsBadBuy = as.factor(IsBadBuy))

###########################################################################


#### regression trees #####

my_recipe <- recipe(IsBadBuy ~ ., data = train) %>%
  step_mutate_at(all_predictors(), fn=factor) %>%
  step_lencode_mixed(all_factor_predictors(), outcome = vars(IsBadBuy)) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) 

prepped <- prep(my_recipe, verbose = TRUE)
new_data <- bake(prepped, new_data = NULL)


my_mod <- rand_forest(mtry  = 1, min_n = 10,trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod) %>%
  fit(data=train)

preds <- predict(wf, new_data = test, type = "prob")

head(preds)


final <- preds %>% dplyr::select(.pred_1)
colnames(final)[1] <- "IsBadBuy"

kaggle_submission <- bind_cols(test %>% select(Id), final) %>%
  rename(IsBadBuy = IsBadBuy)

# Write to CSV
vroom_write(kaggle_submission, file = "./rand.forest.csv", delim = ",")



############################ EDA #############################################

### Check for imbalancing
train %>%
  count(IsBadBuy) %>%
  ggplot(aes(x = IsBadBuy, y = n, fill = IsBadBuy)) +
  geom_col() +
  geom_text(aes(label = n), vjust = -0.5) +
  scale_fill_viridis_d() +
  labs(title = "Class balance: IsBadBuy", y = "Count", x = NULL)

##### boxplot ######
ggplot(train, aes(x = IsBadBuy, y = VehicleAge, fill = IsBadBuy)) +
  geom_violin(trim = FALSE) +
  scale_fill_viridis_d() +
  labs(title = "VehicleAge by IsBadBuy")

##### mosaic plot ########

ggplot(data=train) + geom_mosaic(aes(x=product(Make), fill=IsBadBuy))












