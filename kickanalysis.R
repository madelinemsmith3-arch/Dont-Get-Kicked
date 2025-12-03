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


# setwd("C:\\Users\\madel\\OneDrive\\Documents\\Stat 348\\DontGetKicked\\Dont-Get-Kicked")

##################################################################

# read in the data
train <- vroom("./train.csv")
test <- vroom("./test.csv")

train <- train %>%
  mutate(IsBadBuy = as.factor(IsBadBuy))

head(train)


###########################################################################

# xgboost

library(xgboost)

train <- train %>%
  mutate(across(where(is.character), ~ na_if(.x, "?"))) %>%
  mutate(across(where(is.character), ~ na_if(.x, "")))

test <- test %>%
  mutate(across(where(is.character), ~ na_if(.x, "?"))) %>%
  mutate(across(where(is.character), ~ na_if(.x, "")))

char_to_num <- c(
  "MMRCurrentAuctionAveragePrice",
  "MMRCurrentAuctionCleanPrice",
  "MMRCurrentRetailAveragePrice",
  "MMRCurrentRetailCleanPrice"
)

train <- train %>%
  mutate(across(all_of(char_to_num), ~ as.numeric(.)))

test <- test %>%
  mutate(across(all_of(char_to_num), ~ as.numeric(.)))

train <- train %>%
  mutate(across(where(is.character), ~ as.factor(.)))

test <- test %>%
  mutate(across(where(is.character), ~ as.factor(.)))

clean_levels <- function(x) {
  x %>% 
    str_replace_all("[^A-Za-z0-9]", "_") %>%   # convert weird chars to _
    str_replace_all("_+", "_") %>%            # collapse repeats
    str_replace_all("^_|_$", "") %>%          # trim edges
    na_if("")                                 # blank → NA
}

train <- train %>%
  mutate(across(where(is.factor), clean_levels))

test <- test %>%
  mutate(across(where(is.factor), clean_levels))




set.seed(123)
kick_folds <- vfold_cv(train, v = 5, strata = IsBadBuy)

kick_recipe <- recipe(IsBadBuy ~ ., data = train) %>%
  update_role(RefId, new_role = "ID") %>%   # do NOT model RefId
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors())

xgb_model <- boost_tree(
  trees = tune(),
  learn_rate = tune(),
  mtry = tune(),
  tree_depth = tune(),
  loss_reduction = tune(),
  min_n = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

kick_wf <- workflow() %>%
  add_recipe(kick_recipe) %>%
  add_model(xgb_model)

xgb_grid <- grid_latin_hypercube(
  trees(),
  learn_rate(),
  mtry(range = c(5, 50)),
  tree_depth(range = c(3, 10)),
  loss_reduction(),
  min_n(),
  size = 20
)

set.seed(123)
xgb_tuned <- tune_grid(
  kick_wf,
  resamples = kick_folds,
  grid = xgb_grid,
  metrics = metric_set(roc_auc)
)

best_xgb <- select_best(xgb_tuned, "roc_auc")

final_wf <- finalize_workflow(kick_wf, best_xgb)

final_fit <- final_wf %>%
  fit(data = train)

test_pred <- predict(final_fit, new_data = test, type = "prob") %>%
  bind_cols(test %>% select(RefId)) %>%
  transmute(RefId, IsBadBuy = .pred_1)

write_csv(test_pred, "xgboost")


###########################################################################

### support vector machines

train <- train %>%
  mutate(across(where(is.character), ~ na_if(.x, "?"))) %>%
  mutate(across(where(is.character), ~ na_if(.x, "")))

test <- test %>%
  mutate(across(where(is.character), ~ na_if(.x, "?"))) %>%
  mutate(across(where(is.character), ~ na_if(.x, "")))

char_to_num <- c(
  "MMRCurrentAuctionAveragePrice",
  "MMRCurrentAuctionCleanPrice",
  "MMRCurrentRetailAveragePrice",
  "MMRCurrentRetailCleanPrice"
)

train <- train %>%
  mutate(across(all_of(char_to_num), ~ as.numeric(.)))

test <- test %>%
  mutate(across(all_of(char_to_num), ~ as.numeric(.)))

train <- train %>%
  mutate(across(where(is.character), ~ as.factor(.)))

test <- test %>%
  mutate(across(where(is.character), ~ as.factor(.)))


kick_recipe <- recipe(IsBadBuy ~ ., data = train) %>%
  update_role(RefId, new_role = "ID") %>%
  step_mutate(
    PurchDate = mdy(PurchDate),
    PurchaseYear = year(PurchDate),
    PurchaseMonth = month(PurchDate)
  ) %>%
  step_rm(PurchDate) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors())

svm_model <- svm_rbf(
  cost = tune(),
  rbf_sigma = tune()) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

kick_wf <- workflow() %>%
  add_recipe(kick_recipe) %>%
  add_model(svm_model)

set.seed(123)
kick_folds <- vfold_cv(train, v = 5, strata = IsBadBuy)

svm_grid <- grid_regular(
  cost(range = c(-5, 5)),
  rbf_sigma(range = c(-5, 1)),
  levels = 6
)

svm_tuned <- tune_grid(
  kick_wf,
  resamples = kick_folds,
  grid = svm_grid,
  metrics = metric_set(roc_auc)
)

best_svm <- select_best(svm_tuned, "roc_auc")
best_svm

final_fit <- finalize_workflow(kick_wf, best_svm) %>%
  fit(train)

test_preds <- predict(final_fit, test, type = "prob") %>%
  bind_cols(test %>% select(RefId))

submission <- test_preds %>% 
  transmute(RefId, IsBadBuy = .pred_1)

write_csv(submission, "svm_rbf")


##########################################################################


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












