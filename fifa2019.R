# Read data
  
library("readr")
fifa19 <- as.data.frame(read_csv("./data/fifa.csv"))

# Data Preprocessing
## Transform `Value` into a standard numeric.

fifa19$Value <- substr(fifa19$Value,2,200)
fifa19$ValueNum <- sapply(as.character(fifa19$Value), function(x) {
  unit <- substr(x, nchar(x), nchar(x))
  if (unit == "M") return (as.numeric(substr(x, 1, nchar(x)-1)) * 1000000)
  if (unit == "K") return (as.numeric(substr(x, 1, nchar(x)-1)) * 1000)
  as.numeric(x)
})
rownames(fifa19) <- make.names(fifa19$Name, unique = TRUE)

# Feature selection
##Let's select only features related to player characteristics.

fifa19_selected <- fifa19[,c(4, 8, 14:18, 55:88, 90)]
fifa19_selected$`Preferred Foot` <- factor(fifa19_selected$`Preferred Foot`)

# Feature engineering
##Value is skewed. Will be much easier to model sqrt(Value).

fifa19_selected$ValueNum <- sqrt(fifa19_selected$ValueNum)
fifa19_selected <- na.omit(fifa19_selected)
colnames(fifa19_selected) <- make.names(colnames(fifa19_selected))

# Create a gbm model
## Let's use `gbm` library to create a `gbm` model with 250 trees 3 levels deep.

set.seed(1313)
library("gbm")
fifa_gbm <- gbm(ValueNum~.-Overall,
                data = fifa19_selected, 
                n.trees = 250, 
                interaction.depth = 3)

# Create a DALEX explainer
## Let's wrap gbm model into a DALEX explainer.

library("DALEX")
fifa_gbm_exp <- explain(fifa_gbm, 
                        data = fifa19_selected, 
                        y = fifa19_selected$ValueNum^2, 
                        predict_function = function(m,x){predict(m, x, n.trees = 250)^2})

# Feature Importance explainer
## Calculate Feature Importnace explainer.

library("ingredients")
fifa_feat <- ingredients::feature_importance(fifa_gbm_exp)
plot(fifa_feat, max_vars = 12)




# Partial Dependency explainer
## Calculate Partial Dependency explainer.

fifa19_pd <- ingredients::partial_dependency(fifa_gbm_exp, variables = "Age")
plot(fifa19_pd)

library("ggplot2")
library("scales")
plot(fifa19_pd) +  
  scale_y_continuous(labels = dollar_format(suffix = "€", prefix = ""), name = "Estimated value", limits = 1000000*c(0.1,3), breaks = 1000000*c(0.1,1,2, 3))


# Break Down explainer
## Calculate Break Down explainer.

library("iBreakDown")
fifa_cr_gbm <- break_down(fifa_gbm_exp, new_observation = fifa19_selected["Cristiano.Ronaldo",])
plot(fifa_cr_gbm)

fifa_cr_gbm$label = "Break Down for Cristiano Ronaldo (GBM model)"
plot(fifa_cr_gbm, digits = 0) +  
  scale_y_continuous(labels = dollar_format(suffix = "€", prefix = ""), name = "Estimated value", limits = 10000000*c(0.1,10), breaks = 10000000*c(2.5,5,7.5,10))


fifa_ws_gbm <- break_down(fifa_gbm_exp, new_observation = fifa19_selected["W..Szczęsny",])
fifa_ws_gbm$label = "Break Down for Wojciech Szczęsny (GBM model)"
plot(fifa_ws_gbm, digits = 0) +  
  scale_y_continuous(labels = dollar_format(suffix = "€", prefix = ""), name = "Estimated value", limits = 10000000*c(0.1,4), breaks = 10000000*c(0.1, 2, 4))


# Ceteris Paribus explainer
## Calculate Ceteris Paribus explainer.

fifa19_cp_pg <- ingredients::ceteris_paribus(fifa_gbm_exp, new_observation = fifa19_selected["Cristiano.Ronaldo",], variables = "Age", variable_splits = list(Age = seq(18,45,0.1)))
plot(fifa19_cp_pg) +  
  scale_y_continuous(labels = dollar_format(suffix = "€", prefix = ""), name = "Estimated value", limits = 10000000*c(6,12), breaks = 10000000*c(6, 7.5, 9, 10.5, 12))




# modelStudio app
## Calculate modelStudio application.

library("modelStudio")
options(
    parallelMap.default.mode        = "socket",
    parallelMap.default.cpus        = 8,
    parallelMap.default.show.info   = FALSE,
    digits = 0
)
fifa19_ms <- modelStudio(fifa_gbm_exp, 
                         new_observation = fifa19_selected[c("L..Messi","R..Lewandowski", "W..Szczęsny", "P..Gulácsi","A..Szalai", "Cristiano.Ronaldo", "Neymar.Jr"), ], B = 5, digits = 0)
r2d3::save_d3_html(fifa19_ms, file = "fifa19.html")

