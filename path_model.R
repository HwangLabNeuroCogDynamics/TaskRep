# Do path model to look at structure of representations (Figure 5)

library(lavaan)

#####
df <- read.csv('Data/singel_trial_RSA_betas.csv')
model1 <- 'level: 1
rt ~ c*probe_task_b + e*context_b
probe_task_b ~ a*feature_b + b*identity_b
feature_b ~ d*context_b
identity_b ~ context_b + feature_b
diff_feature_vs_identity_on_task := a-b
indrect := c*a*d
total := e + (c*a*d)
diff := indrect - total
level: 2
rt ~~ probe_task_b + identity_b + feature_b + context_b
probe_task_b ~~ feature_b + identity_b + context_b
identity_b ~~ context_b + feature_b
feature_b ~~ context_b
'
model1<- sem(model1, data = df, cluster ='subject')
summary(model1, standardized = TRUE)
fitMeasures(model1)

model2 <- 'level: 1
rt ~ probe_task_b
probe_task_b ~ identity_b
feature_b ~~ feature_b
context_b ~~ context_b
level: 2
rt ~~ probe_task_b + identity_b + feature_b + context_b
probe_task_b ~~ feature_b + identity_b + context_b
identity_b ~~ context_b + feature_b
feature_b ~~ context_b
'

model2<- sem(model2, data = df, cluster = 'subject')
summary(model2)
fitMeasures(model2)

anova(model1, model2)

model1 <- 'level: 1
rt ~ c*probe_task_b
probe_task_b ~ a*feature_b + b*identity_b
feature_b ~ d*context_b + identity_b
context_b ~ identity_b
diff_feature_vs_identity_on_task := a-b
level: 2
rt ~~ probe_task_b + identity_b + feature_b + context_b
probe_task_b ~~ feature_b + identity_b + context_b
identity_b ~~ context_b + feature_b
feature_b ~~ context_b
'
model1<- sem(model1, data = df, cluster ='subject')
summary(model1, standardized = TRUE)

####