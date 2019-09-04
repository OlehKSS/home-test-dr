**Question 1**

*Imagine the credit risk use case above. You have two models: a logistic regression
model with an F1-score of 0.60 and a neural network with an F1-score of 0.63. Which
model would you recommend for the bank and why?*

I would recommend for the bank to use the logistic regression model, since the
difference between the performance of these two models is around 5%. The logistic
regression model is relatively simpler than the neural network one, it is more
resilient to overfitting, it allows an interpretation of the model parameters, and
can be used to create a variable selection scheme to remove unnecessary features.

**Question 2**

*A customer wants to know which features matter for their dataset. They have several
models created in the DataRobot platform such as random forest, linear regression, and
neural network regressor. They also are pretty good at coding in Python so wouldn't
mind using another library or function. How do you suggest this customer gets feature
importance for their data?*

For the random forests model, the customer can determine feature importance
based on how much, in average, each feature decreases the weighted impurity in the model.
Features can be ranked based on these values.

The customer can analyze linear model coefficients to see which of them have the biggest
positive or negative influence on the final output and rank their importance according to
those values.

Permutation importance can be utilized for all the three models, but it will require additional trainings
on the shuffled data for the selected classifiers.
