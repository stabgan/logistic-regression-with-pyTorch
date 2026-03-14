# Logistic Regression

### Logistic regression predicts the probability of an outcome that can only have two values (i.e. a dichotomy). The prediction is based on the use of one or several predictors (numerical and categorical). A linear regression is not appropriate for predicting the value of a binary variable for two reasons:		
- A linear regression will predict values outside the acceptable range (e.g. predicting probabilities
outside the range 0 to 1)
- Since the dichotomous experiments can only have one of two possible values for each experiment, the residuals will not be normally distributed about the predicted line.
### On the other hand, a logistic regression produces a logistic curve, which is limited to values between 0 and 1. Logistic regression is similar to a linear regression, but the curve is constructed using the natural logarithm of the “odds” of the target variable, rather than the probability. Moreover, the predictors do not have to be normally distributed or have equal variance in each group.		
![](http://www.saedsayad.com/images/LogReg_1.png)

In the logistic regression the constant (b0) moves the curve left and right and the slope (b1) defines the steepness of the curve. By simple transformation, the logistic regression equation can be written in terms of an odds ratio.
![](http://www.saedsayad.com/images/Logistic_odd.png)
		
Finally, taking the natural log of both sides, we can write the equation in terms of log-odds (logit) which is a linear function of the predictors. The coefficient (b1) is the amount the logit (log-odds) changes with a one unit change in x. 
![](http://www.saedsayad.com/images/Logit.png)

As mentioned before, logistic regression can handle any number of numerical and/or categorical variables.
![](http://www.saedsayad.com/images/LogReg_eq.png)

There are several analogies between linear regression and logistic regression. Just as ordinary least square regression is the method used to estimate coefficients for the best fit line in linear regression, logistic regression uses maximum likelihood estimation (MLE) to obtain the model coefficients that relate predictors to the target. After this initial function is estimated, the process is repeated until LL (Log Likelihood) does not change significantly. 	

![](http://www.saedsayad.com/images/LogReg_mle.png)

--------------------------------------------------------------------------------------------------------------------

#### Here I used my model in the MNIST dataset .

I previously applied logistic regression in scikit learn , but doing it in pyTorch let me explore much more.

The Accuraacy of my project is :

![](https://image.ibb.co/nnc7fn/Screen_Shot_2018_02_15_at_9_20_01_PM.png)
