# Natural-Language-Processing-NLP---Quality-Evaluation-System-based-on-Comments
In this project we will use linear regression to predict the popularity of comments on Reddit. Reddit (www.reddit.com) is a popular website where users can form interest-based communities, post content (e.g., images, links to news articles), and participate in thread-based discussions.

# ABSTRACT
In this project we’re going to apply machine learning algorithms on a dataset that is provided by reddit.com. The datasets have different features such as children, controversiality, is_root and text. Our task is to figure out a model that will predict the popularity of a comment. we’ll start by defining linear regression algorithms-closed form and gradient descent and compare the results by implementing them considering, different features and different parameters for each task. Our project is divided into three tasks: first, we will split our dataset into training, validation and test datasets and extract the desired features for every task. Then, we’ll be implementing the closed form and gradient descent algorithms. Finally, we’ll compare the results of these algorithms on the validation set to check the performance and stability of our models. Getting the best model that we have and running it on the test set will examine our trained model when it comes to unseen dataset. We found that the gradient descent approach was slower than the closed-form approach for the dataset provided and we analysed how decay plays a role in gradient descent.
# RESULTS
Task 1
After implementing the two algorithms, we start by taking into consideration the first three features (children, is_root, controvasilty), and compare the performance and accuracy of these algorithms by calculating the runtime and the value of errors.
Gradient descent
In the below table, we are changing the value of η and β to get different decay learning values, then we calculate the runtime, loss and steps for each pair values of η and β.

Table 1
β	η	Runtime	Steps
10e-3	10e-2	0.01847	117
10e-3	10e-3	0.018992	175
10e-3	10e-4	It takes a long time	It takes a long time
10e-3	10e-5	1.800957918	36333
10e-3	10e-6	It takes a long time	It takes a long time
10e-3	10e-7	It takes a long time	It takes a long time
10e-4	10e-2	0.04224681	118
10e-4	10e-3	0.0176391	190
10e-4	10e-4	0.0517199	608
10e-4	10e-5	0.3444151	6604
10e-4	10e-6	9.12999901	186161
10e-5	10e-2	0.0242769	118
10e-5	10e-3	0.0203499	190
10e-5	10e-4	0.0482759	590
10e-5	10e-5	0.2966492	5581
10e-5	10e-6	2.437783	47315











As we can see from the above table, for 10e-4 and 10e-5 pair of η and beta we’re having the best performance with 9.9914e-9 error and 0.3444151 runtime.
Mean square error = 1.34161955 
The Stability of closed form does not depend on any hyperparameters. It is straightforward calculation. But, the stability of Gradient descent mainly depends on the learning rate value which in turn depends on the hyperparameters eta and beta in our case as shown in [3]. When the learning rate is too big, it will not reach the local minimum because it just bounces back and forth between the convex function of gradient descent. If learning rate is very small, gradient descent will eventually reach the local minimum, but it will take too much time. Please refer to table 1 to see the number of steps and runtime taken for different values of η and β.
 
Comparing the runtime and stability (for values mentioned above β=10e-5 and η=10e-4) of gradient descent and closed form algorithms, we can conclude that the closed form has less MSE (better performance) and runtime than the gradient descent method.
Table 2
TRAINING	VALIDATION
Closed Form:
MSE= 1.084674
Runtime= 0.214557	
MSE=1.011749
Runtime= 0.2057149
Gradient Descent:
MSE=1.34161955
Runtime= 0.3444151	
MSE=1.27693395
Runtime= 18.5808629
N.B: we’re taking the best values of η and β that we got in Table 1
From the above results, we can notice that the runtime of gradient descent is more since we are calculating the decay learning rate for each step. Moreover, the closed have a better accuracy since the loss is less than in gradient descent.
Task 2
Below are the results of closed form algorithm.
Table 3
TRAINING	VALIDATION
NO TEXT:
MSE= 1.084674
Runtime= 0.214557	
MSE=1.011749
Runtime= 0.2057149
60 words:
MSE=1.061161
Runtime= 0.275542	
MSE=0.904507
Runtime= 0.26471018
160 words:
MSE=1.0467629
Runtime= 0.284834	
MSE=0.912364
Runtime=0.365887

For all features, no text feature and 60 frequent word feature, MSE on validation is slightly better than the training set. It is not entirely possible all the time but here we have values on validation due to some random noise.  In these cases, the model neither overfits nor underfits.
Task 3
Below are the values of MSE for closed form applied on training set after adding the two new features;
Closed_without new feature	1.0467629
Closed_ with square feature	1.0005977
Closed_ with exp feature	1.0366922
Closed with both new feature	1.0003905

We can see that the performance of the model improved by more than 4% after adding the two new features.
Task 4:
Now we will run our model after adding the two new features on the test set to see how generalise is the model and how accurate it is for an unseen dataset.
Below are the values of MSE for closed form applied on training, validation and test set after adding the two new features;
Table 5
Closed_training	1.0003905
Closed_validation	0.756792
Closed_test	1.0185782

