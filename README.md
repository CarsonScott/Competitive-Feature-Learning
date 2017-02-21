# Competitive Feature Learning

## Classification

Features are sets of values that become active when a set of matching inputs is received. The error of each feature is calculated and compared to a threshold which determines the state (i.e. active or inactive) of the feature. 

The number of features that respond to a given input is limited by the class size. When an input triggers an excess number of features, the most similar features stay active while the rest become inactive. 

## Learning

Features are updated according to the input. Learning occurs in three steps: First, the thresholds are adjusted (regardless of activation) to match the similarity of each feature. Second, the weights of active features are adjusted to match the importance of each value. Finally, the values of the active features are adjusted to match the input.

![Equations](https://github.com/CarsonScott/Competitive-Feature-Learning/blob/master/img/Equations.PNG)
