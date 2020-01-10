## afinidata recommendation engine

We have developed an *ad hoc* recommendation engine that delivers activities implementing our approach to how children should best develop their skills. At the core of the recommendation engine is a collaborative filtering model, with explicitly parameterized mean and user and item bias. The output of the engine however is not any top ranked set of activities as predicted by the model, as we shall describe later.

### Model specification

The items are structured into three different categories 1, 2 and 3. Based on the ratings $r_{iu}$ that the user $u$ has given to item $i$, which may belong to either of categories A, B and C, we do the following:

- we compute the empirical means $r_u=\{r_u^1, r_u^2, r_u^3\}$, normalize these three numbers by computing $\hat{r}_u^A = (r_u^A - \text{mean}(r_u))/\text{std}(r_u)$;
- from the normalized means we compute probabilities given by $p_u^A  = \exp{(-\hat{r}_u^A)}/\sum_B \exp{(-\hat{r}_u^B)}$, which we use as the probability that an item is chosen from category A; the higher the ranking the lower the probability, that is, we favor activities from categories in which the user has given the lowest rankings in average;
- we use the aforementioned probability distribution over categories in order to select a category,
- from the chosen category we filter the activities that have A) already been ranked as well as B) activities out of the age range for the user; if no activities are left, we remove filter B, repeating already seen activities; 

### Model tests

We developed some methods that allow us to simulate the process by which a user ranks activities. This methods are contained in [model_testing.ipynb](https://github.com/paloderosa/afinidata_recommender/blob/master/model_testing.ipynb). These methods perform the following tasks:

- after the model is trained, a recommendation is generated;
- we rank the corresponding item;
- the ranking is stored and the item is considered as seen;
- a new recommendation is generated and so on.

This process eventually exhausts available activities, first from particular categories and later from all categories. When a particular category has been exhausted, we can see that such category is excluded from the domain of the probability distribution over categories. Furthermore, when all categories have been exhausted, we now select from all categories according to the probability distribution over all categories.

The addition of responses should reflect on the collaborative filtering model, which we can observe if we retrain our model.