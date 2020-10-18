# ml-pipeline-recommendation-system

## System design

Let's remember the standard steps of any ML project in Prod

```
1. Design the project
2. Collect the data
3. Exploratory Data Analysis (EDA) and Benchmarking
4. Decide the tools for the project
5. Off-line testing
6. Implement the selected solution in production
7. On-line testing
8. Monitor
```

This document starts at 5. I assume we have already trained, validated and
tested the final solution (off-line testing). I also assume that we have a
potential second candidate for which we will perform an on-line (A/B) test.

We cannot fully reproduce a real-life scenario and production environment
within a company, but we can try to be as realistic as possible.

With that in mind, we will build a recommendation system as follows:
initially, we will use all but 50k interactions (for example) as if this was
the dataset we are given by the company we work for, to build the
recommendation system. The remaining 50k interactions, will be used to build a
simulation of a live system. As I mentioned before, the simulation will not be
entirely realistic, simply because we cannot fully reproduce the conditions
within a given company in general and in particular, 50k interactions expand
for months, at least for the Movies dataset.

The data used to build the algorithm will have to be stored in a realistic way
so that that, as new observations/interactions arrive during the before
mentioned simulation, these are added to the existing dataset. Therefore, we
need to think carefully of the features we use.

We will retrain 4-5 times, every 10k approx interactions. The process can be
described as follows:

1. We will generate a synthetic time step and control the number of iterations
that call the prediction service (e.g API) per unit time. For example, say we
want to have 10000 user-item interactions spread throughout 5 hours, so that
we would have the service running for approximately a day. We could set the
already mentioned time step so that we have most of the time 1 interaction per
second, but suddenly we have 1000 in 10 secs to test scalability. All these
numbers we will be carefully calculated in due time.

2. When we reach, for example, 10k interactions, we will re-train the
algorithm in a separate process. Again, we need to set up time steps so that
the retraining process is fast compared to each retraining period. For
example, if we decided to set the time steps so that 10000 calls to the
service take 5 hours, retraining should take a few minutes, or in general, a
lot less than 5 hours

3. Once the retraining is done we will over-write the current model with the
one retrained if the performance metric improves. This means that performance
will be, of course, monitored.

4. At the same time we will use some of the training blocks (or 2) to run an
A/B test. To keep things "simple" we will set this up so that the challenger
algorithm does not win and we do not have to overwrite with the new algorithm

Once the 50k (or whatever how many we decide to use) interactions have been
"exhausted" the simulation will finish. By that time, the reader will have a
good idea on how to A/B test, re-train and monitor live algorithms. At the
same time, hopefully they will also have a lot of code they can use for their
daily jobs.

There are a couple of comments worth mentioning. Here are ignoring drift and
model interpretability. We could perhaps address drift by simply checking if
the distribution of scores, or user gender, or some other property remains the
same, but I find it a bit forced. Regarding to interpretability, if we end up
going for a tabular representation of the data, then this one is relatively
easy (LIME, SHAP, interpretML). If not, then becomes more complex, and
possibly redundant.

Let me just add a few lines regarding the cold start problem. During these 50k
interactions, we expect to see users and items that we have never seen before.
For items, we will assume that item information is available prior having to
make the recommendation, and we will treat that item as its most similar item.
For users is a bit different, since in most real world cases the user will
have registered by simply providing email (or similar) and name and we will
have no other information whatsoever. For those users we will recommend the
most popular items. Note that, for example, the multinomial variational
autoencoder (aka mult-VAE) does not care about users, but learns to encode
user histories. Therefore, with this algorithm we will not face the user cold
start problem.
