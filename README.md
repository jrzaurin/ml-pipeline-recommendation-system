# ml-pipeline-recommendation-system

### Requirements

- To run the pipeline with Luigi, you need to put this directory on PYTHONPATH. 
- s3cmd is required for step 01 - uploading stuff to s3


### Decisions

- in order to be able to collaborate on the same dataset we decided to save outputs for every stage of data processing on s3 where it can be accessed by everyone. 
- I chose parquet format because it is columnar and data is partitioned before saving. Both of these properties make it easy to read the data in small chunks, without requiring massive amounts of RAM and without wasting time downloading unneded parts
- There comes a point in a data project when it gets complex enough that it pays to have some kind of job orchestration. In my experience this time is about 7 minutes into the project so first thing after downloading data I set up Luigi
- in a production run jobs sometimes fail and they need to be retried. But you only very rarely need to rerun a successful job. That should only happen when you discover a critical bug in job code - not something that should happen every day. In an exploratory phase of a data science project on the other hand - it happens all the time. That's what exploration is - running experiments on a dataset and modifying your code in response to your findings. Unfortunately Luigi lacks a simple way to clean a job output - you have to do it manually. That's why I created a simple mixin for luigi tasks that adds that functionality. It is used like this:

```
luigi --module x03_parquetify ParquetifyReviews --local-scheduler --cleanup
```

Mario also adds other convenient methods - reading info on the latest successful job run:

```python
>>> from x04_train_test_split import TRAIN_SET_REVIEWS_JOB
>>> pprint.pprint(TRAIN_SET_REVIEWS_JOB.run_info())
{'cpu_count': 4,
 'elapsed': 281.7859261035919,
 'end': '2020-10-05 15:33:00.628751',
 'mem GiB': 15.67,
 'start': '2020-10-05 15:28:18.842825'}
```

and a quick way to load job outputs as dask dataframe:

```python
>>> df = TRAIN_SET_REVIEWS_JOB.load_parquet()
```
