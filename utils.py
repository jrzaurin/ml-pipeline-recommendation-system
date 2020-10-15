import pyarrow.parquet as pq
from time import time
import json
import os
from datetime import datetime
from multiprocessing import cpu_count
from psutil import virtual_memory
import pathlib
import shutil

import s3fs
import dask.dataframe as dd
import boto3
import luigi
from luigi.contrib.s3 import S3Target
from pyspark import SparkContext, SQLContext

BUCKET = 'recsys-1'
RAW_REVIEWS_PATH = 's3://' + BUCKET + '/raw/reviews_split/part-*'
RAW_METADATA_PATH = 's3://' + BUCKET + '/raw/metadata_split/part-*'
JOBS_TEMP_DIR = pathlib.Path(__file__).parent.absolute() / 'data/jobs'


def s3_path(path_suffix):
    return os.path.join('s3://' + BUCKET, path_suffix)


def clean_s3_dir(dir_name):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(BUCKET)
    bucket.objects.filter(Prefix=dir_name + '/').delete()


def write_s3_file(path, content):
    s3 = boto3.resource('s3')
    s3.Object(BUCKET, path).put(Body=content)


def upload_directory(local_dir, target_dir):
    client = boto3.client('s3')
    for root, dirs, files in os.walk(local_dir):
        for path in files:
            full_path = (Path(root) / path)
            relative_path = full_path.relative_to(directory)
            target_path = target_dir / relative_path
            client.upload_file(str(full_path), BUCKET, str(target_path))


def download_s3_file(orig_path, local_path):
    client = boto3.client('s3')
    client.download_file(BUCKET, orig_path, local_path)


def download_dir(prefix, local):
    """Copies an entire 'directory' from s3 to local
    https://stackoverflow.com/questions/31918960
    params:
    - prefix: pattern to match in s3
    - local: local path to folder in which to place files
    """
    client = boto3.client('s3')
    keys = []
    dirs = []
    next_token = ''
    base_kwargs = {
        'Bucket': BUCKET,
        'Prefix': prefix,
    }
    while next_token is not None:
        kwargs = base_kwargs.copy()
        if next_token != '':
            kwargs.update({'ContinuationToken': next_token})
        results = client.list_objects_v2(**kwargs)
        contents = results.get('Contents')
        for i in contents:
            k = i.get('Key')
            if k[-1] != '/':
                keys.append(k)
            else:
                dirs.append(k)
        next_token = results.get('NextContinuationToken')
    for d in dirs:
        dest_pathname = os.path.join(local, d)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
    for k in keys:
        dest_pathname = os.path.join(local, k)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
        client.download_file(BUCKET, k, dest_pathname)


def print_parquet_schema(s3_uri):
    s3 = s3fs.S3FileSystem()
    uri = s3_uri + '/part.0.parquet'
    dataset = pq.ParquetDataset(uri, filesystem=s3)
    table = dataset.read()
    print(str(table).split('metadata')[0])


def s3_uri_2_spark(s3_uri):
    return 's3a' + s3_uri.lstrip('s3')


def start_spark(app_name='Bob'):
    sc = SparkContext('local', app_name)
    sql = SQLContext(sc)
    return sc, sql


class Mario(object):
    """
    Mixin for use with luigi.Task

    TODO: add method to print the schema of the output parquet
    """
    cleanup = luigi.BoolParameter(significant=False, default=False)

    def complete(self):
        if self.cleanup and not hasattr(self, 'just_finished_running'):
            return False
        else:
            return super().complete()

    def output_dir(self):
        raise NotImplementedError

    def local_path(self, file_name=None):
        if file_name is not None:
            return JOBS_TEMP_DIR / self.output_dir / file_name
        else:
            return JOBS_TEMP_DIR / self.output_dir()

    def clean_local_dir(self):
        shutil.rmtree(self.local_path(), ignore_errors=True)

    def backup_local_dir(self):
        upload_local_dir(self.local_path(), self.output_dir())

    def get_local_output(self, file_name):
        path = self.local_path(file_name)
        if not os.path.exists(path):
            download_s3_file(self.output_dir() + '/' + file_name, path)
        return path

    def full_output_dir(self, subdir=None):
        if subdir is None:
            return s3_path(self.output_dir())
        else:
            return s3_path(self.output_dir() + '/' + subdir)

    def output(self):
        return S3Target(
            s3_path(
                os.path.join(
                    self.output_dir(),
                    '_SUCCESS.json')))

    def clean_output(self):
        clean_s3_dir(self.output_dir())
        self.clean_local_dir()

    def run_info(self):
        with self.output().open('r') as f:
            return json.load(f)

    def _run(self):
        raise NotImplementedError

    def load_parquet(self, subdir=None, sqlc=None):
        """loads parquet output as dask dataframe of as spark dataframe if SQLContext is provided"""
        if sqlc is None:
            return dd.read_parquet(self.full_output_dir(subdir))
        else:
            return sqlc.read.parquet(
                s3_uri_2_spark(
                    self.full_output_dir(subdir)))

    def save_parquet(self, df, subdir=None):

        output_path = self.full_output_dir(subdir)
        df.to_parquet(output_path)
        print(output_path)
        print_parquet_schema(output_path)

    def run(self):
        self.clean_output()

        start = time()
        self._run()
        end = time()

        mem = virtual_memory()
        mem_gib = round(mem.total / 1024 ** 3, 2)
        run_info = {
            'start': str(datetime.fromtimestamp(start)),
            'end': str(datetime.fromtimestamp(end)),
            'elapsed': end - start,
            'cpu_count': cpu_count(),
            'mem GiB': mem_gib
        }

        with self.output().open('w') as f:
            f.write(json.dumps(run_info))

        self.just_finished_running = True
