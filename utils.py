import pyarrow.parquet as pq
from time import time
import json
import os
from datetime import datetime
from multiprocessing import cpu_count
from psutil import virtual_memory
import pathlib
import shutil
from pathlib import Path
import colorama
from colorama import Fore

import s3fs
import dask.dataframe as dd
import boto3
import luigi
from luigi.contrib.s3 import S3Target
import pyspark
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
            relative_path = full_path.relative_to(local_dir)
            target_path = target_dir / relative_path
            client.upload_file(str(full_path), BUCKET, str(target_path))


def download_s3_file(orig_path, local_path):
    client = boto3.client('s3')
    client.download_file(BUCKET, orig_path, local_path)


def download_dir(remote_dir, local_dir):
    """copies a remote directory onto a local one
    """
    local_dir = Path(local_dir)
    if not remote_dir.endswith('/'):
        remote_dir += '/'

    s3 = boto3.resource('s3')
    client = boto3.client('s3')
    bucket = s3.Bucket(BUCKET)

    for page in bucket.objects.filter(Prefix=remote_dir).pages():
        for f in page:
            relative_path = f.key[len(remote_dir):]
            destination = local_dir / relative_path
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            if not os.path.exists(destination):
                client.download_file(BUCKET, f.key, str(destination))


def find_parquet_file(s3_directory):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(BUCKET)
    # strip the "s3://" bit
    prefix = s3_directory[len(BUCKET) + 6:]
    obs = bucket.objects.filter(Prefix=prefix)
    parquet_file = None
    for page in obs.pages():
        if parquet_file is not None:
            break
        for line in page:
            key = line.key
            if key.endswith('parquet'):
                parquet_file = key
                print(parquet_file)
                break
    return parquet_file


def print_parquet_schema(s3_directory):
    parquet_file = find_parquet_file(s3_directory)
    if parquet_file is None:
        raise ValueError('No parquet file found in %s' % s3_directory)
    s3 = s3fs.S3FileSystem()
    uri = 's3://' + BUCKET + '/' + parquet_file
    dataset = pq.ParquetDataset(uri, filesystem=s3)
    table = dataset.read()
    print(str(table).split('metadata')[0])


def s3_uri_2_spark(s3_uri):
    return 's3a' + s3_uri.lstrip('s3')


sc = None
sqlc = None


def start_spark(app_name='Bob'):
    global sc
    global sqlc
    if sc is not None:
        return sc, sqlc
    sc = SparkContext('local', app_name)
    sqlc = SQLContext(sc)
    return sc, sqlc


def get_reqs_list(job):
    reqs = job.requires()
    try:
        return list(reqs)
    except TypeError as te:
        if reqs is None:
            return []
        else:
            return [reqs]

def what_needs_to_run(job):
    if job.complete():
        return None

    for parent in get_reqs_list(job):
        what_needs = what_needs_to_run(parent)
        if what_needs is not None:
            return what_needs
    return job

def run_everything(job):
    while what_needs_to_run(job) is not None:
        what_needs_to_run(job).run()


def print_dag(job):
    colorama.init()
    visited = set()
    def aux(job, indent=0):
        if job.output_dir() in visited:
            return

        if job.complete():
            s = ' ' * indent + Fore.GREEN + 'o ' + Fore.RESET + str(job) + ' ' + Fore.GREEN + job.output_dir() + Fore.RESET
        else:
            s = ' ' * indent + Fore.RED + 'x ' + Fore.RESET + str(job) + ' ' + Fore.RED + job.output_dir() + Fore.RESET
        print(s)
        visited.add(job.output_dir())
        for child in get_reqs_list(job):
            aux(child, indent + 2)

    aux(job)
    print()
    needs_to_run = what_needs_to_run(job)
    if needs_to_run is None:
        print('nothing to run')
    else:
        print('next to run:\n', str(needs_to_run))


class Mario(object):
    """
    Mixin for use with luigi.Task

    TODO: add method to print the schema of the output parquet
    """
    cleanup = luigi.BoolParameter(significant=False, default=False)
    dry = luigi.BoolParameter(significant=False, default=False)

    def complete(self):
        if self.cleanup and not hasattr(self, 'just_finished_running'):
            return False
        else:
            return super().complete()

    def output_dir(self):
        raise NotImplementedError

    def local_path(self, file_name=None):
        if file_name is not None:
            path = JOBS_TEMP_DIR / self.output_dir() / file_name
        else:
            path = JOBS_TEMP_DIR / self.output_dir()

        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def clean_local_dir(self):
        shutil.rmtree(self.local_path(), ignore_errors=True)

    def backup_local_dir(self):
        upload_directory(self.local_path(), self.output_dir())

    def sync_output_to_local(self):
        download_dir(self.output_dir(), self.local_path())

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

    def load_parquet(self, subdir=None, sqlc=None, from_local=False):
        """loads parquet output as dask dataframe of as spark dataframe if SQLContext is provided"""
        if from_local:
            output_dir = str(self.local_path(subdir))
            self.sync_output_to_local()
        else:
            output_dir = self.full_output_dir(subdir)

        print('loading output from ', output_dir)
        if sqlc is None:
            return dd.read_parquet(output_dir)
        else:
            if from_local:
                return sqlc.read.parquet(output_dir)
            else:
                return sqlc.read.parquet(s3_uri_2_spark(output_dir))

    def save_parquet(self, df, subdir=None):
        output_path = self.full_output_dir(subdir)
        if isinstance(df, pyspark.sql.dataframe.DataFrame):
            df.write.parquet(s3_uri_2_spark(output_path))
        else:
            df.to_parquet(output_path)

        print(output_path)
        print_parquet_schema(output_path)

    def print_output_schema(self, subdir=None):
        print_parquet_schema(self.full_output_dir(subdir))

    def run(self):
        if self.dry:
            print('DRY RUN')
            print_dag(self)
            return

        print('RUNNING')
        print(self)
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
        print(self)
        print('DONE')
        print('Job took %d seconds' % (end - start))
