from time import time
import json
import os
from datetime import datetime
from multiprocessing import cpu_count
from psutil import virtual_memory

import dask.dataframe as dd
import boto3
import luigi
from luigi.contrib.s3 import S3Target 



BUCKET = 'recsys-1'
RAW_REVIEWS_PATH = 's3://' + BUCKET + '/raw/reviews_split/part-*'
RAW_METADATA_PATH = 's3://' + BUCKET + '/raw/metadata_split/part-*'

def s3_path(path_suffix):
    return os.path.join('s3://' + BUCKET, path_suffix)


def clean_s3_dir(dir_name):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(BUCKET)
    bucket.objects.filter(Prefix=dir_name + '/').delete()


def write_s3_file(path, content):
    s3 = boto3.resource('s3')
    s3.Object(BUCKET, path).put(Body=content)


    
class Mario(object):
    """
    Mixin for use with luigi.Task
    """
    cleanup = luigi.BoolParameter(significant=False, default=False)

    def output_dir(self):
        raise NotImplementedError
    
    def full_output_dir(self):
        return s3_path(self.output_dir())

    def output(self):
        if self.cleanup:
            # TODO: there must be a better way to force cleanup
            return S3Target(s3_path(os.path.join(self.output_dir(), 'NOT_SUCCESS.json')))
        return S3Target(s3_path(os.path.join(self.output_dir(), '_SUCCESS.json')))

    def clean_output(self):
        clean_s3_dir(self.output_dir())

    def run_info(self):
        with self.output().open('r') as f:
            return json.load(f)
        
    def _run(self):
        raise NotImplementedError

    def load_parquet(self):
        return dd.read_parquet(s3_path(self.output_dir()))

    def run(self):
        self.clean_output()
        if self.cleanup:
            raise ValueError('output removed. You can run the job again')

        start = time()
        self._run()
        end = time()
        
        mem = virtual_memory()
        mem_gib = round(mem.total / 1024 **3, 2)
        run_info = {
            'start': str(datetime.fromtimestamp(start)),
            'end': str(datetime.fromtimestamp(end)),
            'elapsed': end - start,
            'cpu_count': cpu_count(),
            'mem GiB': mem_gib
        }
        
        with self.output().open('w') as f:
            f.write(json.dumps(run_info))
            
        
