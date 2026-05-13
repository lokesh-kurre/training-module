from glob import glob
from pathlib import Path

import re
import boto3
import botocore.exceptions
from botocore.config import Config

import warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

s3_config = {
    "max_pool_connections": 50,
    "signature_version": "s3v4",
    "retries": {"max_attempts": 1}, 
    "connect_timeout": 1
}

def list_files(directory: str= None, /, recursive: bool= False, s3_client= None, verbose= False, count= 1000, **kwargs):
    assert directory is not None or all(key in kwargs for key in 'Bucket|Prefix'.split("|")), \
        "either positional paramater `directory`{$0} or keyword argument `Bucket` and `Prefix` must be present."
    dir_uri = Path(directory).absolute().as_uri() if directory is not None else f's3://{kwargs["Bucket"]}/{kwargs["Prefix"]}' if directory is not None and '://' not in directory else directory
    try:
        if directory is not None:
            if directory.startswith('s3://'):
                dir_uri = directory
                _, s3_obj_bucket, s3_obj_prefix, _ = re.split(r'^s3://([\w-]+)/(.*)$', directory)
                kwargs['Bucket'] = s3_obj_bucket
                kwargs['Prefix'] = s3_obj_prefix
                directory = None
            
            elif directory.startswith("file://"):
                dir_uri = directory
                _, directory, _ = re.split(r'^file://(.+)$', directory)

            elif '://' not in directory:
                dir_uri = Path(directory).absolute().as_uri()
                
            else:
                raise ValueError("Invalid value passed to `directory`(%s)" % directory)

    except Exception as e:
        if verbose:
            warnings.warn(f'error while listing directory {dir_uri}, err_msg: invalid uri, must be file:///absolute/path/to/dir or s3://bucket/path/to/dir with error {str(e)}', UserWarning)
        return []
    
    else:
        if directory is not None:
            directory = directory.rstrip("/")
            directory += "/**/*" if recursive else "/*"
            yield from glob(directory, recursive= recursive)

        else:
            if s3_client is None:
                s3_client = boto3.client("s3", verify= False, config= Config(**s3_config))
            if count < 0 or recursive and count > 10000:
                warnings.warn(f'trying to all the object in directory {dir_uri} recursive, for larger buckets it may freeze', UserWarning)
            page_iterator = s3_client.get_paginator('list_objects_v2').paginate(**{'PaginationConfig': {} if count < 0 else {'MaxItems': count}, **kwargs, 'Delimiter': '' if recursive else '/', })
            for page in page_iterator:
                yield from (object_summary['Prefix'] for object_summary in page.get('CommonPrefixes', []))
                yield from (object_summary['Key'] for object_summary in page.get('Contents', []))


def read_binary(filepath: str= None, /, stream= False, s3_client= None, verbose= False, **kwargs):
    assert filepath is not None or all(key in kwargs for key in 'Bucket|Key'.split("|")), \
        "either positional paramater `filepath`{$0} or keyword argument `Bucket` and `Key` must be present."
    file_uri = Path(filepath).absolute().as_uri() if filepath is not None else f's3://{kwargs["Bucket"]}/{kwargs["Key"]}' if filepath is not None and '://' not in filepath else filepath
    img_buffer = None
    try:

        if filepath is not None:
            if filepath.startswith('s3://'):
                file_uri = filepath
                _, s3_obj_bucket, s3_obj_key, _ = re.split(r'^s3://([\w-]+)/(.*)$', filepath)
                kwargs['Bucket'] = s3_obj_bucket
                kwargs['Key'] = s3_obj_key
                filepath = None
            
            elif filepath.startswith("file://"):
                file_uri = filepath
                _, filepath, _ = re.split(r'^file://(.+)$', filepath)

            elif '://' not in filepath:
                file_uri = Path(filepath).absolute().as_uri()

            else:
                raise ValueError("Invalid value passed to `filepath`(%s)" % filepath)
            
    except Exception as e:
        if verbose:
            warnings.warn(f'error while parsing file_uri {file_uri}, err_msg: invalid uri, must be file:///absolute/path/to/file or s3://bucket/path/to/file with error {str(e)}', UserWarning)
        return
    try:
        if filepath is not None:
            img_buffer = open(filepath, 'rb')

        else:
            if s3_client is None:
                s3_client = boto3.client("s3", verify= False, config= Config(**s3_config))
            img_buffer = s3_client.get_object(**kwargs)['Body']
        
        return img_buffer if stream else img_buffer.read()
    
    except (FileNotFoundError, botocore.exceptions.ClientError) as e:
        if verbose:
            err_msg = str(e)
            if isinstance(e, FileNotFoundError):
                err_msg = e.args[1]
            warnings.warn(f'error while reading file {file_uri}, err_msg: {err_msg}', UserWarning)

    finally:
        if img_buffer is not None and not stream:
            img_buffer.close()


import cv2
import numpy as np
def read_image(filepath: str= None, /, as_pillow= False, cv2_imdecode_mode= -1, s3_client= None, verbose= False, **kwargs):
    assert filepath is not None or all(key in kwargs for key in 'Bucket|Key'.split("|")), \
        "either positional paramater `filepath`{$0} or keyword argument `Bucket` and `Key` must be present."
    file_uri = Path(filepath).absolute().as_uri() if filepath is not None else f's3://{kwargs["Bucket"]}/{kwargs["Key"]}' if filepath is not None and '://' not in filepath else filepath
        
    stream = read_binary(filepath, stream= True, s3_client= s3_client, verbose= verbose, **kwargs)

    if stream is None:
        return
    
    try:
        if as_pillow:
            from PIL import Image
            img_np = Image.open(stream)
            if img_np is None:
                raise ValueError('pillow image decoding failed')
        else:
            img_np = cv2.imdecode(np.frombuffer(stream.read(), np.uint8), cv2_imdecode_mode)
            if img_np is None:
                raise ValueError('cv2 imdecode failed')
        return img_np
    except Exception as e:
        if verbose:
            warnings.warn(f'error while decoding file {file_uri}, err_msg: {e}', UserWarning)
    finally:
        stream.close()
