import gc
import os

import psutil


def memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory Usage: {mem_info.rss / 1024 ** 2:.2f} MB")


def dataframe_memory_usage(df):
    mem_usage = df.memory_usage(deep=True).sum() / 1024**2
    print(f"DataFrame Memory Usage: {mem_usage:.2f} MB")
    return mem_usage


def garbage_collection():
    gc.collect()
    memory_usage()
