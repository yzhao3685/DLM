
import os
import redis
import time

h, p = os.environ['REDIS_HOST'].split(':')
r = redis.Redis(host=h, port=p, db=0, password=os.environ['REDIS_PASS'])

def set(key, value):
    r = redis.Redis(host=h, port=p, db=0, password=os.environ['REDIS_PASS'])
    r.set(f"{os.environ['EXP_ID']}-{key}", value)

def get(key):
    # print(f"{os.environ['EXP_ID']}-{key}")
    r = redis.Redis(host=h, port=p, db=0, password=os.environ['REDIS_PASS'])
    return r.get(f"{os.environ['EXP_ID']}-{key}")



class RedisLock:
    def __init__(self,lock_key, acquire_timeout=300, lock_timeout=300, redis_client=r):
        self.redis_client = redis_client
        self.lock_key = lock_key
        self.acquire_timeout = acquire_timeout
        self.lock_timeout = lock_timeout
        self.lock_value = None

    def acquire(self):
        self.redis_client = redis.Redis(host=h, port=p, db=0, password=os.environ['REDIS_PASS'])
        end_time = time.time() + self.acquire_timeout
        while time.time() < end_time:
            # Try to acquire the lock
            if self.redis_client.set(self.lock_key, "LOCKED", nx=True, ex=self.lock_timeout):
                self.lock_value = "LOCKED"
                return True
            time.sleep(0.1)
        return False

    def release(self):
        if self.lock_value:
            self.redis_client = redis.Redis(host=h, port=p, db=0, password=os.environ['REDIS_PASS'])
            self.redis_client.delete(self.lock_key)
            self.lock_value = None

    def __enter__(self):
        if not self.acquire():
            raise Exception("Couldn't acquire the lock")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
