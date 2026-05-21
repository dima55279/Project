from contextlib import contextmanager
from tqdm import tqdm
import time


class ProgressManager:

    @staticmethod
    def track(iterable,
              desc="Processing",
              total=None):

        return tqdm(
            iterable,
            desc=desc,
            total=total,
            ncols=100
        )

    @staticmethod
    @contextmanager
    def timer(name):

        start = time.time()

        print(f"[START] {name}")

        yield

        end = time.time()

        print(
            f"[DONE] {name} "
            f"({end - start:.2f}s)"
        )