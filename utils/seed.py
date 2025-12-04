import random


def new_seed() -> int:
    # 32-bit unsigned integer range
    return random.randint(0, 2**32 - 1)
