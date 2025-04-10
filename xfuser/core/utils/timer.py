import torch
import time


def gpu_timer_decorator(func):
    def wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        start_time = time.time()
        result = func(*args, **kwargs)
        torch.cuda.synchronize()
        end_time = time.time()

        if torch.distributed.get_rank() == 0:
            print(
                f"{func.__name__} took {end_time - start_time} seconds to run on GPU."
            )
        return result

    return wrapper

def func_timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        if args and hasattr(args[0], '__class__'):
            class_name = args[0].__class__.__name__
            if func.__name__ == "get_cos_sin":
                sig = inspect.signature(func)
                bound_arguments = sig.bind(*args, **kwargs)
                bound_arguments.apply_defaults()
                args_str = ", ".join(f"{name}={value!r}" for name, value in bound_arguments.arguments.items())
                print(f"Calling {func.__name__}({args_str})")
            print(f"{class_name}.{func.__name__} took {elapsed_time:.3f} seconds")
        else:
            print(f"{func.__name__} took {elapsed_time:.3f} seconds")
        return result
    return wrapper
