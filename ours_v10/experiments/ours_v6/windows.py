"""Window schedule shared by the worker and v5 backbone."""


def make_windows(count, window_size=60, overlap=30):
    if count < 1 or window_size < 1 or not 0 <= overlap < window_size:
        raise ValueError("require N>0, W>0, 0<=overlap<W")
    stride = window_size - overlap
    windows = []
    for start in range(0, count, stride):
        windows.append((start, min(start + window_size, count)))
        if windows[-1][1] == count:
            break
    return windows
