import os
from urllib import request
from tqdm import tqdm


def maybe_download(model_path_or_name: str) -> str:
    """Download and cache model weights from a given URL if not already present.

    This function checks whether the model weights specified by the `model_path_or_name`
    parameter have already been downloaded and cached. If the model weights are not
    present, they will be downloaded from the given URL and stored in a local directory
    for future use.

    Parameters
    ----------
    model_path_or_name : str
        The path to the model weights file or the URL to download the model weights.

    Returns
    -------
    str
        The local file path where the model weights are stored or the same
        input `model_path_or_name` if it already represents a local file path.
    """
    if model_path_or_name.startswith('https://'):
        root_dir = os.path.expanduser('~/.vision_counter')
        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)

        f = os.path.join(root_dir, model_path_or_name.split('/')[-1])
        print(f'Downloading model weights to {f}')
        if not os.path.isfile(f):
            with request.urlopen(model_path_or_name) as response, open(f, 'wb') as f_opened:
                for data in tqdm(response):
                    f_opened.write(data)
        return f
    return model_path_or_name