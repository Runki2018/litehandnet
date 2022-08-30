from torchvision.transforms import functional as F

class ToTensor:
    """Transform image to Tensor.

    Required key: 'img'. Modifies key: 'img'.

    Args:
        results (dict): contain all information about training.
    """

    def __call__(self, results):
        if isinstance(results['img'], (list, tuple)):
            results['img'] = [F.to_tensor(img) for img in results['img']]
        else:
            results['img'] = F.to_tensor(results['img'])

        return results
    
class NormalizeTensor:
    """Normalize the Tensor image (CxHxW), with mean and std.

    Required key: 'img'. Modifies key: 'img'.

    Args:
        mean (list[float]): Mean values of 3 channels.
        std (list[float]): Std values of 3 channels.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, results):
        if isinstance(results['img'], (list, tuple)):
            results['img'] = [
                F.normalize(img, mean=self.mean, std=self.std)
                for img in results['img']
            ]
        else:
            results['img'] = F.normalize(
                results['img'], mean=self.mean, std=self.std)

        return results
 

class Compose:
    """Compose a data pipeline with a sequence of transforms.

    Args:
        transforms (list[dict | callable]): Either config
          dicts of transforms or transform objects.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
            dict: Transformed data.
        """
        for t in self.transforms:
            data = t(data)
            if data is None:
                raise ValueError(f"data is None, and {t.__name__=}")
        return data

    def __repr__(self):
        """Compute the string representation."""
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += f'\n    {t}'
        format_string += '\n)'
        return format_string