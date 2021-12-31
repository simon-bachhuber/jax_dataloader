from functools import reduce
from typing import Union, Sequence
import jax.numpy as jnp 
import jax 

class Dataloader:
    def __init__(self, 
        dataset,
        bs: Union[int, Sequence[int]],
        shuffle: bool=True, 
        drop_last_batch: bool=True
    ):
        """Naive dataloader implementation with support for 
        - shuffeling with explicit key, 
        - dropping the last batch if it is incomplete,
        - parallelism using `jax.vmap`.

        Args:
            dataset (object): Class that implements `__len__` and `__call__(idx)`
            bs (Union[int, Sequence[int]]): Batch size 
            shuffle (bool, optional): If `False` calling `dataloader.shuffle(key)` is a null-operation. Defaults to True.
            drop_last_batch (bool, optional): If `True` last batch is dropped if its batch size would be incomplete. Defaults to True.

        Raises:
            Exception: Requested batch size exceeds dataset-length
        """
        self.dataset = jax.vmap(dataset)
        self._shuffle = shuffle
        N = len(dataset)
        self._order = jnp.arange(0, N)
        self._exhausted_batches = 0

        if isinstance(bs, int):
            self._reshape = lambda arr: arr  
        else:
            self._reshape = lambda arr: arr.reshape(bs+(-1,))
            bs = reduce(lambda a,b:a*b, bs, 1)

        if bs>N:
            raise Exception(f"The requested batch size of {bs} is not possible with a dataset of length {N}")

        if drop_last_batch:
            self._slices = [slice(i,i+bs) for i in range(0,N-bs+1,bs)]
        else: 
            self._slices = [slice(i,i+bs) for i in range(0,N,bs)]

    def shuffle(self, key, returns_new_key=False):
        if returns_new_key:
            key, subkey = jax.random.split(key)

        if self._shuffle:
            self._order = jax.random.permutation(subkey, self._order)

        if returns_new_key:
            return key  

    def __len__(self):
        return len(self._slices)

    def __iter__(self):
        self._exhausted_batches = 0
        return self 

    def __next__(self):
        
        i = self._exhausted_batches

        if i<len(self):
            batch = self.dataset(self._order[self._slices[i]])
            self._exhausted_batches += 1
            return jax.tree_util.tree_map(self._reshape, batch)
        else:
            raise StopIteration

if __name__ == "__main__":
    class Dataset:
        def __init__(self, word: str):
            self.data = word

        def __call__(self, idx: int) -> tuple:
            X = self.data[idx]
            y = self.data[idx]
            return X,y
        
        def __len__(self) -> int:
            return len(self.data)

    dataset_train = Dataset(jnp.arange(5))

    dl_train = Dataloader(dataset_train, bs=2)

    key = jax.random.PRNGKey(0)
    for _ in range(3):
        for X, y in dl_train:
            print(X, "|", y)
        
        print("Shuffle..")
        key = dl_train.shuffle(key, returns_new_key=True)