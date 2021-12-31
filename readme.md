## Jax Dataloader
Naive dataloader implementation with support for 
- shuffeling with explicit key, 
- multiple batch-axes
- dropping the last batch if it is incomplete,
- parallelism using `jax.vmap`.

First, define your dataset as a class that implements `__call__` and `__len__`
```python
class Dataset:
    def __init__(self, data):
        self.data = data

    def __call__(self, idx: int) -> tuple[array]:
        X = self.data[idx]
        y = self.data[idx]
        return X,y
    
    def __len__(self) -> int:
        return len(self.data)

dataset_train = Dataset(jnp.arange(5))
```
Then, you are able to use the Dataloader:
```python
from jax_dataloader import Dataloader

dl_train = Dataloader(dataset_train, bs=2)

key = jax.random.PRNGKey(0)
for _ in range(3):
    for X, y in dl_train:
        print(X, "|", y)
    
    print("Shuffle..")
    key = dl_train.shuffle(key, returns_new_key=True)
```
Which then prints:
```shell
[0 1] | [0 1]
[2 3] | [2 3]
Shuffle..
[1 0] | [1 0]
[4 3] | [4 3]
Shuffle..
[0 3] | [0 3]
[4 2] | [4 2]
Shuffle..
```

