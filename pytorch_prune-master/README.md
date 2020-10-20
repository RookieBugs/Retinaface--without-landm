# Incremental pruning for Pytorch

## Usage

step1 Build Sparse Table
```
from pruning import create_sparse_table
...
sparse_dict_list=create_sparse_table(model,input_shape,DEFAULT_SPARSE_STEP=[0.65, 0.7, 0.75, 0.85, 0.88, 0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99])

```
step2 Select a Total Sparse Ratio to Train
```
total_sparse,sparse_dict=sparse_dict_list[i]
```
step3 Inplaced Get a Pruned Model
```
from pruning import pruning
pruning(model,sparse_dict)
```
step4 To Train
```
...
pred=model(data)
loss=criterion(pred,label)
loss.backward()
...
```
step5 Evaluation

Step6 Stop or Next Total Sparse Ratio Stage
