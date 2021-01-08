# tensor-contraction-sequence-searching

## Folder Description:

./SeqSearch is the solution of vanilla search.

./SeqSearchPrune is the solution of advanced search with outer product pruning.



## Example of how to run vanilla search

Type following comands in command line

```
MacBook-Pro:folder$ git clone https://github.com/liangling76/tensor-contraction-sequence-searching.git
MacBook-Pro:folder$ cd SeqSearch
MacBook-Pro:SeqSearch$ mkdir build
MacBook-Pro:SeqSearch$ cd build
MacBook-Pro:build$ cmake ..
MacBook-Pro:build$ make
```

Then some paramters are required to describe the network, target optimization goal and the execution preference. 
Note that the amount of prune portion stands for the portion of preserved connections. The network building is implemented in file ./SeqSearch/src/utils.cpp.

```
MacBook-Pro:build$ ./SeqSearch 
Enter tensor number: 15
Select optimization type (0 MS, 1 MC): 1
Number of threads: 2
base structure: 2
prune_portion: 0.7
extra edges: 73
```


## Paper link
