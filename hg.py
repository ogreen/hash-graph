#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import numba
from numba import cuda
from numba import jit
import cupy as cupy
import time
import ctypes

import cudf as gd
from cudf.core.column import column
from cudf.core.dataframe import DataFrame, Series
from cudf.tests import utils





@cuda.jit
def normailizeHashArray(inputA, hashRange):
    start  = cuda.grid(1)
    stride = cuda.gridsize(1)
    size   = inputA.shape[0]
    for idx in range(start, size, stride):
        inputA[idx] = inputA[idx]%hashRange

@cuda.jit
def resetCounter(counterArray):
    start  = cuda.grid(1)
    stride = cuda.gridsize(1)
    size   = counterArray.shape[0]
    for idx in range(start, size, stride):
        counterArray[idx]=0
        
@cuda.jit
def countHashBins(hashA, counterArray, binSize):
    start  = cuda.grid(1)
    stride = cuda.gridsize(1)
    size   = hashA.shape[0]
    for idx in range(start, size, stride):
        mybin = int(hashA[idx]//binSize)
        # one = 1
        cuda.atomic.add(counterArray,mybin,1)


@cuda.jit
def reOrderInput(inputA,hashA, d_InputAReOrdered, binSize, counterArray, prefixArray):
    start  = cuda.grid(1)
    stride = cuda.gridsize(1)
    size   = inputA.shape[0]
    for idx in range(start, size, stride):
        mybin = int(hashA[idx]//binSize)
        pos = cuda.atomic.add(counterArray,mybin,1) + prefixArray[mybin]
        d_InputAReOrdered[pos]=inputA[idx] # Todo store index


@cuda.jit
def reOrderHash(hashAReordered, counterArray):
    start  = cuda.grid(1)
    stride = cuda.gridsize(1)
    size   = hashAReordered.shape[0]
    for idx in range(start, size, stride):
        cuda.atomic.add(counterArray,hashAReordered[idx],1)

    # idx  = cuda.grid(1)
    # size   = hashAReordered.shape[0]
    # if(idx>=size):
    #   return
    # cuda.atomic.add(counterArray,hashAReordered[idx],1)

@cuda.jit
def fillTable(table,hashAReordered,d_InputAReOrdered, counterArray,prefixArray):
    # start  = cuda.grid(1)
    # stride = cuda.gridsize(1)
    # size   = hashAReordered.shape[0]
    # for idx in range(start, size, stride):
    #     pos = cuda.atomic.add(counterArray,hashAReordered[idx],1)+prefixArray[hashAReordered[idx]]
    #     table[pos] = d_InputAReOrdered[idx]
    idx  = cuda.grid(1)
    size   = hashAReordered.shape[0]
    if(idx>=size):
      return
    pos = cuda.atomic.add(counterArray,hashAReordered[idx],1)+prefixArray[hashAReordered[idx]]
    table[pos] = d_InputAReOrdered[idx]

inputSize = 1<<25
low = 0
high = inputSize 
hashRange = inputSize >> 2
numBins = 1<<16
binSize = (inputSize+numBins-1)//numBins
print(binSize)
print(type(binSize))

inputA = np.random.randint(low,high,inputSize)

d_inputA          = cuda.to_device(inputA)
d_HashA           = cuda.device_array(d_inputA.shape, dtype=np.int32)
d_InputAReOrdered = cuda.device_array(d_inputA.shape, dtype=np.int32)

d_Table = cuda.device_array(d_inputA.shape, dtype=np.int32)



# Arrays used for HashGraph proceess (size of the hash-range)
d_CounterArray    = cuda.device_array(hashRange+1, dtype=np.int32)
d_PrefixSum       = cuda.device_array(hashRange+1, dtype=np.int32)

d_BinCounterArray = cuda.device_array(numBins+1, dtype=np.int32)
d_BinPrefixSum    = cuda.device_array(numBins+1, dtype=np.int32)

print(d_HashA.shape[0])
threads_per_block = 128
blocks_per_grid   = 512


for i in range(4):

  gdf = DataFrame()
  # d_inputA = cuda.to_device(inputA)
  cuda.synchronize()

  gdf["a"] = d_inputA

  start = time.time()


  d_HashA = gdf.hash_columns(["a"])
  normailizeHashArray[blocks_per_grid, threads_per_block](d_HashA,hashRange)

  resetCounter[blocks_per_grid, threads_per_block](d_BinCounterArray)
  countHashBins[blocks_per_grid, threads_per_block](d_HashA,d_BinCounterArray,binSize)
  d_BinPrefixSum = numba.cuda.to_device(cupy.cumsum(d_BinCounterArray,dtype=np.int32))

  # print(numba.cuda.to_device(d_HashA).copy_to_host())
  # print(d_BinCounterArray.copy_to_host())
  # print(numba.cuda.to_device(d_BinPrefixSum).copy_to_host())
  # continue
  
  resetCounter[blocks_per_grid, threads_per_block](d_BinCounterArray)
  reOrderInput[blocks_per_grid, threads_per_block](d_inputA,d_HashA,d_InputAReOrdered, binSize, d_BinCounterArray, d_BinPrefixSum)

  # cuda.synchronize()
  # continue;

  gdf["b"] = d_InputAReOrdered
  d_HashA = gdf.hash_columns(["b"])
  normailizeHashArray[blocks_per_grid, threads_per_block](d_HashA,hashRange)
  
  resetCounter[blocks_per_grid, threads_per_block](d_CounterArray)
  bp2 = (inputSize + (threads_per_block - 1)) // threads_per_block
  # # print(bp2)
  # # print("hi")
  reOrderHash[bp2, threads_per_block](d_HashA, d_CounterArray)
  # reOrderHash[blocks_per_grid, threads_per_block](d_HashA, d_CounterArray)

  d_PrefixSum = numba.cuda.to_device(cupy.cumsum(d_CounterArray,dtype=np.int32))

  resetCounter[blocks_per_grid, threads_per_block](d_CounterArray)
  # fillTable[blocks_per_grid, threads_per_block](d_Table,d_HashA,d_InputAReOrdered,numBins, d_CounterArray,d_PrefixSum)

  fillTable[bp2, threads_per_block](d_Table,d_HashA,d_InputAReOrdered, d_CounterArray,d_PrefixSum)

# # hashA, hashAReordered, binCounts, counterArray, prefixArray
  # print(d_CounterArray.copy_to_host())
#   print(cupy.asnumpy(d_BinPrefixSum))

#   stam = numba.cuda.to_device(d_BinPrefixSum)

#   print(numba.cuda.is_cuda_array(stam))
#   print(numba.cuda.is_cuda_array(d_BinPrefixSum))
#   print(numba.cuda.is_cuda_array(d_CounterArray))

#   print(type(stam))
#   print(type(d_BinPrefixSum))
#   print(type(d_CounterArray))
#   print(d_BinPrefixSum.dtype)
  cuda.synchronize()
  thisTime = time.time()-start
  print('Time taken = %2.6e seconds'%(thisTime))
  print('Rate = %.5e seconds'%((inputSize)/thisTime))
  # print(d_HashA)
# res = d_hash.copy_to_host()
#print(number)
# print(np.sum(res, dtype=np.int64))

# out_one = cupy.asnumpy(gdf.hash_columns(["a"]))
# print(out_one)

