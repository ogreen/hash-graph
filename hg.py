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
def normailizeHashArray(inputArray, hashRange):
    start  = cuda.grid(1)
    stride = cuda.gridsize(1)
    size   = inputArray.shape[0]
    for idx in range(start, size, stride):
        inputArray[idx] = inputArray[idx]%hashRange

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
        mybin = int(hashA[idx]//binSize)+1
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
    idx  = cuda.grid(1)
    size   = hashAReordered.shape[0]
    if(idx<size):
      cuda.atomic.add(counterArray,hashAReordered[idx]+1,1)

@cuda.jit
def fillTable(table,hashAReordered,d_InputAReOrdered, counterArray,prefixArray):
    idx  = cuda.grid(1)
    size   = hashAReordered.shape[0]
    if(idx<size):
      pos = cuda.atomic.add(counterArray,hashAReordered[idx],1)+prefixArray[hashAReordered[idx]]
      table[pos] = d_InputAReOrdered[idx]



class HashGraph:

  threads_per_block = 512
  blocks_per_grid   = 512


  def __init__(self, d_inputA, numBins=1<<14):

    self.numBins = numBins
    self.hashRange = (d_inputA.shape[0]) >> 2
    self.binSize = (self.hashRange+numBins-1)//numBins

    self.d_table           = cuda.device_array(d_inputA.shape, dtype=inputA.dtype)
    self.d_PrefixSum       = cuda.device_array(self.hashRange+1, dtype=np.uintc)
    self.build()

  def build(self):
    d_InputAReOrdered  = cuda.device_array(d_inputA.shape, dtype=inputA.dtype)

    d_HashA            = cuda.device_array(d_inputA.shape, dtype=np.uintc)
    # d_hashA           = cuda.device_array(d_inputA.shape, dtype=np.uintc)

    # Arrays used for HashGraph proceess (size of the hash-range)
    d_CounterArray    = cuda.device_array(self.hashRange+1, dtype=np.uintc)


    d_BinCounterArray = cuda.device_array(self.numBins+1, dtype=np.uintc)
    d_BinPrefixSum    = cuda.device_array(self.numBins+1, dtype=np.uintc)


    start = time.time()

    gdf = DataFrame()

    gdf["a"] = d_inputA
    gdf["b"] = d_InputAReOrdered
    d_HashA = gdf.hash_columns(["a"])
    normailizeHashArray[HashGraph.blocks_per_grid,HashGraph.threads_per_block](d_HashA,self.hashRange)

    resetCounter[HashGraph.blocks_per_grid, HashGraph.threads_per_block](d_BinCounterArray)
    countHashBins[HashGraph.blocks_per_grid, HashGraph.threads_per_block](d_HashA,d_BinCounterArray,self.binSize)
    d_BinPrefixSum = numba.cuda.to_device(cupy.cumsum(d_BinCounterArray,dtype=np.uintc))

    resetCounter[HashGraph.blocks_per_grid, HashGraph.threads_per_block](d_BinCounterArray)
    reOrderInput[HashGraph.blocks_per_grid, HashGraph.threads_per_block](d_inputA,d_HashA,d_InputAReOrdered, self.binSize, d_BinCounterArray, d_BinPrefixSum)

    # continue;
    d_hashA = gdf.hash_columns(["b"])
    normailizeHashArray[HashGraph.blocks_per_grid, HashGraph.threads_per_block](d_hashA,self.hashRange)
      
    
    resetCounter[HashGraph.blocks_per_grid, HashGraph.threads_per_block](d_CounterArray)
    bp2 = (inputSize + (HashGraph.threads_per_block - 1)) // HashGraph.threads_per_block
    reOrderHash[bp2, HashGraph.threads_per_block](d_hashA, d_CounterArray)


    self.d_PrefixSum = numba.cuda.to_device(cupy.cumsum(d_CounterArray,dtype=np.uintc))
    resetCounter[HashGraph.blocks_per_grid, HashGraph.threads_per_block](d_CounterArray)
    fillTable[bp2, HashGraph.threads_per_block]            (self.d_table,d_hashA,d_InputAReOrdered, d_CounterArray,self.d_PrefixSum)

    cuda.synchronize()

    thisTime = time.time()-start
    print('Time taken = %2.6e seconds'%(thisTime))
    print('Rate = %.5e keys/sec'%((inputSize)/thisTime))




inputSize = 1<<28
low = 0
high = inputSize 
# hashRange = inputSize >> 2
# numBins = 1<<14
# binSize = (hashRange+numBins-1)//numBins

inputA = np.random.randint(low,high,inputSize,dtype=np.int32)


d_inputA           = cuda.device_array(inputA.shape, dtype=inputA.dtype)
d_inputA           = cuda.to_device(inputA)


for i in range(10):

  start = time.time()
  hg = HashGraph(d_inputA)
  thisTime = time.time()-start
  print('Time taken = %2.6e seconds'%(thisTime))
  print('Rate = %.5e keys/sec'%((inputSize)/thisTime))



