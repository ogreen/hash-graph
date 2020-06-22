#!/usr/bin/env python
# coding: utf-8



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
def reOrderInput(inputA,hashA, d_InputAReOrdered, binSize, counterArray, prefixArray, indices, needIndices):
    start  = cuda.grid(1)
    stride = cuda.gridsize(1)
    size   = inputA.shape[0]
    for idx in range(start, size, stride):
        mybin = int(hashA[idx]//binSize)
        pos = cuda.atomic.add(counterArray,mybin,1) + prefixArray[mybin]
        d_InputAReOrdered[pos]=inputA[idx] # Todo store index
        if(needIndices==True):
            indices[pos]=idx

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

@cuda.jit(device=True)
def compAndFindFirst(val1, val2, array,index):
    if (val1==val2):
        array[index] = 1
        return False


#queryType = 0 // Check existance
#queryType = 1 // Count number of instances
#queryType = 2 // 


@cuda.jit
def queryKernel(tableA,prefixArrayA,tableB,prefixArrayB,returnArray, queryType):
    idx  = cuda.grid(1)
    size   = prefixArrayA.shape[0]-1
    if(idx<size):
        sizeAList=prefixArrayA[idx+1]-prefixArrayA[idx];
        sizeBList=prefixArrayB[idx+1]-prefixArrayB[idx];
        
        # queryFunc = GetFuncByFlag(0)

        if(sizeBList==0 or sizeAList==0):
            return

        for b in range(sizeBList):
            bVal = tableB[prefixArrayB[idx]+b]

            for a in range(sizeAList):
                aVal = tableA[prefixArrayA[idx]+a]
                if(queryType==0):
                    if(aVal==bVal):
                        returnArray[prefixArrayB[idx]+b]=1
                        break
                    continue
                if(queryType==1):
                    if(aVal==bVal):
                        returnArray[prefixArrayB[idx]+b]=returnArray[prefixArrayB[idx]+b]+1
                    continue
                # if(queryType==2):
                #     if(aVal==bVal):
                #         returnArray[prefixArrayB[idx]+b]=1
                #         break
                #     continue



class HashGraph:

  threads_per_block = 512
  blocks_per_grid   = 512


  def __init__(self, d_inputA, indices=False, numBins=1<<14):

    self.numBins = numBins
    self.hashRange = (d_inputA.shape[0]) >> 2
    self.binSize = (self.hashRange+numBins-1)//numBins

    # self.d_table           = cuda.device_array(d_inputA.shape, dtype=inputA.dtype)
    # self.d_PrefixSum       = cuda.device_array(self.hashRange+1, dtype=np.uintc)
    if (indices==False):
        self.d_table, self.d_PrefixSum = self.build(d_inputA,False)
    else:
        self.d_table, self.d_PrefixSum, d_indices = self.build(d_inputA,True)


  def build(self, d_input, indices=False):
    d_InputAReOrdered  = cuda.device_array(d_input.shape, dtype=inputA.dtype)

    d_HashA            = cuda.device_array(d_input.shape, dtype=np.uintc)
    # d_hashA           = cuda.device_array(d_input.shape, dtype=np.uintc)

    # Arrays used for HashGraph proceess (size of the hash-range)
    d_CounterArray    = cuda.device_array(self.hashRange+1, dtype=np.uintc)


    d_BinCounterArray = cuda.device_array(self.numBins+1, dtype=np.uintc)
    d_BinPrefixSum    = cuda.device_array(self.numBins+1, dtype=np.uintc)

    d_table           = cuda.device_array(d_input.shape, dtype=inputA.dtype)
    d_PrefixSum       = cuda.device_array(self.hashRange+1, dtype=np.uintc)

    # d_indices         = None
    # if(indices==True):
    d_indices = cuda.device_array(d_input.shape, dtype=np.uintc)

    start = time.time()

    gdf = DataFrame()

    gdf["a"] = d_input
    gdf["b"] = d_InputAReOrdered
    d_HashA = gdf.hash_columns(["a"])
    normailizeHashArray[HashGraph.blocks_per_grid,HashGraph.threads_per_block](d_HashA,self.hashRange)

    resetCounter[HashGraph.blocks_per_grid, HashGraph.threads_per_block](d_BinCounterArray)
    countHashBins[HashGraph.blocks_per_grid, HashGraph.threads_per_block](d_HashA,d_BinCounterArray,self.binSize)
    d_BinPrefixSum = numba.cuda.to_device(cupy.cumsum(d_BinCounterArray,dtype=np.uintc))

    resetCounter[HashGraph.blocks_per_grid, HashGraph.threads_per_block](d_BinCounterArray)
    reOrderInput[HashGraph.blocks_per_grid, HashGraph.threads_per_block](d_input,d_HashA,d_InputAReOrdered, self.binSize, d_BinCounterArray, d_BinPrefixSum, d_indices, indices)

    # continue;
    d_hashA = gdf.hash_columns(["b"])
    normailizeHashArray[HashGraph.blocks_per_grid, HashGraph.threads_per_block](d_hashA,self.hashRange)
      
    
    resetCounter[HashGraph.blocks_per_grid, HashGraph.threads_per_block](d_CounterArray)
    bp2 = (inputSize + (HashGraph.threads_per_block - 1)) // HashGraph.threads_per_block
    reOrderHash[bp2, HashGraph.threads_per_block](d_hashA, d_CounterArray)


    d_PrefixSum = numba.cuda.to_device(cupy.cumsum(d_CounterArray,dtype=np.uintc))
    resetCounter[HashGraph.blocks_per_grid, HashGraph.threads_per_block](d_CounterArray)
    fillTable[bp2, HashGraph.threads_per_block]            (d_table,d_hashA,d_InputAReOrdered, d_CounterArray,d_PrefixSum)

    cuda.synchronize()

    thisTime = time.time()-start
    print('Time taken = %2.6e seconds'%(thisTime))
    print('Rate = %.5e keys/sec'%((inputSize)/thisTime))
    if(indices==False):
        return d_table, d_PrefixSum
    else:
        return d_table, d_PrefixSum, d_indices


  def queryExistance(self, d_inputB):
    d_tableB, d_PrefixSumB, d_indicesB = self.build(d_inputB,True)

    # d_flagArray = cuda.device_array(d_inputB.shape, dtype=np.int8)
    d_flagArray = cupy.zeros(d_inputB.shape, dtype=np.int8)

    bp2 = (self.hashRange + (HashGraph.threads_per_block - 1)) // HashGraph.threads_per_block


    queryKernel[bp2, HashGraph.threads_per_block](self.d_table, self.d_PrefixSum, d_tableB, d_PrefixSumB, d_flagArray, 0)
    return d_flagArray

  def queryCount(self, d_inputB):
    d_tableB, d_PrefixSumB, d_indicesB = self.build(d_inputB,True)

    # d_flagArray = cuda.device_array(d_inputB.shape, dtype=np.int8)
    d_countArray = cupy.zeros(d_inputB.shape, dtype=np.int8)

    bp2 = (self.hashRange + (HashGraph.threads_per_block - 1)) // HashGraph.threads_per_block


    queryKernel[bp2, HashGraph.threads_per_block](self.d_table, self.d_PrefixSum, d_tableB, d_PrefixSumB, d_countArray, 1)
    return d_countArray


inputSize = 1<<28
low = 0
high = inputSize 
# hashRange = inputSize >> 2
# numBins = 1<<14
# binSize = (hashRange+numBins-1)//numBins

np.random.seed(123)



for i in range(5):

    inputA = np.random.randint(low,high,inputSize,dtype=np.int32)
    # inputB = np.random.randint(low,high,inputSize,dtype=inputA.dtype)
    inputB = inputA

    d_inputA           = cuda.device_array(inputA.shape, dtype=inputA.dtype)
    d_inputA           = cuda.to_device(inputA)

    d_inputB           = cuda.device_array(inputB.shape, dtype=inputA.dtype)
    d_inputB           = cuda.to_device(inputB)


    start = time.time()
    hg = HashGraph(d_inputA)
    thisTime = time.time()-start
    print('Time taken = %2.6e seconds'%(thisTime))
    print('Rate = %.5e keys/sec'%((inputSize)/thisTime))


    start = time.time()
    flagArray   = hg.queryExistance(d_inputB);
    coubntArray = hg.queryCount(d_inputB);
    thisTime = time.time()-start
    print(flagArray.sum())
    print('Time taken = %2.6e seconds'%(thisTime))
    print('Rate = %.5e keys/sec'%((inputSize)/thisTime))

    print("")

