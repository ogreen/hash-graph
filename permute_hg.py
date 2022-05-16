import time
from numba import cuda
import numpy as np
import cupy as cp
from hg import HashGraph
import cudf as cudf

import rmm
pool = rmm.mr.PoolMemoryResource(
                rmm.mr.CudaMemoryResource(),
                initial_pool_size=2**33,
                maximum_pool_size=2**35
)
rmm.mr.set_current_device_resource(pool)

inputSize = 10e0 #1<<22
low = 0
high = inputSize 
# hashRange = inputSize >> 2
# numBins = 1<<14
# binSize = (hashRange+numBins-1)//numBins


def mainPermute():
        # inputA = np.arange(low,inputSize,step=1,dtype=np.int32) 
        # d_inputA = cp.asnumpy(inputA)

        d_inputA = cp.arange(low,inputSize,step=1,dtype=np.int32) 

        # d_inputA           = cuda.device_array(inputA.shape, dtype=inputA.dtype)
        # d_inputA           = cuda.to_device(inputA)


        cuda.synchronize()
        start = time.time()
        hg1 = HashGraph(d_inputA,True,seedType="user-defined",seed=211)
        thisTime = time.time()-start
        print('Time taken = %2.6e seconds'%(thisTime))
        print('Rate = %.5e keys/sec'%((inputSize)/thisTime))

        # print(hg1.d_table[0:16])
        print(hg1.d_indices[0:16])


        hgDet1 = HashGraph(d_inputA,True,seedType="deterministic")
        print("Determinstic")
        print(hgDet1.d_indices[0:16])
        hgDet2 = HashGraph(d_inputA,True,seedType="deterministic")
        print("Determinstic")
        print(hgDet2.d_indices[0:16])

        hgUD1 = HashGraph(d_inputA,True,seedType="user-defined",seed=0)
        print("User defined - seed ")
        print(hgUD1.d_indices[0:16])
        print("User defined - different seed")
        hgUD2 = HashGraph(d_inputA,True,seedType="user-defined",seed=231)
        print(hgUD2.d_indices[0:16])
        print("Random seed")
        hgRand1 = HashGraph(d_inputA,True,seedType="random")
        print(hgRand1.d_indices[0:16])
        print("Random seed")
        hgRand2 = HashGraph(d_inputA,True,seedType="random")
        print(hgRand2.d_indices[0:16])

        # cudf.Series(hgDet1).head()
        permuteUnique = (cudf.DataFrame(hgRand2.d_indices))
        # print(permuteUnique)
        # print(type(permuteUnique))
        print(permuteUnique[0].nunique())
if __name__ == "__main__":
    mainPermute()
