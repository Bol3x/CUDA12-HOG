# GROUP 3 - CUDA12-HOG

## CEPARCO - S11
### Members
#### Ryan Onil Barrion, Michael Robert Geronimo, Jan Carlo Roleda, Frances Danielle Solis 

## Live demo video - https://youtu.be/gwUNDz368SY

## a.) Discussion of parallel algorithms implemented in your program

The following algorithms were parallelized using CUDA:
1. Computation of Gradients
2. Binning of Gradient Orientations from 8x8 pixel blocks
3. Normalization of the Gradients

Computing the gradients can be done per-element, as each magnitude and angle can be computed independent of other threads. Distributing these magnitudes into respective orientation bins can be done per-element as well, but will require synchronization between memory writes to prevent race conditions. Finally, L2 normalization of the orientation bins into a long vector sequence can be done after the binning process is completed. This is performed with a 2x2 block, each cell consisting of the 9 orientation bins obtained from the 8x8 pixel blocks, resulting in a (rows / 8 - 1) * (cols / 8 - 1) * 36 element vector.

## b.) Execution time comparison between sequential and parallel
### System Specifications of device used 
![image](https://github.com/Bol3x/CUDA12-HOG/assets/115066447/8a21600d-9bb5-4ba8-916e-c67b1c0b83c6)

### -1st Test (100 runs)
![image](https://github.com/Bol3x/CUDA12-HOG/assets/115066447/282a6597-846e-4b32-a254-687e1a8e207a)

#### Figure 1. Sample Result of HOG Algorithm with input shiba_inu_69 
![image](https://github.com/Bol3x/CUDA12-HOG/assets/115066447/5254af6f-a088-40a9-885e-90dd3fd41c0d)


### -2nd Test (100 runs)
![image](https://github.com/Bol3x/CUDA12-HOG/assets/115066447/9a88260c-b56f-40ed-be61-ee02591d8ace)


#### Figure 2. Sample result of HOG Algorithms for shiba_inu_72.jpg
![image](https://github.com/Bol3x/CUDA12-HOG/assets/115066447/ffea2029-344d-4831-884e-534721d9e3c5)


## c.) Detailed Analysis and Discussion of Results
The testing of the algorithms was implemented both in C++ and CUDA and were both run on the same device with the provided specs in table 1. The algorithm was run 100 times with the execution of both the serial and parallel implementation timed using CUDA’s cudaEvent_t class. To compare between the two, the speedup was calculated by taking the sequential (C++) execution time over the parallelized (CUDA) execution time.

In table 2, the dimensions had a ratio of 2 x 1 wherein the width is double the height. As the dimensions were increased, the speedup also increased, with the speedup at a limit of approximately 15 times over the sequential execution. In table 3, the image utilizes a dimension of square where the width and the height is of equal value. The speedup also increased and reached up to 14 times over the sequential execution.

The implemented algorithms in CUDA distributed 1 thread per 1-2 pixels whereas fastHOG used 1 thread per 8 pixel column in a block. The implemented algorithms in CUDA employed unified memory to facilitate the displaying of the HOG features, and storing the L2 norm coefficients for each 2x2 feature block.


### Improvements
Some improvements that could be considered for this project is to incorporate shared memory usage since using shared memory is noted to be faster than global memory due to being on-chip. It was not used in this case due to difficulties in proper implementation and cross-thread block dependencies. It would be particularly useful in the orientation binning of each 8x8 pixel block. 

Furthermore, implementing a reduction sum in the computation of the L2 norm should also be considered to improve complexity, as it was implemented in the normalization of the fastHOG algorithm. Due to the abovementioned issues with cross-thread block dependencies, difficulties were observed when trying to implement shared memory to compute with a reduction sum. Currently, the L2 norm coefficients are computed on each 36th thread in a sequential manner, increasing idle time for other threads and being generally inefficient.
