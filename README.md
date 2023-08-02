# GROUP 3 - CUDA12-HOG

## CEPARCO - S11
### Members
#### Ryan Onil Barrion, Michael Robert Geronimo, Jan Carlo Roleda, Frances Danielle Solis 

## Live demo video - https://youtu.be/gwUNDz368SY

## a.) Discussion of parallel algorithms implemented in your program

The following algorithms were parallelized using CUDA:
1. Computation of Gradients
2. Orientation of binning 
3. Normalization of the Gradients

//todo explain the parallelization

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
   it is important to note that the testing of the HOG algorithms that was implemented both in C++ and CUDA were both ran on the same device in which the specs was provided.The algorithm was ran 100 times and the outputs of both the serial and parallel implementation  were timed using CUDA’s cudaEvent_t classes. To determine the comparison between the two, the speedup formula was used which is calculated by taking the old execution time and dividing it to the new execution time.

   In table 1, the dimensions had a ratio of 2 x 1 wherein the width is double the height. 



