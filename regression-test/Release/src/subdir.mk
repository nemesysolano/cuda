################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/regression-test.cu 

OBJS += \
./src/regression-test.o 

CU_DEPS += \
./src/regression-test.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I"/home/rsolano/Documents/workspaces/cuda/matrix" -I"/home/rsolano/Documents/workspaces/cuda/regression" -O3 -gencode arch=compute_50,code=sm_50  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I"/home/rsolano/Documents/workspaces/cuda/matrix" -I"/home/rsolano/Documents/workspaces/cuda/regression" -O3 --compile --relocatable-device-code=false -gencode arch=compute_50,code=compute_50 -gencode arch=compute_50,code=sm_50  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


