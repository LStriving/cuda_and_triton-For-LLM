# cuda_and_triton-For-LLM
Awesome Learning materials for CUDA and Triton coding for LLM

---

## **1. Official Documentation and Guides**
### **CUDA**
- **NVIDIA CUDA Toolkit Documentation**: Comprehensive guides for CUDA programming, including best practices and examples.  
  [Link](https://docs.nvidia.com/cuda/) 
- **CUDA C++ Programming Guide**: Detailed explanations of CUDA architecture and programming models.  
  [Link](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) 

### **Triton**
- **Triton Documentation**: Official documentation for Triton, including installation, kernel writing, and advanced features.  
  [Link](https://triton-lang.org/) 
- **Triton User Guide**: A practical guide to using Triton for GPU programming, with examples and tutorials.  
  [Link](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/getting_started/trtllm_user_guide.html) 

---

## **2. Code Repositories**
### **CUDA**
- **NVIDIA/cuda-samples**: A collection of CUDA samples demonstrating various features and optimizations.  
  [Link](https://github.com/NVIDIA/cuda-samples) 
- **Awesome-CUDA-Triton-HPC**: A curated list of CUDA, Triton, and HPC projects, including LLM-related resources.  
  [Link](https://github.com/coderonion/awesome-cuda-triton-hpc) 

### **Triton**
- **TensorRT-LLM Backend**: A repository for serving LLMs with Triton Inference Server using TensorRT-LLM.  
  [Link](https://github.com/NVIDIA/TensorRT-LLM) 
- **Liger Kernel**: A collection of Triton kernels optimized for LLM training, including RMSNorm, RoPE, and Flash Attention.  
  [Link](https://github.com/linkedin/Liger-Kernel) 

---

## **3. Tutorials and Videos**
### **CUDA**
- **CUDA Crash Course (YouTube)**: A series of videos by CoffeeBeforeArch covering CUDA programming basics and optimizations.  
  [Link](https://www.youtube.com/playlist?list=PLxNPSjHT5qvtYRVdNN1yDcdSl39uHV_sU) 
- **CUDA Programming Guide in Chinese**: A Chinese translation of the CUDA programming guide for non-English speakers.  
  [Link](https://github.com/HeKun-NVIDIA/CUDA-Programming-Guide-in-Chinese) 

### **Triton**
- **GPU MODE Lecture 14: Practitioners Guide to Triton**: A detailed lecture on Triton, including kernel writing and benchmarking.  
  [Link](https://christianjmills.com/posts/cuda-mode-notes/lecture-014/) 
- **Triton Tutorials (NVIDIA)**: Tutorials for deploying LLMs with Triton Inference Server, including Phi-3 and other models.  
  [Link](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/README.html) 

---

## **4. Advanced Topics**
### **CUDA-Free Inference for LLMs**
- **PyTorch Blog on CUDA-Free Inference**: A blog post discussing how to achieve 100% Triton-based inference for LLMs like Llama3-8B and Granite-8B.  
  [Link](https://pytorch.org/blog/cuda-free-inference-for-llms/) 
- **Triton SplitK GEMM Kernel**: A custom Triton kernel for matrix multiplication, optimized for LLM inference.  
  [Link](https://pytorch.org/blog/cuda-free-inference-for-llms/) 

### **Performance Optimization**
- **GenAI-Perf**: A tool for benchmarking LLM performance on Triton Inference Server.  
  [Link](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/getting_started/llm.html) 
- **Triton Metrics**: Learn how to query Triton metrics for GPU and request statistics.  
  [Link](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/getting_started/trtllm_user_guide.html) 

---

## **5. Community and Learning Resources**
- **Triton Chinese Documentation**: The first complete Chinese translation of Triton documentation, making it accessible to Chinese developers.  
  [Link](https://triton.hyper.ai/) 
- **Triton Resources on GitHub**: A curated list of Triton resources, including tutorials, kernels, and academic papers.  
  [Link](https://github.com/rkinas/triton-resources) 

