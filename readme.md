## FreeDiff: Progressive Frequency Truncation for Image Editing with Diffusion Models

This is the official implementation of the ECCV 24 paper "FreeDiff: Progressive Frequency Truncation for
Image Editing with Diffusion Models". 


A more detailed introduction of this project(This readme file), more examples of editing of different types of editing in a ipynb file, our results and the datasets will be released gradually.


## Usage

### Requirements
We implement our method with a similar code structure to [Prompt-to-Prompt](https://github.com/google/prompt-to-prompt). The code runs on Python 3.10.12 with Pytorch 2.1.0 and Diffusers 0.27.2. We believe that mild version alterations of Pytorch and Diffusers will not affect the code much.

### Checkpoints
We mainly examine our method on public available pretrained stable diffusion models SD v1-4("CompVis/stable-diffusion-v1-4") and SD v1-5("runwayml/stable-diffusion-v1-5").


## Acknowledgements
We thank the awesome research work [Prompt-to-Prompt](https://github.com/google/prompt-to-prompt)

