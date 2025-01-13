# NICE
Pytorch implementation of [Nonlinear Independent Components Estimation](https://arxiv.org/pdf/1410.8516). 

NICE introduced normalizing flows, models that learn differentiable, bijective mappings between data space and a simple latent space. Their bijective approach to generative modeling and clever design
puts them among the most fascinating classes of deep learning models. While NICE's performance is weak by today's standards, its descendants (RealNVP, GLOW, and others) learn more expressive mappings
and generate more impressive samples.

While I aim to reconstruct the authors' description of NICE as closely as possible, I found the prescribed hyperparameters for Adam to cause instability during training, so a learning rate of 0.0002
and default Adam hyperparameters are used instead.

## Usage

### Quick Start

Run `python main.py` to initialize a NICE model, load pre-trained weights, generate images, and save them to the assets folder.

### Train

Run `python main.py --train --epochs=<your_epochs>` Add `--fresh` to start training from scratch instead of loading pretrained weights. Additional flags are available for hyperparameters. Pretrained weights are held in `nice_model_weights.pth` and your weights will be stored in the same place by default.

You'll notice that the loss (**negative** of the NICE criterion, ie negative log-likelihood of the transformed data under the base distribution) dips below 0 and continues to decrease. 

Upon inspection of the NICE criterion, a positive result is actually expected. While the pdf of the base distribution never exceeds one (thus keeping its log negative, and its negative log positive), 
the other addend of the loss, the log Jacobian determinant, can be (very) positive. The Jacobian determinant is the product of the exponentiated values in the diagonal scaling matrix S, since it 
remains a constant 1 through all coupling layers. These exponentiated S values are only bounded by being positive, so the log of their product can also be positive. For large values, 
this term dominates log-likelihood, making it positive, and the negative log-likelihood negative.

![NICE_Criterion](assets/NICE_criterion.png)

### Experiment

Run `python experiments.py` to generate the visualizations seen in the next section. As noted by the authors, the latent vector elements with the lowest corresponding exp(S_ii) coefficients 
are the most influential. Identifying and modifiying these influential latents open the door for a number of experiments. See section 3.3 of the paper and `experiment.py` for details. 

![multi_img_exp](assets/multi_img_exp)
![simg_img_exp](assets/single_img_exp)

# RealNVP

In production...








