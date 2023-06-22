# RoboCAT
Implementation of Deepmind's <a href="https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/robocat-a-self-improving-robotic-agent/robocat-a-self-improving-foundation-agent-for-robotic-manipulation.pdf">RoboCat</a>, Self-Improving Foundation Agent for Robotic Manipulation




### Architecture
The RoboCat architecture is based on the Gato model and a VQ-GAN encoder. The Gato model is an autoregressive transformer model, while the VQ-GAN encoder is pretrained on a diverse set of images for faster training and better generalization.



### Algorithmic Pseudocode
```
Pretrain VQ-GAN encoder on a diverse set of images.
Tokenize proprioceptive observations, agent actions, and images using the pretrained VQ-GAN encoder.
Train the RoboCat agent using a dataset containing tokenized trajectories and a token prediction loss.
Fine-tune the RoboCat agent on a small dataset of new episodic experience (100-1000 demonstrations).
Deploy the fine-tuned RoboCat agent to gather additional data for self-improvement.
Train a new iteration of the RoboCat agent using the updated dataset.
Deploy the RoboCat agent on real-world robotic embodiments.
```

## Requirements
To reproduce the paper, the following requirements are needed:

Training and task specification

Vision-based tabletop object manipulation tasks
Goal-conditioned agent represented by a policy 𝜋(𝑎𝑡 | 𝑜𝑡, 𝑔𝑡)
Autoregressive transformer model for 𝜋(𝑎𝑡 | 𝑜𝑡, 𝑔𝑡)
Dataset D of trajectories transformed into a dataset of tokenized trajectories Dˆ
Trajectories augmented with goal images (hindsight goals or semantically-equivalent goals)
Architecture and pretraining

Transformer architecture based on Gato (Reed et al., 2022)
Tokenization of proprioceptive observations and agent actions
Image tokenization using a pretrained and frozen VQ-GAN (Esser et al., 2021)
Pretraining VQ-GAN encoder on a diverse collection of images
Training the agent model using a dataset Dˆ and a standard token prediction loss
Fine-tuning and self-improvement

Collecting demonstrations per task via teleoperation
Fine-tuning the generalist RoboCat agent on the demonstrations
Self-improvement by deploying fine-tuned policies to collect additional trajectories
Hindsight goal relabeling and constructing a new training dataset
Training a new generalist RoboCat agent using the updated dataset
Components and Requirements
Dataset

A diverse collection of images for pretraining the VQ-GAN encoder
Trajectories for training the generalist RoboCat agent
Demonstrations for fine-tuning and self-improvement
VQ-GAN

Pretrained and frozen VQ-GAN for image tokenization
Encoder for encoding input images into a series of latent vectors
Codebook of quantized embeddings for discretization
Gato-based Transformer Model

Autoregressive transformer model for representing the goal-conditioned agent policy
Tokenization of proprioceptive observations, agent actions, and images
Training using a standard token prediction loss
Fine-tuning and Self-improvement Process

Teleoperation for collecting demonstrations per task
Fine-tuning the generalist RoboCat agent on the demonstrations
Deploying fine-tuned policies to collect additional trajectories
Hindsight goal relabeling and constructing a new training dataset
Training a new generalist RoboCat agent using the updated dataset

Success Detector or Reward Function
The success detector or reward function is an essential component in the system, as it helps determine if a trajectory is successful or not. This information is crucial for goal selection and self-improvement data collection.

Goal Selection: When augmenting tokenized trajectories with goal images, the success detector is used to identify successful episodes. This allows the system to choose between hindsight goals (last image of the same episode) or semantically-equivalent goals (last image of a different successful episode).

Self-improvement Data Collection: During the self-improvement process, the success detector or reward function is used to identify successful trajectories for a given task. This information is necessary for constructing the self-improvement dataset, which is then used to train the next iteration of the generalist RoboCat agent.

To implement a success detector or reward function, you need to define a set of criteria that determine whether a trajectory is successful or not. These criteria will depend on the specific tasks and the dataset being used. For example, in a task where the goal is to insert an apple into a bowl, the success criteria could be based on the position and orientation of the apple and the bowl in the final state of the trajectory.

In some cases, it might be beneficial to use a learned reward model instead of a handcrafted success detector. A learned reward model can be trained on a set of demonstrations or expert trajectories to predict the success of a given trajectory. This approach can be more flexible and adaptive to different tasks and scenarios.


To detect task success during real-world deployment, vision-based reward models are trained. These models are trained on human demonstrations and data from policies trained to perform the task. The episodes are annotated via a crowd-sourcing interface, where annotators mark the time step after which the task is solved in each episode (if at all), resulting in binary annotations. These annotations are then used to train a binary classifier that can detect task success from image observations at any given time step.

## Real-world Deployment Challenges
Two main challenges arise during real-world deployment: success classification and task resets. Success detection is necessary for the hindsight goal relabeling of semantically-equivalent goals and for determining when a reset is needed. Task resets are required when the agent has completed a task or reached an undesired state, and the environment needs to be reset to a valid starting state.



# Appendex list:

D. VQ-GAN Training Details
Datasets

RoboCat-lim VQ-GAN: Trained on ImageNet, DeepMind Control Suite, Metaworld, domain randomised sim sawyer red-on-blue-stacking data, sim panda data, and real sawyer red-on-blue stacking data.
RoboCat VQ-GAN: Trained on additional data from a simulated version of the YCB fruit lifting task, NIST-i gear task images (real and sim), and real sawyer data of a red-on-blue agent being run with random objects in the bin, including YCB fruits and vegetables.
Model architecture and loss

Derived from Esser et al. (2021), combining convolutional layers with Transformer attention layers to encode the image.
Encoder: ResNet with 2 groups of 2 blocks, followed by 3 attention layers.
Decoder: 3 attention layers followed by a ResNet with 2 groups of 3 blocks.
Vector quantiser: 64 embeddings of dimension 128.
Loss: Weighted as (0.25 * discretisation_loss + l2_reconstruction_loss + 0.1 * log_laplace_loss).
Trained with a batch size of 64 for 1 million training steps.
Ablations for VQ-GAN design choices

Comparisons between VQ-GAN tokeniser and patch ResNet tokeniser.
VQ-GAN tokeniser performs better for generalisation and adaptation.
Predicting future image pixels does not provide a similar advantage.
E. Training and Fine-tuning Parameters
Training parameters

AdamW optimiser with linear warmup and cosine schedule decay.
Batch size of 256 and sequence length of 1024 tokens.
Stochastic depth during pretraining.
Fine-tuning parameters

Adam optimiser with a constant learning rate of 1e-5.
Batch size of 32 and sequence length of 1024 tokens.
Dropout with a rate of 0.1.
F. Success Detection and Policy Pools
Success detection

Treat success detection for each task as a per-time step binary classification problem.
Annotation-efficient annotation procedure to mark transition points between solved and unsolved states.
Policy pools as generalised reset policies

Provides resets for non-self-resetting tasks, generates diverse initial conditions for evaluation, and makes efficient use of robot time by interleaving the evaluation of multiple policies at a time.
Groups together a set of diverse tasks and allows scheduling episodes of different tasks to be run in sequence.
Generates a wider variety of initial conditions than a single fixed reset policy.
Components and Requirements
Datasets

Diverse datasets for training the VQ-GAN and RoboCat models.
VQ-GAN

Pretrained and frozen VQ-GAN for image tokenization.
Encoder and decoder architecture for encoding input images into a series of latent vectors.
Codebook of quantized embeddings for discretization.
Training and Fine-tuning Parameters

Optimizers, learning rates, batch sizes, and sequence lengths for training and fine-tuning the models.
Success Detection and Policy Pools

Success detection mechanism for determining the success of a trajectory.
Policy pools for handling resets, generating diverse initial conditions, and efficiently using robot time.


# Roadmap

* Functional prototype

* Train on massive datasets

* Finetune as specified on paper

* Release as paid API

* Integrate more modalities like hearing, 3d mapping, nerfs, videos, lidar, locomotion, and the whole lot!


## Citations

```bibtex
@article{Bousmalis2023RoboCat,
    title   = {RoboCat: A Self-Improving Foundation Agent for Robotic Manipulation},
    author  = {Konstantinos Bousmalis*, Giulia Vezzani*, Dushyant Rao*, Coline Devin*, Alex X. Lee*, Maria Bauza*, Todor Davchev*, Yuxiang Zhou*, Agrim Gupta*,1, Akhil Raju, Antoine Laurens, Claudio Fantacci, Valentin Dalibard, Martina Zambelli, Murilo Martins, Rugile Pevceviciute, Michiel Blokzijl, Misha Denil, Nathan Batchelor, Thomas Lampe, Emilio Parisotto, Konrad Żołna, Scott Reed, Sergio Gómez Colmenarejo, Jon Scholz, Abbas Abdolmaleki, Oliver Groth, Jean-Baptiste Regli, Oleg Sushkov, Tom Rothörl, José Enrique Chen, Yusuf Aytar, Dave Barker, Joy Ortiz, Martin Riedmiller, Jost Tobias Springenberg, Raia Hadsell†, Francesco Nori† and Nicolas Heess},
    journal = {ArXiv},
    year    = {2023}
}
```



