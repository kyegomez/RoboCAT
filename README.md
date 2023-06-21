# RoboCAT
Implementation of Deepmind's RoboCat: "Self-Improving Foundation Agent for Robotic Manipulation" An next generation robot LLM


Implementation of Deepmind's <a href="https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/robocat-a-self-improving-robotic-agent/robocat-a-self-improving-foundation-agent-for-robotic-manipulation.pdf">RoboCat</a>, Self-Improving Foundation Agent for Robotic Manipulation




Research Paper Implementation Document
Architecture
The RoboCat architecture is based on the Gato model and a VQ-GAN encoder. The Gato model is an autoregressive transformer model, while the VQ-GAN encoder is pretrained on a diverse set of images for faster training and better generalization.

Algorithmic Pseudocode
Pretrain VQ-GAN encoder on a diverse set of images.
Tokenize proprioceptive observations, agent actions, and images using the pretrained VQ-GAN encoder.
Train the RoboCat agent using a dataset containing tokenized trajectories and a token prediction loss.
Fine-tune the RoboCat agent on a small dataset of new episodic experience (100-1000 demonstrations).
Deploy the fine-tuned RoboCat agent to gather additional data for self-improvement.
Train a new iteration of the RoboCat agent using the updated dataset.
Deploy the RoboCat agent on real-world robotic embodiments.
File Tree
RoboCat/
│
├── data/
│   ├── demonstrations/
│   ├── images/
│   └── trajectories/
│
├── models/
│   ├── gato/
│   ├── vq_gan/
│   └── robocat/
│
├── src/
│   ├── tokenization.py
│   ├── training.py
│   ├── fine_tuning.py
│   ├── self_improvement.py
│   └── deployment.py
│
└── main.py


## Requirements
To reproduce the paper, the following requirements are needed:

Python 3.7 or higher
PyTorch 1.9 or higher
HuggingFace Transformers library
VQ-GAN library
A diverse set of images for pretraining the VQ-GAN encoder
A dataset of tokenized trajectories for training the RoboCat agent
A dataset of 100-1000 demonstrations for fine-tuning the RoboCat agent
Real-world robotic embodiments for deployment
Additional Sections
Reward Models
To detect task success during real-world deployment, vision-based reward models are trained. These models are trained on human demonstrations and data from policies trained to perform the task. The episodes are annotated via a crowd-sourcing interface, where annotators mark the time step after which the task is solved in each episode (if at all), resulting in binary annotations. These annotations are then used to train a binary classifier that can detect task success from image observations at any given time step.

## Real-world Deployment Challenges
Two main challenges arise during real-world deployment: success classification and task resets. Success detection is necessary for the hindsight goal relabeling of semantically-equivalent goals and for determining when a reset is needed. Task resets are required when the agent has completed a task or reached an undesired state, and the environment needs to be reset to a valid starting state.




## Citations

```bibtex
@article{Bousmalis2023RoboCat,
    title   = {RoboCat: A Self-Improving Foundation Agent for Robotic Manipulation},
    author  = {Konstantinos Bousmalis*, Giulia Vezzani*, Dushyant Rao*, Coline Devin*, Alex X. Lee*, Maria Bauza*, Todor Davchev*, Yuxiang Zhou*, Agrim Gupta*,1, Akhil Raju, Antoine Laurens, Claudio Fantacci, Valentin Dalibard, Martina Zambelli, Murilo Martins, Rugile Pevceviciute, Michiel Blokzijl, Misha Denil, Nathan Batchelor, Thomas Lampe, Emilio Parisotto, Konrad Żołna, Scott Reed, Sergio Gómez Colmenarejo, Jon Scholz, Abbas Abdolmaleki, Oliver Groth, Jean-Baptiste Regli, Oleg Sushkov, Tom Rothörl, José Enrique Chen, Yusuf Aytar, Dave Barker, Joy Ortiz, Martin Riedmiller, Jost Tobias Springenberg, Raia Hadsell†, Francesco Nori† and Nicolas Heess},
    journal = {ArXiv},
    year    = {2023}
}
```


