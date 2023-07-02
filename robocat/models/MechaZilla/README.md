# MechCore

[![GitHub license](https://img.shields.io/github/license/kyegomez/MechCore)](https://github.com/kyegomez/MechCore/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/kyegomez/MechCore)](https://github.com/kyegomez/MechCore/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/kyegomez/MechCore)](https://github.com/kyegomez/MechCore/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/kyegomez/MechCore)](https://github.com/kyegomez/MechCore/pulls)

MechCore is an open-source suite of robotic transformers models designed to simplify the training, finetuning, and inference processes. With its plug-and-play functionality, MechCore provides an easy-to-use interface to effortlessly utilize a variety of robotic transformers models. Whether you're working on robotics research, autonomous systems, or AI-driven robotic applications, MechCore offers a comprehensive toolkit to enhance your projects.

## Key Features

- Easy integration and plug-and-play functionality.
- Diverse range of pre-trained robotic transformers models.
- Efficient training and finetuning pipelines.
- Seamless inference capabilities.
- Versatile and customizable for various robotic applications.
- Active community and ongoing development.

## Architecture

MechCore is built on a modular architecture, enabling flexibility and extensibility for different use cases. The suite consists of the following components:

1. **Model Library**: MechCore provides a comprehensive model library that includes various pre-trained robotic transformers models. These models are designed to tackle a wide range of robotics tasks, such as perception, motion planning, control, and more. Some of the available models in MechCore include VC-1, RT-1, ROBOTCAT, KOSMOS-X, and many others.

2. **Training and Finetuning**: MechCore offers a streamlined training and finetuning pipeline. You can easily train models from scratch or finetune existing models using your own datasets. The suite provides efficient data preprocessing, augmentation, and optimization techniques to enhance the training process.

3. **Inference**: MechCore allows you to conduct seamless inference using the trained models. You can deploy the models in real-world scenarios, robotics applications, or integrate them into existing systems for robotic perception, decision-making, and control.

## Getting Started

To get started with MechCore, follow the instructions below:

1. Clone the MechCore repository:

```bash
git clone https://github.com/kyegomez/MechCore.git
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Choose the desired model from the model library.

4. Utilize the provided examples and code snippets to train, finetune, or conduct inference using the selected model.

5. Customize the models and pipelines according to your specific requirements.

## Roadmap

The future development of MechCore includes the following milestones:

- Expansion of the model library with additional pre-trained robotic transformers models.
- Integration of advanced optimization techniques and model architectures.
- Support for more diverse robotic applications and tasks.
- Enhanced documentation, tutorials, and code examples.
- Community-driven contributions and collaborations.

Stay tuned for exciting updates and improvements in MechCore!

## Model Directory

Sure! Here's an example of a table-like format in the README.md file, showcasing the models, their tasks, and other metadata:

## Model Directory

| Model | Description | Tasks | Key Features | Code and Resources |
|-------|-------------|-------|--------------|--------------------|
| RT-1  | Robotics Transformer for real-world control at scale | Picking and placing items, opening and closing drawers, getting items in and out of drawers, placing elongated items upright, knocking objects over, pulling napkins, opening jars, and more | - Transformer architecture with image and action tokenization <br> - EfficientNet-B3 model for image tokenization <br> - Token compression for faster inference <br> - Supports a wide range of tasks and environments | [Project Website](https://ai.googleblog.com/2022/12/rt-1-robotics-transformer-for-real.html?m=1) <br> [RT-1 Code Repository](https://github.com/kyegomez/MechaZilla/tree/master/models/rt1) |
| Gato  | Generalist Agent for multi-modal, multi-task robotics | Playing Atari games, image captioning, chatbot interactions, real-world robot arm manipulation, and more | - Multi-modal support for text, images, proprioception, continuous actions, and discrete actions <br> - Serialized tokenization of data for processing with a transformer neural network <br> - Flexibility to output different modalities based on context | [Published Paper](https://arxiv.org/pdf/2205.06175) <br> [Gato Code Repository](https://github.com/kyegomez/MechaZilla/tree/master/models/GATO) |


## Datasets Directory


This section provides an overview of the datasets used in the project. The datasets are divided into two categories: control datasets used to train Gato and vision & language datasets used for vision and language tasks.

### Control Datasets

## GATO

| Dataset | Tasks |
|---------|-------|
| DM Lab | 254 |
| ALE Atari | 51 |
| ALE Atari Extended | 28 |
| Sokoban | 1 |
| BabyAI | 46 |
| DM Control Suite | 30 |
| DM Control Suite Pixels | 28 |
| DM Control Suite Random Small | 26 |
| DM Control Suite Random Large | 26 |
| Meta-World | 45 |
| Procgen Benchmark | 16 |
| RGB Stacking simulator | 1 |
| RGB Stacking real robot | 1 |
| Modular RL | 38 |
| DM Manipulation Playground | 4 |
| Playroom | 1 |

### Vision / Language Datasets

| Dataset | Tasks |
|---------|-------|
| MassiveText | N/A |
| M3W | N/A |
| ALIGN | N/A |
| MS-COCO Captions | N/A |
| Conceptual Captions | N/A |
| LTIP | N/A |
| OKVQA | N/A |
| VQAV2 | N/A |

## PALM-E Datasets

| Dataset | Tasks |
|---------|-------|
| Webli (Chen et al., 2022) | N/A |
| VQ2A (Changpinyo et al., 2022) | N/A |
| VQG (Changpinyo et al., 2022) | N/A |
| CC3M (Sharma et al., 2018) | N/A |
| Object Aware (Piergiovanni et al., 2022) | N/A |
| OKVQA (Marino et al., 2019) | N/A |
| VQAv2 (Goyal et al., 2017) | N/A |
| COCO (Chen et al., 2015) | N/A |
| Wikipedia text | N/A |
| (robot) Mobile Manipulator, real | N/A |
| (robot) Language Table (Lynch et al., 2022), sim and real | N/A |
| (robot) TAMP, sim | N/A |


Please note that the dataset descriptions provided are a summary. To access the datasets from Hugging Face, please visit the Hugging Face website and search for the respective dataset names.


## Share with Friends

Help us spread the word about MechCore by sharing it with your friends and colleagues on various social media platforms:

[![Share on Twitter](https://img.shields.io/twitter/url?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMechCore)](https://twitter.com/intent/tweet?text=Check%20out%20MechCore%2C%20an%20open-source%20suite%20of%20robotic%20transformers%20models%20for%20training%2C%20finetuning%2C%20and%20inference%20in%20robotics%20applications.%20%23MechCore%20%23OpenSource%20%23Robotics%20%23AI%20%23MachineLearning&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMechCore)

[![Share on LinkedIn](https://img.shields.io/badge/Share%20on-LinkedIn-blue)](https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMechCore&title=MechCore%20-%20Open-Source%20Suite%20of%20Robotic%20Transformers%20Models&summary=Check%20out%20MechCore%2C%20an%20open-source%20suite%20of%20robotic%20transformers%20models%20for%20training%2C%20finetuning%2C%20and%20inference%20in%20robotics%20applications.&source=)

[![Share on Facebook](https://img.shields.io/badge/Share%20on-Facebook-blue)](https://www.facebook.com/sharer.php?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMechCore&t=MechCore%20-%20Open-Source%20Suite%20of%20Robotic%20Transformers%20Models)

[![Share on Reddit](https://img.shields.io/badge/Share%20on-Reddit-orange)](https://www.reddit.com/submit?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMechCore&title=MechCore%20-%20Open-Source%20Suite%20of%20Robotic%20Transformers%20Models)

[![Share on WhatsApp](https://img.shields.io/badge/Share%20on-WhatsApp-green)](https://wa.me/?text=Check%20out%20MechCore%2C%20an%20open-source%20suite%20of%20robotic%20transformers%20models%20for%20training%2C%20finetuning%2C%20and%20inference%20in%20robotics%20applications.%0A%0AGitHub%20Repository%3A%20https%3A%2F%2Fgithub.com%2Fkyegomez%2FMechCore)


[![Share on Pinterest](https://img.shields.io/badge/Share%20on-Pinterest-red)](https://www.pinterest.com/pin/create/button/?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMechCore&media=&description=MechCore%20-%20Open-Source%20Suite%20of%20Robotic%20Transformers%20Models)

[![Share on Tumblr](https://img.shields.io/badge/Share%20on-Tumblr-blue)](https://www.tumblr.com/share/link?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMechCore&name=MechCore%20-%20Open-Source%20Suite%20of%20Robotic%20Transformers%20Models&description=Check%20out%20MechCore%2C%20an%20open-source%20suite%20of%20robotic%20transformers%20models%20for%20training%2C%20finetuning%2C%20and%20inference%20in%20robotics%20applications.)

[![Share on Hacker News](https://img.shields.io/badge/Share%20on-Hacker%20News-orange)](https://news.ycombinator.com/submitlink?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMechCore&t=MechCore%20-%20Open-Source%20Suite%20of%20Robotic%20Transformers%20Models)

[![Share on VK](https://img.shields.io/badge/Share%20on-VK-blue)](https://vk.com/share.php?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMechCore&title=MechCore%20-%20Open-Source%20Suite%20of%20Robotic%20Transformers%20Models)

We appreciate your support in sharing MechCore and making it accessible to the wider community!

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](https://github.com/kyegomez/MechCore/blob/main/LICENSE).


# Roadmap

[Integrate CHATGPT FOR ROBOTICS](https://www.microsoft.com/en-us/research/uploads/prod/2023/02/ChatGPT___Robotics.pdf)

* Integrate Kosmos-X, Kosmos-2, PALM-E, ROBOCAT, any other robotic models we should integrate? Let me know!

* Integrate embedding provider for RT-1

* Integrate flash attention for RT-1

* Integrate flasha Attention for GATO

* 