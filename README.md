# Large Language Models as Agents

The idea behind this project is to adapt small (of a few billion parameters) Large Language Models as helpful agents for scientific communities. 
The first use-case is to get a model able to answer questions like: "Which software/code is used to analyse and model AGN data in very-high-energy astrophysics?".
The necessity behind this approach is to facilitate access to software even for scientist who do not directly work on it, therefore do not know the name but know what they want to use the software for.
The dataset used to train the models will be created from papers published on arXiv, which contain many software citations.

I am developing two approaches at the moment:
- Reformatting paper abstracts with the utilities already developed by Microsoft at [AdaptLLM](https://github.com/microsoft/LMOps/tree/main/adaptllm) for reading comprehension tasks;
  - Fine-tune a model on a specific domain with the reading-comprehension tasks;
- Extracting software mentions and URLs from the paper bodies;
  - Fine-tune a model with keywords and software citations to make it understand the connection between the domains and the developed software;

The two approaches may not exclude each other