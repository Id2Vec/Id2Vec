# Id2Vec

Welcome to Id2Vec project, here are the source code and data for the project.


## Dataset

We provide a simple sample dataset in the project, which could help you test the pipeline with low time cost.

For the whole dataset, please download it from this link: [c# and js dataset](https://drive.google.com/file/d/11TdY4C_Ute2w-HLs1fUr_2JRxAog-ZGg/view?usp=sharing)


## Running model

1. Model training:

1.1 Run build_dataset.py to convert the original dataset into the format which is required by the model.

1.2 Set the hyper paremeters {train_data_file, eval_data_file} in train.sh, which should be path_to_new_pt_train.jsonl and path_to_new_pt_valid.jsonl.

1.3 The eval dataset for IdBench task and rename reference task has already been put in the dataset folder.

We packaged the whole pieline and you can easily run the model by setting the hyper parameters in the train.sh and run it.

eg: <code>CUDA_VISIBLE_DEVICES=0,1 sh src/train.sh</code>

2. Model evaluation:

Setting the hyper parameters in the eval.sh and run it.

eg: <code>CUDA_VISIBLE_DEVICES=0,1 sh src/eval.sh</code>

<!--
**Id2Vec/Id2Vec** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
