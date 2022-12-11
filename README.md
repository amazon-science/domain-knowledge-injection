## Injecting domain knowledge in language models for task-oriented dialogue systems

This project contains the code to reproduce the results in:

> [EMNLP 2022] [Injecting domain knowledge in language models for task-oriented dialogue systems](https://www.amazon.science/publications/injecting-domain-knowledge-in-language-models-for-task-oriented-dialogue-systems)

> By Denis Emelin, Daniele Bonadiman, Sawsan Alqahtani, Yi Zhang, Saab Mansour

```
@Inproceedings{Emelin2022,
 author = {Denis Emelin and Daniele Bonadiman and Sawsan Alqahtani and Yi Zhang and Saab Mansour},
 title = {Injecting domain knowledge in language models for task-oriented dialogue systems},
 year = {2022},
 url = {https://www.amazon.science/publications/injecting-domain-knowledge-in-language-models-for-task-oriented-dialogue-systems},
 booktitle = {EMNLP 2022},
}
```

## Abstract

Pre-trained language models (PLM) have advanced the state-of-the-art across NLP applications, but lack domain-specific knowledge that does not naturally occur in pre-training data. Previous studies augmented PLMs with symbolic knowledge for different downstream NLP tasks. However, knowledge bases (KBs) utilized in these studies are usually large-scale and static, in contrast to small, domain-specific, and modifiable knowledge bases that are prominent in real-world task-oriented dialogue (TOD) systems. In this paper, we showcase the advantages of injecting domain-specific knowledge prior to fine-tuning on TOD tasks. To this end, we utilize light-weight adapters that can be easily integrated with PLMs and serve as a repository for facts learned from different KBs. To measure the efficacy of proposed knowledge injection methods, we introduce Knowledge Probing using Response Selection (KPRS) – a probe designed specifically for TOD models. Experiments1 on KPRS and the response generation task show improvements of knowledge injection with adapters over strong baselines.
***

## Required libraries

- nltk
- numpy
- sacrebleu
- tensorboardX
- torch
- transformers
- tqdm
- word2number

## Codebase

- `create_kprs_benchmark` contains python scripts used to construct the KPRS benchmark files, as well as their perturbed variant for the planned knowledge-update experiments
- `data_handling` contains python scripts used to create and modify data used across all performed experiments (excluding the KPRS benchmark)
- `experiments` contains python scripts relevant to the experiments performed as part of the project
- `training_scripts_ec` contains bash scripts used to run experiments on the EC2 instance
- `memory_adapter` contains code relevant to the memory-network adapter variant that we decided not to pursue further

### create_kprs_benchmark

- `combine_training_dialogues` helper script used to aggregate all MultiWoZ 2.2 training files into a single file for convenience
- `collect_contexts` collects dialogue contexts and system responses from the MultiWoZ 2.2 train / test / dev data used to construct the KPRS samples
- `create_samples` generates KPRS samples based on the collected dialogue contexts and system responses, by identifying sets of high-likelihood distractor items in the MultiWoZ 2.2 databases  
- `filter_train` filters the KPRS training data by removing samples that contain entities mentioned in dev / test samples
- `perturb_databases` creates perturbed variants of MultiWoZ 2.2 databases by reassigning entity names across all database entries (used in the knowledge-update experiments)
- `create_perturbed_samples` creates perturbed KPRS samples, where positive responses are consistent with the perturbed databases, while the negative responses are consistent with the original databases (used in the knowledge-update experiments)
- `sample_for_manual_eval` helper script used to sample KPRS samples for manual quality control
- `util`: a collection of utility functions, primarily for managing MultiWoZ databases 

### data_handling

- `prepare_db_facts` derives atomic and composite facts for the adapter training / knowledge injection step from the MultiWoZ 2.2 databases
- `create_dialogue_state_tracking_data` derives dialogue state tracking samples from the MultiWoZ 2.2 train / dev / test data
- `create_response_generation_data` derives response generation samples from the MultiWoZ 2.2 train / dev / test data
- `annotate_response_generation_samples` annotates response generation targets with sets of entities that are supported by the databases and appropriate given the dialogue context
- `merge_samples` combines single-domain and multi-domain samples into a single file (used in the multi-domain experiments)
- `util` a collection of utility functions, primarily for managing MultiWoZ databases 

### experiments

- `adapter_model` implements the adapter-enhanced language model as well as the various methods for combining adapter and language model representations
- `adapter_generation_utils` a modified variant of the HuggingFace Transformers generation adjusted to support the adapter model
- `pretrain_adapter` trains adapters on facts derived from database contents; also used to train the sequentially fine-tuned baselines
- `finetune_on_downstream_task` fine-tunes adapter models and baselines on down-stream tasks
- `evaluate_kprs` defines the evaluation methods for the KPRS task
- `evaluate_dialogue_state_tracking` defines the evaluation methods for dialogue state tracking
- `evaluate_response_generation` defines the evaluation methods for response generation
- `util` a collection of utility functions used primarily for data preparation and serving

### training_scripts

Script names are meant to be self-explanatory. `X_multidomain` scripts are used to run multi-domain experiments.

***

## Running experiments

To run experiments, execute the corresponding bash script in the `training_scripts_ec` directory

- Specify the target domain in the `--active_domains` argument
    - Supported domains are `restaurant`, `hotel`, `attraction`, `train` for single-domain experiments and `mixed` for multi-domain experiments
- For pretraining adapters, specify the fact format in the `--fact_format` argument
    - Supported formats are `atomic`, `composite`, and `atomic_and_composite`
- For fine-tuning, add the `--plm_only` argument to fine-tune the LM without adapters
- To specifiy the combination function, use the `--adapter_combo_method` argument
- Supported methods are `mean`, `gate`, `gate_hidden`, `concatenate`, `expert`, `attention`, and `gru` for use with single adapters, and `mean_multi`, `gate_multi` and `gate_hidden_multi` for use with multiple active adapters
- To load-in pre-trained adapter models, specify the relevant checkpoint in the `--adapter_model_checkpoint` argument
    - When using multiple adapters, provide paths to all pre-trained models (only the adapter parameters will be loaded in)

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
