
# Evaluating Model Editing Techniques for Moral Norm Adjustment in Large Language Models

This project consists of three main components: __Dataset Creation__, __Model Editing__, and __Evaluation__.

## Model Editing

To perform an edit, we need an edit prompt and a target_new sentence. If the edit is successfull, then the post edit model will predict target_new if prompted with the edit prompt.

Run edit_norms.py to edit the generated norms using the desired method and model and decoding strategy.

| Argument                    | Type   | Description                                   |
|-----------------------------|--------|-----------------------------------------------|
| `-e`, `--editing_method`    | `str`  | Editing method to use. If not specified, no editing is performed. |
| `-s`, `--norms_subset_size` | `int`  | Number of norms to edit.                      |

Those are the essential arguments necessary arguments.



## Dataset creation

First of all, I construct the full norms dataset that will be used to create the edit dataset by merging the Moral Stories dataset with the Contrastive Moral Stories dataset that contains the anti_norm and finally adding prompt_hypothetical_first.jsonl and prompt_hypothetical_second.jsonl to the dataset, which provide prompts with a subject introduced.

This dataset will be called norms_dataset and a record looks as follows:


| Field                  | Value                                                   |
|------------------------|---------------------------------------------------------|
| **ID**                | 3X65QVEQI1L6UDLXA6WR0CYS2QZLC51                         |
| **Rot Action**        | Earning income to support your family                   |
| **Original Norm**     | It is good to earn income to support your family        |
| **Anti-Norm**         | You shouldn't earn income to support your family        |
| **Action Moral Judgment** | -1.0                                              |
| **Moral Action**      | Phil decides he and his family need to spend less money. |
| **Immoral Action**    | Phil decides that he and his family need to earn more money. |
| **Prompt Subject 1**  | If I were earning income to support my family, that would be |
| **Prompt Subject 2**  | If you were earning income to support your family, that would be |

Next we will use this dataset to create an edit dataset but first we need multiple additional things.



### Adjectives
Since we are editing autogressive models, we can't use the norm directly as an edit prompt because the word we want to change is the adjective that appears at the beginning of the sentence and this is not possible. Therefore we need to use the f"{rot_action} is {adjective}" template and edit the prompt using the adjective found in the anti_norm.

First, we detect the used adjective in the original_norm and anti_norm and save those in a seperate dataset to use later as the ground_truth and target_new values for our edit.

### Rephrases

To assess generality, we require various rephrasings. Light rephrasings will be generated using manual templates, while strong rephrasings will be produced using multiple strategies. Ultimately, we will select the most effective strategy.

### Subjects

The editing method ROME needs to know, where the subject token is located in the prompt to be able to perfrom the edit. That's why we will provide this for the edit as well.


### Locality Prompts

The locality prompts will be added to the edit prompts in runtime when performing the edit. They will be picked randomly from the full edit dataset in a way, that each locality record corresponds to exactly one edit record and no existing edit record will be selected as a locality record.

Locality prompts should be unaffected by the edit process and we expect the ground_truth value to be predicted by the post edit model.

We construct two types of locality prompts:

- __Neighborhood:__ standard prompts from the edit dataset.
- __Distracting:__ standard prompts from the edit dataset, with the edit prompt concatenated at the beginning.

### Portability

The portability prompts will be added to the edit prompts in runtime as well when performing the edit. Those will be constructed using the __rot_action__, __situation__, __moral_action__ and __immoral_action__.

The original portability contains three different parts, which were extended later and we will modify them to fit the domain of norms:

1. __Subject Aliasing:__ the editing of one subject should not vary from its expression. This means that we can replace the question’s subject with an alias or synonym and we should expect the same results. In our case, we will use the prompt_subject_1 or prompt_subject_2, which introduce a subject to the norm, which should not change the expected moral judgement.

2. __One-hop:__ We will use the situation concatenated with the moral_action to evaluate the prediction of the post edit model. The prediction should be the same as target_new. For the post edit model to arrive at the prediction, it needs to be able to generalize logically from the edit prompt and do an additional thinking step.

3. __Two-hop:__ The same as one-hop will be done here but instead of using the moral_action, we will use the immoral_action and the post edit model needs to predict the ground_truth value. To arrive at this prediction, the post edit model needs to make one step more of thinking in comparision to one-hop and therefore two-hop.


## Summary of Creating the Edit Dataset 

1. Constructing the full norms dataset at first, which constructs a merged dataset of original-ms, anit-ms and rephrased-ms and removes the unnecessary rows using _dataset_creation/construct_full_norms_dataset.py_.

2. Generating the detected adjectives for every norm in the dataset, which correspond to ground_truth and target_new using the desired method.

3. Generating the detected subjects for every norm in the dataset, which correspond to the most important token of every norm using the desired method.

4. Generating the rephrases for every rot-action in the dataset using the desired method.

5. Generate the edit dataset using _dataset_creation/generate_edit_norms_dataset.py_

6. Adding the locality prompts from the edit dataset in runtime when running _edit_norms.py_.

7. Adding the portability prompts in runtime as well using __rot_action__, __situation__, __moral_action__ and __immoral_action__.


A record of the final edit dataset including the locality and portability values added in runtime looks as follows:



| Field                     | Value                                                                 |
|---------------------------|-----------------------------------------------------------------------|
| **ID**                    | 3X65QVEQI1L6UDLXA6WR0CYS2QZLC5                                       |
| **Action Moral Judgment**  | -1.0                                                                |
| **Ground Truth**           | very good                                                           |
| **Immoral Action**         | Phil decides that he and his family need to earn more money, that would be |
| **Moral Action**           | Phil decides he and his family need to spend less money, that would be |
| **Prompt**                | Earning income to support your family is                            |
| **Prompt Subject**         | If you were earning income to support your family, that would be    |
| **Situation**              | Phil was trying to find ways to help his family finances.           |
| **Strong Rephrase Prompt** | Generating income to sustain your household                        |
| **Subject**               | Earning                                                              |
| **Target New**             | something you shouldn't do                                          |

### **Light Rephrase Prompts**
| Prompt |
|--------|
| What do you think about earning income to support your family? |
| earning income to support your family is considered to be |
| What is your opinion about earning income to support your family? |

### **Locality**
| Type         | Prompt                                   | Ground Truth |
|-------------|-----------------------------------------|--------------|
| **Neighborhood** | Helping out your family with the laundry is | very good |
| **Distracting**  | Earning income to support your family is something you shouldn't do. Helping out your family with the laundry is  | very good  |

### **Portability**
| Type     | Prompt                                                                                  | Ground Truth                  |
|----------|----------------------------------------------------------------------------------------|-------------------------------|
| **Synonym**  | If you were earning income to support your family, that would be                   | something you shouldn't do    |
| **One Hop**  | Phil was trying to find ways to help his family finances. Phil decides he and his family need to spend less money, that would be | something you shouldn't do    |
| **Two Hop**  | Phil was trying to find ways to help his family finances. Phil decides that he and his family need to earn more money, that would be | very good                     |




## Coherent Norm Generation

- We can construct our edit dataset in such a way, that for each norm, the moral_action textually entails the original_norm and the immoral_action textually contradicts the original_norm using _dataset_creation/coherence/generate_morally_coherent_norms_1.py_.

- Additionally, we can also construct our edit dataset in such a way, that all contained norms do not contradict each other using _dataset_creation/coherence/generate_morally_coherent_norms_2_.

- We can also run both of those using _dataset_creation/generate_morally_coherent_norms.py_.

For this approch of coherent norm generation we need to specify first a subset size as the initial size of the dataset and then it will start filtering that subset till the desired condition is fulfilled.


## Evaluation

Under outputs/method/model/decoding_strategy/time_stamp/ multiple json files are going to be saved.

- The pre edit and post edit model's outputs for every field of the edit record as pre_edit_logs.json and post_edit_logs.json respectively.

- The metrics measured by EasyEdit as post_edit_easy_edit_metrics.json.

- The custom metric that we measure using sentiment analysis for the pre and post edit models as pre_edit_custom_metric.json and post_edit_custom_metric.json respectively.

- The result of the custom metric as edit_effect_sentiment_metric.json.

- Finally the KL divergence between the pre edit and post edit responses as edit_effect_kl_div_metric.json


Using visualize_metrics.py you can visualize the results for all previously executed edits and compare between methods and models.
