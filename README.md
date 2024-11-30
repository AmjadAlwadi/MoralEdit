Private Machine Learning stuff

## Dataset creation
1. Construct the full norms dataset at first, which constructs a merged dataset of original-ms, anit-ms and rephrased-ms and removes the unnecessary rows using static_dataset_creation\construct_norms_dataset.py.

2. Generate the detected adjectives for every norm in the dataset, which correspond to ground_truth and target_new using the desired method.

3. Generate the detected subjects for every norm in the dataset, which correspond to the most important token of every norm using the desired method.

4. Generate the rephrases for every rot-action in the dataset using the desired method.

5. Generate the locality dataset.

6. Generate the portability dataset.

7. Generate morally coherent norms using dynamic_dataset_creation\generate_morally_coherent_norms.py.

8. Finally, generate some norms to edit using dynamic_dataset_creation\generate_edit_norms_dataset.py.

## Editing Norms
Run edit_norms.py to edit the generated norms using the desired method and model and decoding strategy.

## Results
The outputs and a metrics_summary file containing the calculated metrics are going to be saved in outputs/method/model/decoding_strategy/.

Using visualize_metrics.py you can visualize the results for all previously executed edits and compare between methods and models.