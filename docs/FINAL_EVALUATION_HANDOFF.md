# Final evaluation handoff

Use this checklist when the team agrees to freeze model selection. It is a coordination template, not a report of test results.

## 1. Freeze one candidate per member

Each member supplies one row before the final run. Record alternatives or AudioSet-pretrained experiments separately.

| Member | Model / pretraining | Checkpoint path or artifact reference | Configuration / run reference | Validation accuracy | Ready? |
|---|---|---|---|---|---|
| Member 1 | | | | | |
| Member 2 | | | | | |
| Member 3 | | | | | |
| Member 4 | | | | | |

Existing committed candidates are `checkpoints/best_cnn.pth`, `checkpoints/best_transfer_resnet18.pth` and `checkpoints/best_transfer_mobilenet_v2.pth`. See the [transfer handoff guide](../reports/member4/README.md) for loading and recorded checkpoint hashes. Do not rename or replace these files to complete the table.

## 2. Before evaluating

- Agree on the selected checkpoints, exact Git commit, Python/package versions and evaluation owner.
- Confirm the shared source-disjoint split: training folds 1–8 (7,000 clips after filtering), validation fold 9 (816 clips), test fold 10 (837 clips).
- Verify checkpoint loading on validation data and confirm the class index order against the dataset metadata.
- Record any missing checkpoint, model-specific frontend, or external pretraining requirement. Do not silently substitute another model.
- Agree on the evaluation script and metrics before accessing the test labels.

## 3. Report the final comparison

Run the approved checkpoints on the same test examples. Report accuracy, macro precision, macro recall and macro F1 (unweighted means over all ten classes), with the same zero-division convention for every model. Include a confusion matrix with true classes as rows and predicted classes as columns, using the same label order.

| Model | Test accuracy | Macro precision | Macro recall | Macro F1 | Confusion matrix reference |
|---|---|---|---|---|---|
| | | | | | |

Save the prediction records, class order, evaluation configuration and commit identifier with the results. Report any execution failures or missing predictions rather than comparing different subsets silently. Keep AudioSet-pretrained experiments labeled separately from the required comparison.

Do not select or retune models using fold 10. A lower test score than validation is a valid result. Until the team completes the final evaluation, leave the test table empty.
