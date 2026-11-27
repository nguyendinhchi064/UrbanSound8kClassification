import nbformat as nbf
from pathlib import Path
from nbclient import NotebookClient

root = Path.cwd()


def md(s):
    return nbf.v4.new_markdown_cell(s)


def code(s):
    return nbf.v4.new_code_cell(s)


intro = """from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image
ROOT = Path.cwd()
if not (ROOT / 'reports').exists():
    ROOT = ROOT.parent
assert (ROOT / 'reports/member4/model_comparison.csv').exists()
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})
"""
nb = nbf.v4.new_notebook(
    cells=[
        md(
            "# Transfer learning | ImageNet CNNs\n**Member 4 · Mahij Khantigiri Gosai**\n\nWe compare four pretrained CNN architectures using the same fixed UrbanSound8K split and input representation as the team's custom CNN. This notebook reads the recorded training evidence and does not retrain models or evaluate fold 10."
        ),
        code(
            intro
            + """import json
base = json.loads((ROOT/'reports/member4/baseline_reference.json').read_text())
split = json.loads((ROOT/'reports/member4/split_audit.json').read_text())
display(pd.DataFrame(split).T)
results = pd.read_csv(ROOT/'reports/member4/model_comparison.csv')
print(f"{len(results)} completed ImageNet training runs; validation reference: {base['accuracy']:.2f}%")
"""
        ),
        md(
            "## 1. Architecture and adaptation\nAll models use the unresized 128 × 173 log-Mel representation, replicated into three channels, with ImageNet normalization. A new dropout and ten-class linear head replaces the pretrained classifier. Frozen, partial and full stages progressively expose backbone parameters to training."
        ),
        code("""runs = results[results.seed.eq(42) & results.name.str.endswith(('_frozen','_partial'))]
stages = runs.pivot(index='arch', columns='stage', values='accuracy')[['frozen','partial']]
stages['gain_pp'] = stages.partial - stages.frozen
display(stages.round(2))
display(Image(filename=str(ROOT/'figures/transfer/submission_stage_comparison.png')))
"""),
        md(
            "## 2. Fine-tuning and selected checkpoint\nThe strongest individual ImageNet model is regularized ResNet18 (seed 2024). The recorded checkpoints are selected by validation accuracy and lower validation loss in case of ties."
        ),
        code("""cols = ['name','stage','seed','accuracy','macro_f1','best_epoch','training_seconds']
display(results[cols].head(10).round(3))
run = results.iloc[0]['name']
history = pd.read_csv(ROOT/'reports/member4'/run/'training_history.csv')
fig, ax = plt.subplots(1,2,figsize=(11,3.8))
for a,m in zip(ax,['accuracy','loss']):
    a.plot(history.epoch,history['train_'+m],label='Train')
    a.plot(history.epoch,history['val_'+m],label='Validation')
    a.axvline(results.iloc[0].best_epoch,color='gray',linestyle=':',label='Selected epoch')
    a.set(xlabel='Epoch',ylabel=m.title()); a.legend(frameon=False)
fig.tight_layout()
plt.show()
"""),
        md("## 3. Computational cost and reproducibility"),
        code("""bench = pd.read_csv(ROOT/'reports/member4/final_benchmark.csv')
display(bench[bench.input_shape.eq('1x128x173')][['run','parameters','estimated_macs','latency_ms','checkpoint_mib']].round(3))
display(Image(filename=str(ROOT/'figures/transfer/submission_efficiency.png')))
display(pd.read_csv(ROOT/'reports/member4/seed_comparison.csv').round(3))
display(pd.read_csv(ROOT/'reports/member4/seed_statistics.csv').round(3))
"""),
        md("## 4. Selected model: class-level results"),
        code("""cm = pd.read_csv(ROOT/'reports/member4/resnet18_repeat_full_2024_confusion.csv',index_col=0)
fig,ax=plt.subplots(figsize=(8,6))
im=ax.imshow(cm,cmap='Blues')
ax.set_xticks(range(len(cm)),cm.columns,rotation=45,ha='right')
ax.set_yticks(range(len(cm)),cm.index)
ax.set(xlabel='Predicted',ylabel='Actual',title='ResNet18 — validation')
for i in range(len(cm)):
    for j in range(len(cm)):
        ax.text(j,i,str(cm.iloc[i,j]),ha='center',va='center',color='white' if cm.iloc[i,j]>cm.values.max()/2 else 'black',fontsize=8)
fig.colorbar(im,ax=ax); fig.tight_layout(); plt.show()
display(pd.read_csv(ROOT/'reports/member4/resnet18_repeat_full_2024_class_metrics.csv',index_col=0).round(3))
"""),
        md(
            "## Conclusion\nResNet18 achieves 87.50% validation accuracy versus 86.89% for the existing custom CNN. Its three-seed average is 86.68%, so the best-run gain is not a robust improvement. MobileNetV2 is the smaller checkpoint and lower-MAC alternative. We retain the held-out test fold for the team's final evaluation. See the [submission report](../reports/member4_report.md) and [reproduction guide](../reports/member4/README.md)."
        ),
    ]
)
nb2 = nbf.v4.new_notebook(
    cells=[
        md(
            "# Audio-pretrained transfer | Optional extension\n**Member 4 · Mahij Khantigiri Gosai**\n\nThis separate investigation uses external AudioSet pretraining and model-specific audio frontends. Scores are validation-only and not a controlled comparison of CNN architecture alone."
        ),
        code(
            intro
            + """audio = pd.read_csv(ROOT/'reports/member4_audio/model_comparison.csv')
panns = pd.read_csv(ROOT/'reports/member4_panns/classifier_comparison.csv')
ast = pd.read_csv(ROOT/'reports/member4_ast/frozen_comparison.csv')
ensemble = pd.read_csv(ROOT/'reports/member4_ast/final_ensemble_comparison.csv')
print(f'{len(audio)} audio CNN runs, {len(panns)} PANNs classifier fits, {len(ast)} AST classifier fits')
"""
        ),
        md("## 1. EfficientAT: frozen and fine-tuned"),
        code("""display(audio[['name','stage','seed','accuracy','macro_f1','total_parameters']].round(3))
display(Image(filename=str(ROOT/'figures/audio_transfer/comparison.png')))
display(pd.read_csv(ROOT/'reports/member4_audio/seed_comparison.csv').round(3))
"""),
        md(
            "## 2. Frozen PANNs and AST representations\nThe frozen classifiers are trained only on the training partition. AST final-block tuning is a separate optional experiment."
        ),
        code("""display(panns.round(3))
display(ast.head(6).round(3))
display(pd.read_csv(ROOT/'reports/member4_ast/tail_history.csv').round(3))
"""),
        md("## 3. Multi-seed and CNN/transformer ensembles"),
        code("""display(ensemble.round(3))
display(Image(filename=str(ROOT/'figures/audio_transfer/ast_ensemble_comparison.png')))
display(Image(filename=str(ROOT/'figures/audio_transfer/ast_selected_confusion.png')))
display(pd.read_csv(ROOT/'reports/member4_ast/selected_class_metrics.csv',index_col=0).round(3))
"""),
        md(
            "## Conclusion\nEfficientAT MN20 achieves 90.20% as a single CNN; the best exploratory ensemble reaches 90.81% (741/816 correct). External pretraining overlap cannot be ruled out. The 95% stretch objective was not attained. No test-set metrics are reported; the original CNN contribution remains the required handoff. See [Member 4 report](../reports/member4_report.md)."
        ),
    ]
)
for filename, book in [
    ("member4_transfer_learning.ipynb", nb),
    ("member4_audio_transfer_extension.ipynb", nb2),
]:
    book.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    book.metadata.language_info = {"name": "python", "version": "3.12"}
    NotebookClient(
        book,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
    ).execute()
    path = root / "notebooks" / filename
    nbf.write(book, path)
    print(path, len(book.cells), path.stat().st_size, flush=True)
