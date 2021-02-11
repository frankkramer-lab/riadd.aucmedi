import os
import pandas as pd
from plotnine import *
from sklearn.metrics import roc_curve, auc
import numpy as np

path_res = "results/"


df_results = pd.DataFrame(data=[], dtype=np.float64, columns=["arch", "class", "FPR", "TPR"])
df_auroc = pd.DataFrame(data=[], dtype=np.float64, columns=["arch", "class", "auroc"])

for file in os.listdir(path_res):
    if not file.endswith(".predictions.csv") : continue
    arch = file.split(".")[0]

    dt = pd.read_csv(os.path.join(path_res, file))

    # Identify columns
    cols = dt.columns
    cols = set([s[3:] for s in cols.tolist()[1:]])

    # Iterate over each class
    for c in cols:
        gt = dt["gt_" + c].tolist()
        pdprob = dt["pd_" + c].tolist()
        # Compute AUROC
        fpr, tpr, _ = roc_curve(gt, pdprob)
        auroc = auc(fpr, tpr)
        auroc_series = pd.Series([arch, c, auroc], index=["arch", "class", "auroc"])
        df_auroc = df_auroc.append(auroc_series, ignore_index=True)
        # Add to result dataframe
        results_class = pd.DataFrame(data={"arch": arch, "class": c, "FPR": fpr, "TPR": tpr})
        df_results = df_results.append(results_class, ignore_index=True)

print("Mean AUROC:")
print(df_auroc.groupby("arch").mean())

# Plot roc results
fig = (ggplot(df_results, aes("FPR", "TPR", color="class"))
           + geom_line(size=1.5)
           + geom_abline(intercept=0, slope=1, color="black",
                         linetype="dashed")
           + facet_wrap("arch")
           + ggtitle("Multi-label Classification Performance by ROC")
           + xlab("False Positive Rate")
           + ylab("True Positive Rate")
           + scale_x_continuous(limits=[0, 1])
           + scale_y_continuous(limits=[0, 1])
           + scale_color_discrete(name="Classification")
           + theme_bw(base_size=28))
# Store figure to disk
fig.save(filename="plot.ROC_curve.png",
         path="results/", width=40, height=20, dpi=200, limitsize=False)

# Plot auroc results
df_auroc.fillna(value=0, inplace=True)
fig = (ggplot(df_auroc, aes("class", "auroc", fill="class"))
              + geom_col(stat='identity', width=0.6,
                         position = position_dodge(width=0.6))
              + ggtitle("AUROC Comparison")
              + facet_wrap("arch")
              + xlab("Class")
              + ylab("AUROC")
              + coord_flip()
              + scale_y_continuous(limits=[0, 1])
              + scale_fill_discrete(name="Classification")
              # + scale_fill_brewer(palette=2)
              + theme_bw(base_size=28))
# Store figure to disk
fig.save(filename="plot.AUROC_barplot.png",
        path="results/", width=40, height=20, dpi=200, limitsize=False)
