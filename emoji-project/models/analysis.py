import matplotlib.pyplot as plt

models = ['KNN', 'DTC', 'Hard Ensemble', 'Soft Ensemble', 'SVC']
macro_f1_avg = [0.15, 0.19, 0.20, 0.22, 0.27]
weighted_f1_avg = [0.22, 0.29, 0.29, 0.32, 0.39]

fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.35
index = range(len(models))

bar1 = ax.bar(index, macro_f1_avg, bar_width, label='Macro F1 Avg')
bar2 = ax.bar([i + bar_width for i in index], weighted_f1_avg, bar_width, label='Weighted F1 Avg')

ax.set_xlabel('Model', fontsize='14')
ax.set_ylabel('F1 Score Averages', fontsize='14')
ax.set_title('F1 Score Averages per Model', fontsize='14')
ax.set_xticks([i + bar_width/2 for i in index])
ax.set_xticklabels(models)
ax.legend()


def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


add_labels(bar1)
add_labels(bar2)

plt.show()
