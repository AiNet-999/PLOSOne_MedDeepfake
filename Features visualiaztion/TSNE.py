import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)
features = feature_extractor.predict(X_train)
y_pred1 = model.predict(X_train)
y_pred1_classes = np.argmax(y_pred1, axis=1)
Y_train_classes = np.argmax(Y_train, axis=1)

scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
features_tsne = tsne.fit_transform(features_normalized)


class_labels = ['Real', 'FM', 'FB']
colors = ['royalblue', 'darkorange', 'forestgreen']
markers = ['o', '^', 's']


plt.figure(figsize=(10, 8))
for i, (class_name, color, marker) in enumerate(zip(class_labels, colors, markers)):
    plt.scatter(
        features_tsne[Y_train_classes == i, 0],
        features_tsne[Y_train_classes == i, 1],
        label=class_name,
        color=color,
        marker=marker,
        edgecolors='k',
        s=60,
        alpha=0.85
    )


plt.xlabel('t-SNE Component 1', fontsize=14)
plt.ylabel('t-SNE Component 2', fontsize=14)


plt.legend(fontsize=12, loc='best', edgecolor='black')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
plt.tight_layout()


plt.savefig("tsne_2d.png", bbox_inches='tight', pad_inches=0.2, dpi=300)
plt.show()
