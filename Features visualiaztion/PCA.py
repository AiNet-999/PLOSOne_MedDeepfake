import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
feature_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)
features = feature_extractor.predict(X_train)
y_pred1 = model.predict(X_train)
y_pred1_classes = np.argmax(y_pred1, axis=1)
Y_train_classes = np.argmax(Y_train, axis=1)

explained_variance = pca.explained_variance_ratio_
print("Explained variance ratio for each component:", explained_variance)
print("Total explained variance:", sum(explained_variance))
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

pca = PCA(n_components=3, random_state=42)
features_pca = pca.fit_transform(features_normalized)

class_labels = ['Real', 'FM', 'FB']
colors = ['royalblue', 'darkorange', 'forestgreen']
markers = ['o', '^', 's']
fig = plt.figure(figsize=(10, 8))  
ax = fig.add_subplot(111, projection='3d', elev=25, azim=45)  


for i, (class_name, color, marker) in enumerate(zip(class_labels, colors, markers)):
    ax.scatter(
        features_pca[Y_train_classes == i, 0],  
        features_pca[Y_train_classes == i, 1],  
        features_pca[Y_train_classes == i, 2],  
        label=class_name,  
        color=color,  
        marker=marker,  
        edgecolors='k',  
        s=60,  
        alpha=0.85  
    )


ax.set_xlim(features_pca[:, 0].min() - 0.5, features_pca[:, 0].max() + 0.5)
ax.set_ylim(features_pca[:, 1].min() - 0.5, features_pca[:, 1].max() + 0.5)
ax.set_zlim(features_pca[:, 2].min() - 0.5, features_pca[:, 2].max() + 0.5)


ax.set_xlabel('PCA Component 1', fontsize=14, labelpad=15)
ax.set_ylabel('PCA Component 2', fontsize=14, labelpad=15)
ax.set_zlabel('PCA Component 3', fontsize=14, labelpad=15) 

ax.legend(fontsize=12, loc='best', edgecolor='black')


ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)


plt.savefig("PCA_visualization.png", bbox_inches='tight', pad_inches=0.2, dpi=300)


plt.show()
