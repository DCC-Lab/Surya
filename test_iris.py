import seaborn as sns
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
print(iris.keys())


# Rename classes using the iris target names
iris.frame["target"] = iris.target_names[iris.target]
_ = sns.pairplot(iris.frame, hue="target")