from IPython.display import Image, display
from sklearn.tree import export_graphviz
import graphviz

def show_tree(model, feature_names=None, class_names=None,
              save2fpath=None, return_graph=False,
              width=1000, height=None, impurity=False, filled=True,
              proportion=True, rounded=True,
              special_characters=True):
    graph = graphviz.Source(export_graphviz(
        model, impurity=impurity, filled=filled,
        proportion=proportion, rounded=rounded,
        feature_names=feature_names,
        class_names=class_names,
        special_characters=special_characters
    ))

    img = graph.pipe(format='png')
    display(Image(img, width=width, height=height))

    if not save2fpath is None:
        with open(save2fpath, 'wb') as file:
            file.write(img)

    if return_graph:
        return graph