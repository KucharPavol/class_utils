#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from IPython.display import Image, display
from sklearn.tree import export_graphviz
import graphviz

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.base import is_classifier
import matplotlib.pyplot as plt
import pandas as pd

from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from pdpbox import pdp, get_dataset, info_plots
import eli5

def show_tree(model, feature_names=None, class_names=None,
              save2fpath=None, return_graph=False):    
    graph = graphviz.Source(export_graphviz(
        model, impurity=False, filled=True,
        proportion=True, rounded=True,
        feature_names=feature_names,
        class_names=class_names,
        special_characters=True
    ))
    
    img = graph.pipe(format='png')
    display(Image(img))

    if not save2fpath is None:
        with open(save2fpath, 'wb') as file:
            file.write(img)

    if return_graph:
        return graph

class Explainer:
    def __init__(self,
         model,
         df_train,
         categorical_inputs,
         categorical_imputer,
         numeric_inputs,
         numeric_imputer,
         input_preproc,
         class_names=None,
         **kwargs
    ): 
        """
        Args:
            categorical_imputer: The imputer that is to be used for categorical columns.
                The imputer is not allowed to add new columns or change order of the
                existing ones.

            numeric_imputer: The imputer that is to be used for numeric columns.
                The imputer is not allowed to add new columns or change order of the
                existing ones.
        """
        self.model = model
        self.categorical_inputs = categorical_inputs
        self.categorical_imputer = categorical_imputer
        self.numeric_inputs = numeric_inputs
        self.numeric_imputer = numeric_imputer
        self.input_preproc = input_preproc
        class_names = [str(c) for c in class_names]
        
        self.interpret_preproc = make_column_transformer(
            (make_pipeline(
                # wrap in a function transformer to prevent being refitted
                FunctionTransformer(categorical_imputer.transform, validate=False),
                OrdinalEncoder()),
             categorical_inputs),

             # wrap in a function transformer to prevent being refitted
            (FunctionTransformer(numeric_imputer.transform, validate=False),
             numeric_inputs)
        )
        
        xx_train = self.interpret_preproc.fit_transform(
            df_train[self.categorical_inputs+self.numeric_inputs])
        
        if xx_train.shape[1] != len(categorical_inputs) + len(numeric_inputs):
            raise ValueError("Imputers are not allowed to add new columns or to change their order.")
        
        self.ordenc = self.interpret_preproc.transformers_[0][1][1]
        
        try:
            cat_name_idx = {
                k: v for k, v in enumerate(
                    self.ordenc.categories_
                )
            }
            
            self.categorical_names = {
                k: v for k, v in zip(
                    categorical_inputs,
                    self.ordenc.categories_
                )
            }
        except AttributeError:
            cat_name_idx = {}
            self.categorical_names = {}
        
        self.explainer = LimeTabularExplainer(
            xx_train,
            feature_names=categorical_inputs+numeric_inputs,
            class_names=class_names,
            categorical_features=range(len(categorical_inputs)),
            categorical_names=cat_name_idx,
            mode="classification" if is_classifier(self.model) else "regression",
            **kwargs
        )
        
        self.full_model = make_pipeline(
            FunctionTransformer(self._preproc_fn),
            self.model
        )
        
    def _preproc_fn(self, x):
        df_inst = pd.DataFrame(
            x, columns=self.categorical_inputs+self.numeric_inputs
        )

        if self.categorical_inputs:
            df_inst[self.categorical_inputs] = self.ordenc.inverse_transform(
                df_inst[self.categorical_inputs].values
            )

        x_preproc = self.input_preproc.transform(df_inst)
            
        return x_preproc

    def explain(self, df_inst):
        if len(df_inst.shape) == 1:
            df_inst = df_inst.to_frame().transpose()

        x_inst = self.interpret_preproc.transform(df_inst)[0]
        
        return self.explainer.explain_instance(
            x_inst,
            self.full_model.predict_proba if hasattr(self.full_model, 'predict_proba')
                else self.full_model.predict
        )
    
    def pdp_plot(self, df_inputs, feature_name, **kwargs):
        xx_test = pd.DataFrame(
            self.interpret_preproc.transform(df_inputs),
            columns=self.categorical_inputs+self.numeric_inputs
        )
        
        pdp_goals = pdp.pdp_isolate(
            model=self.full_model,
            dataset=xx_test,
            model_features=self.categorical_inputs+self.numeric_inputs,
            feature=feature_name
        )
        
        pdp.pdp_plot(pdp_goals, feature_name, **kwargs)
        
        try:
            t = self.categorical_names[feature_name]
            plt.xticks(pdp_goals.feature_grids, t)
        except KeyError:
            pass
        
        return pdp_goals
    
    def permutation_importance(self, df_inputs, df_outputs, **kwargs):
        xx = self.interpret_preproc.transform(
            df_inputs[self.categorical_inputs+self.numeric_inputs])
        yy = df_outputs
        
        perm = eli5.sklearn.PermutationImportance(
            self.full_model, **kwargs
        ).fit(xx, yy)
        
        display(eli5.show_weights(perm,
            feature_names=self.categorical_inputs+self.numeric_inputs))
        
        return perm
