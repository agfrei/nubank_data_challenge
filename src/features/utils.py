from IPython.core.interactiveshell import InteractiveShell
from IPython.display import set_matplotlib_formats

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def check_categorical_variables(df: pd.DataFrame):
    pass

def notebook_definitions():
    InteractiveShell.ast_node_interactivity = "all"
    pd.options.display.max_columns = None

def plot_definitions():
    set_matplotlib_formats('pdf', 'png')
    pd.options.display.float_format = '{:.2f}'.format
    rc={'savefig.dpi': 75, 'figure.autolayout': False, 'figure.figsize': [12, 8], 'axes.labelsize': 18,\
       'axes.titlesize': 18, 'font.size': 18, 'lines.linewidth': 2.0, 'lines.markersize': 8, 'legend.fontsize': 16,\
       'xtick.labelsize': 16, 'ytick.labelsize': 16}

    sns.set(style='dark',rc=rc)

    default_color = '#56B4E9'
    colormap = plt.cm.cool

# with open(model_name, 'wb') as model_file:
#     pickle.dump(self.model, model_file)
# with open(model_name, 'rb') as model_file:
#     self.model = pickle.load(model_file)

def random_forest_model(self) -> RandomForestClassifier:
        params = {}
        params['n_estimators'] = 200
        params['max_depth'] = 6
        params['min_samples_split'] = 70
        params['min_samples_leaf'] = 30

        return RandomForestClassifier(**params)

def svc_model(self) -> SVC:
        params = {}
        params['C'] = 0.1
        params['kernel'] = 'poly'
        params['gamma'] = 27

        return SVC(**params)


