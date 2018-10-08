"""Train a model for predict default."""
from model_default_ensemble import DefaultEnsemble

if __name__ == '__main__':
    print('Trainnig: Ensemble for Default')
    model = DefaultEnsemble()
    model.train(file_name='acquisition_train.csv')
