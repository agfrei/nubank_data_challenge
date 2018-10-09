nubank-data-challenge
==============================

Nubank data challenge

Project Organization
------------
This project is based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>

This is a good pactice to try to standardize projects.

    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── deliverable        <- Deliverable for this challenge, including PDFs and CSVs
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. 
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── features       <- Scripts to turn raw data into features for modeling
        │   └── build_features.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
                              predictions
            

## How to run this project
- `make train`: Run all trainning scripts for all models
    - Trainning models can be done individually by `make train_default`, `make train_fraud`, `make train_spend`
- `make predict`: Run all predict scripts for all models and stores the deliverable CSVs in the folder `deliverable`
    - Predicting with the models can be done individually by `make predict_default`, `make predict_fraud`, `make predict_spend`
    - The prediction is done by the last trainned model, but you can choose another one (in `models` folder) and pass to the command like: `make predict_default model=2018-10-05T16-05-47_default_xgboost.bin`
- `make credit_approve`: Creates a CSV for credit approval and stores on `deliverable/approve_limit/approve_limit_submission.csv`


## Important notes
Although a very good model is very important, due to time constraints, I decided to sacrifice a bit of accuracy metrics in order to develop a more robust and complete solution.

The current solution, as it is structured, can be easily adapted to put into production, incorporating into a micro service for example. Your code is clean, simple and easy to maintain.

All the rules cited in this document are flexible and can be changed once the scenario changes, or even after we measure performance in the real world, which will be done constantly.

All notebooks in the `notebook` folder was used as draft, just to support the development of this code and should not be considered for evaluation (even if there isn't 'draft' in the name).


## Next Steps
- The tuning of the hyperparameters was not exhaustive and can be improved
- Other models can be tested.
- The pre-procecessing pipeline can be improved
- The process of feature selection can be improved
- Resolve some TODOs on code

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
