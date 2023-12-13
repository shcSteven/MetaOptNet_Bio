python run.py exp.name=metaopt_lr_sgd lr=0.1 method=metaoptnet dataset=tabula_muris
python run.py exp.name=metaopt_lr_sgd lr=0.01 method=metaoptnet dataset=tabula_muris 
python run.py exp.name=metaopt_lr_sgd lr=0.001 method=metaoptnet dataset=tabula_muris


python run.py exp.name=swissprot_mean_sgd lr=0.1 method=metaoptnet dataset=swissprot


python run.py exp.name=scale_tabula_svm lr=0.1 method=metaoptnet dataset=tabula_muris
