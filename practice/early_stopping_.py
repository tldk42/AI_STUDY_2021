"""
early_stopping: 
    bool, default=False
    Whether to use early stopping to terminate training when validation
    score is not improving. If set to True, it will automatically set aside
    a stratified fraction of training data as validation and termiearly_stoppingnate
    training when validation score returned by the `score` method is not
    improving by at least tol for n_iter_no_change consecutive epochs.
"""