def chain_classifier(model, x_train, y_train, x_validation=None, y_validation=None, 
                     x_test=None, y_test=None, validate=False, test=False, display=False, tune=False, param=None):

    print('Model = ', model, '\n')
    train_scores = []
    validation_scores = []
    test_scores = []
    model_details = []
    
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.metrics import accuracy_score, f1_score
    
    for label in cols_target:

        print('... Processing {} \n'.format(label))
        
        y_train_label = y_train[label]
        y_validation_label = y_validation[label]
        y_test_label = y_test[label]
        
        # To tune the model
        if tune:
            model = RandomizedSearchCV(model, param, n_jobs=-1, cv=10)
        
        # train the model using x_train & y_train
        model.fit(x_train,y_train_label)
        
        # compute the training results
        y_train_pred = model.predict(x_train)
        
        if display:
            print('Training Accuracy is {}'.format(accuracy_score(y_train_label, y_train_pred)))
            print('Training F1Score is {} \n'.format(f1_score(y_train_label, y_train_pred)))
        
        # Append scores
        to_append = (accuracy_score(y_train_label, y_train_pred), f1_score(y_train_label, y_train_pred))
        train_scores.append(to_append)
        
        # Adding predictions as features
        x_train = add_feature(x_train, y_train_pred)
        
        if validate:
            # compute validation results
            y_validation_pred = model.predict(x_validation)
            
            if display:
                print('Validation Accuracy is {}'.format(accuracy_score(y_validation_label, y_validation_pred)))
                print('Validation F1Score is {} \n'.format(f1_score(y_validation_label, y_validation_pred)))
            
            # Adding prediction as feature
            x_validation = add_feature(x_validation, y_validation_pred)
            
            
            # Append scores
            to_append = (accuracy_score(y_validation_label, y_validation_pred), f1_score(y_validation_label, y_validation_pred))
            validation_scores.append(to_append)

        if test:
            # compute test results
            y_test_pred = model.predict(x_test)
            
            if display:
                print('Test Accuracy is {}'.format(accuracy_score(y_test_label, y_test_pred)))
                print('Test F1Score is {} \n'.format(f1_score(y_test_label, y_test_pred)))
            
            # append Scores
            to_append = (accuracy_score(y_test_label, y_test_pred), f1_score(y_test_label, y_test_pred))
            test_scores.append(to_append)
            
            # Adding prediction as feature
            x_test = add_feature(x_test, y_test_pred)
        
        model_details.append(model)
        
        
    scores = (train_scores, validation_scores, test_scores)    
        
    return scores, model_details
