from analysis_func import load_data, preprocess_data, scale_data, create_sequences, build_rnn, fit_regressor, plot_predictions, rolling_origin_predict

def main():
    train_path = "../data/weather/AarhusSydObservations/data_partitions/Partition_test_120_train_750_train.csv"
    test_path = "../data/weather/AarhusSydObservations/data_partitions/Partition_test_120_train_750_test.csv"

    # Load data
    train, test = load_data(train_path, test_path)
    
    # Preprocess data
    dataset_train, dataset_test = preprocess_data(train, test)
    
    # Scale data
    scaled_train, scaled_test, scaler = scale_data(dataset_train, dataset_test)

    # Create sequences
    X_train, y_train = create_sequences(scaled_train, look_back=30, predict_forward=12)
    X_test, y_test = create_sequences(scaled_test, look_back=30, predict_forward=12)

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)

    # build and fit RNN model
    regressor = build_rnn((X_train.shape[1], 1))
    regressor = fit_regressor(regressor, X_train, y_train, epochs=1, batch_size=1)
    regressor.summary()

    # make predictions
    #dataframe_ = rolling_origin_predict(regressor, scaled_train, scaled_test, scaler)

    # save dataframe to csv
    #dataframe_.to_csv("RNN_predictions.csv", index=False)

    # predictions with X_test data
    y_RNN = regressor.predict(X_test)
    print(":", y_RNN.shape)
    print(y_RNN)
    y_RNN_O = scaler.inverse_transform(y_RNN)

    # plot and save plot
    plot_predictions(train, test, y_RNN_O)

if __name__ == "__main__":
    main()