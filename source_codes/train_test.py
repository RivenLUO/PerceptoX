import os
import utils


def simple_training(train_data, train_label,
                    model_config, val_split, save_dir, epochs=50, batch_size=8):

    # Create folder to store results
    folder_path = utils.safe_folder_creation(save_dir)

    # Configure model
    model = model_config

    # Train model
    model_history = model.fit(train_data, train_label,
                              epochs=epochs,
                              batch_size=batch_size,
                              validation_split=val_split)

    # Save model and plots of the performance
    model_path = os.path.join(folder_path, 'model.h5')
    model.save(model_path)
    utils.plot_model_accuracy(model_history, folder_path)
    utils.plot_model_loss(model_history, folder_path)

    # Save model weights and structure
    weights_path = os.path.join(folder_path, 'weights.h5')
    model.save_weights(weights_path)
    json_path = os.path.join(folder_path, 'structure.json')
    utils.save_structure_in_json(model, json_path)
