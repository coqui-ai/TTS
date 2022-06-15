import argparse
import pickle

import numpy as np
import tensorflow as tf
from clearml import Task
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import KFold, cross_val_score


def load_data(train, labels):
    t  = np.load(train)
    l = np.load(labels)
    return t,l

def compute_metrics(y_pred, y_true):
    acc = accuracy_score(y_true.argmax(axis=1),y_pred.argmax(axis=1))
    print(f"Model Accuracy on untrained data: {acc}")
    conf_matrix = confusion_matrix(y_true.argmax(axis=1),y_pred.argmax(axis=1))
    print(f"Confusion matrix on {y_true.shape[-1]} classes: \n {conf_matrix}")
    f1 = f1_score(y_true.argmax(axis=1),y_pred.argmax(axis=1),average='weighted')
    print(f"Weighted f1 score: {f1}")

def test_model(test,test_labels,model):
    test, y_true = load_data(test, test_labels)
    y_pred = model.predict(test, verbose=1)
    compute_metrics(y_pred,y_true)

def baseline_model(class_num):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(1024, input_dim=512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(class_num, activation='softmax'))
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_data",
        nargs='?',
    )
    parser.add_argument(
        "train_labels",
        nargs='?',
    )
    parser.add_argument(
        "test_data",
        nargs='?',
    )
    parser.add_argument(
        "test_labels",
        nargs='?',
    )
    parser.add_argument(
            '--eval',
            action='store_true'
    )
    parser.add_argument(
            '--load_ckpt',
            required=False,
    )
    parser.add_argument(
            '--predict',
            action='store_true'
    )
    parser.add_argument(
            'class_weights',
    )
    parser.add_argument(
            'epoch',
    )
    parser.add_argument(
        "--clearml_project",
        default="YourTTS-sprint2",
    )
    parser.add_argument(
        "--clearml_task",
        default="attribute-classifier",
    )
    parser.add_argument(
        "label",
    )


    args = parser.parse_args()
    task = Task.init(
        project_name=args.clearml_project,
        task_name=f"{args.clearml_task}-{args.label}",
    )
    train, labels = load_data(args.train_data, args.train_labels)
    model = baseline_model(len(np.unique(labels,axis=0)))
    with open(args.class_weights, 'rb') as f:
        class_weights = pickle.load(f)
    if not args.eval and not args.predict:
        print(f"Training on: {train.shape} and testing on {labels.shape}")
        EPOCHS = int(args.epoch)
        checkpoint_filepath = "checkpoints/checkpoint"
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        verbose=1,
        mode='max',
        save_best_only=True)
        model.fit(epochs=EPOCHS,x=train,y=labels,validation_split=0.2,batch_size=100, callbacks=[model_checkpoint_callback], class_weight=class_weights)
    else:
        model.load_weights(args.load_ckpt)
    test_model(args.test_data, args.test_labels,model)
    if(args.predict):
        test = np.load(test)
        y_pred = model.predict(test, verbose=1)
        num_labels = np.where(y_pred==1)[1]
        np.save("predicted_labels.npy",num_labels)
if __name__ == "__main__":
    main()
#    kfold = KFold(n_splits=10, shuffle=True)
#    results = cross_val_score(estimator, train, labels, cv=kfold)
#    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
