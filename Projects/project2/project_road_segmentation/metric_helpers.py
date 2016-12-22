from keras.models import load_model
from post_padding_second import extract_data_sec, extract_labels_sec
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

def generate_roc(model_name, post_model_name, train_range):
    # ********** PARAMETERS ***************************************************************
    # Size of the train images
    IMG_SIZE = 400
    # patches used in training of the first CNN
    PATCH_UNIT = 8
    # patch window used in training of post processing CNN
    PATCH_WINDOW = 21  # MUST BE ODD
    #******* EXCTRACT DATA ***********************************************************
    data_dir = 'training/'
    train_data_filename = data_dir + 'images/'
    train_labels_filename = data_dir + 'groundtruth/'
    file_str = "satImage_%.3d"
    post_model_path = 'models/POST/' + post_model_name

    print("Extracting data")
	# Extract the data from images by predicting with first CNN and padd the results
    pred = extract_data_sec(model_name, train_range, train_data_filename, file_str, PATCH_UNIT, PATCH_WINDOW, IMG_SIZE)
    Y = extract_labels_sec(train_range, train_labels_filename, PATCH_UNIT, PATCH_WINDOW)

    print("Predicting probabilities")
    # load post_modal and predict probabilities
    post_model = load_model(post_model_path)
    y_scores = post_model.predict_proba(pred, verbose=1)

    print("Building ROC")
    # build roc curve for road class
    fpr, tpr, thresholds = roc_curve(Y[:,0], y_scores[:,0], pos_label=1)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
