# Creating FastText train and test files


def fastText_files(representation):
    """
    This function creates and saves the test and train file
    from the test, train and dev split of the dataset (named test, dev and train),
    using the "primary_level_3" level labels, and the chosen text representation.
    
    Possible representations: 'baseline_text', 'lemmas',
    'upos', 'xpos', 'ner', 'dependency', 'lowercase', 'lowercase_nopunctuation'
    
    The function returns a list of the following elements:
        - labels - which can be used for prediction and evaluation.
        - train file path
        - test file path
    
    Args:
        representation (str): the name of the key (from the dataset)
                                of the text representation we want to use
    """
    # First create the dataframes from each split:
    
    train_df = pd.DataFrame(data=train, columns=[representation, "primary_level_3"])
    # Renaming columns to `text` and `labels`
    train_df.columns = ["text", "labels"]
    
    test_df = pd.DataFrame(data=test, columns=[representation, "primary_level_3"])
    test_df.columns = ["text", "labels"]
    
    print("The shape of the dataframes:")
    print(train_df.shape, test_df.shape)
    
    # Then create CSV files which FastText can read
    
    train_file_content=""

    for labels, text in train_df.loc[:, ["labels", "text"]].values:
        label = f"__label__{labels}"
        train_file_content += f"""{label} {text}\n"""
    
    train_path = ""
    train_path = representation + "-fasttext.train"

    with open(train_path,"w") as train_file:
        train_file.write(train_file_content)
    
    train_example = open(train_path,"r").read(1000)
    print("Created train file:")
    print(train_example)
    
    test_file_content=""
    
    for labels, text in test_df.loc[:, ["labels", "text"]].values:
        label = f"__label__{labels}"
        test_file_content += f"""{label} {text}\n"""
    
    test_path = ""
    test_path = representation + "-fasttext.test"
    
    with open(test_path,"w") as test_file:
        test_file.write(test_file_content)
    
    test_example = open(test_path,"r").read(1000)
    print("Created test file:")
    print(test_example)
    
    
    # Finally, create a list of labels which can be used for prediction and evaluation.
    # Let's inspect the labels:
    all_df_labels = train_df["labels"].unique().tolist()
    
    for i in test_df["labels"].unique().tolist():
        if i not in all_df_labels:
            all_df_labels.append(i)

    print(f"Number of all labels: {len(all_df_labels)}")
    
    # Create a final list of labels in a FastText-appropriate format:
    LABELS = train_df.labels.unique().tolist()
    LABELS = [f"__label__{i}" for i in LABELS]
    
    return_list = [LABELS, train_path, test_path]
    print(f"The function returned the following list: {return_list}")
    
    return return_list

def parse_test_file(path: str):
    """Reads fasttext formatted file and returns labels, texts."""
    with open(path, "r") as f:
        content = f.readlines()
    pattern = "{label} {text}\n"
    p = parse.compile(pattern)

    labels, texts = list(), list()
    for line in content:
        rez = p.parse(line)
        if rez is not None:
            labels.append(rez["label"])
            texts.append(rez["text"])
        else:
            print("error parsing line ", line)
    return labels, texts

def prediction_to_label(prediction):
    """Transforms predictions as returned by fasttext into pure labels."""
    return np.array(prediction[0])[:, 0]

def plot_cm(save=False, title=None):
    """
    Plots confusion matrix for prediction on the test set.
    Takes the predictions, named as y_pred, true values, named as y_true,
    and labels, named as LABELS.
    
    Arguments:
        save: whether the confusion matrix is saved. Defaults to False.
        title: the title of the confusion matrix. Defaults to None.
    """
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    plt.figure(figsize=(9, 9))
    plt.imshow(cm, cmap="Oranges")
    classNames = LABELS
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=90)
    plt.yticks(tick_marks, classNames)
    microF1 = f1_score(y_true, y_pred, labels=LABELS, average ="micro")
    macroF1 = f1_score(y_true, y_pred, labels=LABELS, average ="macro")

    print(f"{microF1:0.4}")
    print(f"{macroF1:0.4}")

    metrics = f"{microF1:0.4}, {macroF1:0.4}"
    if title:
        plt.title(title +";\n" + metrics)
    else:
        plt.title(metrics)
    plt.tight_layout()
    if save:
        plt.savefig(save)
    plt.show()
    return microF1, macroF1

def train_FastText(representation):
    """
    The function uses the created FT_train_file and FT_test_file
    and performs five runs of training and evaluation of the model.
    It plots a confusion matrix for each run.

    Args:
        representation (str): the name of the key (from the dataset)
                                of the text representation we want to use
    
    """
    results = []

    for i in range(5):
        model = ft.train_supervised(input=FT_train_file,
                                    epoch = 350,
                                    lr = 0.7,
                                    wordNgrams=1,
                                    verbose = 2
                                                )
        # Parse the test files so that labels and texts are separated
        y_true, y_texts = parse_test_file(FT_test_file)

        # Evaluate te model on test data
        y_pred = model.predict(y_texts)
        y_pred = prediction_to_label(y_pred)

        # Plot the confusion matrix:
        m, M = plot_cm(save=False, title=f"Run: {i}")
        
        rezdict = dict(
            microF1=m,
            macroF1=M,
            run=i,
            experiment= representation,
        )
        results.append(rezdict)
        final_results.append(rezdict)
    
    # Calculate the average micro and macro F1 for the 5 runs:
    mi = []
    ma = []
    
    for i in results:
    mi.append(i['microF1'])
    ma.append(i["macroF1"])

    print(f"micro F1: {np.array(mi).mean():0.03} +/- {np.array(mi).std():0.02}")
    print(f"macro F1: {np.array(ma).mean():0.03} +/- {np.array(ma).std():0.02}")