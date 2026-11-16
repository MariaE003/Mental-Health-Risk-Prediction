import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def prepare_data(data_path,output_dir):

    data = pd.read_csv(data_path)

    X = data.drop(columns=["Student_ID", "Mental_Health_Risk"])
    y = data["Mental_Health_Risk"]

    # train and temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    # validation and test 
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)

    dataset = {
        "X_train" :X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train":y_train,
        "y_val": y_val,
        "y_test": y_test
    }

    for name,dataset in dataset.items():
        dataset.to_csv(output_dir / f"{name}.csv",index=False)

    print("Data preparation completed!")
    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    return True