import zipfile
import gzip
import shutil
from pathlib import Path
from pandas import read_csv, read_excel
from urllib.request import urlretrieve
from sklearn.preprocessing import OrdinalEncoder


def download(url, zip_file):
    try:
        urlretrieve(url, zip_file)
    except Exception as e:
        raise RuntimeError(f"Failed to download from {url}: {e}") from e


def extract(zip_file, extract_path):
    file_type = zip_file.suffix
    try:
        if file_type == ".zip":
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(extract_path)
        elif file_type == ".gz":
            with gzip.open(zip_file, "rb") as f_in, open(extract_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            raise ValueError(f"Unknown file type: {file_type}")

    except Exception as e:
        raise RuntimeError(f"Failed to extract {zip_file}: {e}") from e


def download_and_extract(url, zip_file, extract_path):
    download(url, zip_file)
    extract(zip_file, extract_path)


def load_adult(dataset_dir):
    file = dataset_dir / "adult.data"

    if not file.exists():
        file.parent.mkdir(parents=True, exist_ok=True)
        zip_file = file.parent / "adult.zip"

        download_and_extract(
            "https://archive.ics.uci.edu/static/public/2/adult.zip",
            zip_file,
            file.parent,
        )

    X = read_csv(file, header=None)
    X.columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]

    y = X.pop("income").map({" >50K": 1, " <=50K": 0})

    objcol = X.select_dtypes(exclude=["float", "int"]).columns
    X[objcol] = OrdinalEncoder().fit_transform(X[objcol])

    return X, y


def load_bank(dataset_dir):
    file = dataset_dir / "bank-full.csv"

    if not file.exists():
        file.parent.mkdir(parents=True, exist_ok=True)
        zip_file = file.parent / "bank+marketing.zip"

        download_and_extract(
            "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
            zip_file,
            file.parent,
        )

        zip_sub_file = file.parent / "bank.zip"
        extract(zip_sub_file, file.parent)

    X = read_csv(file, sep=";")

    y = X.pop("y").map({"yes": 1, "no": 0})

    objcol = X.select_dtypes(exclude=["float", "int"]).columns
    X[objcol] = OrdinalEncoder().fit_transform(X[objcol])

    return X, y


def load_credit(dataset_dir):
    file = dataset_dir / "default of credit card clients.xls"

    if not file.exists():
        file.parent.mkdir(parents=True, exist_ok=True)
        zip_file = file.parent / "default+of+credit+card+clients.zip"

        download_and_extract(
            "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
            zip_file,
            file.parent,
        )

    X = read_excel(file, header=1)
    y = X.pop("default payment next month")

    return X, y


def load_blood(dataset_dir):
    file = dataset_dir / "transfusion.data"

    if not file.exists():
        file.parent.mkdir(parents=True, exist_ok=True)
        zip_file = file.parent / "blood+transfusion+service+center.zip"
        download_and_extract(
            "https://archive.ics.uci.edu/static/public/176/blood+transfusion+service+center.zip",
            zip_file,
            file.parent,
        )

    X = read_csv(file)
    X.columns = ["Recency", "Frequency", "Monetary", "Time", "Donated_Blood"]
    y = X.pop("Donated_Blood")
    return X, y


def load_cancer(dataset_dir):
    file = dataset_dir / "wdbc.data"

    if not file.exists():
        file.parent.mkdir(parents=True, exist_ok=True)
        zip_file = file.parent / "breast+cancer+wisconsin+diagnostic.zip"

        download_and_extract(
            "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip",
            zip_file,
            file.parent,
        )

    X = read_csv(file, header=None)
    X.columns = [
        "ID",
        "Diagnosis",
        "radius1",
        "texture1",
        "perimeter1",
        "area1",
        "smoothness1",
        "compactness1",
        "concavity1",
        "concave_points1",
        "symmetry1",
        "fractal_dimension1",
        "radius2",
        "texture2",
        "perimeter2",
        "area2",
        "smoothness2",
        "compactness2",
        "concavity2",
        "con2cave_points",
        "symmetry2",
        "fractal_dimension2",
        "radius3",
        "texture3",
        "perimeter3",
        "area3",
        "smoothness3",
        "compactness3",
        "concavity3",
        "concave_points3",
        "symmetry3",
        "fractal_dimension3",
    ]
    y = X.pop("Diagnosis").map({"M": 1, "B": 0})
    return X, y


def load_ecoli(dataset_dir):
    file = dataset_dir / "ecoli.data"

    if not file.exists():
        file.parent.mkdir(parents=True, exist_ok=True)
        zip_file = file.parent / "ecoli.zip"

        download_and_extract(
            "https://archive.ics.uci.edu/static/public/39/ecoli.zip",
            zip_file,
            file.parent,
        )

    X = read_csv(file, sep="\s+", header=None)
    X.columns = ["sequence", "mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2", "class"]
    y = X.pop("class").map(
        {
            "cp": 0,
            "im": 1,
            "pp": 2,
            "imU": 3,
            "om": 4,
            "omL": 5,
            "imL": 6,
            "imS": 7,
        }
    )

    X[["sequence"]] = OrdinalEncoder().fit_transform(X[["sequence"]]).astype(int)

    return X, y


def load_house(dataset_dir):
    file = dataset_dir / "train.csv"

    if not file.exists():
        print(
            "Please manually download the `train.csv` file from:\n"
            "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data?select=train.csv \n"
            "and place it under the folder `dataset/house/`."
        )

    X = read_csv(file)

    y = X.pop("SalePrice")

    objcol = X.select_dtypes(exclude=["float", "int"]).columns
    X[objcol] = OrdinalEncoder().fit_transform(X[objcol])

    return X, y


def load_data(dataset, strictly_positive=False):
    PROJECT_ROOT = Path(__file__).parent
    dataset_dir = PROJECT_ROOT / f"dataset/{dataset}"

    uci_data = {
        "adult": lambda: load_adult(dataset_dir),
        "bank": lambda: load_bank(dataset_dir),
        "credit": lambda: load_credit(dataset_dir),
        "blood": lambda: load_blood(dataset_dir),
        "cancer": lambda: load_cancer(dataset_dir),
        "ecoli": lambda: load_ecoli(dataset_dir),
        "house": lambda: load_house(dataset_dir),
    }

    if dataset not in uci_data:
        raise ValueError(f"Unknown dataset: {dataset}")

    X, y = uci_data[dataset]()

    if strictly_positive:
        X = X.loc[:, (X > 0).all(axis=0)]
    return X, y
