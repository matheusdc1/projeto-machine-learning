import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN = "Qualidade_Ambiental"


def load_data(filepath: str) -> pd.DataFrame:
    """
    Carrega o dataset a partir de um arquivo CSV.
    """
    return pd.read_csv(filepath)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza a limpeza inicial dos dados.

    Etapas:
    - Converte a coluna Pressao_Atm para numérica.
    - Valores inválidos, como 'erro_sensor', viram NaN.
    """
    df = df.copy()
    df["Pressao_Atm"] = pd.to_numeric(df["Pressao_Atm"], errors="coerce")
    return df


def split_features_target(df: pd.DataFrame, target_column: str = TARGET_COLUMN):
    """
    Separa variáveis preditoras (X) e variável alvo (y).
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def split_train_test(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Divide os dados em treino e teste com estratificação na variável alvo.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Cria o pipeline de pré-processamento para variáveis numéricas.

    Etapas:
    - Imputação pela mediana para valores ausentes.
    - Padronização dos dados.
    """
    numeric_features = X.select_dtypes(include=["float64", "int64"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
        ]
    )


if __name__ == "__main__":
    df = load_data("data/dataset_ambiental.csv")
    df = clean_data(df)

    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    preprocessor = build_preprocessor(X)

    print("Dataset carregado com sucesso.")
    print("Formato do dataset limpo:", df.shape)
    print("Formato de X_train:", X_train.shape)
    print("Formato de X_test:", X_test.shape)
    print("Classes da variável alvo:")
    print(y.value_counts())
    print("Pré-processador criado com sucesso:", preprocessor)