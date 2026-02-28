import pandas as pd
import numpy as np

# Pré-processamento
from sklearn.preprocessing import StandardScaler

# Divisão dos dados
from sklearn.model_selection import train_test_split

# Modelos de Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Avaliação dos modelos
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Salvar modelos
import joblib

# Gerenciar pastas
import os


# Criação das funções
# Carregar os dados
def importar_base(path):
    df = pd.read_csv(path)
    return df

# Limpeza dos dados
def qualidade_df(df):

    # Verificar nulos
    nulls = df.isnull().sum().sum()

    print(f"Valores nulos encontrados: {nulls}")

    # Verificar duplicados (sem remover)
    duplicates = df.duplicated().sum()

    print(f"Registros duplicados: {duplicates}")

    return df

# Separar variáveis X e y

def split_features_target(df, target):

    X = df.drop(target, axis=1)
    y = df[target]

    return X, y

# Encoding
def encode_features(X):

    X_encoded = pd.get_dummies(X, drop_first=True)

    return X_encoded

# Scaling
def scale_df(X):

    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X)

    # Reconstruir DataFrame com colunas
    X_scaled = pd.DataFrame(
        X_scaled_array,
        columns=X.columns,
        index=X.index
    )

    return X_scaled, scaler


# Split Treino/Teste
def split_train_test(X, y):

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

# Treinar Modelos
def treinar_modelos(X_train, X_test, y_train, y_test):

    modelos = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier()
    }

    resultado = {}

    for nome, modelo in modelos.items():

        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        resultado[nome] = {
            "modelo": modelo,
            "acuracia": acc
        }

        print(f"{nome}: {acc:.4f}")

    return resultado

# Melhor Modelo
def melhor_modelo(resultado):

    nome_melhor = max(
        resultado,
        key=lambda x: resultado[x]["acuracia"]
    )

    modelo_melhor = resultado[nome_melhor]["modelo"]
    acc_melhor = resultado[nome_melhor]["acuracia"]

    print(f"\nMelhor modelo: {nome_melhor}")
    print(f"Acurácia: {acc_melhor:.4f}")

    return modelo_melhor, nome_melhor, acc_melhor

#Avaliar Modelo
def avaliar_modelo(modelo, X_test, y_test):

    y_pred = modelo.predict(X_test)

    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))

    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    acc = accuracy_score(y_test, y_pred)

    print(f"\nAcurácia Final: {acc:.4f}")

    return acc


# Salvar Modelos
def salvar_modelo(modelo, scaler, colunas):

    if not os.path.exists("models"):
        os.makedirs("models")

    joblib.dump(modelo, "models/model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(colunas, "models/colunas.pkl")

    print("Modelo, scaler e colunas salvos com sucesso.")



def main():

    # 1. Carregar dados
    df = importar_base(
        'https://raw.githubusercontent.com/LaisST/FIAP_Tech-Challenge_Fase4/refs/heads/main/bases/Obesity.csv'
    )

    # 2. Qualidade dos dados
    df = qualidade_df(df)

    # 3. Separar X e y
    X, y = split_features_target(df, 'Obesity')

    # 4. Encoding
    X = encode_features(X)

    # Salvar colunas
    colunas = X.columns

    # 5. Scaling
    X_scaled, scaler = scale_df(X)

    # 6. Split treino/teste
    X_train, X_test, y_train, y_test = split_train_test(X_scaled, y)

    # 7. Treinar vários modelos
    resultados = treinar_modelos(
        X_train, X_test,
        y_train, y_test
    )

    # 8. Selecionar melhor
    modelo_melhor, nome_melhor, acc_melhor = melhor_modelo(resultados)

    # 9. Avaliação final
    avaliar_modelo(modelo_melhor, X_test, y_test)

    # 10. Salvar
    salvar_modelo(modelo_melhor, scaler, colunas)

    print("\nPipeline finalizado com sucesso.")


if __name__ == "__main__":
    main()
