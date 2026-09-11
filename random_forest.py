import os
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk

from tkinter import filedialog

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    log_loss
)
# Cargar el dataset

def seleccionar_dataset():

    root = tk.Tk()
    root.withdraw()

    archivo = filedialog.askopenfilename(
        title="Selecciona el dataset",
        filetypes=[
            ("Archivos CSV", "*.csv"),
            ("Todos los archivos", "*.*")
        ]
    )

    root.destroy()

    return archivo


print("Selecciona el archivo CSV del dataset...")

DATASET_PATH = seleccionar_dataset()

if not DATASET_PATH:
    print("No se seleccionó ningún archivo.")
    exit()

print(f"\nDataset seleccionado:")
print(DATASET_PATH)

df = pd.read_csv(DATASET_PATH)

print("\nDataset cargado correctamente.")
print(f"Número de registros: {len(df)}")
print(f"Número de variables: {len(df.columns)}")

# Preparación de los datos

# El ID no aporta información para la clasificación
X = df.drop(columns=["id", "diagnosis"])

# Convertir:
# M = 1 (Maligno)
# B = 0 (Benigno)
y = df["diagnosis"].map({
    "M": 1,
    "B": 0
})


# Split del dataset para entrenamiento y prueba 

# 80% temporal, 20% prueba final
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Del 80% restante:
# 75% -> entrenamiento
# 25% -> validación
#
# Resultado final:
# 60% entrenamiento
# 20% validación
# 20% prueba

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)

print("\nDivisión del dataset:")
print(f"Datos de entrenamiento: {len(X_train)}")
print(f"Datos de validación: {len(X_val)}")
print(f"Datos de prueba: {len(X_test)}")

# Creación de random forest

model = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)


# Entrenamiento

print("\nEntrenando Random Forest...")

model.fit(X_train, y_train)

print("Entrenamiento terminado.")


# Predicciones

y_pred = model.predict(X_test)


# Creación de carpeta de resultados


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)

# Training vs evaluation acurracy

print("\nGenerando gráfica de entrenamiento y validación...")

n_estimators_values = range(10, 201, 10)

training_accuracy = []
validation_accuracy = []

for n in n_estimators_values:

    rf = RandomForestClassifier(
        n_estimators=n,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    train_prediction = rf.predict(X_train)
    validation_prediction = rf.predict(X_val)

    train_acc = accuracy_score(
        y_train,
        train_prediction
    )

    validation_acc = accuracy_score(
        y_val,
        validation_prediction
    )

    training_accuracy.append(train_acc)
    validation_accuracy.append(validation_acc)


plt.figure(figsize=(10, 6))

plt.plot(
    n_estimators_values,
    training_accuracy,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    n_estimators_values,
    validation_accuracy,
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Número de árboles (n_estimators)")
plt.ylabel("Accuracy")

plt.title(
    "Training vs Validation Accuracy"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "training_validation_accuracy.png"
    ),
    dpi=900
)

plt.close()

print(
    "Gráfica guardada: "
    "results/training_validation_accuracy.png"
)

# Training vs validation loss

print("\nGenerando gráfica de loss...")

training_loss = []
validation_loss = []

for n in n_estimators_values:

    rf = RandomForestClassifier(
        n_estimators=n,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    train_probabilities = rf.predict_proba(
        X_train
    )

    validation_probabilities = rf.predict_proba(
        X_val
    )

    train_loss = log_loss(
        y_train,
        train_probabilities
    )

    validation_loss_value = log_loss(
        y_val,
        validation_probabilities
    )

    training_loss.append(train_loss)
    validation_loss.append(
        validation_loss_value
    )


plt.figure(figsize=(10, 6))

plt.plot(
    n_estimators_values,
    training_loss,
    marker="o",
    label="Training Loss"
)

plt.plot(
    n_estimators_values,
    validation_loss,
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Número de árboles (n_estimators)")
plt.ylabel("Log Loss")

plt.title(
    "Training vs Validation Loss"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "training_validation_loss.png"
    ),
    dpi=300
)

plt.close()

print(
    "Gráfica guardada: "
    "results/training_validation_loss.png"
)

# Métricas

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n========== RESULTADOS ==========")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

print("\nMatriz de confusión:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\nReporte de clasificación:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Benigno", "Maligno"]
))


# Matriz de confusión 

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Benigno", "Maligno"]
)

disp.plot()

plt.title("Matriz de Confusión - Random Forest")
plt.tight_layout()

os.makedirs("results", exist_ok=True)

plt.savefig(
    "results/matriz_confusion.png",
    dpi=900
)
plt.show()


# Importancia de las variables

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("\n========== VARIABLES MÁS IMPORTANTES ==========")

print(feature_importance.head(10).to_string(index=False))


# Guardar resultados

with open("results/resultados.txt", "w", encoding="utf-8") as file:

    file.write("RESULTADOS RANDOM FOREST\n")
    file.write("========================\n\n")

    file.write(f"Registros totales: {len(df)}\n")
    file.write(f"Datos de entrenamiento: {len(X_train)}\n")
    file.write(f"Datos de prueba: {len(X_test)}\n\n")

    file.write("Métricas:\n")
    file.write(f"Accuracy:  {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall:    {recall:.4f}\n")
    file.write(f"F1-Score:  {f1:.4f}\n\n")

    file.write("Matriz de confusión:\n")
    file.write(str(cm))
    file.write("\n\n")

    file.write("Variables más importantes:\n")
    file.write(feature_importance.head(10).to_string(index=False))

print("\nResultados guardados en results/resultados.txt")
