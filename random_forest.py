import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# ==========================================
# 1. CARGAR DATASET
# ==========================================

df = pd.read_csv("data/breast-cancer.csv")

print("Dataset cargado correctamente.")
print(f"Número de registros: {len(df)}")
print(f"Número de variables: {len(df.columns)}")


# ==========================================
# 2. PREPARAR LOS DATOS
# ==========================================

# El ID no aporta información para la clasificación
X = df.drop(columns=["id", "diagnosis"])

# Convertir:
# M = 1 (Maligno)
# B = 0 (Benigno)
y = df["diagnosis"].map({
    "M": 1,
    "B": 0
})


# ==========================================
# 3. DIVISIÓN ENTRENAMIENTO / PRUEBA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nDivisión del dataset:")
print(f"Datos de entrenamiento: {len(X_train)}")
print(f"Datos de prueba: {len(X_test)}")


# ==========================================
# 4. CREAR RANDOM FOREST
# ==========================================

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


# ==========================================
# 5. ENTRENAMIENTO
# ==========================================

print("\nEntrenando Random Forest...")

model.fit(X_train, y_train)

print("Entrenamiento terminado.")


# ==========================================
# 6. PREDICCIONES
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 7. MÉTRICAS
# ==========================================

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


# ==========================================
# 8. MATRIZ DE CONFUSIÓN
# ==========================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Benigno", "Maligno"]
)

disp.plot()

plt.title("Matriz de Confusión - Random Forest")
plt.tight_layout()

plt.savefig(
    "results/matriz_confusion.png",
    dpi=300
)

plt.show()


# ==========================================
# 9. IMPORTANCIA DE VARIABLES
# ==========================================

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


# ==========================================
# 10. GUARDAR RESULTADOS
# ==========================================

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
