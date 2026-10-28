# ---------------------------------------------------------------------
# IMPORTACIÓN DE LIBRERÍAS
# ---------------------------------------------------------------------
import streamlit as st                      # Framework para construir la interfaz web
import pandas as pd                         # Manejo de tablas (DataFrames)
import numpy as np                          # Operaciones numéricas
import matplotlib.pyplot as plt             # Gráficas base
import seaborn as sns                       # Gráficas estadísticas (histogramas, heatmaps)
import sklearn                              # Se importa para mostrar su versión en la ficha técnica
from sklearn.model_selection import train_test_split, cross_val_score  # Partición de datos y validación cruzada
from sklearn.preprocessing import StandardScaler                       # Estandarización de variables
from sklearn.linear_model import LogisticRegression                    # Algoritmo clásico elegido
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, roc_auc_score)  # Métricas de evaluación

# =====================================================================
# 0. CONFIGURACIÓN GENERAL Y ESTILO
# =====================================================================
# Título de la pestaña del navegador, ícono y diseño de ancho completo
st.set_page_config(page_title="Predicción de Enfermedad Cardíaca - ML",
                   page_icon="🩺", layout="wide")

# Colores que se usan en todas las gráficas: azul = sin enfermedad, naranja = con enfermedad
COLOR_SANO, COLOR_ENFERMO = "#2b5c8f", "#d95f02"

# CSS personalizado: reduce el margen superior, redondea las tarjetas de métricas
# y define el estilo de la caja informativa (.caja-info)
st.markdown("""
<style>
    .block-container {padding-top: 1.5rem;}
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.35);
        border-radius: 12px; padding: 12px 16px;
    }
    .caja-info {
        border-left: 5px solid #2b5c8f; padding: 10px 16px;
        background: rgba(43,92,143,0.10); border-radius: 6px; margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Variables (columnas) que usa el modelo como entrada, según lo hallado en el EDA
FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca", "cp", "exang"]
# Nombres legibles en español para mostrar en gráficas y tablas
NOMBRES = {"age": "Edad", "trestbps": "Presión en reposo", "chol": "Colesterol",
           "thalach": "FC máxima", "oldpeak": "Depresión ST", "ca": "Vasos coloreados",
           "cp": "Tipo de dolor de pecho", "exang": "Angina por ejercicio"}

# =====================================================================
# 1. CARGA Y PREPROCESAMIENTO
# =====================================================================
# @st.cache_data guarda el resultado en memoria para no recargar el Excel en cada interacción
@st.cache_data
def cargar_datos_crudos():
    # Prueba primero la ruta de Google Drive y luego el archivo en la carpeta local
    for ruta in ["/content/drive/MyDrive/APRENDIZAJE/heart_disease.xlsx", "heart_disease.xlsx"]:
        try:
            return pd.read_excel(ruta, sheet_name="heart_disease")  # Lee la hoja del Excel
        except Exception:
            continue  # Si esa ruta falla, intenta con la siguiente
    return None  # Si ninguna ruta funcionó, devuelve None

@st.cache_data
def limpiar_datos(df_raw):
    df = df_raw.copy()                                          # Copia para no modificar los datos originales
    df["ca"] = df["ca"].fillna(df["ca"].mode()[0])              # Imputa los faltantes de 'ca' con la moda
    df["thal"] = df["thal"].fillna(df["thal"].mode()[0])        # Imputa los faltantes de 'thal' con la moda
    df["target_bin"] = (df["target"] > 0).astype(int)           # Variable objetivo binaria: 0 = sano, 1 = enfermo (grados 1 a 4)
    df["enfermo_label"] = df["target_bin"].map({0: "Sin enfermedad", 1: "Con enfermedad"})  # Etiqueta legible del diagnóstico
    df["sexo_label"] = df["sex"].map({1: "Hombre", 0: "Mujer"})  # Etiqueta legible del sexo
    return df

df_raw = cargar_datos_crudos()                                  # Datos tal como vienen en el Excel
if df_raw is None:                                              # Si no se encontró el archivo...
    st.error("No se encontró el archivo heart_disease.xlsx. Verifica la ruta de carga.")  # ...muestra el error
    st.stop()                                                   # ...y detiene la app
df = limpiar_datos(df_raw)                                      # Datos ya limpios y con columnas auxiliares

# =====================================================================
# 2. ENTRENAMIENTO DEL MODELO (una sola vez, compartido por todas las pestañas)
# =====================================================================
# @st.cache_resource guarda el modelo entrenado para no reentrenar en cada clic
@st.cache_resource
def entrenar_modelo(df):
    X, y = df[FEATURES], df["target_bin"]                       # X = variables de entrada, y = diagnóstico binario
    # Separa 80% entrenamiento / 20% prueba, manteniendo la proporción de clases (stratify)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()                                   # Escalador: media 0 y desviación 1
    X_tr_s, X_te_s = scaler.fit_transform(X_tr), scaler.transform(X_te)  # Ajusta con entrenamiento y aplica a ambos
    model = LogisticRegression(random_state=42, max_iter=1000)  # Regresión Logística (hasta 1000 iteraciones)
    model.fit(X_tr_s, y_tr)                                     # Entrena el modelo
    y_pred = model.predict(X_te_s)                              # Clase predicha en el conjunto de prueba
    y_proba = model.predict_proba(X_te_s)[:, 1]                 # Probabilidad de tener enfermedad
    # Validación cruzada de 5 folds sobre todos los datos para medir la estabilidad del accuracy
    cv = cross_val_score(LogisticRegression(random_state=42, max_iter=1000),
                         scaler.transform(X), y, cv=5, scoring="accuracy")
    # Devuelve en un diccionario todo lo que necesitan las pestañas
    return {
        "model": model, "scaler": scaler, "y_test": y_te, "y_pred": y_pred,
        "y_proba": y_proba, "n_train": len(X_tr), "n_test": len(X_te), "cv": cv,
        "acc": accuracy_score(y_te, y_pred),                    # Accuracy: % de aciertos
        "prec": precision_score(y_te, y_pred),                  # Precision: de los que predijo enfermos, cuántos lo eran
        "rec": recall_score(y_te, y_pred),                      # Recall: de los enfermos reales, cuántos detectó
        "f1": f1_score(y_te, y_pred),                           # F1: balance entre precision y recall
        "auc": roc_auc_score(y_te, y_proba),                    # AUC-ROC: capacidad de separar las dos clases
        "cm": confusion_matrix(y_te, y_pred),                   # Matriz de confusión
    }

M = entrenar_modelo(df)                                         # Entrena (o recupera de caché) el modelo
model, scaler = M["model"], M["scaler"]                         # Atajos al modelo y al escalador

# =====================================================================
# 3. BARRA LATERAL
# =====================================================================
with st.sidebar:                                                # Todo lo de este bloque va en la barra lateral
    st.title("🩺 CardioML")                                     # Nombre de la aplicación
    st.caption("Aprendizaje Automático · CUN")                  # Texto pequeño con la asignatura
    st.markdown("**Modelo:** Regresión Logística")
    st.markdown("**Dataset:** UCI Heart Disease (Cleveland)")
    st.divider()                                                # Línea separadora
    st.metric("Accuracy del modelo", f"{M['acc']:.1%}")         # Tarjeta con el accuracy
    st.metric("AUC-ROC", f"{M['auc']:.3f}")                     # Tarjeta con el AUC
    st.divider()
    st.caption("⚠️ Herramienta académica. No reemplaza el criterio médico.")  # Aviso de uso

st.title("🩺 Sistema de Apoyo al Diagnóstico de Enfermedad Cardíaca")  # Título principal de la página

# Crea las 7 pestañas principales y las asigna a una variable cada una
tabs = st.tabs(["🏠 Proyecto", "🔍 EDA", "📏 Métricas", "🔮 Predicción",
                "📊 Resultados", "ℹ️ Info del modelo", "📋 Datos"])
t_proy, t_eda, t_met, t_pred, t_res, t_info, t_datos = tabs

# =====================================================================
# 4. DESCRIPCIÓN DEL PROYECTO
# =====================================================================
with t_proy:
    st.subheader("Descripción del proyecto")
    c1, c2, c3 = st.columns(3)                                  # Tres columnas: problema, dataset, algoritmo
    with c1:
        st.markdown("### 🎯 Problema")
        st.write("Las enfermedades cardiovasculares son una de las principales causas de muerte "
                 "en el mundo. Detectarlas a tiempo a partir de datos clínicos simples puede "
                 "apoyar la decisión médica y la priorización de pacientes.")
    with c2:
        st.markdown("### 🗂️ Dataset")
        # Usa f-string para insertar el número real de pacientes y de variables
        st.write(f"**UCI Heart Disease (Cleveland)**: {len(df)} pacientes y "
                 f"{df_raw.shape[1]} variables clínicas (edad, colesterol, presión, "
                 "frecuencia cardíaca, tipo de dolor, etc.). La variable objetivo indica "
                 "presencia o ausencia de enfermedad.")
    with c3:
        st.markdown("### 🤖 Algoritmo")
        st.write("**Regresión Logística**: modelo clásico de clasificación binaria, "
                 "interpretable y entrega una probabilidad de riesgo. Se aplicó "
                 "estandarización de variables.")
    st.divider()
    st.markdown("### 🧭 ¿Cómo usar la app?")
    p1, p2, p3, p4 = st.columns(4)                              # Cuatro cuadros con los pasos de uso
    p1.info("**1. Explora** los datos en la pestaña EDA")
    p2.info("**2. Evalúa** el modelo en Métricas")
    p3.info("**3. Predice** con datos de un paciente")
    p4.info("**4. Interpreta** el resultado en Resultados")

# =====================================================================
# 5. EDA
# =====================================================================
with t_eda:
    st.subheader("Análisis exploratorio de datos")
    k1, k2, k3, k4 = st.columns(4)                              # Cuatro indicadores clave (KPIs)
    k1.metric("Pacientes", len(df))                             # Total de pacientes
    k2.metric("Con enfermedad", f"{df['target_bin'].mean()*100:.1f}%")  # % de pacientes enfermos
    k3.metric("Edad promedio", f"{df['age'].mean():.1f} años")  # Edad media
    k4.metric("Colesterol promedio", f"{df['chol'].mean():.1f} mg/dl")  # Colesterol medio
    st.divider()

    # Sub-pestañas dentro del EDA
    sub1, sub2, sub3, sub4 = st.tabs(["📈 Distribución", "❓ Valores faltantes",
                                       "🔗 Correlaciones", "🧮 Estadísticas"])
    with sub1:
        cols_num = ["age", "trestbps", "chol", "thalach", "oldpeak"]  # Variables numéricas a explorar
        a, b = st.columns([1, 3])                               # Columna de controles (angosta) y de gráfica (ancha)
        with a:
            # Lista desplegable para elegir la variable; format_func muestra el nombre en español
            var = st.selectbox("Variable numérica", cols_num, format_func=lambda x: f"{NOMBRES[x]} ({x})")
            sep = st.checkbox("Separar por diagnóstico", value=True)  # Casilla: colorear por diagnóstico
            bins = st.slider("Número de bins", 5, 50, 20)       # Deslizador: cantidad de barras del histograma
        with b:
            fig, ax = plt.subplots(figsize=(7, 3.8))            # Crea la figura
            if sep:
                # Histograma separado por diagnóstico, con curva de densidad (kde)
                sns.histplot(data=df, x=var, hue="enfermo_label", bins=bins, kde=True,
                             palette=[COLOR_SANO, COLOR_ENFERMO], ax=ax)
            else:
                # Histograma general de la variable
                sns.histplot(data=df, x=var, bins=bins, kde=True, color=COLOR_SANO, ax=ax)
            ax.set_xlabel(NOMBRES[var])                         # Etiqueta del eje X en español
            st.pyplot(fig)                                      # Muestra la gráfica en la app
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Diagnóstico por sexo**")
            fig, ax = plt.subplots(figsize=(5, 3.2))
            # Conteo de pacientes por sexo, separado por diagnóstico
            sns.countplot(data=df, x="sexo_label", hue="enfermo_label",
                          palette=[COLOR_SANO, COLOR_ENFERMO], ax=ax)
            ax.set_xlabel("")
            st.pyplot(fig)
        with c2:
            st.markdown("**Diagnóstico por tipo de dolor de pecho**")
            fig, ax = plt.subplots(figsize=(5, 3.2))
            # Conteo de pacientes por tipo de dolor de pecho (cp), separado por diagnóstico
            sns.countplot(data=df, x="cp", hue="enfermo_label",
                          palette=[COLOR_SANO, COLOR_ENFERMO], ax=ax)
            st.pyplot(fig)
    with sub2:
        faltantes = df_raw.isna().sum()                         # Cuenta nulos por columna en los datos originales
        # Tabla con cantidad y porcentaje de faltantes
        tabla = pd.DataFrame({"Faltantes": faltantes,
                              "% del total": (faltantes / len(df_raw) * 100).round(2)})
        tabla = tabla[tabla["Faltantes"] > 0]                   # Deja solo las columnas con faltantes
        if tabla.empty:
            st.success("El dataset no tiene valores faltantes.")
        else:
            c1, c2 = st.columns(2)
            c1.markdown("**Antes de la limpieza**")
            c1.dataframe(tabla, use_container_width=True)       # Muestra la tabla de faltantes
            c2.markdown("**Tratamiento aplicado**")
            # Caja informativa con el tratamiento (usa la clase CSS .caja-info definida arriba)
            c2.markdown('<div class="caja-info">Imputación con la <b>moda</b> en las variables '
                        '<code>ca</code> y <code>thal</code>, por ser variables categóricas/discretas.</div>',
                        unsafe_allow_html=True)
            # Verifica que tras la limpieza no queden nulos
            c2.metric("Faltantes después de limpiar", int(df[FEATURES + ["thal"]].isna().sum().sum()))
    with sub3:
        # Se toman solo las columnas numéricas (sin la objetivo binaria) para la correlación
        num = df.select_dtypes(include=np.number).drop(columns=["target_bin"], errors="ignore")
        fig, ax = plt.subplots(figsize=(9, 5.5))
        # Mapa de calor de correlaciones (rojo = positiva, azul = negativa)
        sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        st.pyplot(fig)
        # Correlación de cada variable con 'target', ordenada de mayor a menor valor absoluto
        corr_t = num.corr()["target"].drop("target").sort_values(key=abs, ascending=False)
        st.markdown("**Variables más correlacionadas con el diagnóstico**")
        st.bar_chart(corr_t)                                    # Gráfica de barras de esas correlaciones
    with sub4:
        # Tabla de estadísticas descriptivas (transpuesta con .T para leerla mejor)
        st.dataframe(df[["age", "trestbps", "chol", "thalach", "oldpeak"]].describe().round(2).T,
                     use_container_width=True)

# =====================================================================
# 6. MÉTRICAS DEL MODELO (clasificación)
# =====================================================================
with t_met:
    st.subheader("Métricas de desempeño (conjunto de prueba)")
    m1, m2, m3, m4, m5 = st.columns(5)                          # Una tarjeta por métrica
    m1.metric("Accuracy", f"{M['acc']:.2%}")
    m2.metric("Precision", f"{M['prec']:.2%}")
    m3.metric("Recall", f"{M['rec']:.2%}")
    m4.metric("F1 Score", f"{M['f1']:.2%}")
    m5.metric("AUC-ROC", f"{M['auc']:.3f}")
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Matriz de confusión")
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        # Matriz de confusión como mapa de calor, con los conteos dentro de cada celda
        sns.heatmap(M["cm"], annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Sin enf.", "Con enf."], yticklabels=["Sin enf.", "Con enf."])
        ax.set_xlabel("Predicho"); ax.set_ylabel("Real")        # Ejes: lo predicho vs. lo real
        st.pyplot(fig)
    with c2:
        st.markdown("#### Curva ROC")
        fpr, tpr, _ = roc_curve(M["y_test"], M["y_proba"])      # Puntos de la curva ROC
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        ax.plot(fpr, tpr, color=COLOR_ENFERMO, lw=2, label=f"AUC = {M['auc']:.3f}")  # Curva del modelo
        ax.plot([0, 1], [0, 1], "--", color="gray")             # Línea de referencia (azar)
        ax.set_xlabel("Tasa de falsos positivos"); ax.set_ylabel("Tasa de verdaderos positivos")
        ax.legend()
        st.pyplot(fig)
    # Desempaqueta la matriz: verdaderos negativos, falsos positivos, falsos negativos, verdaderos positivos
    tn, fp, fn, tp = M["cm"].ravel()
    st.markdown("#### 🧠 Interpretación")
    # Texto automático que interpreta los resultados con los números reales del modelo
    st.info(f"De {M['n_test']} pacientes de prueba, el modelo detectó correctamente **{tp}** con "
            f"enfermedad (verdaderos positivos) y **{tn}** sin enfermedad. Hubo **{fn}** casos "
            f"enfermos no detectados (falsos negativos), el error más crítico en salud, y "
            f"**{fp}** falsas alarmas. Validación cruzada (5 folds): "
            f"**{M['cv'].mean():.2%} ± {M['cv'].std():.2%}**.")

# =====================================================================
# 7. PREDICCIÓN INTERACTIVA
# =====================================================================
with t_pred:
    st.subheader("🔮 Evaluación de un nuevo paciente")
    # Deslizador del umbral: a partir de qué probabilidad se considera "alto riesgo"
    umbral = st.slider("Umbral de decisión para clasificar como 'alto riesgo'", 0.10, 0.90, 0.50, 0.05,
                       help="Bajar el umbral detecta más casos enfermos, pero genera más falsas alarmas.")
    # Formulario: los datos solo se procesan al presionar el botón
    with st.form("form_paciente"):
        c1, c2, c3 = st.columns(3)
        with c1:
            i_age = st.slider("Edad (años)", 20, 100, 55)                          # Edad
            i_trestbps = st.slider("Presión arterial en reposo (mm Hg)", 80, 220, 130)  # Presión arterial
            # Tipo de dolor de pecho: se guarda el número (1 a 4) pero se muestra el texto
            i_cp = st.selectbox("Tipo de dolor de pecho", [1, 2, 3, 4],
                                format_func=lambda x: {1: "1: Angina típica", 2: "2: Angina atípica",
                                                       3: "3: Dolor no anginoso", 4: "4: Asintomático"}[x])
        with c2:
            i_chol = st.slider("Colesterol sérico (mg/dl)", 100, 600, 240)         # Colesterol
            i_thalach = st.slider("Frecuencia cardíaca máxima (bpm)", 60, 220, 150)  # Frecuencia cardíaca máxima
            # Angina por ejercicio: 0 = No, 1 = Sí
            i_exang = st.selectbox("Angina inducida por ejercicio", [0, 1],
                                   format_func=lambda x: "Sí" if x == 1 else "No")
        with c3:
            i_oldpeak = st.slider("Depresión del ST (oldpeak)", 0.0, 10.0, 1.0, 0.1)  # Depresión del segmento ST
            i_ca = st.selectbox("Vasos coloreados por fluoroscopia (ca)", [0, 1, 2, 3])  # Vasos principales afectados
        # Botón de envío del formulario
        enviar = st.form_submit_button("Calcular riesgo diagnóstico", type="primary")

    if enviar:  # Solo se ejecuta cuando el usuario presiona el botón
        # Arma una tabla de una fila con los datos del paciente, en el mismo orden que FEATURES
        paciente = pd.DataFrame([[i_age, i_trestbps, i_chol, i_thalach, i_oldpeak, i_ca, i_cp, i_exang]],
                                columns=FEATURES)
        p_s = scaler.transform(paciente)                        # Estandariza con el mismo escalador del entrenamiento
        prob = model.predict_proba(p_s)[0][1]                   # Probabilidad de enfermedad para este paciente
        # Guarda todo en session_state para usarlo en la pestaña Resultados
        st.session_state["paciente"] = paciente
        st.session_state["paciente_s"] = p_s
        st.session_state["prob"] = prob
        st.session_state["umbral"] = umbral

    if "prob" in st.session_state:  # Si ya hay una predicción calculada, la muestra
        prob, um = st.session_state["prob"], st.session_state["umbral"]
        st.divider()
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.metric("Probabilidad de enfermedad", f"{prob:.1%}")  # Probabilidad en porcentaje
        with col_b:
            if prob >= um:                                      # Si supera el umbral: alto riesgo
                st.error(f"⚠️ **ALTO RIESGO** de enfermedad cardíaca (umbral {um:.0%}).")
            else:                                               # Si no: bajo riesgo
                st.success(f"🟢 **BAJO RIESGO** según el modelo (umbral {um:.0%}).")
            st.progress(float(prob))                            # Barra de progreso con la probabilidad
        st.caption("Revisa la pestaña **📊 Resultados** para ver qué factores influyeron.")

# =====================================================================
# 8. VISUALIZACIÓN DE RESULTADOS
# =====================================================================
with t_res:
    st.subheader("Visualización de resultados")
    st.markdown("#### 🌍 Importancia global de variables")
    # Coeficientes del modelo: indican cuánto empuja cada variable hacia "enfermo" (+) o "sano" (-)
    coefs = pd.Series(model.coef_[0], index=[NOMBRES[f] for f in FEATURES]).sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    # Barras horizontales: naranja si el coeficiente es positivo, azul si es negativo
    ax.barh(coefs.index, coefs.values,
            color=[COLOR_ENFERMO if v > 0 else COLOR_SANO for v in coefs.values])
    ax.axvline(0, color="gray", lw=1)                           # Línea vertical en cero
    ax.set_xlabel("Coeficiente (variables estandarizadas)")
    st.pyplot(fig)
    st.caption("Naranja: aumenta el riesgo · Azul: lo disminuye.")
    st.divider()
    st.markdown("#### 👤 ¿Por qué este resultado para el paciente evaluado?")
    if "prob" not in st.session_state:                          # Si aún no se ha hecho una predicción
        st.info("Primero calcula una predicción en la pestaña **🔮 Predicción**.")
    else:
        prob = st.session_state["prob"]
        # Contribución de cada variable = valor estandarizado del paciente × coeficiente
        contrib = pd.Series(st.session_state["paciente_s"][0] * model.coef_[0],
                            index=[NOMBRES[f] for f in FEATURES]).sort_values()
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 2.6))
            # Barras con la probabilidad de cada clase
            ax.barh(["Sin enfermedad", "Con enfermedad"], [1 - prob, prob],
                    color=[COLOR_SANO, COLOR_ENFERMO])
            ax.set_xlim(0, 1)
            for i, v in enumerate([1 - prob, prob]):            # Escribe el porcentaje junto a cada barra
                ax.text(min(v + 0.02, 0.85), i, f"{v:.1%}", va="center")
            ax.set_title("Probabilidad de predicción")
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(5, 3.4))
            # Barras con lo que aportó cada variable al resultado de este paciente
            ax.barh(contrib.index, contrib.values,
                    color=[COLOR_ENFERMO if v > 0 else COLOR_SANO for v in contrib.values])
            ax.axvline(0, color="gray", lw=1)
            ax.set_title("Contribución de cada variable")
            st.pyplot(fig)
        top = contrib.abs().sort_values(ascending=False).index[0]  # Variable con mayor peso en este caso
        st.info(f"La variable que más pesó en este caso fue **{top}**.")
        st.markdown("**Paciente vs. promedio del dataset**")
        # Tabla que compara los valores del paciente con el promedio del dataset
        comp = pd.DataFrame({"Paciente": st.session_state["paciente"].iloc[0],
                             "Promedio dataset": df[FEATURES].mean().round(1)})
        comp.index = [NOMBRES[i] for i in comp.index]           # Nombres en español para las filas
        st.dataframe(comp, use_container_width=True)

# =====================================================================
# 9. INFORMACIÓN DEL MODELO
# =====================================================================
with t_info:
    st.subheader("Ficha técnica del modelo")
    c1, c2 = st.columns(2)
    with c1:
        # Ficha técnica con datos reales del entrenamiento (versión de sklearn, variables, partición, validación cruzada)
        st.markdown(f"""
- **Algoritmo:** Regresión Logística (scikit-learn {sklearn.__version__})
- **Tarea:** Clasificación binaria (0 = sin enfermedad, 1 = con enfermedad)
- **Variables de entrada:** {', '.join(FEATURES)}
- **Preprocesamiento:** imputación con moda (`ca`, `thal`) y estandarización (`StandardScaler`)
- **Partición:** {M['n_train']} entrenamiento / {M['n_test']} prueba (80/20, estratificada, `random_state=42`)
- **Validación cruzada (5 folds):** {M['cv'].mean():.2%}
""")
    with c2:
        st.markdown("**Hiperparámetros**")
        st.json({k: str(v) for k, v in model.get_params().items()})  # Muestra los hiperparámetros del modelo
    # Aviso de limitaciones del modelo
    st.warning("Limitaciones: dataset pequeño (un solo centro hospitalario) y pocas variables. "
               "El modelo es un apoyo académico y no sustituye el diagnóstico clínico.")
    # Cita de la fuente de datos
    st.caption("Fuente de datos: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). "
               "UCI Heart Disease Dataset. UCI Machine Learning Repository.")

# =====================================================================
# 10. DATOS
# =====================================================================
with t_datos:
    st.subheader("Base de datos clínica")
    # Selector múltiple para filtrar por diagnóstico (por defecto muestra ambos grupos)
    filtro = st.multiselect("Filtrar por diagnóstico", df["enfermo_label"].unique(),
                            default=list(df["enfermo_label"].unique()))
    vista = df[df["enfermo_label"].isin(filtro)]                # Aplica el filtro a la tabla
    st.dataframe(vista, use_container_width=True)               # Muestra la tabla filtrada
    # Botón para descargar la tabla filtrada como CSV
    st.download_button("⬇️ Descargar datos filtrados (CSV)", vista.to_csv(index=False).encode("utf-8"),
                       "heart_disease_filtrado.csv", "text/csv")
