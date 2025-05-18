import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.dates import DateFormatter

# ======================================================================
# STYLE ÉNERGIE (BLANC/VERT)
# ======================================================================
BG_COLOR = "#ffffff"  # Blanc
FG_COLOR = "#333333"  # Gris foncé
PRIMARY_COLOR = "#2ecc71"  # Vert émeraude
SECONDARY_COLOR = "#3498db"  # Bleu pour complément
TEXT_BG = "#f9f9f9"  # Gris très clair
TEXT_FG = "#333333"  # Gris foncé
PLOT_BG = "#ffffff"  # Fond blanc pour les graphiques

# ======================================================================
# GÉNÉRATION DE DONNÉES SYNTHÉTIQUES
# ======================================================================
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=200)
temperature = 20 + 10 * np.sin(np.linspace(0, 3*np.pi, 200)) + np.random.normal(0, 2, 200)
humidity = 50 + 20 * np.cos(np.linspace(0, 3*np.pi, 200)) + np.random.normal(0, 5, 200)
day_of_week = np.array([d.weekday() for d in dates])
is_weekend = np.array([1 if d >= 5 else 0 for d in day_of_week])
consumption = 100 + 2*temperature - 0.5*humidity + 10*is_weekend + np.random.normal(0, 5, 200)

data = pd.DataFrame({
    'Date': dates,
    'Temperature': temperature,
    'Humidity': humidity,
    'DayOfWeek': day_of_week,
    'IsWeekend': is_weekend,
    'Consumption': consumption
})

# ======================================================================
# INTERFACE GRAPHIQUE
# ======================================================================
root = tk.Tk()
root.title("Energy Consumption Analyzer PRO")
root.geometry("1200x900")
root.configure(bg=BG_COLOR)

style = ttk.Style(root)
style.theme_use("clam")

# Configuration du style étendu
style.configure("TFrame", background=BG_COLOR)
style.configure("TLabel", background=BG_COLOR, foreground=FG_COLOR, font=("Helvetica", 11))
style.configure("Title.TLabel", font=("Helvetica", 24, "bold"), foreground=PRIMARY_COLOR)
style.configure("Subtitle.TLabel", font=("Helvetica", 14), foreground=SECONDARY_COLOR)
style.configure("TButton", font=("Helvetica", 10, "bold"), padding=8, 
                background=PRIMARY_COLOR, foreground="white")
style.configure("Separator.TSeparator", background=SECONDARY_COLOR)
style.map("TButton", 
          background=[("active", "#27ae60")],
          foreground=[("active", "white")])

# Frame d'accueil
frame_welcome = ttk.Frame(root, style="TFrame")
frame_welcome.pack(fill="both", expand=True)

# Éléments d'accueil améliorés
ttk.Label(frame_welcome, text="⚡", font=("Helvetica", 60)).pack(pady=(30, 10))
ttk.Label(frame_welcome, text="ENERGY CONSUMPTION ANALYZER PRO", style="Title.TLabel").pack(pady=5)
ttk.Label(frame_welcome, text="Analyse avancée de la consommation énergétique", style="Subtitle.TLabel").pack(pady=10)

# Features avec icônes améliorées
features_frame = ttk.Frame(frame_welcome)
features_frame.pack(pady=30)

features = [
    ("🌡️", "Analyse température/consommation"),
    ("📈", "Modèles prédictifs avancés"),
    ("🔍", "Clustering des profils de consommation"),
    ("🔄", "Validation croisée robuste"),
    ("📊", "Visualisations interactives")
]

for icon, text in features:
    feature_item = ttk.Frame(features_frame)
    feature_item.pack(side="left", padx=15)
    ttk.Label(feature_item, text=icon, font=("Helvetica", 16)).pack()
    ttk.Label(feature_item, text=text, font=("Helvetica", 11), wraplength=120, justify="center").pack()

# Frame principal
frame_main = ttk.Frame(root)

# Zone de texte avec style amélioré
txt = ScrolledText(frame_main, width=110, height=14, bg=TEXT_BG, fg=TEXT_FG, 
                   font=("Consolas", 10), wrap="word", borderwidth=2, relief="groove")
txt.pack(pady=15, padx=20, fill="both", expand=True)

# Configuration avancée du graphique
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor(PLOT_BG)
ax.set_facecolor(PLOT_BG)
ax.tick_params(colors=FG_COLOR)
for spine in ax.spines.values():
    spine.set_color(FG_COLOR)
ax.title.set_color(PRIMARY_COLOR)
ax.xaxis.label.set_color(FG_COLOR)
ax.yaxis.label.set_color(FG_COLOR)
ax.grid(color='#e0e0e0', linestyle='--')

canvas = FigureCanvasTkAgg(fig, master=frame_main)
canvas.get_tk_widget().pack(pady=10, padx=20, fill="both", expand=True)

# Signature améliorée
ttk.Label(frame_main, 
         text="Créé par Ayoub Mourfik • © 2025 • Version Pro",
         font=("Helvetica", 9, "italic"),
         foreground="#666666",
         background=BG_COLOR).pack(pady=(0, 15))

# ======================================================================
# FONCTIONS D'ANALYSE AMÉLIORÉES
# ======================================================================
def clear_display():
    txt.delete("1.0", tk.END)
    ax.clear()
    ax.set_facecolor(PLOT_BG)
    ax.tick_params(colors=FG_COLOR)
    for spine in ax.spines.values():
        spine.set_color(FG_COLOR)
    ax.grid(color='#e0e0e0', linestyle='--')

def random_forest():
    clear_display()
    X = data[["Temperature", "Humidity", "IsWeekend"]]
    y = data["Consumption"]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    
    txt.insert(tk.END, "=== RANDOM FOREST REGRESSOR ===\n", "title")
    txt.insert(tk.END, f"Score R² moyen (validation croisée): {scores.mean():.3f} ± {scores.std():.3f}\n\n")
    model.fit(X, y)
    txt.insert(tk.END, "Importance des caractéristiques :\n", "subtitle")
    features_importance = sorted(zip(X.columns, model.feature_importances_), 
                              key=lambda x: x[1], reverse=True)
    for name, imp in features_importance:
        txt.insert(tk.END, f"- {name}: {imp:.3f}\n")
    
    preds = model.predict(X)
    ax.scatter(y, preds, alpha=0.6, color=PRIMARY_COLOR, s=60)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "--", color=SECONDARY_COLOR, linewidth=2)
    ax.set_xlabel("Consommation Réelle (kWh)", fontweight="bold")
    ax.set_ylabel("Prédictions (kWh)", fontweight="bold")
    ax.set_title("Performance du Random Forest", pad=20, fontweight="bold", fontsize=12)
    canvas.draw()

def regression_lineaire():
    clear_display()
    X = data[["Temperature", "Humidity", "IsWeekend"]]
    y = data["Consumption"]
    
    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)
    
    txt.insert(tk.END, "=== RÉGRESSION LINÉAIRE MULTIVARIÉE ===\n", "title")
    txt.insert(tk.END, f"Score R² : {r2_score(y, preds):.3f}\n\n")
    txt.insert(tk.END, "Coefficients standardisés :\n", "subtitle")
    txt.insert(tk.END, f"- Intercept : {model.intercept_:.3f}\n")
    
    # Standardisation des coefficients pour comparaison
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_,
        'Abs(Coefficient)': np.abs(model.coef_)
    }).sort_values('Abs(Coefficient)', ascending=False)
    
    for _, row in coef_df.iterrows():
        txt.insert(tk.END, f"- {row['Feature']}: {row['Coefficient']:.3f}\n")
    
    # Graphique amélioré
    ax.scatter(data["Temperature"], y, alpha=0.7, color=PRIMARY_COLOR, 
              s=60, label="Données réelles")
    sorted_idx = np.argsort(data["Temperature"])
    ax.plot(data["Temperature"].iloc[sorted_idx], preds[sorted_idx], 
            color=SECONDARY_COLOR, linewidth=2.5, label="Prédictions")
    ax.set_xlabel("Température (°C)", fontweight="bold")
    ax.set_ylabel("Consommation (kWh)", fontweight="bold")
    ax.set_title("Régression Linéaire: Température vs Consommation", 
                pad=20, fontweight="bold", fontsize=12)
    ax.legend(framealpha=0.9)
    canvas.draw()

def clustering():
    clear_display()
    X = data[["Temperature", "Consumption"]]
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    txt.insert(tk.END, "=== K-MEANS CLUSTERING ===\n", "title")
    txt.insert(tk.END, "Centres des clusters :\n", "subtitle")
    centers_df = pd.DataFrame(kmeans.cluster_centers_, 
                            columns=["Temperature", "Consumption"],
                            index=[f"Cluster {i}" for i in range(3)])
    txt.insert(tk.END, f"{centers_df.round(2)}\n\n")
    
    txt.insert(tk.END, "Répartition des clusters :\n", "subtitle")
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    for i, count in cluster_counts.items():
        txt.insert(tk.END, f"- Cluster {i}: {count} points ({count/len(clusters):.1%})\n")
    
    # Graphique amélioré
    colors = [PRIMARY_COLOR, SECONDARY_COLOR, "#e74c3c"]
    cluster_names = ["Bas", "Moyen", "Haut"]
    for i in range(3):
        cluster_data = data[clusters == i]
        ax.scatter(cluster_data["Temperature"], cluster_data["Consumption"], 
                  color=colors[i], s=70, label=f"{cluster_names[i]} consommation", 
                  alpha=0.8, edgecolor='white')
    
    # Affichage des centres
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
              s=250, marker='*', c='gold', edgecolor='black', 
              label='Centres des clusters')
    
    ax.set_xlabel("Température (°C)", fontweight="bold")
    ax.set_ylabel("Consommation (kWh)", fontweight="bold")
    ax.set_title("Clustering des Profils de Consommation", 
                pad=20, fontweight="bold", fontsize=12)
    ax.legend(framealpha=0.9)
    canvas.draw()

def serie_temporelle_arima():
    clear_display()
    ts = data.set_index("Date")["Consumption"]
    train = ts[:150]
    
    model = ARIMA(train, order=(2,1,2))
    model_fit = model.fit()
    forecast = model_fit.get_forecast(steps=50)
    preds = forecast.predicted_mean
    conf_int = forecast.conf_int()
    
    txt.insert(tk.END, "=== MODÈLE ARIMA (2,1,2) ===\n", "title")
    txt.insert(tk.END, "Paramètres du modèle :\n", "subtitle")
    txt.insert(tk.END, f"{model_fit.params.round(3)}\n\n")
    
    txt.insert(tk.END, "Tests statistiques :\n", "subtitle")
    txt.insert(tk.END, f"AIC: {model_fit.aic:.1f} | BIC: {model_fit.bic:.1f}\n")
    
    # Graphique amélioré
    ax.plot(ts, color=PRIMARY_COLOR, linewidth=2, label="Données réelles")
    ax.plot(preds.index, preds, color=SECONDARY_COLOR, 
            linestyle='--', linewidth=2, label="Prévisions")
    ax.fill_between(conf_int.index, conf_int.iloc[:,0], conf_int.iloc[:,1], 
                   color=SECONDARY_COLOR, alpha=0.15, label="Intervalle de confiance")
    
    ax.set_title("Prévision de Consommation avec ARIMA", 
                pad=20, fontweight="bold", fontsize=12)
    ax.set_ylabel("Consommation (kWh)", fontweight="bold")
    ax.legend(framealpha=0.9)
    
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    fig.autofmt_xdate()
    canvas.draw()

def validation_croisee():
    clear_display()
    X = data[["Temperature", "Humidity", "IsWeekend"]]
    y = data["Consumption"]
    
    models = {
        "Régression Linéaire": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    txt.insert(tk.END, "=== VALIDATION CROISÉE AVANCÉE ===\n", "title")
    
    for name, model in models.items():
        txt.insert(tk.END, f"\n🔹 {name}\n", "subtitle")
        
        # Calcul des scores
        mse_scores = -cross_val_score(model, X, y, cv=5, 
                                    scoring="neg_mean_squared_error")
        r2_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
        
        # Affichage des résultats
        txt.insert(tk.END, f"MSE moyen: {mse_scores.mean():.2f} ± {mse_scores.std():.2f}\n")
        txt.insert(tk.END, f"R² moyen: {r2_scores.mean():.3f} ± {r2_scores.std():.3f}\n")
        
        # Détails par fold
        txt.insert(tk.END, "\nDétails par fold:\n")
        for i, (mse, r2) in enumerate(zip(mse_scores, r2_scores)):
            txt.insert(tk.END, f"Fold {i+1}: MSE={mse:.2f} | R²={r2:.3f}\n")
        
        txt.insert(tk.END, "\n" + "─"*50 + "\n")
    
    # Visualisation des résultats
    model_names = list(models.keys())
    avg_r2 = [cross_val_score(m, X, y, cv=5, scoring="r2").mean() for m in models.values()]
    
    ax.bar(model_names, avg_r2, color=[PRIMARY_COLOR, SECONDARY_COLOR], alpha=0.7)
    ax.set_ylabel("Score R² moyen", fontweight="bold")
    ax.set_title("Comparaison des Modèles par Validation Croisée", 
                pad=20, fontweight="bold", fontsize=12)
    ax.set_ylim(0, 1)
    
    # Ajout des valeurs sur les barres
    for i, v in enumerate(avg_r2):
        ax.text(i, v + 0.03, f"{v:.3f}", ha='center', fontweight='bold')
    
    canvas.draw()

def show_main():
    frame_welcome.pack_forget()
    frame_main.pack(fill="both", expand=True)

# Configuration des tags texte améliorés
txt.tag_configure("title", font=("Helvetica", 13, "bold"), 
                 foreground=PRIMARY_COLOR, spacing3=10)
txt.tag_configure("subtitle", font=("Helvetica", 11, "bold"), 
                 foreground=SECONDARY_COLOR, spacing2=5)

# Boutons principaux améliorés
frame_buttons = ttk.Frame(frame_main)
frame_buttons.pack(pady=15)

buttons = [
    ("🌳 Random Forest", random_forest),
    ("📉 Régression", regression_lineaire),
    ("🔮 Clustering", clustering),
    ("⏳ ARIMA", serie_temporelle_arima),
    ("✅ Validation", validation_croisee)
]

for i, (text, cmd) in enumerate(buttons):
    btn = ttk.Button(frame_buttons, text=text, command=cmd, style="TButton")
    btn.grid(row=0, column=i, padx=8, pady=5, ipadx=5, ipady=3)
    # Ajout d'un tooltip
    btn.bind("<Enter>", lambda e, t=text: txt.insert(tk.END, f"\n▶ {t.split()[1]}...\n"))
    btn.bind("<Leave>", lambda e: None)

# Bouton de démarrage amélioré
start_btn = ttk.Button(frame_welcome, 
                      text="🚀 DÉMARRER L'ANALYSE AVANCÉE", 
                      command=show_main,
                      style="TButton")
start_btn.pack(pady=40, ipadx=25, ipady=8)

root.mainloop()