import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Génération données synthétiques
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=200)
temperature = 20 + 10 * np.sin(np.linspace(0, 3*np.pi, 200)) + np.random.normal(0, 2, 200)
humidity = 50 + 20 * np.cos(np.linspace(0, 3*np.pi, 200)) + np.random.normal(0, 5, 200)
day_of_week = np.array([d.weekday() for d in dates])
consumption = 100 + 2*temperature - 0.5*humidity + 3*day_of_week + np.random.normal(0, 10, 200)
data = pd.DataFrame({
    'Date': dates,
    'Temperature': temperature,
    'Humidity': humidity,
    'DayOfWeek': day_of_week,
    'Consumption': consumption
})

root = tk.Tk()
root.title("Projet IA - Prédiction Consommation Énergie")
root.geometry("900x700")
root.configure(bg='#2e2e2e')

# Style ttk
style = ttk.Style(root)
style.theme_use('clam')

# Couleurs générales
bg_color = '#2e2e2e'
fg_color = '#e0e0e0'
btn_color = '#4a90e2'
btn_hover_color = '#357ABD'
txt_bg_color = '#1e1e1e'
txt_fg_color = '#dcdcdc'

# Style frame
style.configure('TFrame', background=bg_color)

# Style labels
style.configure('TLabel', background=bg_color, foreground=fg_color, font=('Segoe UI', 12))
style.configure('Welcome.TLabel', font=('Segoe UI', 18, 'bold'))

# Style buttons
style.configure('TButton', background=btn_color, foreground='white', font=('Segoe UI', 11, 'bold'), padding=6)
style.map('TButton', background=[('active', btn_hover_color)])

# Frame accueil
frame_welcome = ttk.Frame(root)
frame_welcome.pack(fill='both', expand=True)

welcome_label = ttk.Label(frame_welcome, text="Bienvenue dans le projet IA Consommation Énergie", style='Welcome.TLabel')
welcome_label.pack(pady=50)

welcome_desc = ttk.Label(frame_welcome, text="Cliquez sur 'Commencer' pour accéder au menu des algorithmes")
welcome_desc.pack(pady=20)

def show_main():
    frame_welcome.pack_forget()
    frame_main.pack(fill='both', expand=True)

start_button = ttk.Button(frame_welcome, text="Commencer", command=show_main)
start_button.pack(pady=20)

# Frame principal (menu et résultats)
frame_main = ttk.Frame(root)

txt = ScrolledText(frame_main, width=100, height=10, bg=txt_bg_color, fg=txt_fg_color, insertbackground=fg_color)
txt.pack(pady=10)

# Signature créateur
signature = ttk.Label(frame_main, text="Créé par Ayoub Mourfik", font=('Segoe UI', 9, 'italic'), foreground='#a0a0a0', background=bg_color)
signature.pack(pady=(0,10))

fig, ax = plt.subplots(figsize=(7,4))
fig.patch.set_facecolor(bg_color)
ax.set_facecolor('#3e3e3e')
ax.tick_params(colors=fg_color)
ax.spines['bottom'].set_color(fg_color)
ax.spines['left'].set_color(fg_color)
ax.title.set_color(fg_color)
ax.xaxis.label.set_color(fg_color)
ax.yaxis.label.set_color(fg_color)

canvas = FigureCanvasTkAgg(fig, master=frame_main)
canvas.get_tk_widget().pack()

def clear_display():
    txt.delete('1.0', tk.END)
    ax.clear()
    ax.set_facecolor('#3e3e3e')
    ax.tick_params(colors=fg_color)
    ax.spines['bottom'].set_color(fg_color)
    ax.spines['left'].set_color(fg_color)
    ax.title.set_color(fg_color)
    ax.xaxis.label.set_color(fg_color)
    ax.yaxis.label.set_color(fg_color)

def random_forest():
    clear_display()
    X = data[['Temperature', 'Humidity', 'DayOfWeek']]
    y = data['Consumption']
    split = int(0.7 * len(data))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    txt.insert(tk.END, f"Random Forest - MSE sur test : {mse:.2f}\n\n")
    ax.plot(y_test.values, label='Réel')
    ax.plot(preds, label='Prédit')
    ax.set_title("Random Forest : Consommation réelle vs prédite")
    ax.legend()
    canvas.draw()

def clustering():
    clear_display()
    X = data[['Temperature', 'Humidity', 'Consumption']]
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X)
    data['Cluster'] = clusters
    txt.insert(tk.END, f"Clustering KMeans - Nombre de points par cluster :\n{data['Cluster'].value_counts()}\n\n")
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    for c in range(3):
        cluster_data = data[data['Cluster']==c]
        ax.scatter(cluster_data['Temperature'], cluster_data['Consumption'], 
                   label=f'Cluster {c}', alpha=0.6, color=colors[c])
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Consumption')
    ax.set_title("Clustering KMeans (3 clusters)")
    ax.legend()
    canvas.draw()

def regression_lineaire():
    clear_display()
    X = data[['Temperature', 'Humidity', 'DayOfWeek']]
    y = data['Consumption']
    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    txt.insert(tk.END, f"Régression Linéaire - MSE sur toutes les données : {mse:.2f}\n\n")
    ax.scatter(data['Date'], y, label='Réel', alpha=0.6)
    ax.plot(data['Date'], preds, color='#e74c3c', label='Prédit')
    ax.set_title("Régression Linéaire : Consommation réelle vs prédite")
    ax.legend()
    canvas.draw()

def serie_temporelle_arima():
    clear_display()
    ts = data.set_index('Date')['Consumption']
    model = ARIMA(ts[:150], order=(2,1,2))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=50)
    txt.insert(tk.END, "ARIMA - Prévision consommation 50 jours suivants\n\n")
    ax.plot(ts, label='Données réelles')
    ax.plot(forecast.index, forecast, color='#e67e22', label='Prévision ARIMA')
    ax.set_title("Série Temporelle ARIMA")
    ax.legend()
    canvas.draw()

def validation_croisee():
    clear_display()
    X = data[['Temperature', 'Humidity', 'DayOfWeek']]
    y = data['Consumption']
    model = LinearRegression()
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    mse_scores = -scores
    txt.insert(tk.END, f"Validation croisée (5 folds) - MSE par fold :\n{mse_scores}\n")
    txt.insert(tk.END, f"MSE moyen : {mse_scores.mean():.2f}\n")

# Boutons menu principal
frame_buttons = ttk.Frame(frame_main)
frame_buttons.pack(pady=10)

ttk.Button(frame_buttons, text="Random Forest", command=random_forest).grid(row=0, column=0, padx=10, pady=5)
ttk.Button(frame_buttons, text="Clustering KMeans", command=clustering).grid(row=0, column=1, padx=10, pady=5)
ttk.Button(frame_buttons, text="Régression Linéaire", command=regression_lineaire).grid(row=0, column=2, padx=10, pady=5)
ttk.Button(frame_buttons, text="Série Temporelle ARIMA", command=serie_temporelle_arima).grid(row=0, column=3, padx=10, pady=5)
ttk.Button(frame_buttons, text="Validation Croisée", command=validation_croisee).grid(row=0, column=4, padx=10, pady=5)

root.mainloop()
