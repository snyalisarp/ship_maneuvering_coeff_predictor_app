import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.signal import savgol_filter

# --- 1. ÖZELLİK MÜHENDİSLİĞİ SINIFI ---
class MMGFeatureEngineer:
    def __init__(self, mass, I_zz, x_G=0.0):
        self.m = mass
        self.I_zz = I_zz
        self.x_G = x_G

    def smooth_and_derive(self, data, window_length=11, polyorder=2):
        if len(data) < window_length:
            return np.gradient(data, edge_order=2)
        return savgol_filter(data, window_length, polyorder, deriv=1)

    def process_turn_zigzag(self, df):
        df = df.copy()
        df['dt'] = df['time'].diff().fillna(method='bfill')
        df['delta_rad'] = np.radians(df['rudder_angle'])
        df['psi_rad'] = np.radians(df['heading_angle'])
        df['r_rad'] = np.radians(df['yaw_rate']) 

        df['u_dot'] = self.smooth_and_derive(df['speed_u']) / df['dt']
        df['v_dot'] = self.smooth_and_derive(df['speed_v']) / df['dt']
        df['r_dot'] = self.smooth_and_derive(df['r_rad']) / df['dt']

        u, v, r, delta = df['speed_u'], df['speed_v'], df['r_rad'], df['delta_rad']
        
        df['u_u'], df['v_v'], df['r_r'], df['delta_delta'] = u*u, v*v, r*r, delta*delta
        df['v_abs_v'], df['r_abs_r'] = v*np.abs(v), r*np.abs(r)
        df['n_sq'] = (df['RPM'] / 60.0) ** 2 if 'RPM' in df.columns else 0

        df['X_meas'] = self.m * (df['u_dot'] - v * r - self.x_G * (r**2))
        df['Y_meas'] = self.m * (df['v_dot'] + u * r + self.x_G * df['r_dot'])
        df['N_meas'] = self.I_zz * df['r_dot'] + self.m * self.x_G * (df['v_dot'] + u * r)
        return df

# --- 2. MODELLER VE EĞİTİM FONKSİYONLARI ---

class MMG_PINN(nn.Module):
    def __init__(self):
        super(MMG_PINN, self).__init__()
        self.X_u = nn.Parameter(torch.tensor(-0.05))
        self.X_uu = nn.Parameter(torch.tensor(-0.01))
        self.X_delta = nn.Parameter(torch.tensor(-0.001)) 
        self.X_P = nn.Parameter(torch.tensor(0.01))
        self.Y_v = nn.Parameter(torch.tensor(-0.1)) 
        self.Y_r = nn.Parameter(torch.tensor(0.05))
        self.Y_delta = nn.Parameter(torch.tensor(0.01)) 
        self.N_v = nn.Parameter(torch.tensor(-0.05))
        self.N_r = nn.Parameter(torch.tensor(-0.05)) 
        self.N_rr = nn.Parameter(torch.tensor(-0.01))
        self.N_delta = nn.Parameter(torch.tensor(-0.01)) 
        self.N_P = nn.Parameter(torch.tensor(0.005))

    def forward(self, inputs):
        u, v, r, delta, n_sq, u_u, v_v, r_r, d_d, r_abs_r = [inputs[:, i] for i in range(10)]
        X_p = (self.X_u * u) + (self.X_uu * u_u) + (self.X_delta * d_d) + (self.X_P * n_sq)
        Y_p = (self.Y_v * v) + (self.Y_r * r) + (self.Y_delta * delta)
        N_p = (self.N_v * v) + (self.N_r * r) + (self.N_rr * r_abs_r) + (self.N_delta * delta) + (self.N_P * n_sq)
        return X_p, Y_p, N_p

def train_pinn(model, df, epochs=500, lr=0.01):
    feature_cols = ['speed_u', 'speed_v', 'r_rad', 'delta_rad', 'n_sq', 'u_u', 'v_v', 'r_r', 'delta_delta', 'r_abs_r']
    X_tensor = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    y_tensor = torch.tensor(df[['X_meas', 'Y_meas', 'N_meas']].values, dtype=torch.float32)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {"loss": [], "nr": [], "xu": []}
    for epoch in range(epochs):
        model.train(); optimizer.zero_grad()
        X_p, Y_p, N_p = model(X_tensor)
        loss_data = torch.mean((X_p - y_tensor[:, 0])**2) + torch.mean((Y_p - y_tensor[:, 1])**2) + torch.mean((N_p - y_tensor[:, 2])**2)
        loss_physics = torch.relu(model.X_u)*1000 + torch.relu(model.N_r)*1000 + torch.relu(model.Y_v)*100
        total_loss = loss_data + loss_physics
        total_loss.backward(); optimizer.step()
        history["loss"].append(total_loss.item()); history["nr"].append(model.N_r.item()); history["xu"].append(model.X_u.item())
    return model, history

def evaluate_pinn(model, df):
    model.eval()
    feature_cols = ['speed_u', 'speed_v', 'r_rad', 'delta_rad', 'n_sq', 'u_u', 'v_v', 'r_r', 'delta_delta', 'r_abs_r']
    with torch.no_grad():
        X_t = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        y_true = df[['X_meas', 'Y_meas', 'N_meas']].values
        X_p, Y_p, N_p = model(X_t)
        y_pred = torch.stack([X_p, Y_p, N_p], dim=1).numpy()
    rmses = [np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])) for i in range(3)]
    maes = [mean_absolute_error(y_true[:, i], y_pred[:, i]) for i in range(3)]
    return {"RMSE_X": rmses[0], "RMSE_Y": rmses[1], "RMSE_N": rmses[2], "Avg_RMSE": np.mean(rmses),
            "MAE_X": maes[0], "MAE_Y": maes[1], "MAE_N": maes[2], "Avg_MAE": np.mean(maes)}

# --- 4. STREAMLIT UYGULAMASI ---
st.set_page_config(page_title="Ship Maneuvering AI", layout="wide")
st.title("🚢 MMG Gemi Manevra Katsayı Tahmin Sistemi")

with st.sidebar:
    st.header("⚙️ Ayarlar")
    MASS = st.number_input("Kütle (kg)", value=356.2)
    I_ZZ = st.number_input("I_zz (kg m^2)", value=0.36323)
    X_G = st.number_input("x_G (m)", value=0.0)
    svr_c = st.slider("SVR Düzenlileştirme (C)", 1, 1000, 100)
    p_epochs = st.number_input("PINN Epochs", value=1000, min_value=10, max_value=1000)
    p_lr = st.selectbox("PINN LR", [0.001, 0.01, 0.1], index=1)

        

up_file = st.file_uploader("Manevra Verisi Yükle (CSV)", type="csv")

if up_file:
    df_raw = pd.read_csv(up_file)
    eng = MMGFeatureEngineer(MASS, I_ZZ, X_G)
    df_p = eng.process_turn_zigzag(df_raw)
    
    
    st.success("Veri başarıyla işlendi.")

    t1, t2, t3 = st.tabs(["🚀 Katsayı Tahminleri ve Performans", "📈 PINN Eğitim Analizi", "Proje Kapsamındaki MMG Katsayılarının Anlamları ve İşlevleri"])

    with t1:
        c1, c2, c3 = st.columns(3)
        
        # OLS ANALİZİ
        with c1:
            st.subheader("İstatistiksel (OLS)")
            X_feats = {'X': ['speed_u', 'u_u', 'delta_delta', 'n_sq'], 
                       'Y': ['speed_v', 'r_rad', 'delta_rad'], 
                       'N': ['speed_v', 'r_rad', 'r_abs_r', 'delta_rad', 'n_sq']}
            ols_coeffs = {}
            ols_metrics = {}
            for ax, feat in X_feats.items():
                reg = LinearRegression(fit_intercept=False).fit(df_p[feat], df_p[f'{ax}_meas'])
                preds = reg.predict(df_p[feat])
                for f, val in zip(feat, reg.coef_): ols_coeffs[f"{ax}_{f}"] = val
                ols_metrics[ax] = {"RMSE": np.sqrt(mean_squared_error(df_p[f'{ax}_meas'], preds)), 
                                   "MAE": mean_absolute_error(df_p[f'{ax}_meas'], preds)}
            
            st.write("**Hata Metrikleri**")
            st.dataframe(pd.DataFrame(ols_metrics).T)
            st.write("**Tahmin Edilen Katsayılar**")
            st.dataframe(pd.Series(ols_coeffs, name="Değer"))

        # SVR ANALİZİ
        with c2:
            st.subheader("Makine Öğrenmesi (SVR)")
            svr_coeffs = {}
            svr_metrics = {}
            for ax, feat in X_feats.items():
                scaler = StandardScaler(with_mean=False)
                X_s = scaler.fit_transform(df_p[feat])
                y = df_p[f'{ax}_meas']
                reg = LinearSVR(fit_intercept=False, C=svr_c, dual=True, max_iter=20000).fit(X_s, y)
                real_c = reg.coef_ / scaler.scale_
                for f, val in zip(feat, real_c): svr_coeffs[f"{ax}_{f}"] = val
                preds = reg.predict(X_s)
                svr_metrics[ax] = {"RMSE": np.sqrt(mean_squared_error(y, preds)), "MAE": mean_absolute_error(y, preds)}
            
            st.write("**Hata Metrikleri**")
            st.dataframe(pd.DataFrame(svr_metrics).T)
            st.write("**Tahmin Edilen Katsayılar**")
            st.dataframe(pd.Series(svr_coeffs, name="Değer"))

        # PINN ANALİZİ
        with c3:
            st.subheader("Fizik Bilgili (PINN)")
            if st.button("PINN Modelini Çalıştır"):
                with st.spinner("Optimizasyon devam ediyor..."):
                    model_p, hist = train_pinn(MMG_PINN(), df_p, p_epochs, p_lr)
                    st.session_state.p_model, st.session_state.p_hist = model_p, hist
                
                eval_p = evaluate_pinn(model_p, df_p)
                p_metrics = {"X": {"RMSE": eval_p["RMSE_X"], "MAE": eval_p["MAE_X"]},
                             "Y": {"RMSE": eval_p["RMSE_Y"], "MAE": eval_p["MAE_Y"]},
                             "N": {"RMSE": eval_p["RMSE_N"], "MAE": eval_p["MAE_N"]}}
                
                st.write("**Hata Metrikleri**")
                st.dataframe(pd.DataFrame(p_metrics).T)
                
                p_coeffs = {n: p.item() for n, p in model_p.named_parameters()}
                st.write("**Tahmin Edilen Katsayılar**")
                st.dataframe(pd.Series(p_coeffs, name="Değer"))

    with t2:
        if 'p_hist' in st.session_state:
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].plot(st.session_state.p_hist['loss'], color='orange')
            ax[0].set_title("Eğitim Kaybı (Total Loss)"); ax[0].set_yscale('log')
            ax[1].plot(st.session_state.p_hist['nr'], label='N_r (Sönümleme)')
            ax[1].plot(st.session_state.p_hist['xu'], label='X_u (Direnç)')
            ax[1].set_title("Fiziksel Parametrelerin Evrimi"); ax[1].legend()
            st.pyplot(fig)
        else:
            st.info("PINN analizini başlatmak için 'Tahminler' sekmesindeki butona tıklayın.")
    
    with t3:
        st.info("Bu katsayılar, geminin hidrodinamik davranışını belirleyen fiziksel parametrelerdir.")
    
        col1, col2, col3 = st.columns(3)
    
        with col1:
            st.subheader("🚀 Surge (İleri)")
            st.markdown("""
            * **$X_u$**: Düşük hız direnci.
            * **$X_{uu}$**: **Ana Direnç.** Geminin hız limitini belirleyen temel su sürtünmesi.
            * **$X_{\delta\delta}$**: Manevra sırasındaki hız kaybı.
            * **$X_P$**: Pervanenin ileri itme gücü.""")

        with col2:
            st.subheader("↔️ Sway (Yan)")
            st.markdown("""
            * **$Y_v$**: Yana kaymaya karşı su direnci.
            * **$Y_r$**: Dönüş sırasında oluşan merkezkaç savrulması.
            * **$Y_{\delta}$**: Dümenin gemiyi yana itme kuvveti.
            """)

        with col3:
            st.subheader("🔄 Yaw (Dönüş)")
            st.markdown("""
            * **$N_v$**: Rota stabilitesi (Düz gitme eğilimi).
            * **$N_r$**: Dönüşü dengeleyen sönümleme.
            * **$N_{rr}$**: **Sert Manevra Freni.** Aşırı hızlı dönmeyi engelleyen direnç.
            * **$N_{\delta}$**: Dümenin döndürme kapasitesi.
            * **$N_P$**: Pervane kaynaklı yan çekme.
            """)