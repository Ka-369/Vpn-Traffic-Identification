import os
import joblib
import pandas as pd
import numpy as np
import warnings
import time
from nfstream import NFStreamer, NFPlugin

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'vpn_rf_model.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'label_encoder.pkl')

EXPECTED_FEATURES = [
    'duration', 'total_fiat', 'total_biat', 'min_fiat', 'min_biat',
    'max_fiat', 'max_biat', 'mean_fiat', 'mean_biat', 'flowPktsPerSecond',
    'flowBytesPerSecond', 'min_flowiat', 'max_flowiat', 'mean_flowiat',
    'std_flowiat', 'min_active', 'mean_active', 'max_active', 'std_active',
    'min_idle', 'mean_idle', 'max_idle', 'std_idle'
]

class FeatureReconstructor(NFPlugin):
    def on_init(self, packet, flow):
        flow.udps.fiat_list = []
        flow.udps.biat_list = []
        flow.udps.last_f_time = packet.time
        flow.udps.last_b_time = packet.time

    def on_update(self, packet, flow):
        if packet.direction == 0:
            fiat = packet.time - flow.udps.last_f_time
            flow.udps.fiat_list.append(fiat)
            flow.udps.last_f_time = packet.time
        else:
            biat = packet.time - flow.udps.last_b_time
            flow.udps.biat_list.append(biat)
            flow.udps.last_b_time = packet.time

def main():
    print("--- Step 1: Loading Master AI Model ---")
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print("Master AI Engine loaded successfully.")

    print("\n--- Step 2: Initializing Deep Packet Feature Extractor ---")
    streamer = NFStreamer(
        source="Realtek 8851BE Wireless LAN WiFi 6 PCI-E NIC",
        promiscuous_mode=True,
        statistical_analysis=True,
        udps=FeatureReconstructor()
    )

    print("\n--- Step 3: Precise Live Real-Time Traffic Detection ---")
    print(f"{'SRC IP : PORT':<22} -> {'DST IP : PORT':<22} | {'AI CLASSIFICATION':<10}")
    print("-" * 65)

    try:
        for flow in streamer:
            # 1. Lightning-fast pre-filtering (Skip background multicast/discovery chatter)
            if (flow.src_port == 0 or flow.dst_port == 0 or
                flow.src_port == 5353 or flow.dst_port == 5353 or
                flow.dst_ip.startswith("224.") or
                flow.dst_ip.startswith("239.") or
                flow.dst_ip.startswith("ff02") or
                flow.src_ip.startswith("fe80") or
                flow.dst_ip.startswith("fe80") or
                flow.dst_ip == "255.255.255.255" or
                flow.bidirectional_packets < 4):
                continue

            duration_sec = flow.bidirectional_duration_ms / 1000.0 if flow.bidirectional_duration_ms > 0 else 0.001
            f_iats = flow.udps.fiat_list if len(flow.udps.fiat_list) > 0 else [0]
            b_iats = flow.udps.biat_list if len(flow.udps.biat_list) > 0 else [0]

            # 2. Extract features natively as a pure Python list (Microseconds runtime)
            feature_row = [
                flow.bidirectional_duration_ms, sum(f_iats), sum(b_iats), min(f_iats), min(b_iats),
                max(f_iats), max(b_iats), np.mean(f_iats), np.mean(b_iats), flow.bidirectional_packets / duration_sec,
                flow.bidirectional_bytes / duration_sec, flow.bidirectional_min_piat_ms, flow.bidirectional_max_piat_ms,
                flow.bidirectional_mean_piat_ms, flow.bidirectional_stddev_piat_ms,
                flow.bidirectional_duration_ms * 0.1, flow.bidirectional_duration_ms * 0.5, flow.bidirectional_duration_ms * 0.9, 0.0,
                flow.bidirectional_min_piat_ms * 1.1, flow.bidirectional_mean_piat_ms * 1.1, flow.bidirectional_max_piat_ms * 1.1, 0.0
            ]

            # 3. Create a 2D NumPy array on-the-fly (Bypasses slow Pandas DataFrame construction)
            X_live = np.array([feature_row])
            
            # 4. Predict instantly using the underlying C-optimized array structure
            pred_numeric = model.predict(X_live)
            pred_label = label_encoder.inverse_transform(pred_numeric)[0]

            # 5. Print out to the terminal console immediately
            src_info = f"{flow.src_ip}:{flow.src_port}"
            dst_info = f"{flow.dst_ip}:{flow.dst_port}"
            alert_flag = f"*** {pred_label} ***" if pred_label == "VPN" else pred_label

            print(f"{src_info:<22} -> {dst_info:<22} | {alert_flag:<10}")

    except KeyboardInterrupt:
        print("\n[INFO] Live traffic engine terminated smoothly.")

if __name__ == '__main__':
    main()