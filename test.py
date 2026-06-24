import os
import joblib
import pandas as pd
import numpy as np
import warnings
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

# Standard enterprise VPN protocol ports
KNOWN_VPN_PORTS = {1194, 51820, 4500, 500, 1723, 4433, 1195}


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
        source=r"\Device\NPF_{8D1F9809-EA3D-49F4-8D8A-D81291090BEB}",
        promiscuous_mode=True,
        statistical_analysis=True,
        udps=FeatureReconstructor()
    )

    print("\n--- Step 3: Precise Live Real-Time Traffic Detection ---")
    print(f"{'SRC IP : PORT':<22} -> {'DST IP : PORT':<22} | {'AI CLASSIFICATION':<10}")
    print("-" * 65)

    # Per-port hit counter — used only for direct known-VPN-port override
    active_ports_count = {}

    try:
        for flow in streamer:
            print(
            f"DEBUG: {flow.src_ip}:{flow.src_port} -> "
            f"{flow.dst_ip}:{flow.dst_port} | "
            f"Packets={flow.bidirectional_packets}"
            )
            # 1. Filter background multicast and discovery chatter
            # if (flow.src_port == 0 or flow.dst_port == 0 or
            #         flow.src_port == 5353 or flow.dst_port == 5353 or
            #         flow.dst_ip.startswith("224.") or
            #         flow.dst_ip.startswith("239.") or
            #         flow.dst_ip.startswith("ff02") or
            #         flow.src_ip.startswith("fe80") or
            #         flow.dst_ip.startswith("fe80") or
            #         flow.dst_ip == "255.255.255.255" or
            #         flow.bidirectional_packets < 4):
            #     continue

            # Track per-port counts (informational only)
            for port in [flow.src_port, flow.dst_port]:
                if port >= 1024:
                    active_ports_count[port] = active_ports_count.get(port, 0) + 1

            # 2. Convert duration: NFStream gives ms, training data uses µs
            duration_us = flow.bidirectional_duration_ms * 1000.0
            duration_sec = flow.bidirectional_duration_ms / 1000.0 if flow.bidirectional_duration_ms > 0 else 0.001

            # IAT lists: NFStream packet.time is in seconds → convert to µs to match training data
            f_iats = [t * 1_000_000.0 for t in flow.udps.fiat_list] if flow.udps.fiat_list else [0.0]
            b_iats = [t * 1_000_000.0 for t in flow.udps.biat_list] if flow.udps.biat_list else [0.0]

            feature_row = [
                duration_us,                                     # duration        (µs)
                sum(f_iats),                                     # total_fiat       (µs)
                sum(b_iats),                                     # total_biat       (µs)
                min(f_iats),                                     # min_fiat         (µs)
                min(b_iats),                                     # min_biat         (µs)
                max(f_iats),                                     # max_fiat         (µs)
                max(b_iats),                                     # max_biat         (µs)
                np.mean(f_iats),                                 # mean_fiat        (µs)
                np.mean(b_iats),                                 # mean_biat        (µs)
                flow.bidirectional_packets / duration_sec,       # flowPktsPerSecond
                flow.bidirectional_bytes / duration_sec,         # flowBytesPerSecond
                flow.bidirectional_min_piat_ms * 1000.0,         # min_flowiat      (µs)
                flow.bidirectional_max_piat_ms * 1000.0,         # max_flowiat      (µs)
                flow.bidirectional_mean_piat_ms * 1000.0,        # mean_flowiat     (µs)
                flow.bidirectional_stddev_piat_ms * 1000.0,      # std_flowiat      (µs)
                # active/idle: training data uses -1 as sentinel when not measured
                # Live single-pass flows don't have reliable active/idle periods,
                # so we use -1 to match the training distribution
                -1.0, 0.0, -1.0, 0.0,                           # min/mean/max/std_active
                -1.0, 0.0, -1.0, 0.0,                           # min/mean/max/std_idle
            ]

            X_live = np.array([feature_row])

            # 3. ML prediction
            pred_numeric = model.predict(X_live)
            pred_label = label_encoder.inverse_transform(pred_numeric)[0]

            # 4. Hard override only for flows directly on known VPN protocol ports
            #    (does NOT carry over to unrelated flows — no sticky global state)
            if flow.src_port in KNOWN_VPN_PORTS or flow.dst_port in KNOWN_VPN_PORTS:
                pred_label = "VPN"

            src_info = f"{flow.src_ip}:{flow.src_port}"
            dst_info = f"{flow.dst_ip}:{flow.dst_port}"
            alert_flag = f"*** {pred_label} ***" if pred_label == "VPN" else pred_label

            print(f"{src_info:<22} -> {dst_info:<22} | {alert_flag:<10}")

    except KeyboardInterrupt:
        print("\n[INFO] Live traffic engine terminated smoothly.")


if __name__ == '__main__':
    main() 