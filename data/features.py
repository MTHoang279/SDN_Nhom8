import pandas as pd

df = pd.read_csv('/home/hoang/Downloads/02-21-2018.csv')

for col in ['Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkts/s', 'Bwd Pkts/s',
            'Flow Duration', 'Protocol', 'Dst Port']:
    df[col] = df[col].astype(str).str.replace(',', '').str.replace('.', '', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
                       'Fwd Pkts/s', 'Bwd Pkts/s', 'Flow Duration', 'Protocol', 'Dst Port'])

df['packet_count'] = df['Tot Fwd Pkts'] + df['Tot Bwd Pkts']
df['byte_count'] = df['TotLen Fwd Pkts'] + df['TotLen Bwd Pkts']
df['packet_count_per_second'] = df['Fwd Pkts/s'] + df['Bwd Pkts/s']

# df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
# df = df.dropna(subset=['Timestamp'])  # bỏ dòng không convert được
# df['Timestamp'] = df['Timestamp'].astype('int64') / 10**9
# df['Timestamp'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).astype(int)

label_mapping = {
    'Benign': 0,
    'DDOS attack-LOIC-UDP': 1,
    'DDOS attack-HOIC': 1
}
df['Label'] = df['Label'].map(label_mapping)

# Chọn các feature cuối cùng
selected_features = [
    'Timestamp', 'Dst Port', 'Protocol', 'Flow Duration', 'packet_count', 'byte_count', 
    'packet_count_per_second', 'Label'
]

# selected_features = [
#     'Dst Port', 'Protocol', 'Flow Duration', 'packet_count', 'byte_count', 
#     'packet_count_per_second', 'Label'
# ]
df_selected = df[selected_features]

# Lưu lại file
df_selected.to_csv('processed_data.csv', index=False)
print(df_selected.head())
