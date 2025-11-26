import pandas as pd
import os
import yaml

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    
    source_path = config['data']['source_path']
    output_dir = config['data']['output_dir']
    countries = config['countries']
    
    print(f"Loading data from {source_path}...")
    # Read CSV - using low_memory=False to avoid mixed type warnings on large files
    df_raw = pd.read_csv(source_path, parse_dates=['utc_timestamp'], low_memory=False)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for cc in countries:
        print(f"Processing {cc}...")
        
        # Identify columns
        load_col = f"{cc}_load_actual_entsoe_transparency"
        
        # Try to find wind/solar columns
        # Logic: Prefer total generation, fallback to onshore if total not available
        wind_col = f"{cc}_wind_generation_actual"
        if wind_col not in df_raw.columns:
            wind_col = f"{cc}_wind_onshore_generation_actual"
            
        solar_col = f"{cc}_solar_generation_actual"
        
        cols_to_keep = ['utc_timestamp', load_col]
        rename_map = {'utc_timestamp': 'timestamp', load_col: 'load'}
        
        if wind_col in df_raw.columns:
            cols_to_keep.append(wind_col)
            rename_map[wind_col] = 'wind'
            
        if solar_col in df_raw.columns:
            cols_to_keep.append(solar_col)
            rename_map[solar_col] = 'solar'
            
        # Extract and rename
        df_cc = df_raw[cols_to_keep].copy()
        df_cc.rename(columns=rename_map, inplace=True)
        
        # Drop rows with missing load
        initial_len = len(df_cc)
        df_cc.dropna(subset=['load'], inplace=True)
        dropped_len = initial_len - len(df_cc)
        if dropped_len > 0:
            print(f"  Dropped {dropped_len} rows with missing load.")
            
        # Sort by timestamp
        df_cc.sort_values('timestamp', inplace=True)
        
        # Save to CSV
        output_file = os.path.join(output_dir, f"{cc}_cleaned.csv")
        df_cc.to_csv(output_file, index=False)
        print(f"  Saved to {output_file}")
        
        # Basic sanity check (print head/tail)
        print(f"  Range: {df_cc['timestamp'].min()} to {df_cc['timestamp'].max()}")
        print(f"  Shape: {df_cc.shape}")

if __name__ == "__main__":
    main()
