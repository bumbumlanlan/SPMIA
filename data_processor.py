import pandas as pd
from sklearn.preprocessing import LabelEncoder

def read_excel_file(file_path, sheet_name=0):
    """Read data from Excel file."""
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def preprocess_data(df):
    """Perform data preprocessing."""
    # Remove duplicate rows
    df = df.drop_duplicates()
    print(f"After removing duplicates: {df.shape[0]} rows")
    
    # Handle missing values
    # Option 1: Drop rows with any missing values
    # df = df.dropna()
    
    # Option 2: Fill numeric columns with mean, categorical with mode
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown')
    
    # Strip whitespace from string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    
    # Convert column names to lowercase and replace spaces with underscores
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    print(f"Preprocessing complete: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def rename_columns(df, mapping_output_path="column_mapping.csv"):
    """Rename columns to F0, F1, F2... and save mapping to file."""
    original_columns = df.columns.tolist()
    new_columns = [f"F{i}" for i in range(len(original_columns))]
    
    # Create and save mapping
    mapping_df = pd.DataFrame({
        'original_name': original_columns,
        'new_name': new_columns
    })
    mapping_df.to_csv(mapping_output_path, index=False)
    print(f"Column mapping saved to: {mapping_output_path}")
    
    # Rename columns
    df.columns = new_columns
    print(f"Columns renamed: {list(df.columns)}")
    return df

def label_encode_columns(df):
    """Label encode all non-numeric columns."""
    label_encoders = {}
    
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"Label encoded: {col} -> {len(le.classes_)} unique values")
    
    return df, label_encoders

def save_to_csv(df, output_path):
    """Save DataFrame to CSV file."""
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Data saved to: {output_path}")

def main():
    # Configuration
    input_file = "source.xlsx"
    output_file = "data.csv"
    mapping_file = "column_mapping.csv"
    sheet_name = 0
    
    # Process data
    df = read_excel_file(input_file, sheet_name)
    df = preprocess_data(df)
    
    # Rename columns to F0, F1, F2... and save mapping
    df = rename_columns(df, mapping_file)
    
    # Label encode non-numeric columns
    df, encoders = label_encode_columns(df)
    
    save_to_csv(df, output_file)

if __name__ == "__main__":
    main()
