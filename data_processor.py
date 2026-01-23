import pandas as pd
from sklearn.preprocessing import LabelEncoder

def read_excel_file(file_path, sheet_name=0):
    """Read data from Excel file."""
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def preprocess_data(df):
    """Perform data preprocessing."""

    # Handle missing values
    df = df.dropna()

    # Strip whitespace from string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()

    # Convert column names to lowercase and replace spaces with underscores
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    print(f"Preprocessing complete: {df.shape[0]} rows, {df.shape[1]} columns")
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
    sheet_name = 0

    # Process data
    df = read_excel_file(input_file, sheet_name)
    df = preprocess_data(df)

    # Label encode non-numeric columns
    df, encoders = label_encode_columns(df)
    
    save_to_csv(df, output_file)

if __name__ == "__main__":
    main()
