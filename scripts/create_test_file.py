import pandas as pd
import os

# Create a simple dataframe with test data
data = {
    'Name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown'],
    'Department': ['Engineering', 'Marketing', 'Sales', 'Human Resources'],
    'Position': ['Software Engineer', 'Marketing Manager', 'Sales Representative', 'HR Specialist'],
    'Start Date': ['2020-01-15', '2019-03-22', '2021-07-10', '2018-11-05'],
    'Salary': [85000, 78000, 65000, 72000]
}

df = pd.DataFrame(data)

# Save to Excel file
output_path = 'test_employees.xlsx'
df.to_excel(output_path, index=False)

print(f"Created test file at: {os.path.abspath(output_path)}")