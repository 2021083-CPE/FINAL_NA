import mysql.connector

# Replace with your MySQL server details
host = 'localhost'       # MySQL server host
user = 'detect'       # MySQL username
password = '121602'  # MySQL password

# Create a connection to the MySQL server
connection = mysql.connector.connect(
    host=host,
    user=user,
    password=password,
)

# Create a cursor object to execute SQL queries
cursor = connection.cursor()

# Define the database name
database_name = 'objectdetection'  # Replace with your desired database name

# Create the database if it doesn't exist
cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name}")

# Select the newly created database
cursor.execute(f"USE {database_name}")

# Define the SQL code to create the 'detections' table
create_table_sql = """
CREATE TABLE IF NOT EXISTS detections (
    id INT AUTO_INCREMENT PRIMARY KEY,
    class VARCHAR(255) NOT NULL,
    x_min FLOAT NOT NULL,
    y_min FLOAT NOT NULL,
    x_max FLOAT NOT NULL,
    y_max FLOAT NOT NULL
)
"""

# Create the 'detections' table
cursor.execute(create_table_sql)

# Commit the changes and close the connection
connection.commit()
connection.close()

print("Database and table created successfully.")
