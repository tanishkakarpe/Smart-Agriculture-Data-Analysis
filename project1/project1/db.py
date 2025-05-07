import mysql.connector

def connect_db():
    """Connects to the MySQL database and returns the connection."""
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='',  # <- Replace with your MySQL password
        database='agriculture'
    )

def fetch_crop_data():
    """Fetches crop data from the database."""
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM crop_table")
    result = cursor.fetchall()
    conn.close()
    return result

def insert_crop_data(ID,N, P, K, temperature, humidity, ph, rainfall, crop_name, cluster_label):
    """Inserts new crop prediction into the database."""
    conn = connect_db()
    cursor = conn.cursor()
    query = """
        INSERT INTO crop_table (ID,N, P, K, Temperature, Humidity, ph, Rainfall, crop_name, Cluster_Label)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(query, (ID,N, P, K, temperature, humidity, ph, rainfall, crop_name, cluster_label))
    conn.commit()
    conn.close()
