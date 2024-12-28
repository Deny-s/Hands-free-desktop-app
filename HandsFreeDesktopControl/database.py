# database.py

import sqlite3
import shutil
import os

class Database:
    def __init__(self):
        self.db_file = 'settings.db'
        self.connection = sqlite3.connect(self.db_file)
        self.cursor = self.connection.cursor()
        self.create_tables()

    def create_tables(self):
        # Create tables if they don't exist
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS settings (
                id INTEGER PRIMARY KEY,
                head_sensitivity INTEGER,
                blink_sensitivity TEXT
            )
        ''')
        self.connection.commit()

    def save_settings(self, head_sensitivity, blink_sensitivity):
        # Save settings to the database
        self.cursor.execute('''
            INSERT INTO settings (head_sensitivity, blink_sensitivity)
            VALUES (?, ?)
        ''', (head_sensitivity, blink_sensitivity))
        self.connection.commit()

    def load_settings(self):
        # Load settings from the database
        self.cursor.execute('SELECT * FROM settings ORDER BY id DESC LIMIT 1')
        return self.cursor.fetchone()

    def reset_database(self):
        # Reset database by deleting settings
        self.cursor.execute('DELETE FROM settings')
        self.connection.commit()

    def backup_database(self):
        # Backup the database file
        shutil.copy(self.db_file, 'settings_backup.db')

    def restore_database(self):
        # Restore the database file
        if os.path.exists('settings_backup.db'):
            shutil.copy('settings_backup.db', self.db_file)
            self.connection = sqlite3.connect(self.db_file)
            self.cursor = self.connection.cursor()

    def close(self):
        self.connection.close()
