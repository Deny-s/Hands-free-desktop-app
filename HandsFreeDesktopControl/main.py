# main.py

import sys
from ui import HandsFreeApp
from PyQt5.QtWidgets import QApplication

def main():
    app = QApplication(sys.argv)
    window = HandsFreeApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
