import tkinter as tk
from ui.ui_main import AquaponicsUI


def main():
    root = tk.Tk()
    app = AquaponicsUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()