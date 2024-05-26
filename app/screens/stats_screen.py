from kivy.uix.screenmanager import Screen
from kivy.lang import Builder

from kivy.uix.label import Label
from app.components.popups.popups import SaveConfirmationPopup

kv_string = """
<StatsScreen>:
    GridLayout:
        id: stats_grid
        size_hint_y: 0.8
        pos_hint: {'top':1}
        padding: 20
        cols: 2
    GridLayout:
        cols: 1
        padding: 20
        size_hint_y: 0.2  
        Button:
            text: 'Back to Main Menu'
            on_release: app.root.current = 'main_menu'
            canvas.before:
                Color:
                    rgba: 1, 0, 0, 1
                Line:
                    width: 2.
                    rectangle: self.x, self.y, self.width, self.height
"""

Builder.load_string(kv_string)


class StatsScreen(Screen):
    def __init__(self, **kwargs):
        super(StatsScreen, self).__init__(**kwargs)
        self.stats = None

    def on_enter(self):
        stats_dict = self.get_stats()
        for key, value in stats_dict.items():
            self.ids.stats_grid.add_widget(
                Label(text=str(key), color=(0, 0, 0, 1))
            )
            self.ids.stats_grid.add_widget(
                Label(text=str(value), color=(0, 0, 0, 1))
            )

    def get_stats(self):
        print(self.stats)
        # Return a dictionary of stats for demonstration
        return {"Stat 1": 100, "Stat 2": 200, "Stat 3": 300}

    def confirm_exit(self):
        exit_confirmation_popup = SaveConfirmationPopup(
            save_function=self.save
        )
        exit_confirmation_popup.open()
        return True

    def save(self):
        project = self.manager.get_screen("main_menu").project
        project.save(self.save_path)
